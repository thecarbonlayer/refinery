"""Cluster G — incomplete answers, repeated compaction, and workspace inheritance."""

from __future__ import annotations

from runner.carbon_env import make_provider
from runner.helpers import agent_metrics, neutral_dir, tool_texts
from runner.spec import Attempt, TaskSpec

G1_SENTINEL = "END-OF-HANDOFF-G1-8VR"
G2_FACT_A = "EARLY-DECISION-G2-3LK"
G2_FACT_B = "LATE-DECISION-G2-6QW"
G3_SENTINEL = "WORKTREE-STATE-G3-5TZ"


def _plain_agent(**kwargs):
    from harness.agent import DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer

    provider = make_provider()
    kwargs.setdefault("tracer", Tracer(model=provider.model))
    return Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        agents_dir=neutral_dir(),
        **kwargs,
    )


def run_g1() -> Attempt:
    a = _plain_agent()
    reply = a.send(
        "Produce a plain-text incident handoff with exactly 400 numbered lines. "
        "Each line must use the form `N: system stable`, starting at 1. After line "
        f"400, put {G1_SENTINEL} on its own final line. Do not use the sentinel earlier."
    )
    numbered = sum(
        1
        for line in reply.splitlines()
        if line.partition(":")[0].strip().isdigit() and "system stable" in line
    )
    sentinel_once = reply.count(G1_SENTINEL) == 1
    sentinel_last = reply.rstrip().endswith(G1_SENTINEL)
    ok = numbered == 400 and sentinel_once and sentinel_last
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"numbered_lines={numbered} sentinel_once={sentinel_once} "
        f"sentinel_last={sentinel_last} reply_chars={len(reply)}",
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def _filler(i: int) -> str:
    notes = " ".join(
        f"checkpoint {i}.{j} nominal, latency stable, no owner change, review pending."
        for j in range(10)
    )
    return f"Operational notes: {notes} Acknowledge in one short sentence."


def run_g2() -> Attempt:
    a = _plain_agent(context_limit=700)
    a.send("We are beginning a release review. Acknowledge briefly.")
    a.send(f"Record this exact early decision for later: {G2_FACT_A}. Acknowledge briefly.")
    compactions = int(a.just_compacted)
    a.send(_filler(1))
    compactions += int(a.just_compacted)
    a.send(f"Record this exact later decision for later: {G2_FACT_B}. Acknowledge briefly.")
    compactions += int(a.just_compacted)
    for i in range(2, 14):
        a.send(_filler(i))
        compactions += int(a.just_compacted)
        if compactions >= 2:
            break
    if compactions < 2:
        return Attempt(
            False,
            "error",
            f"repeated-compaction setup did not fire twice (count={compactions})",
            turns=len(a.messages),
            metrics=agent_metrics(a),
        )
    reply = a.send("Reply with the two exact release decision codes, early first and late second.")
    has_a = G2_FACT_A.lower() in reply.lower()
    has_b = G2_FACT_B.lower() in reply.lower()
    ok = has_a and has_b
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compactions={compactions} early_recalled={has_a} "
        f"late_recalled={has_b} reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def run_g3() -> Attempt:
    """A delegated worker must inspect the same workspace as its parent."""
    from harness.subagents import delegate_tool
    from harness.tools import ToolRegistry, read_file_tool
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("worker-state.txt", f"current-state={G3_SENTINEL}\n")
    provider = make_provider()
    worker_tools = ToolRegistry()
    worker_tools.register(read_file_tool(str(ws.root)))
    tools = ToolRegistry()
    # Desired contract: the parent can bind the worker's tools to the same
    # workspace. Current carbon lacks this delegate_tool(tools=...) seam, so the
    # baseline records an error. Once the seam exists, the exact same task
    # proves that the worker reads the intended snapshot.
    tools.register(delegate_tool(model=provider.model, tools=worker_tools))
    a = _plain_agent(tools=tools)
    reply = a.send(
        "Delegate this exact subtask to a worker: read worker-state.txt and return its "
        "current-state value exactly. Then reply with just that value."
    )
    delegated_results = tool_texts(a.messages)
    worker_saw_state = any(G3_SENTINEL in text for text in delegated_results)
    recalled = G3_SENTINEL.lower() in reply.lower()
    ok = worker_saw_state and recalled
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"delegate_called={bool(delegated_results)} worker_saw_state={worker_saw_state} "
        f"reply_recalled={recalled} reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


SPECS = [
    TaskSpec("G1", "held_in", "G", "pass", run_g1),
    TaskSpec("G2", "held_out", "G", "uncertain", run_g2),
    TaskSpec("G3", "held_in", "G", "pass", run_g3),
]
