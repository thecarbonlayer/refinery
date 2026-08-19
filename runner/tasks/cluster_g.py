"""Cluster G — incomplete answers, repeated compaction, and workspace inheritance."""

from __future__ import annotations

from runner.carbon_env import make_provider
from runner.helpers import (
    agent_metrics,
    neutral_dir,
    scripted_approver,
    tool_call_args,
    tool_texts,
)
from runner.spec import Attempt, TaskSpec

G1_SENTINEL = "END-OF-HANDOFF-G1-8VR"
G2_FACT_A = "EARLY-DECISION-G2-3LK"
G2_FACT_B = "LATE-DECISION-G2-6QW"
G3_SENTINEL = "WORKTREE-STATE-G3-5TZ"

# G4 is the HELD-IN repeated-compaction miner. G2 measures the same door but is
# held-out, and mining a held-out task spends the generalization claim it exists to
# make — the coverage table named G2 a miner, which was that rule contradicting
# itself. G4 takes the mining role; G2 keeps the guarding one.
#
# Structurally different from G2 on purpose, or the "guard" is just a second run of
# the miner: G2 carries two homogeneous decision codes and asks for both at the end,
# while G4 carries three HETEROGENEOUS kinds of harness state and a deeper window.
# The kinds are chosen to be the ones a summarizer paraphrases away first, and are
# exactly what the reference harness tracks deterministically rather than entrusting
# to summary prose — modified files, a rejected approach, and the pending next action.
G4_FILES = ("services/ledger/reconcile.py", "services/ledger/schema_v3.sql")
G4_REJECTED = "REJECTED-APPROACH-G4-2WD"
G4_NEXT = "NEXT-ACTION-G4-5HB"

# G5's files are named, not sentinel-coded: the task asks the agent what it DID, and a
# realistic path is what a real session would carry. Distinct from G4's paths so a
# checkpoint carrying one cannot be mistaken for carrying the other.
G5_FILES = ("reconcile_flags.py", "audit_flags.py")


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
    try:
        result = a.run(
            "Produce a plain-text incident handoff with exactly 400 numbered lines. "
            "Each line must use the form `N: system stable`, starting at 1. After line "
            f"400, put {G1_SENTINEL} on its own final line. Do not use the sentinel earlier."
        )
        reply = result.text
        numbered = sum(
            1
            for line in reply.splitlines()
            if line.partition(":")[0].strip().isdigit() and "system stable" in line
        )
        sentinel_once = reply.count(G1_SENTINEL) == 1
        sentinel_last = reply.rstrip().endswith(G1_SENTINEL)
        ok = numbered == 400 and sentinel_once and sentinel_last
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"numbered_lines={numbered} sentinel_once={sentinel_once} "
        f"sentinel_last={sentinel_last} reply_chars={len(reply)}",
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


def _filler(i: int) -> str:
    notes = " ".join(
        f"checkpoint {i}.{j} nominal, latency stable, no owner change, review pending."
        for j in range(10)
    )
    return f"Operational notes: {notes} Acknowledge in one short sentence."


def run_g2() -> Attempt:
    a = _plain_agent(context_limit=700)
    try:
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
        result = a.run(
            "Reply with the two exact release decision codes, early first and late second."
        )
        reply = result.text
        has_a = G2_FACT_A.lower() in reply.lower()
        has_b = G2_FACT_B.lower() in reply.lower()
        ok = has_a and has_b
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compactions={compactions} early_recalled={has_a} "
        f"late_recalled={has_b} reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


def run_g4() -> Attempt:
    """Heterogeneous harness state must survive REPEATED compaction, not just facts.

    Three kinds of state go in, spread across the trajectory rather than clustered:
    the set of modified files, an approach already tried and rejected, and the pending
    next action. All three are the sort of thing a general "be terse but lose nothing"
    summarizer drops or paraphrases first — a file list becomes "some files", a
    rejected approach becomes "explored options" — and all three are load-bearing for
    the next turn.

    Graded per property, not as one boolean, so a partial retention is visible in the
    detail string and the failure says WHICH kind of state was lost.
    """
    a = _plain_agent(context_limit=900)
    try:
        a.send("We are mid-migration on the ledger service. Acknowledge briefly.")
        a.send(
            f"Record the files changed so far, exactly: {G4_FILES[0]} and {G4_FILES[1]}. "
            "Acknowledge briefly."
        )
        compactions = int(a.just_compacted)
        for i in range(1, 4):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
        a.send(
            f"Record that this approach was tried and rejected: {G4_REJECTED} "
            "(it deadlocked under concurrent writes). Acknowledge briefly."
        )
        compactions += int(a.just_compacted)
        for i in range(4, 8):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
        a.send(f"Record the pending next action, exactly: {G4_NEXT}. Acknowledge briefly.")
        compactions += int(a.just_compacted)
        for i in range(8, 18):
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
        result = a.run(
            "Report three things exactly as recorded: the files changed so far, the "
            "approach that was tried and rejected, and the pending next action."
        )
        reply = result.text
        low = reply.lower()
        has_files = all(f.lower() in low for f in G4_FILES)
        has_rejected = G4_REJECTED.lower() in low
        has_next = G4_NEXT.lower() in low
        ok = has_files and has_rejected and has_next
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compactions={compactions} files_recalled={has_files} "
        f"rejected_recalled={has_rejected} next_action_recalled={has_next} "
        f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


def run_g5() -> Attempt:
    """Which files did I actually change? — asked after repeated compaction.

    G4's sibling, and deliberately not a duplicate of it. G4 STATES its facts in
    conversation, so what it measures is whether a summarizer keeps a stated fact. G5
    never states anything: the agent calls ``write_file`` itself, and the file list
    exists only as a property of the tool calls it made. That is the difference between
    "did the summary remember" and "does the harness know", and only the second is
    something the harness can guarantee rather than hope for.

    This is the observer for deterministic file tracking. A strategy that extracts the
    file list from tool calls and re-attaches it verbatim passes here even when the
    summarizer's prose drops it; a strategy that entrusts the list to that prose does
    not. Measured on ``write_file`` because an edit needs matching text to already
    exist, which adds a failure mode that has nothing to do with compaction.

    Setup failures are reported as ``error``, never ``fail``: if the model never made
    the writes, the task never reached the thing it measures.
    """
    from harness.agent import APPROVAL_TOOLS
    from harness.tools import ToolRegistry
    from harness.workspace import Workspace, write_file_tool

    ws = Workspace()
    approvals = [{"tool": "write_file", "decision": "approve"}] * 12
    a = _plain_agent(
        context_limit=900,
        approve=scripted_approver(approvals),
        approval_required=APPROVAL_TOOLS,
    )
    tools = ToolRegistry()
    tools.register(write_file_tool(ws))
    a.tools = tools

    observed: list[str] = []

    def _write(path: str, line: str) -> int:
        """Ask for one file, re-prompting once if the model answered instead of acting.

        A local model sometimes replies "I have created it" without calling anything.
        One retry keeps that from being recorded as a compaction failure — the task
        measures what survives the door, and it cannot measure it if the writes that
        produce the state never happened. Still bounded: two asks, then the setup
        guard reports `error`.

        The observation is recorded HERE, as the call happens, and never re-derived
        from the final transcript. That is not a style preference: the first version
        scanned `a.messages` at the end and saw only the second file, because
        compaction had already summarized the first write's tool call away. Reading
        setup state out of a compacted transcript measures the door, not the setup —
        and it is the same blindness this task exists to expose.
        """
        fired = 0
        for _ in range(2):
            before = len(tool_call_args(a.messages, ("write_file",)))
            a.send(
                f"Create a file named {path} containing exactly the line: {line}. "
                "Use the write_file tool. Do not describe the file — write it."
            )
            fired += int(a.just_compacted)
            new = tool_call_args(a.messages, ("write_file",))[before:]
            if any(path in args for args in new):
                observed.append(path)
                break
        return fired

    try:
        compactions = _write(G5_FILES[0], "reconcile = True")
        for i in range(1, 4):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
        compactions += _write(G5_FILES[1], "audit = True")
        for i in range(4, 18):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
            if compactions >= 2:
                break

        made_both = set(observed) == set(G5_FILES)
        if compactions < 2 or not made_both:
            return Attempt(
                False,
                "error",
                f"setup incomplete: compactions={compactions} observed={observed}",
                approvals=approvals,
                turns=len(a.messages),
                metrics=agent_metrics(a),
            )
        result = a.run("List every file you have created or modified in this session.")
        reply = result.text
        low = reply.lower()
        recalled = [f for f in G5_FILES if f.lower() in low]
        ok = len(recalled) == len(G5_FILES)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compactions={compactions} observed={observed} "
        f"recalled={recalled} reply={reply[:240]!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


def run_g3() -> Attempt:
    """A delegated worker must inspect the same workspace as its parent."""
    from harness.subagents import delegate_tool
    from harness.tools import ToolRegistry, read_file_tool
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("worker-state.txt", f"current-state={G3_SENTINEL}\n")
    provider = make_provider()
    a = _plain_agent()
    worker_tools = ToolRegistry()
    # scratch_root is the PARENT's own session scratch — carbon's own
    # _coding_tools.worker_tools() threads the same value for the same reason
    # (harness/agent.py): a worker asked to read back a scratch:// ref the
    # PARENT's own truncation door produced must resolve it against the scratch
    # the parent actually owns, not an empty/absent one.
    worker_tools.register(read_file_tool(str(ws.root), scratch_root=a.session_env.scratch_root))
    tools = ToolRegistry()
    # The parent binds the worker's tools to its own workspace through
    # delegate_tool(tools=...). That is a CODE seam in carbon (harness/subagents.py),
    # not part of the editable config surface — refinery measures it but cannot tune
    # it, which is the whole grader/graded distinction. The seam exists today, so the
    # prior is `pass` and this task proves the worker reads the intended snapshot
    # rather than a fresh, empty workspace of its own.
    tools.register(delegate_tool(model=provider.model, tools=worker_tools))
    a.tools = tools
    try:
        result = a.run(
            "Delegate this exact subtask to a worker: read worker-state.txt and return its "
            "current-state value exactly. Then reply with just that value."
        )
        reply = result.text
        delegated_results = tool_texts(a.messages)
        worker_saw_state = any(G3_SENTINEL in text for text in delegated_results)
        recalled = G3_SENTINEL.lower() in reply.lower()
        ok = worker_saw_state and recalled
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"delegate_called={bool(delegated_results)} worker_saw_state={worker_saw_state} "
        f"reply_recalled={recalled} reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


SPECS = [
    TaskSpec("G1", "held_in", "G", "uncertain", primitive="response", alias="RSP-1", run=run_g1),
    TaskSpec("G2", "held_out", "G", "uncertain", primitive="compaction", alias="CMP-2", run=run_g2),
    TaskSpec("G3", "held_in", "G", "pass", primitive="subagent", alias="SUB-1", run=run_g3),
    TaskSpec("G4", "held_in", "G", "uncertain", primitive="compaction", alias="CMP-3", run=run_g4),
    TaskSpec("G5", "held_in", "G", "uncertain", primitive="compaction", alias="CMP-4", run=run_g5),
]
