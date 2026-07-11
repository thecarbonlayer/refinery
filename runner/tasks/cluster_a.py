"""Cluster A — context drift / long-session degradation.

Two mechanisms, one held-out each: compaction loss (A1 -> A3) and clamp
truncation (A2 -> A4). A1's fact sits on TURN 2 deliberately — compact() keeps
keep_head=2 messages, so a turn-1 fact survives verbatim and the task would be
vacuous (task-suite-v2: "turn-2 placement is load-bearing"). Context limits are
NEVER lowered here: the default limit is itself the editable knob under test.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from runner.gemma_env import make_provider
from runner.helpers import compacted, neutral_dir
from runner.spec import Attempt, TaskSpec

A1_SENTINEL = "TUESDAY-KEY-9X"
A2_SENTINEL = "K4Q7-ROLLOUT-9Z"
A3_VALUES = {"A": "ALPHA-CODE-D4T", "B": "BRAVO-CODE-M7Q", "C": "CHARLIE-CODE-X2R"}
A4_SENTINEL = "ZQ-PASS-77KD"
A1_MAX_FILLER_TURNS = 22


def _plain_agent(**kwargs):
    from harness.agent import DEFAULT_SYSTEM, Agent

    provider = make_provider()
    return Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        agents_dir=neutral_dir(),
        **kwargs,
    )


def _filler(i: int, sentences: int = 12) -> str:
    """~350-500 tokens of neutral ops prose (short small-talk turns never push
    reported usage past the 4000-token limit — the vacuous-pass bug v1 had)."""
    parts = [
        f"Ops note {i}.{j}: service check {j} nominal, latency within budget, error rate flat, "
        "no follow-up owner assigned yet, revisit after the deploy review window closes."
        for j in range(sentences)
    ]
    return (
        f"Here is section {i} of the ops meeting notes. "
        + " ".join(parts)
        + " Acknowledge briefly in one short sentence."
    )


def run_a1() -> Attempt:
    a = _plain_agent()
    a.send("Quick sync before the deploy review. Please acknowledge briefly.")  # turn 1: neutral
    a.send(  # turn 2: the fact — first message compaction will summarize
        f"Note for later: the deploy key rotates every {A1_SENTINEL}. Acknowledge briefly."
    )
    for i in range(3, 3 + A1_MAX_FILLER_TURNS):
        a.send(_filler(i))
        if compacted(a.messages):
            break
    if not compacted(a.messages):  # setup assertion: compaction must actually fire
        return Attempt(
            False,
            "error",
            f"compaction never fired after {A1_MAX_FILLER_TURNS} filler turns "
            f"(last_tokens={a._last_tokens})",
            turns=len(a.messages),
        )
    reply = a.send("What is the deploy key rotation schedule? Reply with the exact key phrase.")
    is_compacted = compacted(a.messages)
    recalled = A1_SENTINEL.lower() in reply.lower()
    ok = recalled and is_compacted
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compacted={is_compacted} sentinel_recalled={recalled} reply={reply!r}",
        turns=len(a.messages),
    )


def _a2_log() -> str:
    """A2's oversized rollout log (needle on the FINAL line). Module-level so
    the clamp regression test exercises THIS fixture, not a drifting copy."""
    heartbeats = "\n".join(
        f"2026-07-10T02:{i % 60:02d}:{(7 * i) % 60:02d} INFO worker-{i % 8} heartbeat ok seq={i}"
        for i in range(160)
    )
    return heartbeats + f"\nFINAL: the rollout password is {A2_SENTINEL}\n"


def run_a2() -> Attempt:
    from harness.limits import MAX_ITEM_CHARS
    from harness.tools import Tool, ToolRegistry

    log = _a2_log()
    assert len(log) > MAX_ITEM_CHARS, "authoring bug: log must exceed the clamp"
    tools = ToolRegistry()
    tools.register(
        Tool(
            name="fetch_log",
            description="Fetch the full deployment log for the last rollout.",
            parameters={"type": "object", "properties": {}, "required": []},
            func=lambda: log,
        )
    )
    a = _plain_agent(tools=tools)
    reply = a.send(
        "Call the fetch_log tool and tell me what the FINAL line says the rollout "
        "password is. Reply with just the password."
    )
    ok = A2_SENTINEL.lower() in reply.lower()
    called = any(m.get("role") == "tool" for m in a.messages)
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"needle_in_reply={ok} tool_called={called} reply={reply!r}",
        turns=len(a.messages),
    )


def run_a3() -> Attempt:
    a = _plain_agent()
    a.send(
        "Let's plan the API rollout. The API rollout access value is "
        f"{A3_VALUES['A']}. Acknowledge briefly."
    )
    a.send(_filler(1, sentences=8))
    a.send(
        "Switching topics to the database migration. The database migration code is "
        f"{A3_VALUES['B']}. Acknowledge briefly."
    )
    a.send(_filler(2, sentences=8))
    a.send(
        "New topic: the frontend redesign. The frontend redesign ticket key is "
        f"{A3_VALUES['C']}. Acknowledge briefly."
    )
    a.send(_filler(3, sentences=8))
    reply = a.send("What is the database migration code? Reply with just the code.")
    low = reply.lower()
    has_b = A3_VALUES["B"].lower() in low
    has_other = A3_VALUES["A"].lower() in low or A3_VALUES["C"].lower() in low
    ok = has_b and not has_other
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"has_B={has_b} contaminated={has_other} reply={reply!r}",
        turns=len(a.messages),
    )


def run_a4() -> Attempt:
    from harness.limits import MAX_ITEM_CHARS

    d = Path(mkdtemp(prefix="a4-"))
    filler = "\n".join(
        f"note {i:03d}: rollout checklist item {i} reviewed and signed off" for i in range(120)
    )
    body = filler + f"\n\n## Final section\nThe rollout password is {A4_SENTINEL}.\n"
    assert len(body) > MAX_ITEM_CHARS, "authoring bug: file must exceed the clamp"
    notes = d / "notes.txt"
    notes.write_text(body)
    a = _plain_agent()
    reply = a.send(
        f"@{notes} What does the last section say the rollout password is? "
        "Reply with just the password."
    )
    ok = A4_SENTINEL.lower() in reply.lower()
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"needle_in_reply={ok} reply={reply!r}",
        turns=len(a.messages),
    )


SPECS = [
    TaskSpec("A1", "held_in", "A", "fail", run_a1),
    TaskSpec("A2", "held_in", "A", "fail", run_a2),
    TaskSpec("A3", "held_out", "A", "uncertain", run_a3),
    TaskSpec("A4", "held_out", "A", "uncertain", run_a4),
]
