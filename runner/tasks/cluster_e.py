"""Cluster E — information never reaches the model.

The original A2/A4 pair was written against a 4k PREFIX clamp, which could hide a
fact near the end of a moderately large item. That is no longer the shipped policy:
`head_tail` retains a tail, so A2's sentinel survives truncation and A2 now measures
1.000 — its old `fail` prior was never caused by `tool_output`.

These two tasks make the requirement explicit, and each pins a DIFFERENT distinction
so the strategy menu can be compared rather than merely scaled:

- E2's tag sits at the very end, so a tail-preserving policy keeps it and a head-only
  policy loses it, at a FIXED budget. That is the shape a strategy test should have.
- E1's needle sits in the middle, which no shipped strategy reaches at any sane
  budget, and it must be retrieved without flooding the window. So E1 is not a
  setting to tune — it is a standing request for a middle-preserving strategy.

The region no task covers yet is a fact in the HEAD window, between the head a
tail-preserving policy keeps and the larger head a head-only policy keeps. Without it
nothing rewards head-only, so switching away from it looks free while silently losing
every fact that lives there.
"""

from __future__ import annotations

import hashlib

from runner.carbon_env import make_provider
from runner.helpers import agent_metrics, bash_runs, neutral_dir, scripted_approver, tool_texts
from runner.spec import Attempt, TaskSpec

E1_SENTINEL = "RANGE-ACCESS-E1-7QK"
# E1 passes only if the needle is retrieved while delivering less than this share of
# the fixture into the window. Without it the sole way through is raising the clamp
# until a naive whole-file read succeeds, which takes 24% of a 114 KB file into
# context — precisely the "flooding" the task exists to rule out. A fraction of the
# fixture, never a char count, so it survives any fixture resize.
E1_MAX_DELIVERED_SHARE = 0.10

# E2's tag is COMPUTED, never written into the script. The literal form was readable
# in the source, and the assistant duly read it and reported the answer without ever
# running the command — so the task graded "did you obey" instead of "did the tail
# survive". A sha the model cannot evaluate by inspection makes execution the only
# route, and leaves the tag's tail POSITION, which is what E2 actually measures.
E2_TAG_SEED = "e2-root-cause-seed-4417"
E2_SENTINEL = hashlib.sha256(E2_TAG_SEED.encode()).hexdigest()[:12].upper()
E2_PASS_COUNT = 3200


def _plain_agent(*, tools, approvals: list[dict] | None = None):
    from harness.agent import APPROVAL_TOOLS, DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer

    provider = make_provider()
    return Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=tools,
        agents_dir=neutral_dir(),
        approve=scripted_approver(approvals) if approvals is not None else None,
        approval_required=APPROVAL_TOOLS if approvals is not None else set(),
        tracer=Tracer(model=provider.model),
    )


def _large_reference() -> str:
    lines = [
        f"record {i:04d}: component-{i % 17} healthy, owner=team-{i % 9}, status=nominal"
        for i in range(1800)
    ]
    lines[1537] = f"record 1537: release authorization code={E1_SENTINEL}"
    body = "\n".join(lines) + "\n"
    assert len(body) > 100_000
    assert body.find(E1_SENTINEL) > 80_000
    return body


def run_e1() -> Attempt:
    """A late fact in a 100k file, retrieved WITHOUT flooding the window.

    Recalling the code is necessary but not sufficient. Both shipped truncation
    strategies are positional — head, or head plus tail — so neither reaches the
    middle at any sane budget, and the only way a whole-file read delivers this needle
    is a budget large enough to carry 24% of the file. That passes the letter of the
    task and defeats its purpose, so economy is part of the verdict.

    Consequence, stated plainly: no value of any current setting can pass this task.
    That is the intended reading — the missing piece is a middle-preserving strategy
    (summarize the middle, or an index the model can query), which is a Carbon
    capability request, not a number to tune. The task stays red until that exists.
    """
    from harness.tools import ToolRegistry, read_file_tool
    from harness.workspace import Workspace

    body = _large_reference()
    ws = Workspace()
    ws.write("release-records.txt", body)
    tools = ToolRegistry()
    tools.register(read_file_tool(str(ws.root)))
    a = _plain_agent(tools=tools)
    reply = a.send(
        "Find the release authorization code in release-records.txt. "
        "Reply with just the exact code."
    )
    called = any(m.get("role") == "tool" for m in a.messages)
    recalled = E1_SENTINEL.lower() in reply.lower()
    delivered = sum(len(t) for t in tool_texts(a.messages))
    share = delivered / len(body)
    economical = share <= E1_MAX_DELIVERED_SHARE
    ok = called and recalled and economical
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"read_called={called} sentinel_recalled={recalled} "
        f"delivered_share={share:.3f} (limit {E1_MAX_DELIVERED_SHARE}) "
        f"economical={economical} reply={reply[:200]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def _long_test_output() -> str:
    prefix = "\n".join(
        f"PASS test_component_{i:04d} ({(i % 13) + 1}ms)" for i in range(E2_PASS_COUNT)
    )
    body = prefix + f"\nFAILED integration: root-cause-tag={E2_SENTINEL}\n"
    assert len(body) > 80_000
    assert body.rstrip().endswith(E2_SENTINEL)
    return body


def e2_script() -> str:
    """The script E2 asks the model to run — and which the model can freely read.

    The tag is DERIVED here rather than written in, so the text carries the seed and
    the algorithm but not the answer. In the literal form the model simply read the
    tag out of the source and replied without running anything, which made the task
    grade obedience instead of tail survival. A public function so a test can assert
    on the exact bytes the model sees, rather than on this module's source, where the
    seed appears only as an identifier.
    """
    return (
        "import hashlib\n"
        f"for i in range({E2_PASS_COUNT}):\n"
        "    print(f'PASS test_component_{i:04d} ({(i % 13) + 1}ms)')\n"
        f"tag = hashlib.sha256({E2_TAG_SEED!r}.encode()).hexdigest()[:12].upper()\n"
        "print(f'FAILED integration: root-cause-tag={tag}')\n"
    )


E2_POST_PROCESSORS = ("|", ">", "tail", "head", "grep", "sed", "awk")


def _ran_script_plainly(cmd: str) -> bool:
    """Did this command run the script, without the SHELL doing the trimming?

    Two failure modes to avoid, and the obvious predicate hits both. Requiring the
    command to *start with* the invocation reports a false negative whenever the model
    wraps it — a `cd` first, or two lines in one call. That is what happened: the model
    sent ``ls run_tests.py`` and ``python3 run_tests.py`` as a single two-line command,
    ran the script, reported the right tag, and was recorded as never having run it.
    E2 read 0.000 for a reason that had nothing to do with truncation.

    Matching anywhere in the command fixes that but opens the other hole: a
    ``... | tail -5`` invocation preserves the tail by shell, not by policy, which is
    the very thing the task is trying to measure. So those are excluded instead.
    """
    return "python3 run_tests.py" in cmd and not any(op in cmd for op in E2_POST_PROCESSORS)


def run_e2() -> Attempt:
    """The failure at the tail of noisy command output must survive door control."""
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import ToolRegistry
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("run_tests.py", e2_script())
    tools = ToolRegistry()
    tools.register(bash_tool(Sandbox(trusted=True, timeout=60), workdir=str(ws.root)))
    approvals: list[dict] = []
    a = _plain_agent(tools=tools, approvals=approvals)
    reply = a.send(
        "Run `python3 run_tests.py` and report the exact root-cause-tag from the failed test. "
        "Reply with just the tag."
    )
    runs = bash_runs(a.messages, _ran_script_plainly)
    called = bool(runs)
    # A `... | tail` call may let a capable model recover despite the harness, so
    # post-processed invocations are excluded above rather than counted here: the
    # mechanism under test is whether the ORDINARY run kept the useful tail.
    first_result_kept_tail = called and E2_SENTINEL in runs[0][1]
    recalled = E2_SENTINEL.lower() in reply.lower()
    ok = called and first_result_kept_tail and recalled
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"bash_called={called} first_result_kept_tail={first_result_kept_tail} "
        f"sentinel_recalled={recalled} reply={reply[:240]!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


SPECS = [
    TaskSpec("E1", "held_in", "E", "fail", run_e1),
    TaskSpec("E2", "held_out", "E", "pass", run_e2),
]
