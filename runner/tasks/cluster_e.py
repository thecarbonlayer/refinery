"""Cluster E — information never reaches the model.

The original A2/A4 pair proved that a 4k prefix clamp can hide a fact near the
end of a moderately large item. These tasks make the strategy requirement
explicit: the fixtures are intentionally too large for a reasonable global
clamp increase. Passing requires selective or ranged access, or a result policy
that preserves the diagnostically useful tail.
"""

from __future__ import annotations

from runner.carbon_env import make_provider
from runner.helpers import agent_metrics, bash_runs, neutral_dir, scripted_approver
from runner.spec import Attempt, TaskSpec

E1_SENTINEL = "RANGE-ACCESS-E1-7QK"
E2_SENTINEL = "FAILURE-TAIL-E2-9PX"
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
    """A late fact in a 100k file should be retrievable without flooding context."""
    from harness.tools import ToolRegistry, read_file_tool
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("release-records.txt", _large_reference())
    tools = ToolRegistry()
    tools.register(read_file_tool(str(ws.root)))
    a = _plain_agent(tools=tools)
    reply = a.send(
        "Find the release authorization code in release-records.txt. "
        "Reply with just the exact code."
    )
    called = any(m.get("role") == "tool" for m in a.messages)
    ok = called and E1_SENTINEL.lower() in reply.lower()
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"read_called={called} sentinel_recalled={ok} reply={reply[:240]!r}",
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


def run_e2() -> Attempt:
    """The failure at the tail of noisy command output must survive door control."""
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import ToolRegistry
    from harness.workspace import Workspace

    ws = Workspace()
    script = (
        f"for i in range({E2_PASS_COUNT}):\n"
        "    print(f'PASS test_component_{i:04d} ({(i % 13) + 1}ms)')\n"
        f"print('FAILED integration: root-cause-tag={E2_SENTINEL}')\n"
    )
    ws.write("run_tests.py", script)
    tools = ToolRegistry()
    tools.register(bash_tool(Sandbox(trusted=True, timeout=60), workdir=str(ws.root)))
    approvals: list[dict] = []
    a = _plain_agent(tools=tools, approvals=approvals)
    reply = a.send(
        "Run `python3 run_tests.py` and report the exact root-cause-tag from the failed test. "
        "Reply with just the tag."
    )
    runs = bash_runs(a.messages, lambda cmd: cmd.strip().startswith("python3 run_tests.py"))
    called = bool(runs)
    # A second `... | tail` call may let a capable model recover despite the
    # harness. The diagnostic still fails unless the ordinary first run kept
    # the useful tail; that is the mechanism under test.
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
    TaskSpec("E1", "held_in", "E", "uncertain", run_e1),
    TaskSpec("E2", "held_out", "E", "pass", run_e2),
]
