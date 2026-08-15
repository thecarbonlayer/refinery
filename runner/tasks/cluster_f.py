"""Cluster F — tool semantics and execution depth.

F1 catches silent ambiguous edits. F2 is a mutation-sensitive guard for the
agent loop: a harness capped at six tool rounds cannot finish it, while the
current declared budget can.
"""

from __future__ import annotations

import json
from pathlib import Path

from runner.carbon_env import make_provider
from runner.helpers import (
    agent_metrics,
    neutral_dir,
    scripted_approver,
    tool_call_args,
    workspace_kwargs,
)
from runner.spec import Attempt, TaskSpec

F1_SOURCE = """\
def alpha_timeout():
    timeout = 5
    return timeout


def beta_timeout():
    timeout = 5
    return timeout
"""
F1_EXPECTED = F1_SOURCE.replace(
    "def beta_timeout():\n    timeout = 5",
    "def beta_timeout():\n    timeout = 30",
)
F2_SENTINEL = "DEPTH-COMPLETE-F2-4MN"
F2_CODES = ("start", "quartz", "maple", "cobalt", "ember", "delta", "lumen", "orbit", "raven")


def _agent(
    *,
    tools,
    workspace_root: Path | str | None = None,
    approvals: list[dict] | None = None,
):
    """``workspace_root`` is optional here, unlike cluster E's, and the asymmetry
    is the point: F1's tools are bound to a workspace, so anything carbon's
    truncation door writes has to land in that same tree. F2's ``advance`` is pure
    in-memory state with no tree behind it at all — there is no root to name, so
    the default stands and carbon anchors on ``agents_dir`` as it does today.
    """
    from harness.agent import APPROVAL_TOOLS, DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer

    provider = make_provider()
    return Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=tools,
        agents_dir=neutral_dir(),
        **(workspace_kwargs(workspace_root) if workspace_root is not None else {}),
        approve=scripted_approver(approvals) if approvals is not None else None,
        approval_required=APPROVAL_TOOLS if approvals is not None else set(),
        tracer=Tracer(model=provider.model),
    )


def run_f1() -> Attempt:
    from harness.tools import ToolRegistry, read_file_tool
    from harness.workspace import Workspace, edit_file_tool

    ws = Workspace()
    ws.write("timeouts.py", F1_SOURCE)
    tools = ToolRegistry()
    tools.register(read_file_tool(str(ws.root)))
    tools.register(edit_file_tool(ws))
    approvals: list[dict] = []
    a = _agent(tools=tools, workspace_root=ws.root, approvals=approvals)
    reply = a.send(
        "In timeouts.py, change only beta_timeout's timeout from 5 to 30. "
        "Do not change alpha_timeout."
    )
    actual = (ws.root / "timeouts.py").read_text()
    edit_args = tool_call_args(a.messages, ("edit_file",))
    ambiguous_call = False
    for raw in edit_args:
        try:
            old = str(json.loads(raw).get("old", ""))
        except json.JSONDecodeError:
            continue
        if old and F1_SOURCE.count(old) > 1:
            ambiguous_call = True
    ok = actual == F1_EXPECTED and bool(edit_args)
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"exact_edit={actual == F1_EXPECTED} edit_called={bool(edit_args)} "
        f"ambiguous_call_attempted={ambiguous_call} reply={reply[:200]!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def run_f2() -> Attempt:
    from harness.tools import Tool, ToolRegistry

    state = {"index": 0, "correct_calls": 0}

    def advance(code: str) -> str:
        i = state["index"]
        expected = F2_CODES[i]
        if code != expected:
            return f"error: expected code {expected!r}; retry with that exact code"
        state["correct_calls"] += 1
        state["index"] += 1
        if state["index"] == len(F2_CODES):
            return f"sequence complete; final receipt={F2_SENTINEL}"
        return f"accepted; call advance again with code={F2_CODES[state['index']]}"

    tools = ToolRegistry()
    tools.register(
        Tool(
            name="advance",
            description=(
                "Advance a stateful workflow by supplying the exact code returned "
                "by the previous step. Start with code 'start'."
            ),
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
            func=advance,
        )
    )
    a = _agent(tools=tools)
    reply = a.send(
        "Complete the advance workflow. Start with code 'start', follow every returned "
        "instruction, and reply with just the final receipt."
    )
    recalled = F2_SENTINEL.lower() in reply.lower()
    ok = state["correct_calls"] == len(F2_CODES) and recalled
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"correct_calls={state['correct_calls']}/{len(F2_CODES)} "
        f"receipt_recalled={recalled} reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


SPECS = [
    TaskSpec("F1", "held_in", "F", "pass", run_f1),
    TaskSpec("F2", "held_out", "F", "pass", run_f2),
]
