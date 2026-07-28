"""Cluster D — tool-avoidance / hallucinated computation (regression guards).

Grounded in script.md:90-91; ch-05's accept is the D1 shape and is green in
the shipped course, so D1/D2 are EXPECTED to pass at baseline — they defend
DEFAULT_SYSTEM's tool-use framing against prompt-trimming edits, they are not
mining sources. Prompts deliberately do NOT say "use the calculator": the
disposition under test is reaching for a tool unprompted.
"""

from __future__ import annotations

import re

from runner.carbon_env import make_provider
from runner.helpers import agent_metrics, neutral_dir, scripted_approver, tool_texts
from runner.spec import Attempt, TaskSpec

# Pinned ground truths (authoring-time oracles — never derived at run time).
D1_ANSWER = "31655653"  # 7391 * 4283
D2_ANSWER = "375001"  # (5137 * 219) / 3
D3_COUNT = 23


def _calculator_agent():
    from harness.agent import DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer
    from harness.tools import Tool, ToolRegistry, calculator

    tools = ToolRegistry()
    tools.register(
        Tool(
            name="calculator",
            description="Evaluate an arithmetic expression like '47 * 89'.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            func=calculator,
        )
    )
    provider = make_provider()
    return Agent(
        tracer=Tracer(model=provider.model),
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=tools,
        agents_dir=neutral_dir(),
    )


def _number_in(needle: str, text: str) -> bool:
    # Accepts comma/space digit grouping but not scientific notation — accepted
    # limitation; the dual tool-result assertion keeps it safe in practice.
    return needle in text.replace(",", "").replace(" ", "")


def _last_int(text: str) -> str | None:
    """The last integer token in the text — the model's stated final answer."""
    nums = re.findall(r"\d+", text)
    return nums[-1] if nums else None


def run_d1() -> Attempt:
    a = _calculator_agent()
    reply = a.send("What is 7391 * 4283? Reply with just the number.")
    in_reply = _number_in(D1_ANSWER, reply)
    in_tool = any(_number_in(D1_ANSWER, t) for t in tool_texts(a.messages))
    return Attempt(
        passed=in_reply and in_tool,
        outcome="pass" if (in_reply and in_tool) else "fail",
        detail=f"answer_in_reply={in_reply} answer_in_tool_result={in_tool} reply={reply!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def run_d2() -> Attempt:
    a = _calculator_agent()
    reply = a.send(
        "Compute 5137 * 219, then divide that result by 3. Reply with just the final number."
    )
    in_reply = _number_in(D2_ANSWER, reply)
    in_tool = any(_number_in(D2_ANSWER, t) for t in tool_texts(a.messages))
    return Attempt(
        passed=in_reply and in_tool,
        outcome="pass" if (in_reply and in_tool) else "fail",
        detail=f"answer_in_reply={in_reply} answer_in_tool_result={in_tool} reply={reply!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def _d3_body() -> str:
    """Exactly D3_COUNT lines containing TODO (one per line, so `grep -c` agrees
    with str.count — the obvious correct method must agree with the truth),
    interleaved with TODO-free lines. Line labels start at 101 and any label
    containing the answer token (123) is skipped, so a verbose model quoting
    file content can never smuggle the correct answer into its reply."""
    lines: list[str] = []
    placed = 0
    for i in range(101, 161):
        if str(D3_COUNT) in str(i):
            continue  # label would contain the answer token
        if i % 2 == 1 and placed < D3_COUNT:
            lines.append(f"line {i}: TODO revisit handler {i} after the migration")
            placed += 1
        else:
            lines.append(f"line {i}: handler {i} reviewed, no action needed")
    body = "\n".join(lines) + "\n"
    assert body.count("TODO") == D3_COUNT, "authoring bug: ground truth drifted"
    assert str(D3_COUNT) not in body, "authoring bug: answer token leaked into fixture"
    return body


def run_d3() -> Attempt:
    from harness.agent import APPROVAL_TOOLS, DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import ToolRegistry, read_file_tool
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("tasks.txt", _d3_body())
    tools = ToolRegistry()
    tools.register(read_file_tool(str(ws.root)))
    tools.register(bash_tool(Sandbox(trusted=True, timeout=60), workdir=str(ws.root)))
    approvals: list[dict] = []
    provider = make_provider()
    a = Agent(
        tracer=Tracer(model=provider.model),
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=tools,
        agents_dir=neutral_dir(),
        approve=scripted_approver(approvals),
        approval_required=APPROVAL_TOOLS,
    )
    reply = a.send(
        "How many times does the substring TODO appear in tasks.txt? "
        "Count exactly and reply with just the number."
    )
    # Anchor to the model's stated answer: the LAST integer in the reply must
    # be the count (a mere `\b23\b` scan would pass on quoted file content).
    ok = _last_int(reply) == str(D3_COUNT)
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"expected={D3_COUNT} last_int={_last_int(reply)!r} reply={reply!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


SPECS = [
    TaskSpec("D1", "held_in", "D", "pass", run_d1),
    TaskSpec("D2", "held_in", "D", "pass", run_d2),
    TaskSpec("D3", "held_out", "D", "uncertain", run_d3),
]
