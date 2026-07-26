"""Cluster H — context-overflow and transient-provider recovery.

These deterministic harness fault injections exercise retry policy without
spending live model calls and keep the hard retry bound observable.
"""

from __future__ import annotations

from harness.agent import Agent
from harness.observability import Tracer
from model import LLMResponse, Provider

from runner.helpers import agent_metrics, neutral_dir
from runner.spec import Attempt, TaskSpec

H1_SENTINEL = "TRANSIENT-RECOVERED-H1-4KT"
H2_SENTINEL = "OVERFLOW-RECOVERED-H2-7QW"


def run_h1() -> Attempt:
    state = {"calls": 0}

    def responder(messages, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("503 temporarily unavailable")
        return LLMResponse(content=H1_SENTINEL)

    provider = Provider("fake://transient", "fault-injection", responder=responder)
    agent = Agent(
        provider=provider,
        model=provider.model,
        agents_dir=neutral_dir(),
        tracer=Tracer(model=provider.model),
    )
    try:
        reply = agent.send("Return the recovery receipt.")
    except Exception as exc:  # fail_fast is a measurable candidate, not suite infra failure
        reply = f"error: {exc}"
    ok = H1_SENTINEL in reply and state["calls"] == 2
    return Attempt(
        ok,
        "pass" if ok else "fail",
        f"provider_calls={state['calls']} reply={reply!r}",
        turns=len(agent.messages),
        metrics=agent_metrics(agent),
    )


def run_h2() -> Attempt:
    state = {"main": 0, "summary": 0}

    def responder(messages, **kwargs):
        is_summary = bool(
            messages
            and messages[0].get("role") == "system"
            and "context summarizer" in str(messages[0].get("content", ""))
        )
        if is_summary:
            state["summary"] += 1
            return LLMResponse(content="FAULT-INJECTION CHECKPOINT")
        state["main"] += 1
        if state["main"] == 1:
            raise RuntimeError("maximum context length exceeded")
        return LLMResponse(content=H2_SENTINEL)

    provider = Provider("fake://overflow", "fault-injection", responder=responder)
    agent = Agent(
        provider=provider,
        model=provider.model,
        agents_dir=neutral_dir(),
        tracer=Tracer(model=provider.model),
    )
    agent.messages = [{"role": "user", "content": f"old context {i}"} for i in range(10)]
    try:
        reply = agent.send("Continue after recovering the window.")
    except Exception as exc:
        reply = f"error: {exc}"
    ok = H2_SENTINEL in reply and state == {"main": 2, "summary": 1} and agent.compaction_count == 1
    return Attempt(
        ok,
        "pass" if ok else "fail",
        f"calls={state} compactions={agent.compaction_count} reply={reply!r}",
        turns=len(agent.messages),
        metrics=agent_metrics(agent),
    )


def run_h3() -> Attempt:
    state = {"calls": 0}

    def responder(messages, **kwargs):
        state["calls"] += 1
        raise RuntimeError("429 rate limit")

    provider = Provider("fake://bounded", "fault-injection", responder=responder)
    agent = Agent(provider=provider, model=provider.model, agents_dir=neutral_dir())
    raised = False
    try:
        agent.send("This provider will not recover.")
    except RuntimeError:
        raised = True
    ok = raised and 1 <= state["calls"] <= 5
    return Attempt(
        ok,
        "pass" if ok else "fail",
        f"raised={raised} provider_calls={state['calls']} hard_max=5",
        turns=len(agent.messages),
        metrics=agent_metrics(agent),
    )


SPECS = [
    TaskSpec("H1", "held_in", "H", "pass", run_h1),
    TaskSpec("H2", "held_out", "H", "pass", run_h2),
    TaskSpec("H3", "held_in", "H", "pass", run_h3),
]
