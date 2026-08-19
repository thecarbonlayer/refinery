"""Cluster H — context-overflow and transient-provider recovery.

These deterministic harness fault injections exercise retry policy without
spending live model calls and keep the hard retry bound observable.

Carbon is imported inside each task, never at module scope: importing the task
registry must not bind carbon's config, because the loop applies a candidate to
carbon's working tree and depends on a fresh subprocess to pick the edit up.

The provider here is scripted, so token and cost telemetry would be a
structural zero rather than a measurement — these tasks report mechanism
metrics only (see ``agent_metrics(include_cost=False)``).
"""

from __future__ import annotations

from runner.helpers import agent_metrics, neutral_dir
from runner.spec import Attempt, TaskSpec

H1_SENTINEL = "TRANSIENT-RECOVERED-H1-4KT"
H2_SENTINEL = "OVERFLOW-RECOVERED-H2-7QW"


def _fault_agent(responder, scheme: str):
    """An agent wired to a scripted fault-injecting provider.

    Always tracered, for consistency across the cluster and so that any call
    which does complete is recorded. Note this changes no number for H3, whose
    provider never succeeds: carbon increments call counters only on completion,
    so an untraced H3 reported the same zeros. H1 and H2 are where a missing
    tracer would actually lose a completed call.
    """
    from harness.agent import Agent
    from harness.harness_config import CONFIG
    from harness.observability import Tracer
    from model import Provider

    provider = Provider(scheme, "fault-injection", responder=responder)
    return Agent(
        provider=provider,
        model=provider.model,
        agents_dir=neutral_dir(),
        tracer=Tracer(model=provider.model),
        # Explicit, though it matches carbon's own default: `Agent.__init__` binds
        # that default at import time, so a config edit to `default_context_limit`
        # would not reach this agent and any sweep over the knob would be a silent
        # no-op that proved nothing.
        context_limit=CONFIG.default_context_limit,
    )


def run_h1() -> Attempt:
    from model import LLMResponse

    state = {"calls": 0}

    def responder(messages, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("503 temporarily unavailable")
        return LLMResponse(content=H1_SENTINEL)

    agent = _fault_agent(responder, "fake://transient")
    result = None
    try:
        result = agent.run("Return the recovery receipt.")
        reply = result.text
    except Exception as exc:  # fail_fast is a measurable candidate, not suite infra failure
        reply = f"error: {exc}"
    finally:
        agent.close()  # the storage contract says close ends the scratch lifecycle
    ok = H1_SENTINEL in reply and state["calls"] == 2
    return Attempt(
        ok,
        "pass" if ok else "fail",
        f"provider_calls={state['calls']} reply={reply!r}",
        turns=len(agent.messages),
        metrics=agent_metrics(agent, result=result, include_cost=False),
    )


def _is_summarizer_call(messages: list[dict]) -> bool:
    """Structural, text-free detection of carbon's compaction call.

    ``compact()`` sends exactly ``[system, user]`` — its instructions plus the
    serialized transcript. A main-turn payload here cannot take that shape:
    ``_fault_agent`` passes no ``system=``, so carbon prepends no system message,
    and H2's seeded history is entirely ``user`` messages. That last part is the
    real guarantee, not ``keep_head > 0``: ``_clean_cut`` snapping can drive
    ``head_end`` to 0 and put the summary note first, but only when a tool message
    forces the snap, and this fixture has none. Keep the history tool-free.

    Deliberately matches SHAPE, never prompt text: `compaction_prompt` is an
    editable knob, so a candidate that legitimately rewrites it must not make
    this fixture misread a summarizer call as a main call and report a recovery
    regression that never happened.
    """
    return [m.get("role") for m in messages] == ["system", "user"]


def run_h2() -> Attempt:
    from harness.harness_config import CONFIG
    from model import LLMResponse

    state = {"main": 0, "summary": 0}
    # Counters sampled AT each main call, so the compaction caused by the overflow can
    # be isolated from any pre-turn compaction. Totals cannot do that: a legitimate
    # pre-turn compaction satisfies a `>= 1` total on its own, which let a carbon that
    # retried WITHOUT compacting pass wherever the pre-turn door happened to fire.
    at_main: list[tuple[int, int]] = []
    holder: dict = {}

    def responder(messages, **kwargs):
        if _is_summarizer_call(messages):
            state["summary"] += 1
            return LLMResponse(content="FAULT-INJECTION CHECKPOINT")
        state["main"] += 1
        agent_now = holder.get("agent")
        at_main.append((getattr(agent_now, "compaction_count", 0), state["summary"]))
        if state["main"] == 1:
            raise RuntimeError("maximum context length exceeded")
        return LLMResponse(content=H2_SENTINEL)

    agent = _fault_agent(responder, "fake://overflow")
    holder["agent"] = agent
    # Seed enough history that compaction can shrink it: `compact()` returns its
    # input unchanged when len(messages) <= keep_head + keep_tail, and both fields
    # are bounded only "positive", so a fixed 10 messages made recovery a no-op for
    # any legal window summing to >= 11. Deriving the size fixes that, but it cannot
    # make the fixture safe on its own — a bigger history is likelier to trip the
    # PRE-TURN compaction door — which is why the verdict is a delta, not a total.
    # This only ensures the overflow branch has something to compact.
    policy = CONFIG.compaction
    seeded = max(10, (policy.keep_head + policy.keep_tail) * 2)
    agent.messages = [{"role": "user", "content": f"old context {i}"} for i in range(seeded)]
    result = None
    try:
        result = agent.run("Continue after recovering the window.")
        reply = result.text
    except Exception as exc:
        reply = f"error: {exc}"
    finally:
        agent.close()  # the storage contract says close ends the scratch lifecycle
    # The mechanism as a DELTA across the two main calls: between the call that
    # overflowed and the retry that succeeded, exactly one compaction happened and it
    # really summarized. `main == 2` is stable because overflow recovery never
    # consults the retry policy.
    #
    # Three earlier forms failed, each instructively. Exact TOTALS (`compaction_count
    # == 1`) collapsed H2 to 0.0 across hundreds of legal settings, because
    # `Agent.run` calls `_maybe_compact()` before the turn and a seeded history past
    # `default_context_limit * trigger_fraction` compacts twice. Pure inequalities
    # fixed that but lost the tooth: a carbon that retried WITHOUT compacting still
    # passed wherever the pre-turn door fired, since that one legitimate compaction
    # satisfied every `>= 1`. Only a delta separates "the overflow was recovered" from
    # "something compacted at some point". Verified: invariant across 107 legal
    # compaction/context settings, and red for both broken-recovery variants.
    # A FOURTH form was needed once the compaction menu grew. The delta above was
    # positional — exactly two main calls, and the increment between the first two —
    # which encoded an ordering that only held for the strategy shipped at the time.
    # Under token_budget_checkpoint at keep_head=6/keep_tail=6/trigger_fraction=0.02,
    # instrumenting the run shows recovery attempted once, returning True, and really
    # compacting; the agent recovers and returns the sentinel. Only the BOOKKEEPING
    # differs: the increment lands between the second and third main call, so a rule
    # written around call positions reads a healthy recovery as a failure.
    #
    # So compare the ends, not the neighbours: the compaction the overflow caused must
    # appear somewhere between the first main call and the last. That keeps both teeth
    # the earlier forms were built for. A pre-turn compaction is already baked into
    # at_main[0], so it cannot satisfy this on its own; and a carbon that takes the
    # overflow branch while compacting nothing never moves the counter at all. Both are
    # pinned by the broken-recovery guards in the suite.
    recovered = len(at_main) >= 2 and at_main[-1][0] > at_main[0][0]
    ok = H2_SENTINEL in reply and recovered
    detail_counts = (
        f"compactions_at_main={[c for c, _ in at_main]} summaries_at_main={[s for _, s in at_main]}"
    )
    return Attempt(
        ok,
        "pass" if ok else "fail",
        f"calls={state} compactions={agent.compaction_count} {detail_counts} reply={reply!r}",
        turns=len(agent.messages),
        metrics=agent_metrics(agent, result=result, include_cost=False),
    )


def expected_retry_calls(policy) -> int:
    """Provider calls a permanently-failing endpoint should receive under ``policy``.

    Both fields matter. Carbon gates ``can_retry`` on ``strategy == "backoff"``,
    so ``fail_fast`` makes exactly ONE call no matter what ``max_attempts`` says.
    Comparing against ``max_attempts`` alone reports a failure for behaviour that
    exactly matches a legal policy — and `retry` is guard-only, so a false
    regression here is the only thing this task can contribute.
    """
    return policy.max_attempts if policy.strategy == "backoff" else 1


def run_h3() -> Attempt:
    """Retries stay bounded AND actually happen, measured against live config.

    The bound is derived from the CONFIGURED policy, not carbon's schema ceiling.
    A loose ``1 <= calls <= 5`` passed even when the retry loop never retried at
    all; exact equality catches a loop that gave up early and one that ran away,
    while still tracking any legal ``(strategy, max_attempts)`` pair.
    """
    from harness.harness_config import CONFIG

    state = {"calls": 0}

    def responder(messages, **kwargs):
        state["calls"] += 1
        raise RuntimeError("429 rate limit")

    agent = _fault_agent(responder, "fake://bounded")
    raised = False
    result = None
    try:
        result = agent.run("This provider will not recover.")
    except RuntimeError:
        raised = True
    finally:
        agent.close()  # the storage contract says close ends the scratch lifecycle
    policy = CONFIG.retry
    expected = expected_retry_calls(policy)
    ok = raised and state["calls"] == expected
    return Attempt(
        ok,
        "pass" if ok else "fail",
        f"raised={raised} provider_calls={state['calls']} strategy={policy.strategy} "
        f"max_attempts={policy.max_attempts} expected_calls={expected}",
        turns=len(agent.messages),
        metrics=agent_metrics(agent, result=result, include_cost=False),
    )


SPECS = [
    TaskSpec("H1", "held_in", "H", "pass", primitive="retry", alias="RET-1", run=run_h1),
    TaskSpec("H2", "held_out", "H", "pass", primitive="retry", alias="RET-2", run=run_h2),
    TaskSpec("H3", "held_in", "H", "pass", primitive="retry", alias="RET-3", run=run_h3),
]
