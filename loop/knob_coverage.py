"""Which tasks can actually observe each editable Carbon knob, and in what role.

Lives in ``loop/``, not ``runner/``, deliberately. ``runner_sha`` is a content hash
of the runner package and every recorded result is stamped with it, so anything
kept there invalidates every baseline when it changes. This table is governance
metadata about what the loop may PROPOSE — it measures nothing — and correcting a
row should never cost a re-measurement. It did, twice, before the move.

This is a governance contract, not a suggestion list. A newly editable knob must
name real coverage before the loop is allowed to tune it.

Three roles per knob, and the ordering between them is what the tests enforce:

- **observers** — tasks whose mechanism genuinely routes through this knob AND
  whose verdict some legal value of it can move. "Reads a tool result" is not
  enough: a 45-character result never meets a 4,000-character budget, so the task
  is invariant and the coverage is decorative.
- **miners** ⊆ observers — observers with failure headroom (prior ``fail`` or
  ``uncertain``) that moving the knob could turn green. EVERY miner must have
  headroom, and a knob whose observers all pass declares ``miners: ()`` and joins
  ``GUARD_ONLY_KNOBS`` — requiring it to name a fossil miner instead, as an earlier
  version did, forced the table to state something false.
- **guards** ⊆ observers — observers that can still REGRESS, i.e. whose prior is
  not ``fail``: a task already at 0.0 cannot drop further. Note this admits
  ``uncertain`` priors (D3, G2, A3 below), which are weaker guards than a `pass`
  prior — the earlier wording said "expected to hold", which the registry's own
  priors deny for those three. Weak is not the same as decorative, but do not read
  an `uncertain` guard as a firm one.

``tests/test_knob_coverage.py`` enforces the subset relations, both prior
properties, the sentinel allowlist, and the freshness of the exemption tables.

**What no test can enforce, stated plainly.** The `observers` tuples are authored
assertions about carbon's internals. An audit found six of them false at once, and
padding a row with irrelevant task names satisfies every mechanical check — the
subset rules only relate the roles to each other, never to carbon. A reviewer has
demonstrated a plausible new knob passing the whole contract with no real coverage.
So: this file is a human-audited claim with mechanical guardrails, not a proof.
Changing a row means re-measuring, not editing a tuple.

Nothing in `loop/` reads this table today, which caps the blast radius but also
means these tests are its ONLY enforcement.
"""

from __future__ import annotations

# Sentinels resolved from the live registry by the test, never hand-listed.
# `LIVE_ALL` is every task that drives a real model through carbon's agent — the
# H cluster is excluded because it builds agents with no `system=` and a scripted
# provider, so no system prompt or sampling temperature ever reaches it.
# The other two SPLIT that set by prior, which is what makes the headroom
# property true by construction: a single wildcard covering all 20 tasks let an
# existential check pass on one lucky failing task while quietly filing seven
# already-passing tasks as miners.
LIVE_ALL = "LIVE_MODEL_TASKS"
LIVE_MINERS = "LIVE_MODEL_TASKS_WITH_HEADROOM"
LIVE_GUARDS = "LIVE_MODEL_TASKS_EXPECTED_TO_PASS"

# Only these knobs may use the sentinels above. Every live call carries the system
# prompt and the sampling temperature, so every live task genuinely observes both.
# No other knob is suite-wide — and because the sentinels make every enforced
# property true by construction, letting a new knob claim them is a universal pass.
# A reviewer demonstrated exactly that with a memory-recall knob no task touches.
SUITE_WIDE_KNOBS = frozenset({"system_prompt", "temperature"})

# Tasks whose verdict some legal `tool_output` value can move, i.e. whose
# verification depends on tool-result text. The criterion is NOT "the budget fires
# at baseline": D3's 3,029-char fixture does not fire at the shipped 4,000 budget,
# while F1's 113-char source IS destroyed by a legal budget of 20, breaking the
# exact match `edit_file` needs. Only E1 and E2 are sensitive at PLAUSIBLE budgets,
# which is why the guards sit there.
#
# Two deliberate exclusions. A2 is NOT here: its needle survives every budget down
# to 8,736, and its prior is `fail`, so no legal value moves its verdict either way
# — listing it was the decorative padding this module warns about. C1/C2 are not
# here either, and that is now a property of the code: their leak predicates read
# the RAW tool result (see cluster_c's `recording_tool` wiring), precisely so a
# candidate cannot raise a containment score by clamping away the evidence.
# E4 is sensitive to the STRATEGY rather than the budget: its verdict turns on
# whether the door leaves a recoverable artifact behind, which only an
# offload-shaped strategy does — no budget value alone can move it.
_TOOL_RESULT_READERS = ("D3", "E1", "E2", "E3", "E4", "F1", "F2", "G3")

# Every task that makes any tool call at all, so the per-turn round budget binds
# it. A different mechanism from result truncation, and therefore a different set
# — sharing one list between the two was a false claim about both.
_TOOL_USERS = (
    "A2", "B1", "B2", "B3", "C1", "C2", "C3", "D1", "D2", "D3",
    "E1", "E2", "E3", "E4", "F1", "F2", "G3",
)  # fmt: skip

KNOB_COVERAGE: dict[str, dict[str, tuple[str, ...]]] = {
    # Reaches every live task through the agent's system prompt / sampling call.
    "system_prompt": {
        "observers": (LIVE_ALL,),
        "miners": (LIVE_MINERS,),
        "guards": (LIVE_GUARDS,),
    },
    "temperature": {
        "observers": (LIVE_ALL,),
        "miners": (LIVE_MINERS,),
        "guards": (LIVE_GUARDS,),
    },
    # `deliver()` applies file_injection ONLY on an `@path` match in the user
    # text. A4 is the suite's sole `@path` sender and has no tools, so the
    # injected block is its only route to the answer — see UNGUARDED_KNOBS.
    "file_injection": {"observers": ("A4",), "miners": ("A4",), "guards": ()},
    # The strategy menu, not the budget, is what this row now mines. E1 measures
    # retrieval ECONOMY, which the budget genuinely moves. E3's midpoint needle in
    # OPAQUE command output was this table's standing capability gap — both
    # positional strategies miss the middle at any sane budget, and the budgets
    # that do reach it are flood-shaped — until carbon put an offload strategy on
    # the `tool_output` menu; the weakness E3 reports is now a settable value, so
    # E3 graduates from CAPABILITY_GAPS to miner. Classified honestly: E3's own
    # verifier requires the needle INSIDE the first plain run's inline result, so
    # an excerpt-plus-path configuration leaves E3 itself red, and the only values
    # that flip it green remain flood-sized budgets — which E1's economy verdict
    # prices out. E4 is the task an offload configuration can genuinely turn
    # green, and the proof the mined fix is the recoverable kind rather than the
    # bigger bucket iteration 1 shipped (the clamp raise of iterations/iter-01,
    # PR #1): same opaque-stdout shape, needle astride the midpoint
    # of a stream no honest inline cut can span, pass gated on recovery from the
    # offloaded artifact itself (attribution in cluster_e), so neither a larger
    # budget nor a luckier excerpt can fake it. E2 (tail survival through the
    # inline excerpt) and D3 (a second guard at lower budgets) stay the guards,
    # unchanged: an offload candidate that floods the window or loses the tail
    # dies on them, which is exactly the regression they exist to catch.
    "tool_output": {
        "observers": _TOOL_RESULT_READERS,
        "miners": ("E1", "E3", "E4"),
        "guards": ("E2", "D3"),
    },
    # F2 forces 10 model calls and fails at any budget <= 9 — the only binding
    # observer, so it must be a GUARD: omitting it left the declared guard set to
    # F1 and G3, which need ~3 and ~2 rounds and are therefore dominated by F2.
    "max_tool_steps": {
        "observers": _TOOL_USERS,
        "miners": (),
        "guards": ("F1", "F2", "G3"),
    },
    # A1's window peaks at ~3630 tokens against a 3200 trigger, so compaction
    # fires and the knob can move it. A3 peaks at ~1185 and never compacts: no
    # value turns it green, so it is a guard of the LOWERING direction only, not a
    # miner. G2 is not an observer at all — it pins `context_limit=700` itself.
    # A1 measured 1.000 after Carbon's compaction fix landed, so it no longer mines
    # anything here — it defends the window instead. A3 never compacts at all, so the
    # knob has no failing observer left and is guard-only.
    "default_context_limit": {
        "observers": ("A1", "A3"),
        "miners": (),
        "guards": ("A1", "A3"),
    },
    # Only tasks that actually reach `compact()`. G2 compacts three times at its
    # own 700-token limit, so it sees every sub-field and is the only real guard;
    # it is also a miner, which makes it a weak guard, but a nominal one would be
    # worse. A3 never compacts, so nothing in the object can move it. H2 is NOT an
    # observer: it forces overflow with an injected error and is invariant to
    # strategy, keep_head, keep_tail, trigger_fraction and summary_max_tokens.
    # G4 mines, G2 guards. G2 was previously both, which contradicted the rule that
    # mining uses held-in evidence only: G2 is held-out, so tuning against it spends
    # the generalization test. G4 is the held-in counterpart and carries a different
    # trajectory shape, so passing it does not entail passing G2.
    "compaction": {
        "observers": ("A1", "G2", "G4"),
        "miners": ("G4",),
        "guards": ("A1", "G2"),
    },
    # Same observers, and for a sharper reason: H2 detects the summarizer by
    # payload SHAPE precisely so that rewriting this knob cannot fool it, which by
    # construction makes it blind to the knob. `tests/test_registry.py` asserts
    # that invariance directly.
    "compaction_prompt": {
        "observers": ("A1", "G2", "G4"),
        "miners": ("G4",),
        "guards": ("A1", "G2"),
    },
    # `verify_attempts` is gated on a test command in the task's instruction root,
    # and cluster B is the only cluster that ships one — every other task uses a
    # neutral dir, so the gate never arms. Re-prompts track the knob 1:1.
    "verify_attempts": {
        "observers": ("B1", "B2", "B3"),
        "miners": ("B1", "B2", "B3"),
        "guards": ("B2", "B3"),
    },
    # G1 needs ~1878 completion tokens against 4,096 and fails via
    # `finish_reason=length` when lowered — 30x tighter than the next task
    # (B1's ~52-token tool-call arguments), so it is the binding observer.
    "max_tokens": {"observers": ("G1",), "miners": (), "guards": ("G1",)},
    # H1 is the ONLY observer: it fails under `fail_fast` at any max_attempts, and
    # under `backoff` with max_attempts=1. H2 takes the overflow branch, which
    # returns BEFORE `can_retry`. H3 derives its expectation from the same config
    # carbon reads, so it passes for all ten legal pairs — invariant by construction,
    # which makes it a canary for carbon's retry CODE and not an observer of this
    # knob at all. `base_delay_ms` is observed by nobody; it only feeds `time.sleep`.
    "retry": {"observers": ("H1",), "miners": (), "guards": ("H1",)},
}

# Knobs no observer fails BECAUSE OF: the loop may defend them but has nothing to
# mine. Note the precise claim — `max_tool_steps` has nine observers with failure
# headroom, but none of them fails on tool-round depth, so raising the budget cannot
# turn any of them green. "No failing observer" would be plainly false here.
#
# Each entry states WHY, because a bare set would let a new knob join by adding one
# line and no argument. That is a speed bump, not a gate: a reviewer has shown a
# fabricated-coverage knob passing through either exemption table, and no test in this
# file can tell a true rationale from a plausible one.
GUARD_ONLY_KNOBS: dict[str, str] = {
    "max_tokens": (
        "G1 already produces 400 lines within the shipped budget; nothing fails on length."
    ),
    "max_tool_steps": (
        "F2 completes its 10 calls within the shipped budget; nothing fails on depth."
    ),
    "retry": "H1 already recovers from an injected transient fault; nothing fails on retry policy.",
    "default_context_limit": (
        "A1 and A3 both hold at the shipped window; no observer fails on context size."
    ),
}

# Capability gaps: a task fails, and NO value of any current setting can fix it because
# the strategy it needs is not on the menu. These are the loop's honest output when the
# right answer does not exist yet — a written request for a person, never a setting
# change that approximates one.
#
# Empty today, and the empty state has a history worth keeping: E3 held this slot —
# "needs a tool_output strategy that preserves the MIDDLE of a large result, an offload
# that hands back a retrievable path" — from the suite repair until carbon added such a
# strategy to the `tool_output` menu. The request was answered, so E3 moved to that
# knob's miners (see the row above) with E4 alongside it to prove path-recoverability
# specifically. That is this table working as designed: a gap is a written request for
# a person, and it leaves when the person ships the strategy, never when the task is
# softened to stop asking.
CAPABILITY_GAPS: dict[str, str] = {}

# Tasks that DO the thing a knob acts on and are still deliberately not observers.
#
# Every entry below was already argued in the prose above; this restates it where a
# program can read it. `loop/observed_coverage.py` derives from recorded runs which
# tasks exercise each knob, and without this table it re-raises the same settled
# exclusions on every run — a review queue nobody reads is worse than none, because
# the one genuinely unreviewed entry hides among them.
#
# The bar is the same as for an observer: an argument about carbon's internals, not a
# convenience. An entry here is a claim that NO legal value of the knob can move the
# task, despite the task visibly doing the thing the knob acts on.
DELIBERATE_NON_OBSERVERS: dict[str, dict[str, str]] = {
    "tool_output": {
        "A2": (
            "Needle survives every budget down to 8,736 and the prior is `fail`, so no "
            "legal value moves the verdict either way."
        ),
        "C1": (
            "The leak predicate reads the RAW tool result (cluster_c's `recording_tool` "
            "wiring), precisely so a candidate cannot raise a containment score by "
            "clamping away the evidence."
        ),
        "C2": (
            "Same raw-result wiring as C1, and for the same reason: C2 grades whether a "
            "secret escaped, so reading the truncated copy would let a candidate score "
            "containment by shrinking the budget until the evidence was cut away."
        ),
    },
    "compaction": {
        "H2": (
            "Forces overflow with an injected error and is invariant to strategy, "
            "keep_head, keep_tail, trigger_fraction and summary_max_tokens."
        ),
    },
    "compaction_prompt": {
        "H2": (
            "Detects the summarizer by payload SHAPE, never by prompt text, precisely "
            "so rewriting this knob cannot fool it. Asserted in test_registry.py."
        ),
    },
    "retry": {
        "H2": "Takes the overflow branch, which returns BEFORE `can_retry`.",
        "H3": (
            "Derives its expectation from the same config carbon reads, so it passes "
            "for all ten legal pairs — invariant by construction, a canary for carbon's "
            "retry CODE rather than an observer of the knob."
        ),
    },
}


# Knobs whose only observers are their own miners, so no task can guard them.
# A REAL coverage gap, recorded rather than hidden behind a guard that cannot see
# the knob. Closing it needs a new task, not a table edit.
UNGUARDED_KNOBS: dict[str, str] = {
    "file_injection": (
        "A4 is the suite's only `@path` sender and it is the miner. Guarding this "
        "knob needs a second `@path` task with a passing prior."
    ),
}
