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

`loop/observed_coverage.py` now reads this table and checks it against what the
recorded runs actually show — the first mechanical check on the rows themselves rather
than on how they relate to each other. It cannot confirm a row, only contradict one:
observed activity is NECESSARY for a knob to reach a task, never sufficient. So the
warning above still stands for every row it does not deny.
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

# Tasks whose verdict some legal `tool_output` value can move — classified against
# carbon's TRUE legal domain, which is any positive integer budget plus the strategy
# menu. No plausibility floor.
#
# DECIDED 2026-08-15, after a review round argued the opposite. A floor was written
# (500, from carbon's overflow-recovery `SHRINK_MIN_BUDGET`) to keep the rows small,
# and then marked NOT ENFORCED because nothing in the proposal path honoured it. The
# reviewer's minimum fix was to enforce it — advertise a `min`, reject below-floor
# candidates. The decision went the other way, and the reasoning is worth keeping:
#
#   Enforcing adds a real constraint on what the loop may propose in order to fix a
#   BOOKKEEPING inconsistency. This program has been burned repeatedly by exactly that
#   move — a checked-in-defaults literal, a set of strategy pins — each added to tidy
#   something up and each turning into a false veto that rejected legal candidates. And
#   the system already handles a degenerate budget with evidence rather than decree: a
#   budget of 20 reddens six tasks at once and the acceptance rule refuses it loudly,
#   while E1's economy guard prices out the other direction.
#
# So the rows grow instead. Each addition below is MEASURED, not argued from memory:
# the number is the budget at which that task's largest tool result stops arriving
# intact, obtained by building its real fixture and running its pinned command.
#
# A2 is here for a different reason and would be here under any floor: its exclusion
# argued only about the BUDGET, but `tool_output` also carries the STRATEGY, and a
# proposed `keep_head` drops the tail its sentinel sits in. That is a strategy argument,
# so no budget policy touches it.
#
# Still excluded, and now the only exclusions: C1 and C2 read the RAW tool result
# (cluster_c's `recording_tool` wiring) so a candidate cannot raise a containment score
# by clamping the evidence away — but that protects only their LEAK predicate, and each
# verdict also needs a correct functional reply. They are held OUT of this row pending a
# counterfactual sweep at small budgets, not excluded on a settled argument.
# CMP-7 added 2026-08-20 (contract amendment 4): it runs a fixture emitting 2,964
# characters per call and reads the result back three times. Below that budget the
# noise stops arriving intact, which is a legal value moving what the task measures —
# and in the direction that makes it EASIER, since less bulk means less competition
# for the buried fact. So it is a guard on the row, not a miner: the regression it
# watches for is a budget (or an offload strategy) that changes how much bulk the
# fact has to survive.
_TOOL_RESULT_READERS = (
    "A2", "B1", "B2", "B3", "C3", "CMP-7", "D1", "D2", "D3",
    "E1", "E2", "E3", "E4", "F1", "F2", "G3",
)  # fmt: skip

# The budget below which each task's largest tool result stops arriving intact.
# Measured 2026-08-15 by building the real fixture and running the pinned command; no
# model calls. Kept as the evidence for the rows above, so the classification is
# checkable rather than remembered.
MEASURED_BREAK_BUDGETS: dict[str, int] = {
    "CMP-7": 2964,  # the pinned noise fixture's own output, measured by running it
    "B1": 156,  # largest seeded source; bash output is 232
    "B2": 183,  # largest seeded source; bash output is 331
    "B3": 103,  # largest seeded source; bash output is 154
    "C3": 21,  # runtime.txt; its verdict also runs through the model's write decision
    "D1": 8,  # the answer itself, read back out of the tool result by `tool_texts`
    "D2": 6,  # same shape as D1
    "F1": 113,  # already listed before this pass
}

# Every task that makes any tool call at all, so the per-turn round budget binds
# it. A different mechanism from result truncation, and therefore a different set
# — sharing one list between the two was a false claim about both.
# G5 was missing here until the derived coverage map (`loop/observed_coverage.py`)
# flagged it against the recorded runs: it registers `write_file_tool` and makes two to
# three calls per attempt. It belongs for the reason the list states — a per-turn round
# budget binds anything that takes rounds, and a legal `max_tool_steps` of 1 or 2 would
# bind G5. This is what a hand-maintained table costs: the row was added for compaction
# and nobody re-read this one.
# CMP-7 makes three `bash` calls across three turns and needs a call-then-answer
# round on each, so a `max_tool_steps` of 1 binds it.
_TOOL_USERS = (
    "A2", "B1", "B2", "B3", "C1", "C2", "C3", "CMP-7", "D1", "D2", "D3",
    "E1", "E2", "E3", "E4", "F1", "F2", "G3", "G5",
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
    # A4 mines (needle at the END, so `keep_head` loses it); A5 guards (needle in the
    # HEAD, so a legal `tail_fraction` near 1 loses it). The pair pins opposite ends of
    # the same interval, which is why one could not stand in for the other and why this
    # knob went unguarded until A5 was authored.
    "file_injection": {"observers": ("A4", "A5"), "miners": ("A4",), "guards": ("A5",)},
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
    # Guards are every observer whose prior is not `fail`, which is now most of the row:
    # each can REGRESS under a small enough budget or under `keep_head`, and that is
    # what a guard is for. E3/E4 are excluded from guards by their `fail` priors — a
    # task at 0.0 cannot drop further — and they stay miners because an offload-shaped
    # strategy is what can turn them green.
    "tool_output": {
        "observers": _TOOL_RESULT_READERS,
        "miners": ("E1", "E3", "E4"),
        "guards": (
            "A2",
            "B1",
            "B2",
            "B3",
            "C3",
            "CMP-7",
            "D1",
            "D2",
            "D3",
            "E2",
            "F1",
            "F2",
            "G3",
        ),  # fmt: skip
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
    #
    # CMP-7 added 2026-08-20 (contract amendment 4). It is the ONLY compaction task
    # that runs at the shipped window — G2/G4/G5 and CMP-5/CMP-6 each pin their own
    # `context_limit`, which makes them structurally blind to this knob — so it is the
    # only one whose compaction behavior a change here can move at all. A guard, not a
    # miner: lowering the window compacts it harder, raising it may stop compaction
    # firing and trip the task's own setup guard.
    "default_context_limit": {
        "observers": ("A1", "A3", "CMP-7"),
        "miners": (),
        "guards": ("A1", "A3", "CMP-7"),
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
    #
    # G5 added 2026-08-15, after the derived coverage map flagged it against the runs.
    # Its own commit calls it "the observer that made compaction-v4 measurable" and it
    # never entered this row. The claim: G5 runs at a 900-token limit and requires at
    # least two compactions, and the state it grades is not STATED in conversation the
    # way G4's is — the agent calls `write_file` itself, so the file list exists only as
    # a property of the tool calls it made. A strategy that lifts that list out of the
    # tool calls and re-attaches it passes; one that leaves it to the summarizer's prose
    # does not. That is a discrimination BETWEEN strategies, which is what this row is
    # for.
    #
    # CORRECTED 2026-08-15, same day, after review. G5 was first added here as a MINER
    # and that was wrong twice over. Mechanically: carbon's checkpoint lifts file paths
    # out of `tool_calls` deterministically (`harness/checkpoint.py` `file_ops`) and
    # reattaches them independently of the summary prose — which is exactly what makes
    # G5 a good STRATEGY observer, and the same fact means better wording cannot mine
    # it. Empirically: `iterations/iter-02/clusters.json` already records that no
    # `compaction_prompt` edit fixes the structural failure, and G5 is 3/3 in the v7
    # baseline, so there is nothing to turn green. Its `uncertain` registry prior passes
    # the letter of the miner rule while the measurement denies its substance — a prior
    # is a claim about the suite as authored, not a reading of the current baseline.
    #
    # So: observer and GUARD on both rows. That is the real coverage this table gained —
    # a compaction candidate that loses the file list drops G5 from 1.00, and nothing
    # was watching that before. A weak guard, on the `uncertain` prior, exactly like D3.
    #
    # CMP-5/6/7 added 2026-08-20 (phase2c-guards-contract.md §6). Observers on the
    # same argument G2/G4/G5 already carry — each drives repeated compaction and
    # grades what came back through it, so every sub-field of this object (strategy,
    # keep_head, keep_tail, trigger_fraction, summary_max_tokens) can move their
    # verdicts. GUARDS, never miners, and that asymmetry is the point of the phase:
    # they exist to catch a compaction fix mined from G4 that generalizes only to
    # G4's shape. Mining against them would tune the candidate on the very tasks
    # that are supposed to be able to refuse it.
    #
    # Each covers an axis the existing rows cannot: CMP-5 a SUPERSEDED decision (both
    # G2 and G4 grade retention alone, so a summarizer that keeps every decision and
    # loses their status passes both); CMP-6 MEANING with no sentinel to match on;
    # CMP-7 a fact competing with bulky tool output at the DEFAULT window rather than
    # a pinned small one.
    "compaction": {
        "observers": ("A1", "G2", "G4", "G5", "CMP-5", "CMP-6", "CMP-7"),
        "miners": ("G4",),
        "guards": ("A1", "G2", "G5", "CMP-5", "CMP-6", "CMP-7"),
    },
    # Same observers as compaction had BEFORE the Phase 2c guards, and deliberately
    # not extended with them: this row is about the summarizer's PROMPT text, and the
    # three new tasks have not been measured against a rewritten prompt at all. The
    # contract extends `compaction` only, and a coverage row is an authored claim
    # about carbon's internals — copying it across because the two rows have always
    # matched would be exactly the padding the module docstring warns about.
    # H2 stays out for the sharper reason: it detects the summarizer by
    # payload SHAPE precisely so that rewriting this knob cannot fool it, which by
    # construction makes it blind to the knob. `tests/test_registry.py` asserts
    # that invariance directly.
    "compaction_prompt": {
        "observers": ("A1", "G2", "G4", "G5"),
        "miners": ("G4",),
        "guards": ("A1", "G2", "G5"),
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
    # tool_exposure (carbon seam 3, Select; program/tool-exposure-knob +
    # program/sel-task-authoring). Every tool-carrying agent routes its offered
    # set through `Agent._exposed_specs`, but this row claims only the tasks
    # DESIGNED to discriminate exposure values, plus the roadmap's named
    # cross-section guards. The SEL tasks (cluster_s) each pin one axis: SEL-2
    # is the miner (one relevant tool among 30 plausible decoys — the map's
    # prerequisite shape, `uncertain` prior because nothing has measured a
    # 31-tool registry); SEL-3 (near-duplicate decoy), SEL-4 (exposure order:
    # any query_match k <= 30 removes its needed tool, proven offline through
    # carbon's own selector), and SEL-5 (vocabulary mismatch: any k <= 6
    # removes the only tool that can answer) guard the ways a filtering value
    # can win the miner and quietly break selection. D1/D2 are the roadmap's
    # own guard clause — "the needed calculator must never be selected away" —
    # a registry-of-one shape none of the SEL fixtures covers. NOTE: the SEL
    # section is UNCALIBRATED (tests/test_sel_tasks.py proves its names appear
    # in no committed calibration artifact); this row states reachability, not
    # measured rates, and no SEL number gates anything until the section's own
    # null campaign runs (decision 20).
    "tool_exposure": {
        "observers": ("SEL-2", "SEL-3", "SEL-4", "SEL-5", "D1", "D2"),
        "miners": ("SEL-2",),
        "guards": ("SEL-3", "SEL-4", "SEL-5", "D1", "D2"),
    },
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
    # `tool_output` has NO entries, and the three that were here are gone rather than
    # annotated. This set is executable: `unlisted_with_activity()` removes every task
    # named here from its review queue, so an entry known to be wrong does not merely
    # sit there being wrong — it actively silences the warning about itself.
    #
    #   A2 — claimed a prior of `fail` where the registry says `pass`, and argued only
    #        about the budget while the same knob carries the STRATEGY: a proposed
    #        `keep_head` drops the tail its sentinel sits in.
    #   C1, C2 — raw-result capture protects their LEAK predicate, which is the half the
    #        rationale addressed. Both verdicts also require a correct functional reply,
    #        and carbon sends the post-policy truncated result back to the model, so a
    #        legal value can move the other half.
    #   C3 — the verifier scans files, but the model consumes tool results before
    #        deciding what to write, so the causal path survives. Iteration 3's 16k
    #        candidate moved C3 by +0.4.
    #
    # Removing them does NOT promote them into `KNOB_COVERAGE` — that would be a
    # decision about what the loop may tune. It only stops suppressing an unresolved
    # warning, which is a cleanup. A2 and C3 have since been classified as guards; C1
    # and C2 stay in the queue pending a counterfactual sweep at small budgets.
    #
    # G5 is restored, because its argument is MECHANISM and survives abandoning the
    # floor. Its only tool is `write_file`, and its verdict reads the file list off the
    # tool CALLS, never off their results — so no budget can move it even at a value
    # that would cut the 35-character result to nothing.
    "tool_output": {
        "G5": (
            "Only tool is `write_file`; its verdict reads the file list off the tool "
            "CALLS and never off their results, so no budget moves it at any value. A "
            "mechanism argument, not a size one — it survives the floor being abandoned."
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
#
# EMPTY as of 2026-08-15. `file_injection` was the sole entry — "A4 is the suite's only
# `@path` sender and it is the miner. Guarding this knob needs a second `@path` task
# with a passing prior." A5 is that task: same `@path` mechanism, needle in the HEAD
# window instead of the tail, `pass` prior so it can actually regress. The entry left
# because someone wrote the task, which is the only way an entry here may leave.
UNGUARDED_KNOBS: dict[str, str] = {}
