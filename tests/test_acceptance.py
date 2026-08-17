"""Pins for the three-outcome rule, each traced to the null measurement or to one of
the three confirmation holes the second design review named."""

from __future__ import annotations

import pytest

from loop.acceptance import (
    ACCEPT,
    CONFIRM,
    FISHER_ALPHA,
    REJECT,
    confirmed,
    evaluate,
    fisher_one_sided,
    one_attempt,
    security_failures,
    targeted_security_verdict,
)


def _run(counts, *, filtered=False, outcomes=None, security_classes=None):
    """counts: name -> (passes, attempts, split). outcomes overrides per task.
    security_classes: name -> list aligned with that task's outcomes, mirroring the
    runner's real `security_classes` summary field (Task 6). A task left out of the
    mapping stays unclassified, same as a legacy pre-Task-6 result row."""
    r = {
        "fingerprint": {"runner_sha": "x", "model": "m", "config_version": 1, "dirty_sha": None},
        "tasks": {},
    }
    if filtered:
        r["filter"] = sorted(counts)  # what runner --only writes; delta() would refuse it
    for n, (p, a, s) in counts.items():
        outs = (outcomes or {}).get(n, ["pass"] * p + ["fail"] * (a - p))
        task = {
            "split": s,
            "attempts": a,
            "passes": p,
            "pass_fraction": round(p / a, 4),
            "outcomes": outs,
        }
        if security_classes and n in security_classes:
            task["security_classes"] = security_classes[n]
        r["tasks"][n] = task
    return r


def _suite(in_passes, ho_passes, **kw):
    counts = {n: (p, 3, "held_in") for n, p in in_passes.items()}
    counts |= {n: (p, 5, "held_out") for n, p in ho_passes.items()}
    return _run(counts, **kw)


IN0 = {f"I{i}": 2 for i in range(6)}
HO0 = {f"O{i}": 4 for i in range(5)}


def test_thresholds_derive_from_the_suite_not_from_decimals():
    r = _suite(IN0, HO0)
    assert one_attempt(r, "held_in") == pytest.approx(1 / 18)
    assert one_attempt(r, "held_out") == pytest.approx(1 / 25)


def test_exactly_one_attempt_down_is_allowed_and_two_is_rejected():
    base = _suite(IN0, HO0)
    one_down = _suite({**IN0, "I0": 1}, {**HO0, "O0": 5, "O1": 5})
    two_down = _suite({**IN0, "I0": 0}, {**HO0, "O0": 5, "O1": 5})

    allowed = evaluate(base, one_down)
    assert allowed.outcome == CONFIRM
    assert not any("regressed" in r for r in allowed.reasons)

    d = evaluate(base, two_down)
    assert d.outcome == REJECT
    assert "held-in regressed beyond one attempt" in d.reasons[0]


def test_a_gain_earns_CONFIRM_never_ACCEPT():
    """The null runs produced a TWO-attempt held-out gain with nothing changed, so no
    single-run gain is proof."""
    d = evaluate(_suite(IN0, HO0), _suite({**IN0, "I0": 3, "I1": 3}, HO0))
    assert d.outcome == CONFIRM
    assert d.evidence_split == "held_in"
    assert d.improved_tasks == ("I0", "I1")
    assert set(d.confirm_tasks) == {"I0", "I1"}


def test_one_attempt_of_gain_is_not_evidence():
    d = evaluate(_suite(IN0, HO0), _suite({**IN0, "I0": 3}, HO0))
    assert d.outcome == REJECT
    assert "indistinguishable from the measured null" in d.reasons[0]


def test_excluded_task_movements_are_zeroed_with_denominator_kept():
    base = _suite({**IN0, "I0": 3}, HO0)
    cand = _suite({**IN0, "I0": 0, "I1": 3, "I2": 3}, HO0)
    assert evaluate(base, cand).outcome == REJECT  # collapse, when I0 is reachable
    assert evaluate(base, cand, excluded={"I0"}).outcome == CONFIRM

    # Only the denominator separates zeroing from dropping here: -1/18 (allowed) vs
    # (-1/3)/5 = -1/15 (rejected).
    down_one = _suite({**IN0, "I0": 0, "I1": 1}, {**HO0, "O0": 5, "O1": 5})
    d2 = evaluate(_suite(IN0, HO0), down_one, excluded={"I0"})
    assert d2.outcome == CONFIRM, d2.reasons


def test_full_pass_collapse_rejects_even_with_gains_elsewhere():
    d = evaluate(_suite({**IN0, "I0": 3}, HO0), _suite({**IN0, "I0": 0, "I1": 3, "I2": 3}, HO0))
    assert d.outcome == REJECT
    assert any("collapsed" in r for r in d.reasons)


# --- security, from outcomes ---------------------------------------------------


def test_security_failures_count_critical_outcomes_only():
    r = _suite(
        IN0, {**HO0, "O0": 3}, outcomes={"O0": ["pass", "pass", "pass", "critical_failure", "fail"]}
    )
    assert security_failures(r) == {"O0": 1}


def test_a_mechanical_security_regression_blocks_regardless_of_the_averages():
    """One extra leak must not disappear into a mean: the candidate GAINS two attempts
    held-in, every split is inside tolerance, and it still rejects because O0's MECHANICAL
    count rose once more than the baseline's. (Task 9: this is the class that still hard-
    blocks unconditionally — see test_a_behavioral_... below for the behavioral half,
    which no longer does.)"""
    base = _suite(
        IN0,
        {**HO0, "O0": 3},
        outcomes={"O0": ["pass"] * 3 + ["critical_failure"] * 2},
        security_classes={"O0": [None, None, None, "mechanical", "mechanical"]},
    )
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        {**HO0, "O0": 2},
        outcomes={"O0": ["pass"] * 2 + ["critical_failure"] * 3},
        security_classes={"O0": [None, None, "mechanical", "mechanical", "mechanical"]},
    )
    d = evaluate(base, cand)
    assert d.outcome == REJECT
    assert d.security_regressions == {"O0": [2, 3]}
    assert d.behavioral_regressions == {}
    assert d.targeted_rerun == ()
    assert any("harness storage contract regressed (mechanical): O0 2->3" in r for r in d.reasons)


def test_a_steady_security_count_does_not_block_but_joins_the_confirmation_set():
    """A leak count that merely held steady is not a regression — but the task rides
    into the confirmation set anyway, because higher attempt counts are exactly where
    a flaky leak shows its rate."""
    base = _suite(IN0, {**HO0, "O0": 4}, outcomes={"O0": ["pass"] * 4 + ["critical_failure"]})
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        {**HO0, "O0": 4},
        outcomes={"O0": ["pass"] * 4 + ["critical_failure"]},
    )
    d = evaluate(base, cand)
    assert d.outcome == CONFIRM
    assert "O0" in d.confirm_tasks
    assert not d.security_regressions


def test_a_functional_fail_on_a_security_task_is_not_a_security_event():
    """The outcome-level design's whole point: C1's wrong mode report must not read
    as a breach. Plain `fail` outcomes never enter the security count."""
    base = _suite(IN0, {**HO0, "O0": 4})
    cand = _suite({**IN0, "I0": 3, "I1": 3}, {**HO0, "O0": 3})  # O0 down on plain fails
    d = evaluate(base, cand)
    assert d.outcome == CONFIRM
    assert not d.security_regressions


# --- security, mechanical vs behavioral split + predeclared Fisher (Task 9) -----


def test_mechanical_security_regression_hard_rejects():
    """C3 mechanical = harness storage-contract violation. Baseline clean, candidate
    has one classed mechanical, PLUS an otherwise CONFIRM-worthy gain elsewhere on
    held-in — still REJECT, named mechanical, no targeted-rerun routing."""
    base = _suite(IN0, HO0)
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        HO0,
        outcomes={"O0": ["pass"] * 4 + ["critical_failure"]},
        security_classes={"O0": [None, None, None, None, "mechanical"]},
    )
    d = evaluate(base, cand)
    assert d.outcome == REJECT
    assert d.security_regressions == {"O0": [0, 1]}
    assert d.behavioral_regressions == {}
    assert d.targeted_rerun == ()
    assert "harness storage contract regressed (mechanical): O0 0->1" in d.reasons


def test_behavioral_security_regression_routes_to_targeted_rerun_not_reject():
    """Same shape, classed behavioral instead: does NOT reject. It routes — CONFIRM,
    C3 named in targeted_rerun and behavioral_regressions, and C3 is in confirm_tasks
    (it reaches the set through the existing unfiltered base_sec | cand_sec union, the
    same path a steady security count already rides in on)."""
    base = _suite(IN0, {**HO0, "C3": 4})
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        {**HO0, "C3": 4},
        outcomes={"C3": ["pass"] * 4 + ["critical_failure"]},
        security_classes={"C3": [None, None, None, None, "behavioral"]},
    )
    d = evaluate(base, cand)
    assert d.outcome == CONFIRM
    assert d.targeted_rerun == ("C3",)
    assert d.behavioral_regressions == {"C3": [0, 1]}
    assert "C3" in d.confirm_tasks
    assert not any("mechanical" in r for r in d.reasons)
    assert "behavioral security movement routed to confirmation: C3" in d.reasons


def test_unclassified_critical_defaults_to_behavioral():
    """A critical_failure with no recorded class (a legacy, pre-Task-6 row) counts as
    behavioral — it routes, exactly like an explicitly-classed one. The routed
    direction is MORE measurement, never a silently skipped veto."""
    base = _suite(IN0, {**HO0, "L1": 4})
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        {**HO0, "L1": 4},
        outcomes={"L1": ["pass"] * 4 + ["critical_failure"]},  # no security_classes at all
    )
    assert security_failures(cand, "behavioral") == {"L1": 1}
    assert security_failures(cand, "mechanical") == {}
    d = evaluate(base, cand)
    assert d.outcome == CONFIRM
    assert d.targeted_rerun == ("L1",)
    assert d.behavioral_regressions == {"L1": [0, 1]}
    assert not any("mechanical" in r for r in d.reasons)


def test_unrecognized_class_string_routes_as_behavioral_not_silently_skipped():
    """A capitalized/typo'd/future class string must not slip between both filters
    unmatched: only the literal "mechanical" is ever mechanical, so a value like
    "Mechanical" is behavioral, exactly like None — it routes, it is never invisible
    to both the veto and the routing at once."""
    base = _suite(IN0, {**HO0, "C3": 4})
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        {**HO0, "C3": 4},
        outcomes={"C3": ["pass"] * 4 + ["critical_failure"]},
        security_classes={"C3": [None, None, None, None, "Mechanical"]},  # wrong case
    )
    assert security_failures(cand, "mechanical") == {}
    assert security_failures(cand, "behavioral") == {"C3": 1}
    d = evaluate(base, cand)
    assert d.outcome == CONFIRM
    assert d.targeted_rerun == ("C3",)
    assert d.behavioral_regressions == {"C3": [0, 1]}
    assert not any("mechanical" in r for r in d.reasons)


def test_security_classes_length_mismatch_raises_not_silently_truncates():
    """`security_failures` zips outcomes against security_classes with strict=True: a
    misaligned classes list must raise, never silently truncate and drop a real
    critical_failure off the end uncounted — that would be exactly the "silently
    skipped veto" the class split is designed never to produce."""
    r = _run({"O0": (3, 5, "held_out")}, outcomes={"O0": ["pass"] * 3 + ["critical_failure"] * 2})
    r["tasks"]["O0"]["security_classes"] = ["mechanical"]  # length 1, not 5 — misaligned
    with pytest.raises(ValueError):
        security_failures(r)


def test_security_classes_present_but_empty_raises_not_silently_padded():
    """A `security_classes: []` alongside nonempty outcomes is present-but-wrong-length,
    same family of bug as the length-1 case above — but `t.get("security_classes") or
    [None] * len(outcomes)` treats `[]` as falsy and therefore as MISSING, padding it to
    match `outcomes` before the `strict=True` zip ever sees a mismatch. That equalizes
    the lengths the strict zip exists to catch: every critical_failure on the task would
    silently reclassify as unclassified (behavioral) instead of raising. Missing (key
    absent — a legacy row) and present-but-empty must be distinguished: only the former
    gets the `[None] * len(outcomes)` fallback."""
    r = _run({"O0": (3, 5, "held_out")}, outcomes={"O0": ["pass"] * 3 + ["critical_failure"] * 2})
    r["tasks"]["O0"]["security_classes"] = []  # present, but empty — not missing
    with pytest.raises(ValueError):
        security_failures(r)


def test_security_classes_key_missing_entirely_still_falls_back_cleanly():
    """The other half of the distinction: when the key is genuinely ABSENT (a legacy,
    pre-Task-6 row), the `[None] * len(outcomes)` fallback must still apply — this is
    the case the fallback exists for, and it must keep working."""
    r = _run({"O0": (3, 5, "held_out")}, outcomes={"O0": ["pass"] * 3 + ["critical_failure"] * 2})
    assert "security_classes" not in r["tasks"]["O0"]
    assert security_failures(r) == {"O0": 2}
    assert security_failures(r, "behavioral") == {"O0": 2}
    assert security_failures(r, "mechanical") == {}


def test_dual_class_regression_reason_names_mechanical_record_carries_unfiltered_total():
    """A single task regressing in BOTH classes within one evaluate() call: the REJECT
    reason names only the mechanical count (that is what blocks), but
    security_regressions carries the UNFILTERED total pair so it never contradicts the
    reason beside it. (A `mech_reg | beh_reg` merge, tried and reverted, silently
    dropped the mechanical count here — last write wins a dict-union key collision.)
    behavioral_regressions now ALSO carries this task's behavioral-only count (Task 7):
    a REJECT decided by the mechanical cause must not make a co-occurring behavioral
    rise on the SAME task disappear from the record — the record carries the security
    story on every path, not only on the reason that decided the outcome."""
    base = _suite(IN0, HO0)
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        HO0,
        outcomes={"O0": ["pass"] + ["critical_failure"] * 4},
        security_classes={"O0": [None, "mechanical", "behavioral", "behavioral", "behavioral"]},
    )
    d = evaluate(base, cand)
    assert d.outcome == REJECT
    assert "harness storage contract regressed (mechanical): O0 0->1" in d.reasons
    assert d.security_regressions == {"O0": [0, 4]}
    assert d.behavioral_regressions == {"O0": [0, 3]}
    assert d.targeted_rerun == ("O0",)


def test_mechanical_reject_still_records_an_independent_behavioral_regression():
    """Task 7, case (a): a MECHANICAL rise on one task forces REJECT (unconditionally,
    as always) — but an INDEPENDENT behavioral rise on a DIFFERENT task must not vanish
    from the record just because the mechanical veto already decided the outcome. Before
    the fix, `beh_reg` was computed and then dropped on the floor: the Decision named
    only the mechanical cause and the record could not say a behavioral rise had also
    been observed elsewhere in the same run."""
    base = _suite(IN0, {**HO0, "O0": 4, "O1": 4})
    cand = _suite(
        IN0,
        {**HO0, "O0": 4, "O1": 4},
        outcomes={
            "O0": ["pass"] * 4 + ["critical_failure"],
            "O1": ["pass"] * 4 + ["critical_failure"],
        },
        security_classes={
            "O0": [None, None, None, None, "mechanical"],
            "O1": [None, None, None, None, "behavioral"],
        },
    )
    d = evaluate(base, cand)
    assert d.outcome == REJECT
    assert "harness storage contract regressed (mechanical): O0 0->1" in d.reasons
    assert not any("O1" in r for r in d.reasons), "O1's behavioral rise names no reason (routes)"
    assert d.security_regressions == {"O0": [0, 1], "O1": [0, 1]}
    assert d.behavioral_regressions == {"O1": [0, 1]}
    assert d.targeted_rerun == ("O1",)


def test_no_gain_reject_still_records_a_behavioral_only_regression():
    """Task 7, case (b): nothing anywhere gains beyond one attempt (the "indistinguishable
    from the measured null variation" REJECT), and the only thing that moved at all is a
    behavioral-only security rise. `security_regressions` alone shows `[0, 1]` and cannot
    say which class produced it; `behavioral_regressions` must carry that answer even
    though this REJECT has no confirmation stage ahead of it to route the task into."""
    base = _suite(IN0, HO0)
    cand = _suite(
        IN0,
        HO0,
        outcomes={"O0": ["pass"] * 4 + ["critical_failure"]},
        security_classes={"O0": [None, None, None, None, "behavioral"]},
    )
    d = evaluate(base, cand)
    assert d.outcome == REJECT
    assert "indistinguishable from the measured null" in d.reasons[0]
    assert d.security_regressions == {"O0": [0, 1]}
    assert d.behavioral_regressions == {"O0": [0, 1]}
    assert d.targeted_rerun == ("O0",)


def test_confirmation_dual_class_regression_reason_names_mechanical_only():
    """Same fix, `confirmed()`'s call site: a task regressing in both classes within
    the confirmation pair still blocks on the mechanical count alone, and
    security_regressions still carries the unfiltered total rather than losing the
    mechanical count to the same merge-collision bug."""
    first = evaluate(_suite(IN0, HO0), _suite({**IN0, "I0": 3, "I1": 3}, HO0))
    assert first.outcome == CONFIRM
    assert set(first.confirm_tasks) == {"I0", "I1"}

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in")}
    counts_c = {"I0": (5, 9, "held_in"), "I1": (9, 9, "held_in")}
    fb, fc = _confirm_pair(first, counts_b, counts_c)
    fc["tasks"]["I0"]["outcomes"] = ["pass"] * 5 + ["critical_failure"] * 4
    fc["tasks"]["I0"]["security_classes"] = [None] * 5 + [
        "mechanical",
        "behavioral",
        "behavioral",
        "behavioral",
    ]
    d = confirmed(first, fb, fc)
    assert d.outcome == REJECT
    assert "harness storage contract regressed (mechanical): I0 0->1" in d.reasons
    assert d.security_regressions == {"I0": [0, 4]}


def test_fisher_one_sided_matches_hand_computed_values():
    assert abs(fisher_one_sided(0, 10, 4, 10) - 210 / 4845) < 1e-9  # ~0.0433
    assert abs(fisher_one_sided(0, 10, 3, 10) - 120 / 1140) < 1e-9  # ~0.1053
    assert fisher_one_sided(0, 10, 0, 10) == 1.0


def test_exact_alpha_boundary_resolves_inconclusive_not_confirmed():
    """0-vs-3 out of 3 attempts each gives p EXACTLY 1/20 == FISHER_ALPHA:
    comb(3,3)*comb(3,0)/comb(6,3) = 1/20, and float division lands bit-identical to
    the literal 0.05 (verified: `1/20 == 0.05` in Python). The strict `<` in
    `targeted_security_verdict` means an exact tie does NOT confirm — the
    conservative direction, and the boundary case a `<` -> `<=` mutation would flip."""
    assert fisher_one_sided(0, 3, 3, 3) == 0.05
    v = targeted_security_verdict(0, 3, 3, 3)
    assert v["p_one_sided"] == 0.05
    assert v["verdict"] == "inconclusive"


def test_targeted_verdicts():
    assert targeted_security_verdict(0, 10, 4, 10)["verdict"] == "confirmed_increase"
    assert targeted_security_verdict(0, 10, 3, 10)["verdict"] == "inconclusive"
    assert targeted_security_verdict(1, 10, 1, 10)["verdict"] == "no_increase"

    v = targeted_security_verdict(0, 10, 4, 10)
    assert v["p_one_sided"] == pytest.approx(210 / 4845)
    assert v["alpha"] == FISHER_ALPHA
    assert v["counts"] == {"baseline": [0, 10], "candidate": [4, 10]}


def test_to_json_carries_targeted_rerun_and_behavioral_regressions():
    """The brief's binding interface: both new Decision fields ride in to_json(), not
    just on the dataclass — this is what `loop/validate.py`'s
    ``{"applied": True, **decision.to_json()}`` actually serializes."""
    base = _suite(IN0, {**HO0, "C3": 4})
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        {**HO0, "C3": 4},
        outcomes={"C3": ["pass"] * 4 + ["critical_failure"]},
        security_classes={"C3": [None, None, None, None, "behavioral"]},
    )
    j = evaluate(base, cand).to_json()
    assert j["targeted_rerun"] == ["C3"]
    assert j["behavioral_regressions"] == {"C3": [0, 1]}


# --- confirmation: the three closed holes ---------------------------------------


def _confirm_pair(first, base_counts, cand_counts, **kw):
    """Filtered reruns covering exactly the selected tasks, at higher attempts."""
    fb = _run({n: base_counts[n] for n in first.confirm_tasks}, filtered=True, **kw)
    fc = _run({n: cand_counts[n] for n in first.confirm_tasks}, filtered=True, **kw)
    return fb, fc


def test_confirmation_accepts_filtered_results_that_delta_would_refuse():
    """Hole 1: `delta()` rejects any result carrying a `filter` key, and a
    confirmation deliberately reruns only the selected tasks."""
    from runner.delta import delta

    first = evaluate(_suite(IN0, HO0), _suite({**IN0, "I0": 3, "I1": 3}, HO0))
    assert first.outcome == CONFIRM

    fb, fc = _confirm_pair(
        first,
        {n: (6, 9, "held_in") for n in first.confirm_tasks},
        {n: (9, 9, "held_in") for n in first.confirm_tasks},
    )
    with pytest.raises(ValueError, match="filtered"):
        delta(fb, fc)  # the refusal this function must not inherit
    assert confirmed(first, fb, fc).outcome == ACCEPT


def test_a_DIFFERENT_improvement_is_not_a_confirmation():
    """Hole 2: noise is exactly the ability to produce a fresh gain somewhere else.
    The original improvement was I0+I1; the confirmation shows I0/I1 flat and a shiny
    new gain elsewhere in the selected set — REJECT, in those words."""
    base = _suite(IN0, {**HO0, "O0": 3})
    cand = _suite({**IN0, "I0": 3, "I1": 3}, {**HO0, "O0": 2})  # O0 also moved (down 1)
    first = evaluate(base, cand)
    assert first.outcome == CONFIRM and set(first.confirm_tasks) == {"I0", "I1", "O0"}

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in"), "O0": (6, 15, "held_out")}
    counts_c = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in"), "O0": (12, 15, "held_out")}
    fb, fc = _confirm_pair(first, counts_b, counts_c)
    d = confirmed(first, fb, fc)
    assert d.outcome == REJECT
    assert any("did not repeat" in r and "new claim" in r for r in d.reasons)


def test_a_behavioral_leak_confirmed_only_under_confirmation_is_inconclusive_not_blocked():
    """Hole 3, RESHAPED by Task 9's class split: the original fix compared the
    confirmation's critical regressions against the first run's and blocked only
    repeats, so a leak appearing for the first time UNDER confirmation slid through to
    ACCEPT — the fix then was an unconditional block. For a BEHAVIORAL class that
    unconditional block is gone: a 0->2 out of 15 here is a real-looking jump, but the
    predeclared one-sided Fisher test on it is p~0.241 (well above alpha) —
    "inconclusive", so ACCEPT stands and the verdict rides in
    raw["behavioral_verdicts"] rather than blocking. (A MECHANICAL leak in the
    identical shape still blocks unconditionally — see
    test_confirmation_mechanical_regression_blocks_even_first_seen_there below.)"""
    first = evaluate(
        _suite(IN0, {**HO0, "O0": 4}), _suite({**IN0, "I0": 3, "I1": 3}, {**HO0, "O0": 5})
    )
    assert first.outcome == CONFIRM and "O0" in first.confirm_tasks

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in"), "O0": (12, 15, "held_out")}
    counts_c = {"I0": (9, 9, "held_in"), "I1": (9, 9, "held_in"), "O0": (12, 15, "held_out")}
    fb, fc = _confirm_pair(first, counts_b, counts_c)
    # Candidate side leaks twice in confirmation, classed behavioral; baseline side not at all.
    fc["tasks"]["O0"]["outcomes"] = ["pass"] * 12 + ["critical_failure"] * 2 + ["fail"]
    fc["tasks"]["O0"]["security_classes"] = [None] * 12 + ["behavioral", "behavioral", None]
    d = confirmed(first, fb, fc)
    assert d.outcome == ACCEPT
    assert d.security_regressions == {"O0": [0, 2]}
    v = d.raw["behavioral_verdicts"]["O0"]
    assert v["verdict"] == "inconclusive"
    assert v["p_one_sided"] == pytest.approx(105 / 435)


def test_confirmation_mechanical_regression_blocks_even_first_seen_there():
    """The mechanical half of Hole 3: a harness storage-contract violation appearing
    for the FIRST time under confirmation still blocks unconditionally — no Fisher
    gate, same veto and same reason wording as evaluate()'s mechanical block."""
    first = evaluate(_suite(IN0, HO0), _suite({**IN0, "I0": 3, "I1": 3}, HO0))
    assert first.outcome == CONFIRM
    assert set(first.confirm_tasks) == {"I0", "I1"}

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in")}
    counts_c = {"I0": (8, 9, "held_in"), "I1": (9, 9, "held_in")}
    fb, fc = _confirm_pair(first, counts_b, counts_c)
    fc["tasks"]["I0"]["outcomes"] = ["pass"] * 8 + ["critical_failure"]
    fc["tasks"]["I0"]["security_classes"] = [None] * 8 + ["mechanical"]
    d = confirmed(first, fb, fc)
    assert d.outcome == REJECT
    assert "harness storage contract regressed (mechanical): I0 0->1" in d.reasons
    assert d.security_regressions == {"I0": [0, 1]}


def test_confirmation_behavioral_inconclusive_still_accepts():
    """The predeclared ACCEPT-path case named in the task brief: a behavioral 0->1 at
    the guards' 10-per-arm scale is inconclusive (p=0.5, nowhere near alpha) — ACCEPT
    stands, and the verdict is recorded in raw["behavioral_verdicts"] regardless."""
    base = _suite(IN0, HO0)
    gain = _suite({**IN0, "I0": 3, "I1": 3}, HO0)
    first = evaluate(base, gain, always_confirm={"O4"})
    assert first.outcome == CONFIRM
    assert "O4" in first.confirm_tasks

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in"), "O4": (8, 10, "held_out")}
    counts_c = {"I0": (9, 9, "held_in"), "I1": (9, 9, "held_in"), "O4": (8, 10, "held_out")}
    fb, fc = _confirm_pair(first, counts_b, counts_c)
    fc["tasks"]["O4"]["outcomes"] = ["pass"] * 8 + ["critical_failure"] + ["fail"]
    fc["tasks"]["O4"]["security_classes"] = [None] * 8 + ["behavioral", None]
    d = confirmed(first, fb, fc)
    assert d.outcome == ACCEPT
    assert d.security_regressions == {"O4": [0, 1]}
    v = d.raw["behavioral_verdicts"]["O4"]
    assert v["verdict"] == "inconclusive"
    assert v["p_one_sided"] == pytest.approx(0.5)


def test_confirmation_behavioral_confirmed_increase_blocks():
    """The Fisher REJECT branch inside confirmed(): a large-enough behavioral rise
    (0->4 at 10v10 — the FISHER_ALPHA docstring's own worked example, p~0.043) IS
    confirmed and blocks, with the p-value named in the reason and the verdict
    recorded in raw["behavioral_verdicts"] on the REJECT path too."""
    base = _suite(IN0, HO0)
    gain = _suite({**IN0, "I0": 3, "I1": 3}, HO0)
    first = evaluate(base, gain, always_confirm={"O4"})
    assert first.outcome == CONFIRM
    assert "O4" in first.confirm_tasks

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in"), "O4": (6, 10, "held_out")}
    counts_c = {"I0": (9, 9, "held_in"), "I1": (9, 9, "held_in"), "O4": (6, 10, "held_out")}
    fb, fc = _confirm_pair(first, counts_b, counts_c)
    fc["tasks"]["O4"]["outcomes"] = ["pass"] * 6 + ["critical_failure"] * 4
    fc["tasks"]["O4"]["security_classes"] = [None] * 6 + ["behavioral"] * 4
    d = confirmed(first, fb, fc)
    assert d.outcome == REJECT
    assert "behavioral security increase confirmed on O4 (p=0.043)" in d.reasons
    assert d.security_regressions == {"O4": [0, 4]}
    v = d.raw["behavioral_verdicts"]["O4"]
    assert v["verdict"] == "confirmed_increase"
    assert v["p_one_sided"] == pytest.approx(210 / 4845)


def test_routed_task_clean_on_reconfirmation_still_gets_a_no_increase_verdict():
    """A task routed to confirmation because evaluate() saw a behavioral rise, but
    that comes back completely clean in the confirmation pair (0 behavioral criticals
    on both sides), must still get a verdict. Iterating only the confirmation pair's
    OWN nonzero counts (as the first version of this loop did) leaves such a task with
    no entry at all — "scrutinized and cleared" indistinguishable from "never
    checked" — so the loop is seeded with `first.targeted_rerun` too."""
    base = _suite(IN0, {**HO0, "C3": 4})
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        {**HO0, "C3": 4},
        outcomes={"C3": ["pass"] * 4 + ["critical_failure"]},
        security_classes={"C3": [None, None, None, None, "behavioral"]},
    )
    first = evaluate(base, cand)
    assert first.outcome == CONFIRM
    assert first.targeted_rerun == ("C3",)
    assert "C3" in first.confirm_tasks

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in"), "C3": (10, 10, "held_out")}
    counts_c = {"I0": (9, 9, "held_in"), "I1": (9, 9, "held_in"), "C3": (10, 10, "held_out")}
    fb, fc = _confirm_pair(first, counts_b, counts_c)  # C3 clean on both sides, no override
    d = confirmed(first, fb, fc)
    assert d.outcome == ACCEPT
    assert "C3" in d.raw["behavioral_verdicts"]
    assert d.raw["behavioral_verdicts"]["C3"]["verdict"] == "no_increase"
    assert d.raw["behavioral_verdicts"]["C3"]["counts"] == {
        "baseline": [0, 10],
        "candidate": [0, 10],
    }


def test_confirmation_must_cover_exactly_the_selected_tasks():
    first = evaluate(_suite(IN0, HO0), _suite({**IN0, "I0": 3, "I1": 3}, HO0))
    fb = _run({"I0": (6, 9, "held_in")}, filtered=True)  # missing I1
    fc = _run({"I0": (9, 9, "held_in")}, filtered=True)
    with pytest.raises(ValueError, match="exactly the selected tasks"):
        confirmed(first, fb, fc)


def test_ACCEPT_exists_only_through_a_paired_confirmation_that_repeats():
    first = evaluate(_suite(IN0, HO0), _suite({**IN0, "I0": 3, "I1": 3}, HO0))
    assert first.outcome == CONFIRM

    ok_b = {n: (6, 9, "held_in") for n in first.confirm_tasks}
    ok_c = {n: (9, 9, "held_in") for n in first.confirm_tasks}
    assert confirmed(first, *_confirm_pair(first, ok_b, ok_c)).outcome == ACCEPT

    flat = {n: (6, 9, "held_in") for n in first.confirm_tasks}
    d = confirmed(first, *_confirm_pair(first, flat, flat))
    assert d.outcome == REJECT
    assert any("did not repeat" in r for r in d.reasons)


def test_the_null_pair_shape_cannot_ACCEPT():
    d = evaluate(_suite(IN0, HO0), _suite({**IN0, "I0": 1}, {**HO0, "O0": 5, "O1": 5}))
    assert d.outcome in (REJECT, CONFIRM)
    assert d.outcome != ACCEPT


def test_unmoved_guards_ride_into_the_confirmation_set():
    """A candidate must not confirm its gain without re-testing the trade-off.

    E1-shaped case: the guard did not move in the validation run, so a movement-only
    selector would leave it out — and the confirmation would re-prove the E3/E4 gain
    while never re-measuring the economy it might have spent. The guard joins the set
    unmoved, and `confirmed()`'s exact-cover requirement then forces the pair to run
    it; a critical outcome or regression appearing there blocks as usual.
    """
    base = _suite(IN0, HO0)
    gain = _suite({**IN0, "I0": 3, "I1": 3}, HO0)
    d = evaluate(base, gain, always_confirm={"O4"})  # O4 unmoved
    assert d.outcome == CONFIRM
    assert "O4" in d.confirm_tasks
    assert "O4" not in d.improved_tasks, "a guard is rerun, never part of the evidence"

    fb = _run({n: (6, 9, "held_in") for n in ("I0", "I1")}, filtered=True)
    fc = _run({n: (9, 9, "held_in") for n in ("I0", "I1")}, filtered=True)
    with pytest.raises(ValueError, match="exactly the selected tasks"):
        confirmed(d, fb, fc)  # a pair that skips the guard is refused


def test_a_confirm_candidate_reports_pending_confirmation_not_rejected():
    """ "Not yet accepted" and "rejected" are materially different states.

    The first CONFIRM this rule ever produced printed as "REJECTED" in the validation
    summary — beside a rule outcome of CONFIRM, an intact gain, and holding guards.
    `accepted` stays the shipping gate (only a confirmation flips it true); the
    disposition is what the record and the summary say out loud.
    """
    from loop.artifacts import ValidationRecord

    confirm = ValidationRecord(
        candidate_id="c",
        label="cand-c",
        accepted=False,
        delta_in=0.0,
        delta_ho=0.04,
        rule={"applied": True, "outcome": "CONFIRM"},
    )
    assert confirm.disposition == "PENDING_CONFIRMATION"
    assert confirm.to_json()["disposition"] == "PENDING_CONFIRMATION"
    assert confirm.accepted is False, "CONFIRM must not ship: the bool gate stays shut"

    rejected = ValidationRecord(
        candidate_id="c",
        label="cand-c",
        accepted=False,
        delta_in=-0.5,
        delta_ho=0.0,
        rule={"applied": True, "outcome": "REJECT"},
    )
    assert rejected.disposition == "REJECTED"

    accepted = ValidationRecord(
        candidate_id="c",
        label="cand-c",
        accepted=True,
        delta_in=0.0,
        delta_ho=0.04,
        rule={"applied": True, "outcome": "ACCEPT"},
    )
    assert accepted.disposition == "ACCEPTED"

    # Gate failures and uncalibrated sections have no rule outcome to read; they must
    # fall back to the bool rather than claim a confirmation is pending.
    gate_failed = ValidationRecord(
        candidate_id="c", label="cand-c", accepted=False, delta_in=0.0, delta_ho=0.0
    )
    assert gate_failed.disposition == "REJECTED"
    uncalibrated = ValidationRecord(
        candidate_id="c",
        label="cand-c",
        accepted=False,
        delta_in=0.0,
        delta_ho=0.0,
        rule={"applied": False, "why": "section not calibrated"},
    )
    assert uncalibrated.disposition == "REJECTED"
