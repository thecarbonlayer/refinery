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


def test_confirmation_decisions_carry_behavioral_counts_so_the_pr_cannot_mislabel_them():
    """`pr_body` recovers the mechanical count as total-minus-behavioral.

    `confirmed()` never computed `beh_reg` at all, so every Decision it returned had
    an empty `behavioral_regressions` beside a populated `security_regressions` — and
    the subtraction then rendered a BEHAVIORAL rise as MECHANICAL to the human
    approving the merge. A mechanical rise is the one that rejects a candidate
    outright, so the PR would have stated the opposite of what the rule decided.
    """
    from loop.acceptance import Decision, confirmed

    def res(passes: int, outcomes: list[str], classes: list[str | None]) -> dict:
        return {
            "fingerprint": {"runner_sha": "x"},
            "tasks": {
                "C3": {
                    "split": "held_out",
                    "passes": passes,
                    "attempts": 10,
                    "outcomes": outcomes,
                    "security_classes": classes,
                }
            },
        }

    base = res(10, ["pass"] * 10, [None] * 10)
    cand = res(9, ["pass"] * 9 + ["critical_failure"], [None] * 9 + ["behavioral"])
    first = Decision(
        outcome="CONFIRM",
        reasons=(),
        delta_in=0.0,
        delta_ho=0.05,
        threshold_in=0.01,
        threshold_ho=0.01,
        improved_tasks=("C3",),
        confirm_tasks=("C3",),
    )
    d = confirmed(first, base, cand)
    assert d.security_regressions == {"C3": [0, 1]}
    assert d.behavioral_regressions == {"C3": [0, 1]}, (
        "an empty behavioral dict here makes pr_body derive mechanical=1 and tell the "
        "human a storage-contract breach occurred"
    )


def test_a_written_validation_record_can_be_loaded_back_by_the_pr_command(tmp_path):
    """`to_json` writes DERIVED keys the constructor does not take.

    Adding `disposition` to the serialized record broke `loop pr` outright — it loads
    with `ValidationRecord(**rec_raw)` and died on TypeError before opening anything.
    The record on disk is the durable artifact and may carry more than the dataclass
    accepts, so the loader filters rather than the writer omitting evidence.
    """
    import dataclasses
    import json

    from loop.artifacts import ValidationRecord, write_validation_record

    rec = ValidationRecord(
        candidate_id="c",
        label="cand-c",
        accepted=False,
        delta_in=0.0,
        delta_ho=0.04,
        rule={"applied": True, "outcome": "CONFIRM"},
    )
    out = write_validation_record(rec, tmp_path / "validation-c.json")
    raw = json.loads(out.read_text())
    assert "disposition" in raw, "the derived value must still reach the record"

    fields = {f.name for f in dataclasses.fields(ValidationRecord)}
    loaded = ValidationRecord(**{k: v for k, v in raw.items() if k in fields})
    assert loaded.candidate_id == "c" and loaded.disposition == "PENDING_CONFIRMATION"


# --- section calibration: measured bounds for ONE section (contract §4) ----------


def _measured_calibration(tmp_path, *, arm_fingerprint=None, supported=None):
    """A `SectionCalibration` whose every number was MEASURED, never authored here.

    Four null arms of the contract's shape (§2 — nothing changed between them) are
    written to a scratch results dir, run through `loop.calibrate` (the same code the
    real protocol runs), and the artifact is loaded back through the production
    loader. No threshold in the tests below is a literal: each one reads the bound off
    the calibration and asserts its own movement against that.

    The arms are stamped with the SAME provenance the suites built by `_run()` carry,
    and the loader is handed that fingerprint: freshness is a question about the
    measurements being judged, not about the process doing the judging (contract §4,
    amendment 2026-08-19). `arm_fingerprint` overrides it to build a stale artifact.
    """
    import json

    from loop.calibrate import SUPPORTED, calibrate
    from loop.validate import section_calibration

    judged_fp = _run({})["fingerprint"]
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    fp = arm_fingerprint or {k: judged_fp[k] for k in ("runner_sha", "config_version", "model")}
    # A1/G4/G5 held-in, G2 held-out (contract §1). One arm drifts on A1 and G2 — that
    # drift IS the measurement: the spread with nothing changed between the arms.
    arms = {
        "null-cmp-a": {"A1": 7, "G4": 8, "G5": 9, "G2": 6},
        "null-cmp-b": {"A1": 7, "G4": 8, "G5": 9, "G2": 6},
        "null-cmp-c": {"A1": 8, "G4": 8, "G5": 9, "G2": 7},
        "null-cmp-d": {"A1": 7, "G4": 8, "G5": 9, "G2": 6},
    }
    for label, passes in arms.items():
        (results_dir / f"{label}.json").write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "filter": sorted(passes),
                    "tasks": {
                        n: {
                            "split": "held_out" if n == "G2" else "held_in",
                            "attempts": 10,
                            "passes": p,
                            "pass_fraction": round(p / 10, 4),
                        }
                        for n, p in passes.items()
                    },
                }
            )
        )
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(calibrate(sorted(arms), results_dir, supported or SUPPORTED)))
    return section_calibration("compaction", judged_fp, analysis_path=path)


# The supported set (A1, G4, G5 held-in; G2 held-out) plus BALLAST the section's bound
# was never measured on — X1..X4 exist to make the whole-split mean and the
# supported-set mean different numbers, which is the entire point of the mechanism.
_CMP_COUNTS = {
    "A1": (8, 10, "held_in"),
    "G4": (8, 10, "held_in"),
    "G5": (8, 10, "held_in"),
    "G2": (10, 20, "held_out"),
    "X1": (2, 3, "held_in"),
    "X2": (2, 4, "held_out"),
    "X3": (2, 4, "held_out"),
    "X4": (2, 4, "held_out"),
}


def _cmp_run(moved=None, **kw):
    counts = dict(_CMP_COUNTS)
    for name, passes in (moved or {}).items():
        _, attempts, split = counts[name]
        counts[name] = (passes, attempts, split)
    return _run(counts, **kw)


def test_section_calibration_carries_only_values_the_artifact_measured(tmp_path):
    """Contract §5: no threshold is authored by hand. The dataclass is a view onto the
    artifact — supported set from `per_task`, bounds from `section_noise` — and its
    `source` must not carry a machine path into a committed record."""
    import json

    from loop.acceptance import SectionCalibration

    cal = _measured_calibration(tmp_path)
    analysis = json.loads((tmp_path / "analysis.json").read_text())
    assert isinstance(cal, SectionCalibration)
    assert cal.section == "compaction"
    assert cal.supported == frozenset(analysis["per_task"])
    assert cal.noise_in == analysis["section_noise"]["held_in"]
    assert cal.noise_ho == analysis["section_noise"]["held_out"]
    assert cal.guards == frozenset({"A1", "G2", "G5"})
    assert cal.source.endswith("analysis.json") and "/Users/" not in cal.source


def test_a_supported_set_gain_above_the_measured_bound_confirms_naming_the_task(tmp_path):
    """The mechanism's reason for existing: a gain the whole-split rule cannot see.

    G2 is the section's one held-out supported task. It gains four attempts; the
    whole-split mean divides that by four unsupported tasks the bound was never
    measured on and lands under the one-attempt allowance, so the UNCALIBRATED rule
    finds nothing. Judged on the supported-set mean against the MEASURED bound, the
    same movement is evidence.
    """
    cal = _measured_calibration(tmp_path)
    base, cand = _cmp_run(), _cmp_run({"G2": 14})
    gain = 14 / 20 - 10 / 20
    assert gain > cal.noise_ho, "fixture precondition: the movement must clear the bound"

    assert evaluate(base, cand).outcome == REJECT, (
        "the whole-split rule cannot see this gain — that is what the calibration is for"
    )

    d = evaluate(base, cand, calibration=cal)
    assert d.outcome == CONFIRM
    assert d.evidence_split == "held_out"
    assert d.improved_tasks == ("G2",), "the evidence basis is the SUPPORTED set"
    assert d.delta_ho == pytest.approx(gain), "the supported-set mean decides"
    assert d.threshold_ho == cal.noise_ho and d.threshold_in == cal.noise_in
    assert d.raw["full_split_delta_ho"] == pytest.approx(gain / 4), (
        "the whole-split number is still recorded — the rule may not hide what it stopped reading"
    )
    assert any("measured section noise" in r for r in d.reasons)
    assert set(cal.guards) <= set(d.confirm_tasks), "the section's guards ride in unmoved"


def test_the_same_movement_below_the_measured_bound_is_within_noise(tmp_path):
    cal = _measured_calibration(tmp_path)
    base, cand = _cmp_run(), _cmp_run({"G2": 11})
    assert 0 < 11 / 20 - 10 / 20 < cal.noise_ho, "fixture precondition: under the bound"

    d = evaluate(base, cand, calibration=cal)
    assert d.outcome == REJECT
    assert any("within measured noise" in r for r in d.reasons)
    assert d.improved_tasks == () and d.evidence_split == ""


def test_a_mechanical_security_rise_still_rejects_under_a_calibration(tmp_path):
    """Whole-suite protections are UNCHANGED by the calibration: the leak is on X2, a
    task outside the supported set entirely, and it still blocks unconditionally."""
    cal = _measured_calibration(tmp_path)
    base = _cmp_run()
    cand = _cmp_run(
        {"G2": 14},
        outcomes={"X2": ["pass", "pass", "critical_failure", "fail"]},
        security_classes={"X2": [None, None, "mechanical", None]},
    )
    d = evaluate(base, cand, calibration=cal)
    assert d.outcome == REJECT
    assert "harness storage contract regressed (mechanical): X2 0->1" in d.reasons
    assert d.security_regressions == {"X2": [0, 1]}


def test_a_collapse_on_a_causally_plausible_task_still_vetoes_under_a_calibration(tmp_path):
    cal = _measured_calibration(tmp_path)
    base = _cmp_run({"X2": 4})  # X2 at a full pass in the baseline
    cand = _cmp_run({"G2": 14, "X2": 0})
    d = evaluate(base, cand, calibration=cal)
    assert d.outcome == REJECT
    assert any("full-pass task collapsed to zero: X2" in r for r in d.reasons)


def test_evidence_grade_exclusions_are_context_and_never_zero_a_movement(tmp_path):
    """Contract §4: compaction's exclusions are evidence-grade (`airtight=False` is
    structural — `trigger_fraction` belongs to the knob), so they are recorded and
    caveated, never subtracted."""
    cal = _measured_calibration(tmp_path)
    base, cand = _cmp_run(), _cmp_run({"G2": 14, "X1": 0})
    d = evaluate(base, cand, calibration=cal, unreachable_probable={"X1"})
    assert d.outcome == CONFIRM
    assert d.excluded == (), "an evidence-grade exclusion is not a proof-grade one"
    assert d.raw["unreachable_probable"] == ["X1"]
    assert d.raw["full_split_delta_in"] < 0, "X1's movement is still counted in full"
    context = " ".join(d.reasons)
    assert "unreachable_probable" in context and "X1" in context
    assert "never a verdict" in context, "the caveat must be stated, not just the name"


def test_confirmed_honors_the_measured_bound_where_one_attempt_would_have_accepted(tmp_path):
    """`confirmed()` with the calibration asks the SAME magnitude question the first
    decision asked: the original gain must repeat above the measured section bound.
    A one-attempt grain re-derived from the confirmation's own attempt counts (0.02 at
    50 attempts) would accept a repeat five times smaller than the noise the arms
    actually measured."""
    import dataclasses

    cal = _measured_calibration(tmp_path)
    first = evaluate(_cmp_run(), _cmp_run({"G2": 14}), calibration=cal)
    assert first.outcome == CONFIRM and first.improved_tasks == ("G2",)
    # G4 rides in because it is part of the supported set the bound was measured on
    # (contract §4 amendment) — see the held-in section at the end of this file.
    assert set(first.confirm_tasks) == {"A1", "G2", "G4", "G5"}

    base_counts = _confirm_counts()
    weak = {**base_counts, "G2": (28, 50, "held_out")}  # +0.06, under the measured bound
    assert 0 < 28 / 50 - 25 / 50 < cal.noise_ho
    fb, fc = _confirm_pair(first, base_counts, weak)
    # The same claim WITHOUT the calibration — a first decision that never recorded a
    # calibrated regime, which is what this data would have produced before the
    # mechanism existed. `confirmed()` now refuses to judge the calibrated one this way
    # (see test_a_calibrated_claim_cannot_be_confirmed_against_the_weaker_bar), so the
    # comparison is made against the decision the weaker regime would have written.
    uncalibrated_claim = dataclasses.replace(first, raw={})
    assert confirmed(uncalibrated_claim, fb, fc).outcome == ACCEPT, (
        "one attempt (0.02) would accept it"
    )
    d = confirmed(first, fb, fc, calibration=cal)
    assert d.outcome == REJECT
    assert any("did not repeat" in r for r in d.reasons)
    assert d.threshold_ho == cal.noise_ho

    strong = {**base_counts, "G2": (34, 50, "held_out")}  # +0.18, over the measured bound
    assert 34 / 50 - 25 / 50 > cal.noise_ho
    fb2, fc2 = _confirm_pair(first, base_counts, strong)
    assert confirmed(first, fb2, fc2, calibration=cal).outcome == ACCEPT


def test_a_calibrated_confirmation_still_blocks_a_mechanical_leak(tmp_path):
    cal = _measured_calibration(tmp_path)
    first = evaluate(_cmp_run(), _cmp_run({"G2": 14}), calibration=cal)
    base_counts = _confirm_counts()
    cand_counts = _confirm_counts({"G2": 34})
    fb, fc = _confirm_pair(first, base_counts, cand_counts)
    fc["tasks"]["A1"]["outcomes"] = ["pass"] * 40 + ["critical_failure"] + ["fail"] * 9
    fc["tasks"]["A1"]["security_classes"] = [None] * 40 + ["mechanical"] + [None] * 9
    d = confirmed(first, fb, fc, calibration=cal)
    assert d.outcome == REJECT
    assert "harness storage contract regressed (mechanical): A1 0->1" in d.reasons


# --- the ACCEPT gate under calibration (contract §4, amendments 2026-08-19) -------
#
# Every calibrated test above lands its evidence on HELD-OUT, where the supported set
# has exactly one member (G2) and the supported-set mean is that one task's delta.
# That shape cannot see a denominator bug: 1/1 and a shrunken 1/1 are the same number.
# The tests below put the evidence on HELD-IN, where the supported set has three
# members (A1, G4, G5) against four tasks in the split, so the two denominators are
# genuinely different and a silently dropped task changes the answer.

_CONFIRM_COUNTS = {
    "A1": (40, 50, "held_in"),
    "G4": (40, 50, "held_in"),
    "G5": (40, 50, "held_in"),
    "G2": (25, 50, "held_out"),
}


def _confirm_counts(moved=None, extra=None):
    counts = dict(_CONFIRM_COUNTS)
    for name, passes in (moved or {}).items():
        _, attempts, split = counts[name]
        counts[name] = (passes, attempts, split)
    return counts | (extra or {})


def test_calibrated_evidence_on_held_in_uses_the_supported_denominator(tmp_path):
    """Three supported tasks in a four-task split: the two means divide by 3 and by 4.

    A1 gains two attempts. Over the supported set that is 0.2/3; over the whole split
    it is 0.2/4, under the one-attempt allowance the ballast task X1's coarse grain
    sets. The calibrated rule sees evidence, the uncalibrated one does not, and the
    difference is visible in the denominators rather than in a single task's number.
    """
    cal = _measured_calibration(tmp_path)
    base, cand = _cmp_run(), _cmp_run({"A1": 10})
    assert 0.2 / 3 > cal.noise_in > 0, "fixture precondition: clears the MEASURED bound"

    assert evaluate(base, cand).outcome == REJECT, "0.2/4 is under the one-attempt allowance"

    d = evaluate(base, cand, calibration=cal)
    assert d.outcome == CONFIRM
    assert d.evidence_split == "held_in"
    assert d.improved_tasks == ("A1",)
    assert d.delta_in == pytest.approx(0.2 / 3), "supported-set denominator: 3 tasks"
    assert d.raw["full_split_delta_in"] == pytest.approx(0.2 / 4), "whole split: 4 tasks"


def test_a_supported_task_missing_from_an_arm_refuses_instead_of_shrinking(tmp_path):
    """The bound was measured on a mean over FOUR tasks. Judging a mean over three
    against it silently changes the denominator — the answer moves and nothing in the
    record says why. That is a refusal, not a smaller measurement."""
    cal = _measured_calibration(tmp_path)
    counts = {n: v for n, v in _CMP_COUNTS.items() if n != "G4"}
    base, cand = _run(counts), _run(dict(counts, A1=(10, 10, "held_in")))

    assert evaluate(base, cand).outcome == REJECT  # no calibration, no requirement
    with pytest.raises(ValueError, match="G4"):
        evaluate(base, cand, calibration=cal)


def test_confirm_tasks_gain_the_whole_supported_set_including_the_miner(tmp_path):
    """G4 is the MINER: never a guard, and it did not move. It still has to be rerun —
    it is one of the four tasks the bound itself was measured on, so a confirmation
    that skips it cannot compute the supported-set mean it is judged against."""
    cal = _measured_calibration(tmp_path)
    d = evaluate(_cmp_run(), _cmp_run({"A1": 10}), calibration=cal)
    assert d.outcome == CONFIRM
    assert set(d.confirm_tasks) == {"A1", "G2", "G4", "G5"}
    assert "G4" not in cal.guards and "G4" not in d.improved_tasks


def test_a_collapse_in_the_calibrated_confirmation_blocks_the_accept(tmp_path):
    """CRITICAL, and the reason the veto belongs in `confirmed()` too: X2 is outside
    the supported set, so it enters no calibrated mean and no guard list. Under the
    supported-set judgment alone, a task going 1.00 -> 0.00 inside the confirmation
    pair is invisible — and `confirmed()` is the ONLY road to ACCEPT."""
    cal = _measured_calibration(tmp_path)
    first = evaluate(_cmp_run(), _cmp_run({"G2": 14, "X2": 3}), calibration=cal)
    assert first.outcome == CONFIRM
    assert "X2" in first.confirm_tasks, "it moved, so it rides into the confirmation"

    base_counts = _confirm_counts(extra={"X2": (4, 4, "held_out")})
    repeat = {"G2": 34}  # the original gain repeats above the measured bound
    survives = _confirm_counts(repeat, extra={"X2": (1, 4, "held_out")})
    collapses = _confirm_counts(repeat, extra={"X2": (0, 4, "held_out")})

    ok = confirmed(first, *_confirm_pair(first, base_counts, survives), calibration=cal)
    assert ok.outcome == ACCEPT, "a steep drop short of zero is not the veto's business"

    d = confirmed(first, *_confirm_pair(first, base_counts, collapses), calibration=cal)
    assert d.outcome == REJECT
    assert "full-pass task collapsed to zero: X2" in d.reasons


def test_the_confirmation_collapse_veto_is_not_gated_on_a_calibration(tmp_path):
    """The same protection on the uncalibrated path — this is a whole-suite guarantee
    about the ACCEPT gate, not a compaction feature. (It is new behavior for every
    section: before this, a full-pass task could collapse inside a confirmation and
    the pair would still ACCEPT as long as the split means held.)"""
    first = evaluate(
        _suite(IN0, {**HO0, "O0": 5}), _suite({**IN0, "I0": 3, "I1": 3}, {**HO0, "O0": 4})
    )
    assert first.outcome == CONFIRM and "O0" in first.confirm_tasks

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in"), "O0": (15, 15, "held_out")}
    counts_c = {"I0": (9, 9, "held_in"), "I1": (9, 9, "held_in"), "O0": (0, 15, "held_out")}
    d = confirmed(first, *_confirm_pair(first, counts_b, counts_c))
    assert d.outcome == REJECT
    assert "full-pass task collapsed to zero: O0" in d.reasons


def test_a_calibrated_claim_cannot_be_confirmed_against_the_weaker_bar(tmp_path):
    """A first decision carrying `raw["regime"] == "section_calibration"` was judged
    against a measured bound. Confirming it with no calibration in hand would judge the
    repeat against a one-attempt grain instead — a different, weaker question wearing
    the same word. The record says which regime produced it, so this is detectable, and
    it refuses loudly rather than quietly re-deciding."""
    cal = _measured_calibration(tmp_path)
    first = evaluate(_cmp_run(), _cmp_run({"G2": 14}), calibration=cal)
    assert first.raw["regime"] == "section_calibration"

    base_counts = _confirm_counts()
    cand_counts = _confirm_counts({"G2": 34})
    fb, fc = _confirm_pair(first, base_counts, cand_counts)
    with pytest.raises(ValueError, match="calibrat"):
        confirmed(first, fb, fc)
    assert confirmed(first, fb, fc, calibration=cal).outcome == ACCEPT


def test_a_confirmation_missing_a_supported_task_refuses(tmp_path):
    """Reachable through a record written before `confirm_tasks` carried the supported
    set: `load_first_decision` rebuilds that Decision faithfully, and its confirmation
    pair would then compute the supported-set mean over a short denominator."""
    from loop.acceptance import Decision

    cal = _measured_calibration(tmp_path)
    stale_record = Decision(
        outcome=CONFIRM,
        reasons=(),
        delta_in=0.0,
        delta_ho=0.2,
        threshold_in=cal.noise_in,
        threshold_ho=cal.noise_ho,
        evidence_split="held_out",
        improved_tasks=("G2",),
        confirm_tasks=("A1", "G2", "G5"),  # no G4 — written before the amendment
        raw={"regime": "section_calibration"},
    )
    counts = {n: v for n, v in _CONFIRM_COUNTS.items() if n != "G4"}
    fb, fc = _confirm_pair(stale_record, counts, counts | {"G2": (34, 50, "held_out")})
    with pytest.raises(ValueError, match="G4"):
        confirmed(stale_record, fb, fc, calibration=cal)


def test_behavioral_routing_and_the_fisher_verdict_survive_a_calibration(tmp_path):
    """The third whole-suite protection, end to end under a calibration: a behavioral
    critical on a task outside the supported set does not veto at full-suite counts, it
    ROUTES — and at the confirmation's 10-per-arm counts the predeclared Fisher test
    decides it. 4-vs-0 is the FISHER_ALPHA docstring's own worked example (p~0.043)."""
    cal = _measured_calibration(tmp_path)
    base = _cmp_run()
    cand = _cmp_run(
        {"G2": 14},
        outcomes={"X2": ["pass", "pass", "critical_failure", "fail"]},
        security_classes={"X2": [None, None, "behavioral", None]},
    )
    first = evaluate(base, cand, calibration=cal)
    assert first.outcome == CONFIRM, "a behavioral rise routes, it does not veto here"
    assert first.targeted_rerun == ("X2",)
    assert "X2" in first.confirm_tasks

    base_counts = _confirm_counts(extra={"X2": (6, 10, "held_out")})
    cand_counts = _confirm_counts({"G2": 34}, extra={"X2": (6, 10, "held_out")})
    fb, fc = _confirm_pair(first, base_counts, cand_counts)
    fc["tasks"]["X2"]["outcomes"] = ["pass"] * 6 + ["critical_failure"] * 4
    fc["tasks"]["X2"]["security_classes"] = [None] * 6 + ["behavioral"] * 4
    d = confirmed(first, fb, fc, calibration=cal)
    assert d.outcome == REJECT
    assert "behavioral security increase confirmed on X2 (p=0.043)" in d.reasons
    assert d.raw["behavioral_verdicts"]["X2"]["verdict"] == "confirmed_increase"


def test_a_calibration_from_other_provenance_never_reaches_a_judgment(tmp_path):
    """The loader's amended freshness rule, seen from this side: an artifact measured
    under a different model is not a bound for these measurements, so no calibration
    exists to pass and the section falls back to the whole-split rule."""
    from loop.validate import calibration_status

    judged = _run({})["fingerprint"]
    stale = _measured_calibration(
        tmp_path,
        arm_fingerprint={
            "runner_sha": judged["runner_sha"],
            "config_version": judged["config_version"],
            "model": "a-different-model",
        },
    )
    assert stale is None
    cal, why = calibration_status("compaction", judged, analysis_path=tmp_path / "analysis.json")
    assert cal is None and "model" in why
