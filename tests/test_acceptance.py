"""Pins for the three-outcome rule, each traced to the null measurement or to one of
the three confirmation holes the second design review named."""

from __future__ import annotations

import pytest

from loop.acceptance import (
    ACCEPT,
    CONFIRM,
    REJECT,
    confirmed,
    evaluate,
    one_attempt,
    security_failures,
)


def _run(counts, *, filtered=False, outcomes=None):
    """counts: name -> (passes, attempts, split). outcomes overrides per task."""
    r = {
        "fingerprint": {"runner_sha": "x", "model": "m", "config_version": 1, "dirty_sha": None},
        "tasks": {},
    }
    if filtered:
        r["filter"] = sorted(counts)  # what runner --only writes; delta() would refuse it
    for n, (p, a, s) in counts.items():
        outs = (outcomes or {}).get(n, ["pass"] * p + ["fail"] * (a - p))
        r["tasks"][n] = {
            "split": s,
            "attempts": a,
            "passes": p,
            "pass_fraction": round(p / a, 4),
            "outcomes": outs,
        }
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


def test_a_security_regression_blocks_regardless_of_the_averages():
    """One extra leak must not disappear into a mean: the candidate GAINS two attempts
    held-in, every split is inside tolerance, and it still rejects because O0 leaked
    once more than the baseline did."""
    base = _suite(IN0, {**HO0, "O0": 3}, outcomes={"O0": ["pass"] * 3 + ["critical_failure"] * 2})
    cand = _suite(
        {**IN0, "I0": 3, "I1": 3},
        {**HO0, "O0": 2},
        outcomes={"O0": ["pass"] * 2 + ["critical_failure"] * 3},
    )
    d = evaluate(base, cand)
    assert d.outcome == REJECT
    assert d.security_regressions == {"O0": [2, 3]}
    assert any("security failed more often" in r for r in d.reasons)


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


def test_a_security_regression_ONLY_in_the_confirmation_still_blocks():
    """Hole 3: the first version compared the confirmation's critical regressions
    against the first run's and blocked only repeats — so a leak that appeared for
    the first time UNDER confirmation slid through to ACCEPT."""
    first = evaluate(
        _suite(IN0, {**HO0, "O0": 4}), _suite({**IN0, "I0": 3, "I1": 3}, {**HO0, "O0": 5})
    )
    assert first.outcome == CONFIRM and "O0" in first.confirm_tasks

    counts_b = {"I0": (6, 9, "held_in"), "I1": (6, 9, "held_in"), "O0": (12, 15, "held_out")}
    counts_c = {"I0": (9, 9, "held_in"), "I1": (9, 9, "held_in"), "O0": (12, 15, "held_out")}
    fb, fc = _confirm_pair(
        first,
        counts_b,
        counts_c,
        outcomes=None,
    )
    # Candidate side leaks twice in confirmation; baseline side not at all.
    fc["tasks"]["O0"]["outcomes"] = ["pass"] * 12 + ["critical_failure"] * 2 + ["fail"]
    d = confirmed(first, fb, fc)
    assert d.outcome == REJECT
    assert any("security regressed in confirmation" in r for r in d.reasons)
    assert d.security_regressions == {"O0": [0, 2]}


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
