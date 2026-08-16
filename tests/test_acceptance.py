"""Pins for the three-outcome rule, each threshold traced to the null measurement."""

from __future__ import annotations

import pytest

from loop.acceptance import ACCEPT, CONFIRM, REJECT, confirmed, evaluate, one_attempt


def _run(counts: dict[str, tuple[int, int, str]]) -> dict:
    return {
        "fingerprint": {"runner_sha": "x", "model": "m", "config_version": 1, "dirty_sha": None},
        "tasks": {
            n: {"split": s, "attempts": a, "passes": p, "pass_fraction": round(p / a, 4)}
            for n, (p, a, s) in counts.items()
        },
    }


def _suite(in_passes: dict[str, int], ho_passes: dict[str, int]) -> dict:
    counts = {n: (p, 3, "held_in") for n, p in in_passes.items()}
    counts |= {n: (p, 5, "held_out") for n, p in ho_passes.items()}
    return _run(counts)


IN0 = {f"I{i}": 2 for i in range(6)}  # 6 held-in tasks at 2/3
HO0 = {f"O{i}": 4 for i in range(5)}  # 5 held-out tasks at 4/5


def test_thresholds_derive_from_the_suite_not_from_decimals():
    """1/(tasks x attempts) per split — the shipped suite gives 1/54 and 1/50, and a
    fixture with different structure gives different numbers. Hard-coding 0.0185 would
    freeze today's suite shape into the rule."""
    r = _suite(IN0, HO0)
    assert one_attempt(r, "held_in") == pytest.approx(1 / 18)  # 6 tasks x 3 attempts
    assert one_attempt(r, "held_out") == pytest.approx(1 / 25)  # 5 tasks x 5 attempts


def test_exactly_one_attempt_down_is_allowed_and_two_is_rejected():
    """Six unchanged runs reached exactly one attempt of negative movement per split,
    so one attempt is the measured allowance — strictly beyond it is a regression."""
    base = _suite(IN0, HO0)
    # Both carry a clear two-attempt held-out gain, so the only difference between
    # them is the held-in movement — one attempt versus two. Without the gain the
    # one-down case would reject anyway for LACK OF EVIDENCE, and the test would be
    # conflating the two checks it exists to keep separate.
    one_down = _suite({**IN0, "I0": 1}, {**HO0, "O0": 5, "O1": 5})
    two_down = _suite({**IN0, "I0": 0}, {**HO0, "O0": 5, "O1": 5})

    allowed = evaluate(base, one_down)
    assert allowed.outcome == CONFIRM
    assert not any("regressed" in r for r in allowed.reasons)

    d = evaluate(base, two_down)
    assert d.outcome == REJECT
    assert "held-in regressed beyond one attempt" in d.reasons[0]


def test_a_gain_earns_CONFIRM_never_ACCEPT():
    """The null runs produced a TWO-attempt held-out gain (+0.0400) with nothing
    changed. No single-run gain is proof, so evaluate() can never return ACCEPT."""
    base = _suite(IN0, HO0)
    gain = _suite({**IN0, "I0": 3, "I1": 3}, HO0)  # +2 attempts held-in
    d = evaluate(base, gain)
    assert d.outcome == CONFIRM
    assert d.outcome != ACCEPT


def test_one_attempt_of_gain_is_not_evidence():
    """A single attempt's movement is exactly the measured null variation."""
    base = _suite(IN0, HO0)
    small = _suite({**IN0, "I0": 3}, HO0)  # +1 attempt held-in only
    d = evaluate(base, small)
    assert d.outcome == REJECT
    assert "indistinguishable from the measured null" in d.reasons[0]


def test_no_movement_at_all_is_rejected_not_accepted():
    base = _suite(IN0, HO0)
    d = evaluate(base, _suite(dict(IN0), dict(HO0)))
    assert d.outcome == REJECT


def test_excluded_task_movements_are_zeroed_with_denominator_kept():
    """A collapse on a task the edited section cannot reach is the grader's noise —
    zeroed, not dropped, so the split mean keeps its denominator."""
    base = _suite({**IN0, "I0": 3}, HO0)
    cand = _suite({**IN0, "I0": 0, "I1": 3, "I2": 3}, HO0)  # I0 collapses, I1+I2 gain
    without = evaluate(base, cand)
    assert without.outcome == REJECT, "collapse must reject when the section reaches I0"
    with_excl = evaluate(base, cand, excluded={"I0"})
    assert with_excl.outcome == CONFIRM
    assert "I0" in with_excl.excluded

    # ZEROED, not dropped — and here is a case where only the denominator separates
    # the two. I1 falls one attempt; with I0 zeroed over the full 6-task denominator
    # that is -1/18, exactly the allowance. Dropping I0 makes it (-1/3)/5 = -1/15,
    # past the allowance, and a legal candidate is rejected by the arithmetic of its
    # own exclusion. The gain sits held-out so the held-in movement is the decider.
    down_one = _suite({**IN0, "I0": 0, "I1": 1}, {**HO0, "O0": 5, "O1": 5})
    d2 = evaluate(_suite(IN0, HO0), down_one, excluded={"I0"})
    assert d2.outcome == CONFIRM, d2.reasons
    assert not any("regressed" in r for r in d2.reasons)


def test_full_pass_collapse_rejects_even_with_gains_elsewhere():
    """Kept from the one-number rule: iteration 1 saw A1 collapse 1.0 -> 0.0 while A2
    rose the same amount, leaving Δ_in unchanged."""
    base = _suite({**IN0, "I0": 3}, HO0)
    cand = _suite({**IN0, "I0": 0, "I1": 3, "I2": 3}, HO0)
    d = evaluate(base, cand)
    assert d.outcome == REJECT
    assert any("collapsed" in r for r in d.reasons)


def test_critical_negative_movement_is_flagged_for_confirmation():
    """C3's leaks were real failures whose timing was noise. One extra leak must not
    vanish into a mean, so a critical negative rides along into CONFIRM's reasons even
    when the aggregate is inside tolerance."""
    base = _suite(IN0, HO0)
    cand = _suite({**IN0, "I0": 3, "I1": 3}, {**HO0, "O0": 3})  # gain in, O0 down 1
    d = evaluate(base, cand, critical={"O0"})
    assert d.outcome == CONFIRM
    assert d.critical_regressions == {"O0": pytest.approx(-1 / 5)}
    assert any("critical task moved negative" in r for r in d.reasons)


def test_ACCEPT_exists_only_through_a_paired_confirmation():
    base = _suite(IN0, HO0)
    gain = _suite({**IN0, "I0": 3, "I1": 3}, HO0)
    first = evaluate(base, gain)
    assert first.outcome == CONFIRM

    ok = confirmed(first, base, gain)
    assert ok.outcome == ACCEPT

    flat = _suite(dict(IN0), dict(HO0))
    failed = confirmed(first, base, flat)
    assert failed.outcome == REJECT
    assert "did not repeat" in failed.reasons[0]


def test_a_repeated_critical_regression_blocks_ACCEPT():
    base = _suite(IN0, HO0)
    cand = _suite({**IN0, "I0": 3, "I1": 3}, {**HO0, "O0": 3})
    first = evaluate(base, cand, critical={"O0"})
    assert first.outcome == CONFIRM

    again = confirmed(first, base, cand, critical={"O0"})
    assert again.outcome == REJECT
    assert "critical regression repeated" in again.reasons[0]

    # A DIFFERENT critical task moving in the confirmation is not a repeat.
    other = _suite({**IN0, "I0": 3, "I1": 3}, {**HO0, "O1": 3})
    fresh = confirmed(first, base, other, critical={"O0", "O1"})
    assert fresh.outcome == ACCEPT


def test_the_null_pair_that_fooled_the_old_rule_cannot_ACCEPT():
    """The a->b shape from the real null runs: -1 attempt held-in, +2 attempts
    held-out, nothing changed. The old rule's causal filter ACCEPTED it. Here it may
    reach CONFIRM (the gain is real-looking, that is the point) but never ACCEPT
    without a fresh pair — and evaluate() cannot emit ACCEPT at all."""
    base = _suite(IN0, HO0)
    noise = _suite({**IN0, "I0": 1}, {**HO0, "O0": 5, "O1": 5})
    d = evaluate(base, noise)
    assert d.outcome in (REJECT, CONFIRM)
    assert d.outcome != ACCEPT
