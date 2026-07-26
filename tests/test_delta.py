import pytest

from runner.delta import acceptance, delta


def _results(
    tasks: dict[str, tuple[str, float]], model: str = "carbon", runner_sha: str = "rsha1"
) -> dict:
    """Minimal results-JSON shape: name -> (split, pass_fraction). Attempts
    mirror the real suite (3 held_in, 5 held_out) so parity checks pass."""
    attempts = {"held_in": 3, "held_out": 5}
    return {
        "fingerprint": {
            "gemma_sha": "abc",
            "config_version": 1,
            "model": model,
            "runner_sha": runner_sha,
        },
        "tasks": {
            name: {"split": split, "pass_fraction": frac, "attempts": attempts[split]}
            for name, (split, frac) in tasks.items()
        },
        "summary": {},
    }


BASE = _results({"A1": ("held_in", 0.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)})


def test_delta_computes_split_and_per_task_changes():
    cand = _results({"A1": ("held_in", 2 / 3), "B1": ("held_in", 1.0), "A3": ("held_out", 0.6)})
    d = delta(BASE, cand)
    assert abs(d["delta_in"] - ((2 / 3 + 1.0) / 2 - 0.5)) < 1e-9
    assert abs(d["delta_ho"] - 0.2) < 1e-9
    assert abs(d["per_task"]["A1"] - 2 / 3) < 1e-9
    assert d["per_task"]["B1"] == 0.0
    assert d["regressions"] == {}
    assert d["catastrophic_regressions"] == {}


def test_delta_reports_efficiency_metrics_without_using_them_as_truth_gate():
    baseline = _results({"A1": ("held_in", 0.0), "B1": ("held_out", 0.0)})
    candidate = _results({"A1": ("held_in", 1.0), "B1": ("held_out", 1.0)})
    baseline["summary"]["metrics"] = {"tokens": 1000.0, "cost": 1.0}
    candidate["summary"]["metrics"] = {"tokens": 800.0, "cost": 0.8}
    d = delta(baseline, candidate)
    assert d["metric_delta"] == {"cost": pytest.approx(-0.2), "tokens": -200.0}
    assert d["accepted"] is True


def test_acceptance_rule():
    assert acceptance(0.1, 0.0) == {"accepted": True, "delta_in": 0.1, "delta_ho": 0.0}
    assert acceptance(0.0, 0.0)["accepted"] is False  # max must be > 0
    assert acceptance(0.2, -0.1)["accepted"] is False  # no held-out regression
    assert acceptance(-0.1, 0.2)["accepted"] is False  # no held-in regression


def test_delta_vetoes_full_pass_to_zero_pass_hidden_by_split_average():
    """Iteration 1's real blind spot: one held-in task gains 0->1 while a
    different held-in task collapses 1->0, so aggregate Δ_in stays zero."""
    baseline = _results(
        {
            "miner": ("held_in", 0.0),
            "guard": ("held_in", 1.0),
            "heldout": ("held_out", 0.0),
        }
    )
    candidate = _results(
        {
            "miner": ("held_in", 1.0),
            "guard": ("held_in", 0.0),
            "heldout": ("held_out", 1.0),
        }
    )
    d = delta(baseline, candidate)
    assert d["delta_in"] == 0.0
    assert d["delta_ho"] == 1.0
    assert d["aggregate_accepted"] is True
    assert d["catastrophic_regressions"] == {"guard": -1.0}
    assert d["accepted"] is False


def test_delta_reports_smaller_regression_without_hard_veto():
    baseline = _results(
        {
            "miner": ("held_in", 0.0),
            "noisy": ("held_in", 2 / 3),
            "heldout": ("held_out", 0.0),
        }
    )
    candidate = _results(
        {
            "miner": ("held_in", 1.0),
            "noisy": ("held_in", 1 / 3),
            "heldout": ("held_out", 1.0),
        }
    )
    d = delta(baseline, candidate)
    assert d["regressions"] == {"noisy": -(1 / 3)}
    assert d["catastrophic_regressions"] == {}
    assert d["accepted"] is True


def test_delta_refuses_mismatched_task_sets():
    import pytest

    cand = _results({"A1": ("held_in", 1.0)})
    with pytest.raises(ValueError):
        delta(BASE, cand)


def test_delta_refuses_mismatched_attempt_counts():
    """A candidate re-measured with fewer attempts is a different sample size;
    blending it into a Δ would compare fractions of unequal precision."""
    import pytest

    cand = _results({"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)})
    cand["tasks"]["A1"]["attempts"] = 1
    with pytest.raises(ValueError, match="sample-size mismatch.*A1"):
        delta(BASE, cand)


def test_delta_refuses_mismatched_models():
    import pytest

    cand = _results(
        {"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)},
        model="other-model",
    )
    with pytest.raises(ValueError, match="carbon.*other-model"):
        delta(BASE, cand)


def test_delta_refuses_mismatched_runner_sha():
    """Results measured by two different verifier versions are not
    like-for-like — a verifier fix would masquerade as a harness Δ."""
    import pytest

    cand = _results(
        {"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)},
        runner_sha="rsha2",
    )
    with pytest.raises(ValueError, match="verifier version mismatch"):
        delta(BASE, cand)
    # legacy results with no runner_sha at all vs a stamped one: also refused
    legacy = _results({"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)})
    del legacy["fingerprint"]["runner_sha"]
    with pytest.raises(ValueError, match="verifier version mismatch"):
        delta(legacy, BASE)


def test_delta_accepts_matching_runner_sha():
    cand = _results({"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)})
    d = delta(BASE, cand)
    assert "delta_in" in d and "delta_ho" in d


def test_delta_refuses_missing_fingerprint():
    """Two results that BOTH lack a fingerprint would sail through every
    parity gate on None==None — unattributed measurements must be refused
    outright, in either position."""
    import pytest

    tasks = {"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)}
    for strip in ("del", "empty"):
        bare = _results(tasks)
        if strip == "del":
            del bare["fingerprint"]
        else:
            bare["fingerprint"] = {}
        with pytest.raises(ValueError, match="lacks a fingerprint"):
            delta(bare, BASE)
        with pytest.raises(ValueError, match="lacks a fingerprint"):
            delta(BASE, bare)


def test_delta_refuses_filtered_results():
    """A --only run stamps 'filter' into its results JSON; Δ over a partial
    suite would skew the split means, so both directions must be refused."""
    import pytest

    filtered = _results({"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)})
    filtered["filter"] = ["A1"]
    with pytest.raises(ValueError, match="filtered"):
        delta(filtered, BASE)
    with pytest.raises(ValueError, match="filtered"):
        delta(BASE, filtered)
