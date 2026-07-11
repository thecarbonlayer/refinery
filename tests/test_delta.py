from runner.delta import acceptance, delta


def _results(
    tasks: dict[str, tuple[str, float]], model: str = "gemma", runner_sha: str = "rsha1"
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


def test_acceptance_rule():
    assert acceptance(0.1, 0.0) == {"accepted": True, "delta_in": 0.1, "delta_ho": 0.0}
    assert acceptance(0.0, 0.0)["accepted"] is False  # max must be > 0
    assert acceptance(0.2, -0.1)["accepted"] is False  # no held-out regression
    assert acceptance(-0.1, 0.2)["accepted"] is False  # no held-in regression


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
    with pytest.raises(ValueError, match="gemma.*other-model"):
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
