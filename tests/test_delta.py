import pytest

from runner.delta import acceptance, delta


def _results(
    tasks: dict[str, tuple[str, float]],
    model: str = "carbon",
    runner_sha: str = "rsha1",
    provider_order: str | None = None,
    quantization: str | None = None,
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
            "provider_order": provider_order,
            "quantization": quantization,
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


def test_delta_never_imputes_an_unmeasured_metric_as_zero():
    """A metric absent on one side is NOT a measured zero.

    Subtracting against a default of 0 turned "not measured" into the most
    favourable number available — a large negative cost delta — rendered in the PR
    body under prose saying negative means less work for the same score.
    """
    baseline = _results({"A1": ("held_in", 0.0), "B1": ("held_out", 0.0)})
    candidate = _results({"A1": ("held_in", 1.0), "B1": ("held_out", 1.0)})
    baseline["summary"]["metrics"] = {"tokens": 1000.0, "cost": 0.42}
    candidate["summary"]["metrics"] = {"tokens": 800.0}  # cost not measured at all
    d = delta(baseline, candidate)
    assert d["metric_delta"] == {"tokens": -200.0}
    assert d["metric_not_compared"] == ["cost"]


def test_delta_reports_a_candidate_only_metric_as_not_compared():
    """The symmetric difference is load-bearing, and only one direction was tested.

    With `set(base) - set(cand)`, a metric only the CANDIDATE measured vanished from
    `metric_not_compared` while still being excluded from the intersection-based
    `metric_delta` — so the PR body claimed both runs measured the same metrics over
    a comparison that silently omitted one.
    """
    baseline = _results({"A1": ("held_in", 0.0), "B1": ("held_out", 0.0)})
    candidate = _results({"A1": ("held_in", 1.0), "B1": ("held_out", 1.0)})
    baseline["summary"]["metrics"] = {"tokens": 1000.0}
    candidate["summary"]["metrics"] = {"tokens": 800.0, "retries": 2.0}
    d = delta(baseline, candidate)
    assert d["metric_delta"] == {"tokens": -200.0}
    assert d["metric_not_compared"] == ["retries"]


def test_metric_drift_never_moves_acceptance():
    """AGENTS.md: metrics "expose tradeoffs but never override correctness".

    Nothing enforced it — swapping `catastrophic_regressions` for
    `metric_denominator_drift` in the accept expression left every test green, which
    would quietly promote a diagnostic to a gate.
    """
    baseline = _results({"A1": ("held_in", 0.0), "B1": ("held_out", 0.0)})
    candidate = _results({"A1": ("held_in", 1.0), "B1": ("held_out", 1.0)})
    baseline["summary"] |= {"metrics": {"tokens": 10.0}, "metric_task_counts": {"tokens": 9}}
    candidate["summary"] |= {"metrics": {"tokens": 10.0}, "metric_task_counts": {"tokens": 1}}
    d = delta(baseline, candidate)
    assert d["metric_denominator_drift"] == ["tokens"]  # heavy drift...
    assert d["accepted"] is True  # ...and a clean gain is still accepted


def test_delta_surfaces_attempt_level_denominator_drift():
    """The case task counts cannot see, and the commonest one.

    A candidate whose attempts start ERRORING keeps every task in the population —
    identical task counts — while each mean covers a third as many attempts. That
    rendered a large favourable cost delta beside "denominator drift: none".
    """
    baseline = _results({"A1": ("held_in", 1.0), "B1": ("held_out", 1.0)})
    candidate = _results({"A1": ("held_in", 1.0), "B1": ("held_out", 1.0)})
    baseline["summary"] |= {
        "metrics": {"tokens": 1000.0},
        "metric_task_counts": {"tokens": 2},
        "metric_attempt_counts": {"tokens": 6},
    }
    candidate["summary"] |= {
        "metrics": {"tokens": 670.0},
        "metric_task_counts": {"tokens": 2},  # same tasks...
        "metric_attempt_counts": {"tokens": 2},  # ...a third of the attempts
    }
    d = delta(baseline, candidate)
    assert d["metric_task_counts"]["tokens"] == {"baseline": 2, "candidate": 2}
    assert d["metric_denominator_drift"] == ["tokens"], "attempt-level drift went unreported"
    assert d["metric_attempt_counts"]["tokens"] == {"baseline": 6, "candidate": 2}


def test_delta_surfaces_metric_denominator_drift():
    """Two sides can report the SAME mean over different populations: an attempt
    that raises records ``metrics={}``, so a task the candidate breaks drops out of
    the mean entirely. Diagnostic only, but it must never be invisible."""
    baseline = _results({"A1": ("held_in", 1.0), "B1": ("held_out", 1.0)})
    candidate = _results({"A1": ("held_in", 1.0), "B1": ("held_out", 1.0)})
    baseline["summary"] |= {"metrics": {"tokens": 100.0}, "metric_task_counts": {"tokens": 3}}
    candidate["summary"] |= {"metrics": {"tokens": 100.0}, "metric_task_counts": {"tokens": 2}}
    d = delta(baseline, candidate)
    assert d["metric_delta"] == {"tokens": 0.0}  # identical means...
    assert d["metric_denominator_drift"] == ["tokens"]  # ...over different task counts
    assert d["metric_task_counts"]["tokens"] == {"baseline": 3, "candidate": 2}


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


def test_catastrophic_veto_needs_both_a_full_baseline_and_a_zero_candidate():
    """Pin each conjunct SEPARATELY.

    Mutating either one alone left the suite green, because the other still held:
    `== 1.0` → `> 0.0` needs the candidate to be exactly 0.0 to fire, and `== 0.0` →
    `< 1.0` needs the baseline to be exactly 1.0. Together they promote every negative
    per-task movement to a hard veto — the sampling-noise sensitivity `delta.py`'s own
    comment says was deliberately rejected.
    """
    for label, base, cand in (
        ("full baseline, partial drop", 1.0, 0.5),
        ("partial to zero", 0.5, 0.0),
    ):
        baseline = _results({"gain": ("held_in", 0.0), "drop": ("held_out", base)})
        candidate = _results({"gain": ("held_in", 1.0), "drop": ("held_out", cand)})
        d = delta(baseline, candidate)
        assert d["catastrophic_regressions"] == {}, f"{label} wrongly vetoed"
        assert d["regressions"], f"{label} should still be a visible warning"


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


def test_delta_refuses_mismatched_serving_base():
    """Same model string, different provider or quantization is a different
    serving base in everything but name — a Δ across them measures the serving
    swap, not the edit."""
    import pytest

    tasks = {"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)}
    pinned = _results(tasks, provider_order="deepinfra", quantization="fp8")
    other_provider = _results(tasks, provider_order="together", quantization="fp8")
    other_quant = _results(tasks, provider_order="deepinfra", quantization="bf16")
    unpinned = _results(tasks)
    with pytest.raises(ValueError, match="provider_order"):
        delta(pinned, other_provider)
    with pytest.raises(ValueError, match="quantization"):
        delta(pinned, other_quant)
    with pytest.raises(ValueError, match="provider_order"):
        delta(pinned, unpinned)


def test_delta_accepts_a_matching_serving_pin():
    tasks = {"A1": ("held_in", 1.0), "B1": ("held_in", 1.0), "A3": ("held_out", 0.4)}
    base = _results(tasks, provider_order="deepinfra", quantization="fp8")
    cand = _results(tasks, provider_order="deepinfra", quantization="fp8")
    d = delta(base, cand)
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


def _run(counts: dict[str, tuple[int, int, str]]) -> dict:
    """A results dict from integer counts, with `pass_fraction` stored ROUNDED exactly
    as `suite.py` writes it — the rounding is the thing under test."""
    return {
        "fingerprint": {"runner_sha": "x", "model": "m", "config_version": 1, "dirty_sha": None},
        "tasks": {
            n: {
                "split": s,
                "attempts": a,
                "passes": p,
                "pass_fraction": round(p / a, 4),
            }
            for n, (p, a, s) in counts.items()
        },
    }


def test_movements_that_cancel_give_exactly_zero_not_a_tiny_negative():
    """One task up an attempt, another down an attempt: the split change is exactly 0.

    It was not. `pass_fraction` is stored rounded to 4 places and `split_rate` summed
    those, so 2/3 became 0.6667 while two thirds summed to 0.3333 + 0.3333 = 0.6666.
    Two runs with IDENTICAL integer counts reported Δ_in = -5.56e-06, and the rule
    tests `>= 0`, so a storage-format artefact was read as a real regression. Measured
    across six unchanged runs it fired on 2 of 15 pairs.
    """
    from runner.delta import delta

    before = _run({"T1": (2, 3, "held_in"), "T2": (1, 3, "held_in"), "H": (5, 5, "held_out")})
    after = _run({"T1": (1, 3, "held_in"), "T2": (2, 3, "held_in"), "H": (5, 5, "held_out")})

    d = delta(before, after)
    assert d["delta_in"] == 0.0, f"cancelling movements must be exactly 0, got {d['delta_in']!r}"
    # Not merely close: the sign bit matters, because the rule tests `>= 0`.
    assert not (d["delta_in"] < 0)
    assert d["delta_ho"] == 0.0


def test_identical_integer_counts_give_a_zero_delta_whatever_the_rounding():
    """Same counts on both sides must mean no movement, for every third-shaped rate.

    Thirds are where 4-place rounding loses the most, and 3 attempts is the held-in
    default, so this is the shipped case rather than a corner.
    """
    from runner.delta import delta

    counts = {f"T{i}": (i % 4, 3, "held_in") for i in range(1, 9)}
    counts["H"] = (3, 5, "held_out")
    d = delta(_run(counts), _run(dict(counts)))
    assert d["delta_in"] == 0.0 and d["delta_ho"] == 0.0
    assert all(v == 0.0 for v in d["per_task"].values())


def test_a_real_one_attempt_movement_still_registers():
    """The fix must not flatten genuine movement into zero as well."""
    from runner.delta import delta

    before = _run({"T": (2, 3, "held_in"), "H": (4, 5, "held_out")})
    after = _run({"T": (3, 3, "held_in"), "H": (5, 5, "held_out")})
    d = delta(before, after)
    assert abs(d["delta_in"] - 1 / 3) < 1e-12
    assert abs(d["delta_ho"] - 1 / 5) < 1e-12
