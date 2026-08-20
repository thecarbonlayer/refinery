"""Pins for `loop.calibrate`: measured noise bounds from null (nothing-changed) arms.

Built against the frozen contract (contracts/phase2-calibration-contract.md §2-3):
five null arms (a full-suite baseline plus four --only supported-set subset runs)
feed `calibrate()`, which must refuse a fingerprint mismatch, accept filtered/
subset arms (the whole point — `runner.delta` refuses them), and record what the
CURRENT, uncalibrated `evaluate()` rule does on every arm pair, including the
full-suite-vs-subset pairs whose mismatched task sets/attempts make `evaluate()`
raise.
"""

from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import pytest

from loop.acceptance import CONFIRM, REJECT
from loop.calibrate import (
    COVERAGE_LEVEL,
    calibrate,
    calibrate_model,
    null_gain_quantile,
    null_task_quantile,
)


def _arm(
    label: str,
    passes_by_task: dict[str, tuple[int, int, str]],
    *,
    fingerprint: dict | None = None,
    filtered: bool = False,
) -> dict:
    """A suite-JSON-shaped result: name -> (passes, attempts, split).

    Mirrors `runner/suite.py`'s written shape closely enough for `calibrate()` and
    `loop.acceptance.evaluate()` to consume: `fingerprint`, `tasks{name: {split,
    attempts, passes, pass_fraction}}`, and — when `filtered` — a `filter` key, the
    marker `runner.delta` refuses outright and this tool exists precisely to accept.
    """
    fp = fingerprint or {"runner_sha": "rsha1", "config_version": 7, "model": "carbon-model"}
    tasks = {
        name: {
            "split": split,
            "attempts": attempts,
            "passes": passes,
            "pass_fraction": round(passes / attempts, 4),
        }
        for name, (passes, attempts, split) in passes_by_task.items()
    }
    result = {"fingerprint": fp, "tasks": tasks}
    if filtered:
        result["filter"] = sorted(passes_by_task)
    return result


def _write(results_dir: Path, label: str, result: dict) -> None:
    (results_dir / f"{label}.json").write_text(json.dumps(result))


def test_fingerprint_mismatch_across_arms_is_refused_naming_the_field(tmp_path):
    arm0 = _arm("arm0", {"A1": (2, 3, "held_in")})
    arm1 = _arm(
        "arm1",
        {"A1": (2, 3, "held_in")},
        fingerprint={"runner_sha": "rsha2", "config_version": 7, "model": "carbon-model"},
    )
    _write(tmp_path, "arm0", arm0)
    _write(tmp_path, "arm1", arm1)
    with pytest.raises(ValueError, match="runner_sha"):
        calibrate(["arm0", "arm1"], tmp_path, frozenset({"A1"}))


def test_config_version_mismatch_is_also_refused_naming_the_field(tmp_path):
    arm0 = _arm("arm0", {"A1": (2, 3, "held_in")})
    arm1 = _arm(
        "arm1",
        {"A1": (2, 3, "held_in")},
        fingerprint={"runner_sha": "rsha1", "config_version": 8, "model": "carbon-model"},
    )
    _write(tmp_path, "arm0", arm0)
    _write(tmp_path, "arm1", arm1)
    with pytest.raises(ValueError, match="config_version"):
        calibrate(["arm0", "arm1"], tmp_path, frozenset({"A1"}))


def test_dirty_sha_mismatch_across_arms_is_refused(tmp_path):
    """Two arms can share `runner_sha`/`config_version`/`model` and still be
    measuring DIFFERENT carbon states if the working tree was dirty and edited
    between them -- `dirty_sha` is what tells those apart (contract §2), and
    omitting it from the digest let two such arms read as identical null arms."""
    arm0 = _arm(
        "arm0",
        {"A1": (2, 3, "held_in")},
        fingerprint={
            "runner_sha": "rsha1",
            "config_version": 7,
            "model": "carbon-model",
            "dirty_sha": "d" * 40,
        },
    )
    arm1 = _arm(
        "arm1",
        {"A1": (2, 3, "held_in")},
        fingerprint={
            "runner_sha": "rsha1",
            "config_version": 7,
            "model": "carbon-model",
            "dirty_sha": "e" * 40,
        },
    )
    _write(tmp_path, "arm0", arm0)
    _write(tmp_path, "arm1", arm1)
    with pytest.raises(ValueError, match="dirty_sha"):
        calibrate(["arm0", "arm1"], tmp_path, frozenset({"A1"}))


def test_dirty_sha_none_on_both_arms_is_consistent_not_a_mismatch(tmp_path):
    """A clean tree on both arms (`dirty_sha` absent -> None on both) must not be
    refused -- None==None IS consistent; only a genuine difference is a mismatch."""
    arm0 = _arm("arm0", {"A1": (2, 3, "held_in")})
    arm1 = _arm("arm1", {"A1": (2, 3, "held_in")})
    _write(tmp_path, "arm0", arm0)
    _write(tmp_path, "arm1", arm1)
    result = calibrate(["arm0", "arm1"], tmp_path, frozenset({"A1"}))
    assert result["arms"][0]["dirty_sha"] is None
    assert result["arms"][1]["dirty_sha"] is None


def test_duplicate_arm_labels_are_refused_naming_the_label(tmp_path):
    """Supplying the same label twice would satisfy the two-arm shape from one
    physical result -- a pairwise spread of zero built from one arm compared with
    itself, not two independent null measurements."""
    arm0 = _arm("arm0", {"A1": (2, 3, "held_in")})
    _write(tmp_path, "arm0", arm0)
    with pytest.raises(ValueError, match="arm0"):
        calibrate(["arm0", "arm0"], tmp_path, frozenset({"A1"}))


def test_per_task_max_pairwise_delta_and_counts(tmp_path):
    """An 8/10 -> 6/10 move on one task is the textbook example: |Δ| = 0.2."""
    supported = frozenset({"A1", "G2"})
    arm_a = _arm("null-cmp-a", {"A1": (8, 10, "held_in"), "G2": (5, 10, "held_out")})
    arm_b = _arm("null-cmp-b", {"A1": (6, 10, "held_in"), "G2": (5, 10, "held_out")})
    _write(tmp_path, "null-cmp-a", arm_a)
    _write(tmp_path, "null-cmp-b", arm_b)

    result = calibrate(["null-cmp-a", "null-cmp-b"], tmp_path, supported)

    assert result["per_task"]["A1"]["max_abs_delta"] == pytest.approx(0.2)
    assert result["per_task"]["A1"]["passes_by_arm"] == {
        "null-cmp-a": {"passes": 8, "attempts": 10},
        "null-cmp-b": {"passes": 6, "attempts": 10},
    }
    assert result["per_task"]["G2"]["max_abs_delta"] == 0.0


def test_section_noise_is_max_pairwise_supported_set_mean_delta(tmp_path):
    supported = frozenset({"A1", "G4", "G2"})
    arm_a = _arm(
        "null-cmp-a",
        {"A1": (8, 10, "held_in"), "G4": (6, 10, "held_in"), "G2": (5, 10, "held_out")},
    )
    arm_b = _arm(
        "null-cmp-b",
        {"A1": (6, 10, "held_in"), "G4": (6, 10, "held_in"), "G2": (3, 10, "held_out")},
    )
    _write(tmp_path, "null-cmp-a", arm_a)
    _write(tmp_path, "null-cmp-b", arm_b)

    result = calibrate(["null-cmp-a", "null-cmp-b"], tmp_path, supported)

    # held_in means: (0.8+0.6)/2=0.7 vs (0.6+0.6)/2=0.6 -> |Δ|=0.1
    assert result["section_noise"]["held_in"] == pytest.approx(0.1)
    # held_out means: 0.5 vs 0.3 -> |Δ|=0.2
    assert result["section_noise"]["held_out"] == pytest.approx(0.2)
    assert set(result["section_noise_arms"]["held_in"]) == {"null-cmp-a", "null-cmp-b"}
    assert set(result["section_noise_arms"]["held_out"]) == {"null-cmp-a", "null-cmp-b"}


def test_section_noise_exact_carries_the_true_rational_bound(tmp_path):
    """`section_noise` rounds the bound to a float for display; `section_noise_exact`
    must carry the SAME bound as an exact fraction string, unrounded -- 3/10 is not
    exact in binary (the nearest float64 is slightly BELOW it), so a consumer that
    only had the float and rebuilt a `Fraction` from it would get a threshold that is
    not actually 3/10."""
    from fractions import Fraction

    supported = frozenset({"A1"})
    arm_a = _arm("null-cmp-a", {"A1": (8, 10, "held_in")})
    arm_b = _arm("null-cmp-b", {"A1": (5, 10, "held_in")})
    _write(tmp_path, "null-cmp-a", arm_a)
    _write(tmp_path, "null-cmp-b", arm_b)

    result = calibrate(["null-cmp-a", "null-cmp-b"], tmp_path, supported)

    assert result["section_noise"]["held_in"] == pytest.approx(0.3)
    exact = result["section_noise_exact"]["held_in"]
    num, den = exact.split("/")
    assert Fraction(int(num), int(den)) == Fraction(3, 10)
    # the float round-trip of THIS specific bound is provably not the true value --
    # pinning that the fixture actually exercises the gap the exact field closes.
    assert Fraction(result["section_noise"]["held_in"]) != Fraction(3, 10)


def test_section_noise_exact_is_zero_over_one_when_no_group_covers_the_split(tmp_path):
    """The `not groups` branch (no arm has a UNIFORM attempt count across the whole
    supported set for this split) writes a bound of 0.0 by construction --
    `section_noise_exact` must say the same thing exactly, not leave the split out or
    emit a different shape."""
    supported = frozenset({"A1", "G2"})
    # every arm mixes attempt counts within the split -- no arm ever contributes to a
    # group, so `groups` stays empty and the `not groups` branch fires.
    arm_a = _arm("null-cmp-a", {"A1": (8, 10, "held_in"), "G2": (3, 5, "held_in")})
    arm_b = _arm("null-cmp-b", {"A1": (5, 10, "held_in"), "G2": (2, 5, "held_in")})
    _write(tmp_path, "null-cmp-a", arm_a)
    _write(tmp_path, "null-cmp-b", arm_b)

    result = calibrate(["null-cmp-a", "null-cmp-b"], tmp_path, supported)

    assert result["section_noise_arms"]["held_in"] == []
    assert result["section_noise"]["held_in"] == 0.0
    assert result["section_noise_exact"]["held_in"] == "0/1"


def test_section_noise_excludes_an_arm_whose_attempts_dont_match_the_others(tmp_path):
    """Arm 0 (full suite, 3/5 attempts) never feeds section_noise alongside arms
    a-d (10 attempts): the bound needs equal-attempts arms, exactly as `evaluate()`
    needs equal attempts for a Δ to be like-for-like. `section_noise_arms` must say
    honestly which arms fed the bound."""
    supported = frozenset({"A1", "G4", "G2"})
    arm0 = _arm(
        "arm0",
        {
            "A1": (2, 3, "held_in"),
            "G4": (2, 3, "held_in"),
            "G2": (3, 5, "held_out"),
            "UNSUPPORTED": (1, 3, "held_in"),
        },
    )
    arm_a = _arm(
        "null-cmp-a",
        {"A1": (8, 10, "held_in"), "G4": (6, 10, "held_in"), "G2": (5, 10, "held_out")},
    )
    arm_b = _arm(
        "null-cmp-b",
        {"A1": (6, 10, "held_in"), "G4": (6, 10, "held_in"), "G2": (3, 10, "held_out")},
    )
    for label, r in (("arm0", arm0), ("null-cmp-a", arm_a), ("null-cmp-b", arm_b)):
        _write(tmp_path, label, r)

    result = calibrate(["arm0", "null-cmp-a", "null-cmp-b"], tmp_path, supported)

    assert "arm0" not in result["section_noise_arms"]["held_in"]
    assert "arm0" not in result["section_noise_arms"]["held_out"]
    assert set(result["section_noise_arms"]["held_in"]) == {"null-cmp-a", "null-cmp-b"}
    # same value as the two-arm-only case above
    assert result["section_noise"]["held_in"] == pytest.approx(0.1)
    assert result["section_noise"]["held_out"] == pytest.approx(0.2)


def test_pairwise_outcomes_one_entry_per_unordered_pair(tmp_path):
    supported = frozenset({"A1", "G4"})
    labels = ["null-cmp-a", "null-cmp-b", "null-cmp-c"]
    arms = {
        "null-cmp-a": _arm("null-cmp-a", {"A1": (8, 10, "held_in"), "G4": (5, 10, "held_in")}),
        "null-cmp-b": _arm("null-cmp-b", {"A1": (8, 10, "held_in"), "G4": (5, 10, "held_in")}),
        "null-cmp-c": _arm("null-cmp-c", {"A1": (7, 10, "held_in"), "G4": (5, 10, "held_in")}),
    }
    for label in labels:
        _write(tmp_path, label, arms[label])

    result = calibrate(labels, tmp_path, supported)

    assert len(result["pairwise_outcomes"]) == 3  # C(3,2)
    for entry in result["pairwise_outcomes"].values():
        assert entry["outcome"] in {REJECT, CONFIRM}
        # equal task sets on both sides -- no intersection filtering happened
        assert "task_intersection" not in entry


def test_filtered_subset_arms_are_accepted_not_refused(tmp_path):
    """The tool's whole point: unlike `runner.delta`, a `filter` key must not
    trigger a refusal."""
    supported = frozenset({"A1"})
    arm_a = _arm("null-cmp-a", {"A1": (8, 10, "held_in")}, filtered=True)
    arm_b = _arm("null-cmp-b", {"A1": (6, 10, "held_in")}, filtered=True)
    _write(tmp_path, "null-cmp-a", arm_a)
    _write(tmp_path, "null-cmp-b", arm_b)

    result = calibrate(["null-cmp-a", "null-cmp-b"], tmp_path, supported)

    assert result["per_task"]["A1"]["max_abs_delta"] == pytest.approx(0.2)


def test_full_suite_vs_subset_pair_evaluates_over_the_intersection(tmp_path):
    """Arm 0 (full suite) carries a task the subset arm never ran at all; the pair
    must record `task_intersection` and, since restricting to the intersection
    still leaves mismatched attempts (3 vs 10), `evaluate()` raises inside
    `_parity` -- the tool must catch that and record ERROR honestly rather than
    crash the whole analysis."""
    supported = frozenset({"A1", "G4"})
    full = _arm(
        "baseline-p1-r2",
        {
            "A1": (2, 3, "held_in"),
            "G4": (2, 3, "held_in"),
            "UNSUPPORTED": (1, 3, "held_in"),
        },
    )
    subset = _arm("null-cmp-a", {"A1": (8, 10, "held_in"), "G4": (6, 10, "held_in")})
    _write(tmp_path, "baseline-p1-r2", full)
    _write(tmp_path, "null-cmp-a", subset)

    result = calibrate(["baseline-p1-r2", "null-cmp-a"], tmp_path, supported)

    assert len(result["pairwise_outcomes"]) == 1
    entry = next(iter(result["pairwise_outcomes"].values()))
    assert sorted(entry["task_intersection"]) == ["A1", "G4"]
    assert entry["outcome"] == "ERROR"
    assert "error" in entry and entry["error"]


def test_arms_field_and_computed_at_runner_sha(tmp_path):
    supported = frozenset({"A1"})
    arm_a = _arm("null-cmp-a", {"A1": (8, 10, "held_in")})
    arm_b = _arm("null-cmp-b", {"A1": (6, 10, "held_in")})
    _write(tmp_path, "null-cmp-a", arm_a)
    _write(tmp_path, "null-cmp-b", arm_b)

    result = calibrate(["null-cmp-a", "null-cmp-b"], tmp_path, supported)

    assert result["computed_at_runner_sha"] == "rsha1"
    assert [a["label"] for a in result["arms"]] == ["null-cmp-a", "null-cmp-b"]
    for arm in result["arms"]:
        assert arm["runner_sha"] == "rsha1"
        assert arm["config_version"] == 7
        assert arm["model"] == "carbon-model"


def test_main_writes_analysis_json(tmp_path, monkeypatch):
    import loop.calibrate as calibrate_mod

    supported = frozenset({"A1"})
    arm_a = _arm("null-cmp-a", {"A1": (8, 10, "held_in")})
    arm_b = _arm("null-cmp-b", {"A1": (6, 10, "held_in")})
    _write(tmp_path, "null-cmp-a", arm_a)
    _write(tmp_path, "null-cmp-b", arm_b)

    out_path = tmp_path / "out" / "analysis.json"
    monkeypatch.setattr(calibrate_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(calibrate_mod, "SUPPORTED", supported)
    monkeypatch.setattr(calibrate_mod, "ANALYSIS_PATH", out_path)

    calibrate_mod.main(["null-cmp-a", "null-cmp-b"])

    written = json.loads(out_path.read_text())
    assert written["per_task"]["A1"]["max_abs_delta"] == pytest.approx(0.2)


def test_main_requires_at_least_one_label(monkeypatch):
    import loop.calibrate as calibrate_mod

    with pytest.raises(SystemExit):
        calibrate_mod.main([])


# ---------------------------------------------------------------------------
# Round 2 (contracts/phase2b-calibration-contract.md): `calibrate_model()`,
# the null-MODEL artifact, and the exact-enumeration primitives it is built
# on. `_arm`/`_write` above are reused unchanged -- they already support a
# custom `fingerprint` dict (round-2's provenance needs `gemma_sha`, the
# runner's own field, mapped to `carbon_sha`) and the `filtered` marker
# (round-2's `_check_supported_set` reads exactly this "filter" key to tell a
# `--only` subset arm apart from a full-suite one).
# ---------------------------------------------------------------------------

_MODEL_SUPPORTED = frozenset({"A1", "G2", "G4", "G5"})


def test_null_rate_is_pooled_exact_unreduced_fraction_string(tmp_path):
    """Pooled over ALL arms (contract §1) as a LITERAL "passes/attempts"
    string -- the denominator is the pooled attempt count itself (contract
    §3's ">= 49 held-in, >= 55 for G2"), never reduced the way
    `section_noise_exact` reduces a bound, or that count would be hidden."""
    supported = frozenset({"A1"})
    arm_full = _arm("full-a", {"A1": (6, 10, "held_in")})
    arm_a = _arm("cmp-a", {"A1": (5, 10, "held_in")}, filtered=True)
    arm_b = _arm("cmp-b", {"A1": (5, 10, "held_in")}, filtered=True)
    arm_c = _arm("cmp-c", {"A1": (5, 10, "held_in")}, filtered=True)
    for label, r in (("full-a", arm_full), ("cmp-a", arm_a), ("cmp-b", arm_b), ("cmp-c", arm_c)):
        _write(tmp_path, label, r)

    result = calibrate_model(["full-a", "cmp-a", "cmp-b", "cmp-c"], tmp_path, supported)

    assert result["null_model"]["A1"]["null_rate"] == "21/40"
    assert result["null_model"]["A1"]["per_arm"] == {
        "full-a": [6, 10],
        "cmp-a": [5, 10],
        "cmp-b": [5, 10],
        "cmp-c": [5, 10],
    }
    assert result["coverage_level"] == "0.975"


def test_provenance_refusal_on_carbon_sha_mismatch(tmp_path):
    """`carbon_sha` (contract §1) is sourced from the fingerprint's
    `gemma_sha` -- carbon is the section under test, not the verifier, and
    two arms measuring DIFFERENT carbon revisions are not a null pair."""
    fp1 = {
        "runner_sha": "rsha1",
        "config_version": 7,
        "model": "carbon-model",
        "gemma_sha": "aaaa",
        "dirty_sha": None,
    }
    fp2 = {**fp1, "gemma_sha": "bbbb"}
    arm0 = _arm("full-a", {"A1": (2, 3, "held_in")}, fingerprint=fp1)
    arm1 = _arm("cmp-a", {"A1": (5, 10, "held_in")}, fingerprint=fp2, filtered=True)
    _write(tmp_path, "full-a", arm0)
    _write(tmp_path, "cmp-a", arm1)
    with pytest.raises(ValueError, match="carbon_sha"):
        calibrate_model(["full-a", "cmp-a"], tmp_path, frozenset({"A1"}))


def test_provenance_refusal_reuses_the_fingerprint_field_check(tmp_path):
    """The round-1 4 fields still refuse under `calibrate_model` -- it EXTENDS
    `_check_fingerprints`'s check, it does not reimplement a narrower one."""
    fp1 = {
        "runner_sha": "rsha1",
        "config_version": 7,
        "model": "carbon-model",
        "gemma_sha": "aaaa",
        "dirty_sha": None,
    }
    fp2 = {**fp1, "runner_sha": "rsha2"}
    arm0 = _arm("full-a", {"A1": (2, 3, "held_in")}, fingerprint=fp1)
    arm1 = _arm("cmp-a", {"A1": (5, 10, "held_in")}, fingerprint=fp2, filtered=True)
    _write(tmp_path, "full-a", arm0)
    _write(tmp_path, "cmp-a", arm1)
    with pytest.raises(ValueError, match="runner_sha"):
        calibrate_model(["full-a", "cmp-a"], tmp_path, frozenset({"A1"}))


def test_duplicate_label_refused_naming_the_label_for_calibrate_model(tmp_path):
    """`_check_labels` is shared with `calibrate()` -- the refusal, and its
    wording, must not diverge between the two entry points."""
    arm0 = _arm("full-a", {"A1": (2, 3, "held_in")})
    _write(tmp_path, "full-a", arm0)
    with pytest.raises(ValueError, match="full-a"):
        calibrate_model(["full-a", "full-a"], tmp_path, frozenset({"A1"}))


def test_fifth_task_in_a_subset_run_is_refused(tmp_path):
    """Contract §1's pin: {A1, G2, G4, G5} exactly. A `--only` subset arm
    (``filter`` present) carrying a task outside that set breaks the
    protocol's promise that only the supported set was measured there."""
    full = _arm(
        "full-a",
        {
            "A1": (2, 3, "held_in"),
            "G2": (3, 5, "held_out"),
            "G4": (1, 3, "held_in"),
            "G5": (2, 3, "held_in"),
        },
    )
    bad_subset = _arm(
        "cmp-bad",
        {
            "A1": (5, 10, "held_in"),
            "G2": (5, 10, "held_out"),
            "G4": (2, 10, "held_in"),
            "G5": (6, 10, "held_in"),
            "EXTRA": (3, 10, "held_in"),
        },
        filtered=True,
    )
    _write(tmp_path, "full-a", full)
    _write(tmp_path, "cmp-bad", bad_subset)
    with pytest.raises(ValueError, match="EXTRA"):
        calibrate_model(["full-a", "cmp-bad"], tmp_path, _MODEL_SUPPORTED)


def test_arm_missing_a_supported_task_is_refused(tmp_path):
    """Every arm must cover every supported task -- the pool needs a full
    denominator on each, never a partial one standing in for the whole."""
    incomplete = _arm(
        "full-a",
        {"A1": (2, 3, "held_in"), "G2": (3, 5, "held_out"), "G4": (1, 3, "held_in")},
    )  # missing G5
    _write(tmp_path, "full-a", incomplete)
    with pytest.raises(ValueError, match="G5"):
        calibrate_model(["full-a"], tmp_path, _MODEL_SUPPORTED)


def test_fitness_grain_fails_for_a_fabricated_tight_model(tmp_path):
    """A degenerate (always-0) pooled rate collapses the null distribution of
    (mean_b - mean_a) to a point mass at 0 -- its 97.5% quantile is 0, which
    can never exceed a positive grain (contract §4.1). This is the round-1
    failure mode made structural: a threshold that gates nothing."""
    arm_a = _arm("full-a", {"A1": (0, 3, "held_in")})
    arm_b = _arm("full-b", {"A1": (0, 3, "held_in")})
    _write(tmp_path, "full-a", arm_a)
    _write(tmp_path, "full-b", arm_b)

    result = calibrate_model(["full-a", "full-b"], tmp_path, frozenset({"A1"}))

    grain = result["fitness"]["grain"]
    assert grain["held_in"]["quantile"] == "0/1"
    assert grain["held_in"]["grain"] == "1/3"
    assert grain["held_in"]["pass"] is False
    assert grain["pass"] is False
    assert result["fitness"]["fit"] is False


def test_fitness_goodness_fails_for_an_outlier_arm(tmp_path):
    """One arm at 0/10 pooled against three arms at 9/10 disagrees with the
    single-rate null model badly enough (contract §4.2) that its exact
    two-sided binomial tail sits far below 0.01 -- the other arms' own tails
    stay well clear, so only the outlier is flagged, per task+arm."""
    full = _arm("full-a", {"A1": (9, 10, "held_in")})
    good_a = _arm("cmp-a", {"A1": (9, 10, "held_in")}, filtered=True)
    good_b = _arm("cmp-b", {"A1": (9, 10, "held_in")}, filtered=True)
    outlier = _arm("cmp-c", {"A1": (0, 10, "held_in")}, filtered=True)
    for label, r in (
        ("full-a", full),
        ("cmp-a", good_a),
        ("cmp-b", good_b),
        ("cmp-c", outlier),
    ):
        _write(tmp_path, label, r)

    result = calibrate_model(["full-a", "cmp-a", "cmp-b", "cmp-c"], tmp_path, frozenset({"A1"}))

    goodness = result["fitness"]["goodness"]
    per_arm = goodness["per_task"]["A1"]["per_arm"]
    assert per_arm["cmp-c"]["pass"] is False
    assert Fraction(*(int(x) for x in per_arm["cmp-c"]["tail_p"].split("/"))) < Fraction(1, 100)
    assert per_arm["full-a"]["pass"] is True
    assert per_arm["cmp-a"]["pass"] is True
    assert per_arm["cmp-b"]["pass"] is True
    assert goodness["pass"] is False
    # grain still passes here (a real spread, not a degenerate point mass) --
    # `fit` is false specifically because of goodness, not because grain also
    # failed, proving the two checks are independently recorded.
    assert result["fitness"]["grain"]["pass"] is True
    assert result["fitness"]["fit"] is False


def test_fit_false_does_not_raise_and_is_recorded_not_enforced(tmp_path):
    """`calibrate_model` computes and records `fit`; it never refuses to
    RETURN an unfit artifact -- installing/loading it is the loader's job
    (contract §4: "the artifact is written with fit: false"), not this
    function's."""
    arm_a = _arm("full-a", {"A1": (0, 3, "held_in")})
    arm_b = _arm("full-b", {"A1": (0, 3, "held_in")})
    _write(tmp_path, "full-a", arm_a)
    _write(tmp_path, "full-b", arm_b)

    result = calibrate_model(["full-a", "full-b"], tmp_path, frozenset({"A1"}))  # must not raise

    assert result["fitness"]["fit"] is False
    assert "null_model" in result and "provenance" in result


_REALISTIC_FINGERPRINT = {
    "runner_sha": "rsha1",
    "config_version": 7,
    "model": "carbon-model",
    "gemma_sha": "carbon-realistic-sha",
    "dirty_sha": None,
}


def test_realistic_seven_arm_protocol_is_fit(tmp_path):
    """The full contract §3 shape -- 3 full-suite arms (3/5 attempts) + 4
    `--only` subset arms (10 attempts) -- pools to exactly the counts §3
    promises (49 held-in, 55 for G2) and, with a real (non-degenerate)
    spread, clears every fitness check."""
    full_a = _arm(
        "r2-null-full-a",
        {
            "A1": (2, 3, "held_in"),
            "G2": (3, 5, "held_out"),
            "G4": (1, 3, "held_in"),
            "G5": (2, 3, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
    )
    full_b = _arm(
        "r2-null-full-b",
        {
            "A1": (2, 3, "held_in"),
            "G2": (2, 5, "held_out"),
            "G4": (1, 3, "held_in"),
            "G5": (2, 3, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
    )
    full_c = _arm(
        "r2-null-full-c",
        {
            "A1": (1, 3, "held_in"),
            "G2": (3, 5, "held_out"),
            "G4": (0, 3, "held_in"),
            "G5": (3, 3, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
    )
    cmp_a = _arm(
        "r2-null-cmp-a",
        {
            "A1": (5, 10, "held_in"),
            "G2": (5, 10, "held_out"),
            "G4": (2, 10, "held_in"),
            "G5": (6, 10, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
        filtered=True,
    )
    cmp_b = _arm(
        "r2-null-cmp-b",
        {
            "A1": (6, 10, "held_in"),
            "G2": (6, 10, "held_out"),
            "G4": (1, 10, "held_in"),
            "G5": (7, 10, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
        filtered=True,
    )
    cmp_c = _arm(
        "r2-null-cmp-c",
        {
            "A1": (5, 10, "held_in"),
            "G2": (4, 10, "held_out"),
            "G4": (2, 10, "held_in"),
            "G5": (6, 10, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
        filtered=True,
    )
    cmp_d = _arm(
        "r2-null-cmp-d",
        {
            "A1": (6, 10, "held_in"),
            "G2": (7, 10, "held_out"),
            "G4": (1, 10, "held_in"),
            "G5": (5, 10, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
        filtered=True,
    )
    arms = {
        "r2-null-full-a": full_a,
        "r2-null-full-b": full_b,
        "r2-null-full-c": full_c,
        "r2-null-cmp-a": cmp_a,
        "r2-null-cmp-b": cmp_b,
        "r2-null-cmp-c": cmp_c,
        "r2-null-cmp-d": cmp_d,
    }
    for label, r in arms.items():
        _write(tmp_path, label, r)

    labels = list(arms)
    result = calibrate_model(labels, tmp_path, _MODEL_SUPPORTED)

    assert result["null_model"]["A1"]["null_rate"] == "27/49"
    assert result["null_model"]["G2"]["null_rate"] == "30/55"
    assert result["null_model"]["G4"]["null_rate"] == "8/49"
    assert result["null_model"]["G5"]["null_rate"] == "31/49"
    assert len(result["provenance"]) == 7
    assert result["fitness"]["fit"] is True
    assert result["fitness"]["power"]["per_task"].keys() == _MODEL_SUPPORTED
    assert result["computed_at_runner_sha"] == "rsha1"
    # Positive proof carbon_sha is actually POPULATED from gemma_sha here, not
    # just consistently None on every arm (the mismatch tests only prove the
    # two DIFFER when they should; this proves the mapping itself works).
    for entry in result["provenance"]:
        assert entry["carbon_sha"] == "carbon-realistic-sha"


def test_fitness_power_gain_gate_rows_exist_for_both_splits_and_are_exact_fractions(tmp_path):
    """Contract §4.4 says "each gate", not one -- `per_task` alone only ever
    covered the repeat/guard gate's power (single task, confirmation attempt
    counts). The gain gate `evaluate()` actually judges is the split's
    supported-set MEAN at STANDARD attempt counts; its own power is a
    separate, weaker number a reader must be able to see, not one this
    artifact is allowed to omit just because a per-task number exists nearby.
    """
    full_a = _arm(
        "r2-null-full-a",
        {
            "A1": (2, 3, "held_in"),
            "G2": (3, 5, "held_out"),
            "G4": (1, 3, "held_in"),
            "G5": (2, 3, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
    )
    full_b = _arm(
        "r2-null-full-b",
        {
            "A1": (2, 3, "held_in"),
            "G2": (2, 5, "held_out"),
            "G4": (1, 3, "held_in"),
            "G5": (2, 3, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
    )
    full_c = _arm(
        "r2-null-full-c",
        {
            "A1": (1, 3, "held_in"),
            "G2": (3, 5, "held_out"),
            "G4": (0, 3, "held_in"),
            "G5": (3, 3, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
    )
    cmp_a = _arm(
        "r2-null-cmp-a",
        {
            "A1": (5, 10, "held_in"),
            "G2": (5, 10, "held_out"),
            "G4": (2, 10, "held_in"),
            "G5": (6, 10, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
        filtered=True,
    )
    cmp_b = _arm(
        "r2-null-cmp-b",
        {
            "A1": (6, 10, "held_in"),
            "G2": (6, 10, "held_out"),
            "G4": (1, 10, "held_in"),
            "G5": (7, 10, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
        filtered=True,
    )
    cmp_c = _arm(
        "r2-null-cmp-c",
        {
            "A1": (5, 10, "held_in"),
            "G2": (4, 10, "held_out"),
            "G4": (2, 10, "held_in"),
            "G5": (6, 10, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
        filtered=True,
    )
    cmp_d = _arm(
        "r2-null-cmp-d",
        {
            "A1": (6, 10, "held_in"),
            "G2": (7, 10, "held_out"),
            "G4": (1, 10, "held_in"),
            "G5": (5, 10, "held_in"),
        },
        fingerprint=_REALISTIC_FINGERPRINT,
        filtered=True,
    )
    arms = {
        "r2-null-full-a": full_a,
        "r2-null-full-b": full_b,
        "r2-null-full-c": full_c,
        "r2-null-cmp-a": cmp_a,
        "r2-null-cmp-b": cmp_b,
        "r2-null-cmp-c": cmp_c,
        "r2-null-cmp-d": cmp_d,
    }
    for label, r in arms.items():
        _write(tmp_path, label, r)

    result = calibrate_model(list(arms), tmp_path, _MODEL_SUPPORTED)

    gain_gate = result["fitness"]["power"]["gain_gate"]
    assert gain_gate.keys() == {"held_in", "held_out"}
    for split, row in gain_gate.items():
        for field in ("threshold", "power"):
            num, den = row[field].split("/")
            frac = Fraction(int(num), int(den))
            assert Fraction(0) <= frac <= Fraction(1), (split, field, frac)
        assert row["carrier"] in row["tasks"]
        assert set(row["per_carrier"]) == set(row["tasks"]), (
            "every task on the split can carry the improvement, so every one gets a row"
        )
        floor = min(Fraction(row["per_carrier"][t]["power"]) for t in row["tasks"])
        assert Fraction(row["power"]) == floor, (
            "the published power is the FLOOR across carriers -- a gate advertised at "
            "its luckiest carrier's power overstates what it can detect"
        )
        assert Fraction(row["per_carrier"][row["carrier"]]["power"]) == floor
    # Held-out's only task, G2, is trivially its own carrier. Held-in's three
    # carriers do NOT share a power: G4's near-floor pooled rate makes a +0.2
    # candidate easiest to see there and G5's near-ceiling rate makes it hardest,
    # so the split's headline number is G5's -- the weakest, not the alphabetically
    # first, which is what round 2 published and what this rider corrects.
    assert gain_gate["held_out"]["carrier"] == "G2"
    assert gain_gate["held_in"]["carrier"] == "G5"
    assert (
        Fraction(gain_gate["held_in"]["per_carrier"]["G5"]["power"])
        < Fraction(gain_gate["held_in"]["per_carrier"]["A1"]["power"])
        < Fraction(gain_gate["held_in"]["per_carrier"]["G4"]["power"])
    )
    # Pinned to the review's own reproduction of this exact fraction from
    # this exact pooled pool -- a regression guard on the actual number, not
    # just its shape. A1's per-carrier row is that same number, unchanged by
    # the rider: only which row is published as the split's headline moved.
    assert gain_gate["held_in"]["per_carrier"]["A1"]["power"] == (
        "778070204904630956396140097536/47352336533208097710339703242875"
    )
    assert gain_gate["held_in"]["power"] == (
        "738354301393030840236197035008/47352336533208097710339703242875"
    )
    assert gain_gate["held_in"]["threshold"] == "4/9"


def test_main_model_flag_writes_model_json(tmp_path, monkeypatch):
    import loop.calibrate as calibrate_mod

    supported = frozenset({"A1"})
    arm_full = _arm("full-a", {"A1": (6, 10, "held_in")})
    arm_a = _arm("cmp-a", {"A1": (5, 10, "held_in")}, filtered=True)
    _write(tmp_path, "full-a", arm_full)
    _write(tmp_path, "cmp-a", arm_a)

    out_path = tmp_path / "out" / "model-r2.json"
    monkeypatch.setattr(calibrate_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(calibrate_mod, "SUPPORTED", supported)
    monkeypatch.setattr(calibrate_mod, "MODEL_PATH", out_path)

    calibrate_mod.main(["--model", "full-a", "cmp-a"])

    written = json.loads(out_path.read_text())
    assert written["null_model"]["A1"]["null_rate"] == "11/20"
    assert written["coverage_level"] == "0.975"


def test_main_model_flag_requires_at_least_one_label():
    import loop.calibrate as calibrate_mod

    with pytest.raises(SystemExit):
        calibrate_mod.main(["--model"])


def test_main_without_model_flag_still_writes_round_one_analysis(tmp_path, monkeypatch):
    """`main()`'s existing (no-flag) behavior stays byte-identical -- adding
    `--model` must not change what `python -m loop.calibrate <labels>` does."""
    import loop.calibrate as calibrate_mod

    supported = frozenset({"A1"})
    arm_a = _arm("null-cmp-a", {"A1": (8, 10, "held_in")})
    arm_b = _arm("null-cmp-b", {"A1": (6, 10, "held_in")})
    _write(tmp_path, "null-cmp-a", arm_a)
    _write(tmp_path, "null-cmp-b", arm_b)

    out_path = tmp_path / "out" / "analysis.json"
    monkeypatch.setattr(calibrate_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(calibrate_mod, "SUPPORTED", supported)
    monkeypatch.setattr(calibrate_mod, "ANALYSIS_PATH", out_path)

    calibrate_mod.main(["null-cmp-a", "null-cmp-b"])

    written = json.loads(out_path.read_text())
    assert written["per_task"]["A1"]["max_abs_delta"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# `null_gain_quantile` / `null_task_quantile` -- the exact-enumeration
# primitives, exported for `loop.acceptance` (Task 3) to import directly.
# ---------------------------------------------------------------------------


def test_null_gain_quantile_worked_example():
    """A hand-checkable case: three held-in-shaped tasks, 3v3 attempts each,
    at rates {A1: 1/2, G4: 1/10, G5: 13/20} -- pinned in the task report."""
    rates = {"A1": Fraction(1, 2), "G4": Fraction(1, 10), "G5": Fraction(13, 20)}
    attempts = {"A1": 3, "G4": 3, "G5": 3}

    q = null_gain_quantile(rates, attempts, attempts, COVERAGE_LEVEL)

    assert q == Fraction(4, 9)


def test_null_gain_quantile_stays_within_plus_minus_one():
    """D = mean_b - mean_a is a difference of two means in [0, 1] -- it can
    never leave [-1, 1], whatever the rates or attempt counts. (Catches the
    exact bug this module shipped with once: dividing a pooled RAW pass-count
    difference by task count instead of averaging each task's own fraction
    first, which let a 3-task quantile land outside [-1, 1].)"""
    rates = {"A1": Fraction(1, 2), "G4": Fraction(1, 10), "G5": Fraction(13, 20)}
    attempts = {"A1": 3, "G4": 3, "G5": 3}

    q = null_gain_quantile(rates, attempts, attempts, COVERAGE_LEVEL)

    assert Fraction(-1) <= q <= Fraction(1)


def test_null_gain_quantile_requires_matching_task_keys():
    with pytest.raises(ValueError, match="attempts_a"):
        null_gain_quantile({"A1": Fraction(1, 2)}, {"A1": 3, "G4": 3}, {"A1": 3}, COVERAGE_LEVEL)


def test_null_gain_quantile_rejects_a_level_above_one():
    with pytest.raises(ValueError, match="level"):
        null_gain_quantile({"A1": Fraction(1, 2)}, {"A1": 3}, {"A1": 3}, Fraction(3, 2))


def test_null_task_quantile_matches_the_single_task_gain_quantile():
    """The carrier/guard construction (contract §2) is a specialization of
    the multi-task one, not a separately-derived formula."""
    rate = Fraction(21, 49)

    single = null_task_quantile(rate, 10, 10, COVERAGE_LEVEL)
    via_gain = null_gain_quantile({"A1": rate}, {"A1": 10}, {"A1": 10}, COVERAGE_LEVEL)

    assert single == via_gain


def test_null_gain_quantile_is_deterministic_and_exact_type():
    """No sampling: the same inputs always produce the exact same `Fraction`,
    not merely an approximately-equal float."""
    rates = {"A1": Fraction(21, 49), "G4": Fraction(5, 49)}
    attempts = {"A1": 3, "G4": 3}

    first = null_gain_quantile(rates, attempts, attempts, COVERAGE_LEVEL)
    second = null_gain_quantile(rates, attempts, attempts, COVERAGE_LEVEL)

    assert isinstance(first, Fraction)
    assert first == second


def test_null_gain_quantile_is_fast_at_three_tasks_ten_attempts():
    """Watch-runtime guard: three tasks at ten attempts each (the subset
    arms' own shape, contract §3) must enumerate well under 2 seconds --
    the size this module's own fitness checks and a real confirmation gate
    actually run at."""
    rates = {"A1": Fraction(21, 49), "G4": Fraction(5, 49), "G5": Fraction(30, 49)}
    attempts = {"A1": 10, "G4": 10, "G5": 10}

    start = time.perf_counter()
    null_gain_quantile(rates, attempts, attempts, COVERAGE_LEVEL)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0


# ---------------------------------------------------------------------------
# Exactness hardening (review, pre-Task-3 batch): a float `rate`/`level` must
# never reach the memoized `_binom_pmf` cache -- `hash(0.5) == hash(Fraction
# (1, 2))`, so one float call would otherwise poison every later EXACT call
# at the same (n, rate) for the rest of the process.
# ---------------------------------------------------------------------------


def test_null_gain_quantile_rejects_a_float_rate():
    with pytest.raises(ValueError, match="Fraction"):
        null_gain_quantile({"A1": 0.5}, {"A1": 3}, {"A1": 3}, COVERAGE_LEVEL)


def test_null_gain_quantile_rejects_a_float_level():
    with pytest.raises(ValueError, match="Fraction"):
        null_gain_quantile({"A1": Fraction(1, 2)}, {"A1": 3}, {"A1": 3}, 0.975)


def test_null_task_quantile_rejects_a_float_rate():
    with pytest.raises(ValueError, match="Fraction"):
        null_task_quantile(0.5, 3, 3, COVERAGE_LEVEL)


def test_null_task_quantile_rejects_a_float_level():
    with pytest.raises(ValueError, match="Fraction"):
        null_task_quantile(Fraction(1, 2), 3, 3, 0.975)


def test_null_gain_quantile_rejects_non_positive_attempts():
    with pytest.raises(ValueError, match="A1"):
        null_gain_quantile({"A1": Fraction(1, 2)}, {"A1": 0}, {"A1": 3}, COVERAGE_LEVEL)


def test_rejected_float_call_does_not_poison_the_binom_pmf_cache():
    """The cache-probe case from the review: a float call at (n, rate) that
    gets refused must not leave behind a cached float pmf that a later exact
    call at the SAME (n, rate) would silently reuse -- the rejection has to
    happen BEFORE anything is cached, not after."""
    import loop.calibrate as calibrate_mod

    with pytest.raises(ValueError):
        null_gain_quantile({"A1": 0.5}, {"A1": 3}, {"A1": 3}, COVERAGE_LEVEL)

    result = null_gain_quantile({"A1": Fraction(1, 2)}, {"A1": 3}, {"A1": 3}, COVERAGE_LEVEL)

    assert isinstance(result, Fraction)
    assert result == Fraction(2, 3)
    # And the underlying memoized pmf itself is genuinely exact, not a float
    # that happens to compare equal to a Fraction.
    pmf = calibrate_mod._binom_pmf(3, Fraction(1, 2))
    assert all(isinstance(p, Fraction) for p in pmf)


def test_binom_pmf_rejects_a_float_p_directly():
    import loop.calibrate as calibrate_mod

    with pytest.raises(TypeError, match="Fraction"):
        calibrate_mod._binom_pmf(3, 0.5)


def test_binom_pmf_rejects_a_float_p_even_after_the_exact_form_is_cached():
    """Self-contained (does not rely on test execution order): the exact form
    of a given (n, rate) is called and cached FIRST, inside this same test,
    then the numerically-equal float form is called -- it must still raise,
    proving the type check runs on every call and is not merely a cache-miss
    guard `@cache` skips on a hit (the exact bug this hardening closes:
    `hash(0.5) == hash(Fraction(1, 2))` makes them the same cache key)."""
    import loop.calibrate as calibrate_mod

    exact = calibrate_mod._binom_pmf(3, Fraction(1, 2))
    assert all(isinstance(p, Fraction) for p in exact)

    with pytest.raises(TypeError, match="Fraction"):
        calibrate_mod._binom_pmf(3, 0.5)

    # And the exact entry itself is undisturbed.
    assert calibrate_mod._binom_pmf(3, Fraction(1, 2)) == exact


# ---------------------------------------------------------------------------
# Cross-arm disagreement (review, pre-Task-3 batch): a task's split and
# standard attempt count are properties of the SUITE, not of one arm's run --
# two arms disagreeing must refuse, not silently resolve to whichever arm
# came first in the label list.
# ---------------------------------------------------------------------------


def test_cross_arm_split_disagreement_is_refused(tmp_path):
    arm_a = _arm("full-a", {"A1": (2, 3, "held_in")})
    arm_b = _arm("full-b", {"A1": (2, 3, "held_out")})
    _write(tmp_path, "full-a", arm_a)
    _write(tmp_path, "full-b", arm_b)
    with pytest.raises(ValueError, match="A1"):
        calibrate_model(["full-a", "full-b"], tmp_path, frozenset({"A1"}))


def test_cross_arm_standard_attempts_disagreement_is_refused(tmp_path):
    arm_a = _arm("full-a", {"A1": (2, 3, "held_in")})
    arm_b = _arm("full-b", {"A1": (3, 5, "held_in")})
    _write(tmp_path, "full-a", arm_a)
    _write(tmp_path, "full-b", arm_b)
    with pytest.raises(ValueError, match="A1"):
        calibrate_model(["full-a", "full-b"], tmp_path, frozenset({"A1"}))


def test_zero_standard_attempts_is_refused_naming_the_arm(tmp_path):
    fp = {"runner_sha": "rsha1", "config_version": 7, "model": "carbon-model", "dirty_sha": None}
    zero = {
        "fingerprint": fp,
        "tasks": {"A1": {"split": "held_in", "attempts": 0, "passes": 0, "pass_fraction": 0.0}},
    }
    _write(tmp_path, "full-a", zero)
    with pytest.raises(ValueError, match="full-a"):
        calibrate_model(["full-a"], tmp_path, frozenset({"A1"}))


def test_a_subset_arm_with_zero_attempts_is_refused_before_it_divides_by_zero(tmp_path):
    """The Task 2 review's other rider. `_standard_attempts` only ever inspects the
    UN-filtered arms, so a subset arm recording zero attempts for a supported task
    walks past it. It then reaches `_check_stability`, where a leave-one-arm-out pool
    whose remaining arms all recorded zero attempts builds `Fraction(0, 0)` -- a bare
    `ZeroDivisionError` naming no arm, no task, and nothing a reader could act on.

    Three zero-attempt subset arms plus one real full-suite arm is the smallest shape
    that reproduces it: leaving the full arm out empties the pool. The refusal must
    name the arm and the task instead.
    """
    fp = {"runner_sha": "rsha1", "config_version": 7, "model": "carbon-model", "dirty_sha": None}
    _write(tmp_path, "full-a", _arm("full-a", {"A1": (2, 3, "held_in")}, fingerprint=fp))
    for label in ("cmp-a", "cmp-b", "cmp-c"):
        _write(
            tmp_path,
            label,
            {
                "fingerprint": dict(fp),
                "filter": ["A1"],
                "tasks": {
                    "A1": {"split": "held_in", "attempts": 0, "passes": 0, "pass_fraction": 0.0}
                },
            },
        )
    labels = ["full-a", "cmp-a", "cmp-b", "cmp-c"]
    with pytest.raises(ValueError, match="cmp-a") as exc:
        calibrate_model(labels, tmp_path, frozenset({"A1"}))
    assert "A1" in str(exc.value)
    assert "0 attempts" in str(exc.value)
