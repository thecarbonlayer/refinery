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
from pathlib import Path

import pytest

from loop.acceptance import CONFIRM, REJECT
from loop.calibrate import calibrate


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
