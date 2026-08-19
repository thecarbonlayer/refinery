"""Validation wiring: apply -> run (seam) -> revert-always -> Δ -> acceptance."""

import json
import shutil
import subprocess

import pytest

from loop.artifacts import Candidate
from loop.config_edit import config_path
from loop.validate import require_clean_tree, validate_candidate
from runner.carbon_env import CARBON_ROOT

REAL_CONFIG = CARBON_ROOT / "harness" / "harness_config.json"


def _bumped() -> int:
    """The version an edit should produce: whatever the real config carries now,
    plus one. Hardcoding it pinned these tests to a config that has since been
    bumped in carbon, so they broke on a change that was not theirs."""
    return json.loads(REAL_CONFIG.read_text())["version"] + 1


def _live(field: str):
    """The value carbon's config carries right now.

    Hardcoding `old` pinned these tests to one config exactly as hardcoding
    `version` once did: a legal `max_tokens` candidate broke five tests that had
    nothing to do with it. `max_tokens` is a knob the loop exists to tune.
    """
    return json.loads(REAL_CONFIG.read_text())[field]


OLD_MT = _live("max_tokens")
NEW_MT = OLD_MT * 2  # legal (positive int, no ceiling) and always distinct


CANDIDATE = Candidate(
    id="cand-x",
    cluster_id="CL-1",
    proposer="Fable",
    proposer_detail="test",
    fields={"max_tokens": {"old": OLD_MT, "new": NEW_MT}},
    rationale="r",
    expected_effect="e",
    regression_risk="g",
)

FP = {
    "gemma_sha": "abc",
    "gemma_dirty": False,
    "dirty_sha": None,
    "config_version": 1,
    "model": "m",
    "runner_sha": "r1",
}


def results_json(fractions: dict[str, float], fingerprint=FP, summary=None) -> dict:
    tasks = {}
    for name, frac in fractions.items():
        split = "held_out" if name in {"A3", "A4"} else "held_in"
        n = 5 if split == "held_out" else 3
        tasks[name] = {"split": split, "attempts": n, "pass_fraction": frac}
    return {"fingerprint": dict(fingerprint), "tasks": tasks, "summary": summary or {}}


def _gates_pass(_carbon_root):
    """Harness gates stubbed green: these tests are about Δ and telemetry, and the
    real gate shells out to both repos' suites, which a fixture carbon cannot pass."""
    return {"passed": True, "checks": {}}


def test_validate_carries_every_telemetry_field_onto_the_record(fake_carbon, tmp_path):
    """`validate_candidate` is the only path from delta() to the committed record.
    Dropping a field here — or passing a literal `[]` — silently loses the evidence:
    nothing read these fields, so six such mutations left the suite green.
    """
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    base_summary = {
        "metrics": {"tokens": 1000.0, "cost": 0.42},
        "metric_task_counts": {"tokens": 3, "cost": 3},
        "metric_attempt_counts": {"tokens": 9, "cost": 9},
    }
    cand_summary = {
        "metrics": {"tokens": 670.0},  # cost not measured on this side
        "metric_task_counts": {"tokens": 3},
        "metric_attempt_counts": {"tokens": 3},  # same tasks, a third of the attempts
    }
    fractions = {"A2": 1.0, "A4": 0.8, "D1": 1.0}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(results_json(fractions, summary=base_summary)))

    def fake_runner(label, only, attempts):
        cand_fp = dict(FP, gemma_dirty=True, dirty_sha="d1", config_version=2)
        out = results_json(fractions, fingerprint=cand_fp, summary=cand_summary)
        (results_dir / f"{label}.json").write_text(json.dumps(out))

    record = validate_candidate(
        CANDIDATE,
        baseline_path=baseline,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        run_gates=_gates_pass,
        results_dir=results_dir,
        log=lambda *_: None,
    )
    assert record.metric_delta == {"tokens": -330.0}
    assert record.metric_not_compared == ["cost"]
    assert record.metric_task_counts["tokens"] == {"baseline": 3, "candidate": 3}
    assert record.metric_attempt_counts["tokens"] == {"baseline": 9, "candidate": 3}
    assert record.metric_denominator_drift == ["cost", "tokens"]


@pytest.fixture
def fake_carbon(tmp_path):
    root = tmp_path / "carbon"
    (root / "harness").mkdir(parents=True)
    shutil.copy(REAL_CONFIG, root / "harness" / "harness_config.json")
    for args in (
        ["init", "-b", "main"],
        ["config", "user.email", "t@t"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-m", "seed"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


@pytest.fixture
def baseline(tmp_path):
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(results_json({"A2": 0.0, "A4": 0.0, "D1": 1.0})))
    return p


def test_accepted_candidate_and_revert(fake_carbon, baseline, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    seen = {}

    def fake_runner(label, only, attempts):
        # the edit must be LIVE while the suite runs
        seen["config"] = json.loads(config_path(fake_carbon).read_text())
        cand_fp = dict(FP, gemma_dirty=True, dirty_sha="d1", config_version=2)
        out = results_json({"A2": 1.0, "A4": 0.8, "D1": 1.0}, fingerprint=cand_fp)
        (results_dir / f"{label}.json").write_text(json.dumps(out))

    record = validate_candidate(
        CANDIDATE,
        baseline_path=baseline,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        run_gates=_gates_pass,
        results_dir=results_dir,
        log=lambda *_: None,
    )
    assert seen["config"]["max_tokens"] == NEW_MT and seen["config"]["version"] == _bumped()
    # reverted afterwards, tree clean
    assert json.loads(config_path(fake_carbon).read_text())["max_tokens"] == OLD_MT
    require_clean_tree(fake_carbon)
    assert record.accepted and record.delta_in == 0.5 and record.delta_ho == pytest.approx(0.8)
    assert record.per_task["A2"] == 1.0


def test_regression_is_rejected(fake_carbon, baseline, tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    def fake_runner(label, only, attempts):
        cand_fp = dict(FP, gemma_dirty=True, dirty_sha="d1", config_version=2)
        out = results_json({"A2": 1.0, "A4": 0.0, "D1": 0.0}, fingerprint=cand_fp)
        (results_dir / f"{label}.json").write_text(json.dumps(out))

    record = validate_candidate(
        CANDIDATE,
        baseline_path=baseline,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        run_gates=_gates_pass,
        results_dir=results_dir,
        log=lambda *_: None,
    )
    assert not record.accepted  # D1 regressed: Δ_in mixes +1 and -1 -> 0? no: (1-0)+(0-1)=0
    assert record.delta_in == 0.0 and record.delta_ho == 0.0


def test_runner_crash_still_reverts(fake_carbon, baseline, tmp_path):
    def exploding_runner(label, only, attempts):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        validate_candidate(
            CANDIDATE,
            baseline_path=baseline,
            carbon_root=fake_carbon,
            run_runner=exploding_runner,
            run_gates=_gates_pass,
            results_dir=tmp_path,
            log=lambda *_: None,
        )
    assert json.loads(config_path(fake_carbon).read_text())["max_tokens"] == OLD_MT
    require_clean_tree(fake_carbon)


def test_dirty_tree_refused(fake_carbon, baseline, tmp_path):
    (fake_carbon / "stray.txt").write_text("x")
    with pytest.raises(RuntimeError, match="not clean"):
        validate_candidate(
            CANDIDATE,
            baseline_path=baseline,
            carbon_root=fake_carbon,
            run_runner=lambda *a: None,
            run_gates=_gates_pass,
            results_dir=tmp_path,
            log=lambda *_: None,
        )


def test_a_candidate_that_reddens_either_repo_is_vetoed_before_the_suite_runs(
    fake_carbon, tmp_path
):
    """The gate exists because the task suite cannot see a broken harness.

    A config value can turn carbon's or this repo's own tests red without moving a
    single task score — no task asserts on carbon's invariants or on these fixtures.
    Three such breakages reached a merged branch before the gate existed. So the veto
    must (a) reject, (b) record WHICH check failed rather than a bare False, and
    (c) fire before the suite, since forty minutes of model time on a harness that no
    longer holds together is the expensive half of the mistake.
    """
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(results_json({"A2": 1.0, "A4": 0.8, "D1": 1.0})))
    ran: list[str] = []

    def failing_gates(_carbon_root):
        return {
            "passed": False,
            "checks": {
                "carbon_verify": {"passed": True, "exit_code": 0},
                "refinery_pytest": {
                    "passed": False,
                    "exit_code": 1,
                    "tail": ["FAILED tests/test_registry.py::test_some_invariant"],
                },
            },
        }

    record = validate_candidate(
        CANDIDATE,
        baseline_path=baseline,
        carbon_root=fake_carbon,
        run_runner=lambda label, *_: ran.append(label),
        run_gates=failing_gates,
        results_dir=tmp_path / "results",
        log=lambda *_: None,
    )

    assert record.accepted is False
    assert ran == [], "the suite ran anyway — the veto is supposed to come first"
    assert record.gates["passed"] is False
    assert record.gates["checks"]["refinery_pytest"]["passed"] is False
    assert record.gates["checks"]["carbon_verify"]["passed"] is True


def test_a_passing_gate_is_recorded_too_not_just_a_failing_one(fake_carbon, tmp_path):
    """A record that only carries the veto when it fires cannot distinguish "checked
    and clean" from "never checked" — which is exactly the ambiguity that let three
    breakages through."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    fractions = {"A2": 1.0, "A4": 0.8, "D1": 1.0}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(results_json(fractions)))

    def fake_runner(label, only, attempts):
        (results_dir / f"{label}.json").write_text(json.dumps(results_json(fractions)))

    record = validate_candidate(
        CANDIDATE,
        baseline_path=baseline,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        run_gates=lambda _r: {
            "passed": True,
            "checks": {
                "carbon_verify": {"passed": True, "exit_code": 0},
                "refinery_pytest": {"passed": True, "exit_code": 0},
            },
        },
        results_dir=results_dir,
        log=lambda *_: None,
    )
    assert record.gates["passed"] is True
    assert set(record.gates["checks"]) == {"carbon_verify", "refinery_pytest"}


def test_the_real_gate_refuses_to_recurse_into_pytest():
    """It shells out to `pytest`; called from inside a test that is a nested full run."""
    from loop.validate import run_harness_gates

    with pytest.raises(RuntimeError, match="must not run inside pytest"):
        run_harness_gates()


# --- section calibration: the loader and the rule's section gate (contract §4) ----


def _analysis(
    tmp_path,
    *,
    stamped_sha,
    name="analysis.json",
    config_version=None,
    model=None,
    supported_extra=frozenset(),
):
    """A calibration artifact of contract §3's shape, MEASURED by `loop.calibrate`
    from four fabricated null arms — nothing here is a hand-written threshold.

    The arms' fingerprint is what freshness is judged against (contract §4, amendment
    2026-08-19): it defaults to the provenance `FP` carries, so an artifact built with
    no overrides is fresh for results stamped `FP`. Override any of the three fields to
    build one measured under a different verifier, config version, or model.
    """
    from loop.calibrate import SUPPORTED, calibrate

    results_dir = tmp_path / f"results-{name}"
    results_dir.mkdir()
    fp = {
        "runner_sha": stamped_sha,
        "config_version": FP["config_version"] if config_version is None else config_version,
        "model": FP["model"] if model is None else model,
    }
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
    path = tmp_path / name
    path.write_text(
        json.dumps(calibrate(sorted(arms), results_dir, SUPPORTED | frozenset(supported_extra)))
    )
    return path


# Supported set (A1/G4/G5 held-in, G2 held-out) plus ballast the bound was never
# measured on. X1 moves in the candidate and is the evidence-grade exclusion.
_CMP = {
    "A1": ("held_in", 10, 8),
    "G4": ("held_in", 10, 8),
    "G5": ("held_in", 10, 8),
    "G2": ("held_out", 20, 10),
    "X1": ("held_in", 20, 10),
    "X2": ("held_out", 4, 2),
}


def _cmp_results(moved=None):
    tasks = dict(_CMP)
    for name, passes in (moved or {}).items():
        split, attempts, _ = tasks[name]
        tasks[name] = (split, attempts, passes)
    return {
        "fingerprint": dict(FP),
        "tasks": {
            n: {
                "split": s,
                "attempts": a,
                "passes": p,
                "pass_fraction": round(p / a, 4),
                "outcomes": ["pass"] * p + ["fail"] * (a - p),
            }
            for n, (s, a, p) in tasks.items()
        },
    }


_COVERAGE = {"unreachable_proven": {}, "unreachable_probable": {"X1": -0.05}}


def _cmp_candidate(fields):
    return Candidate(
        id="cmp",
        cluster_id="CL-1",
        proposer="Fable",
        proposer_detail="test",
        fields={f: {"old": 1, "new": 2} for f in fields},
        rationale="r",
        expected_effect="e",
        regression_risk="g",
    )


def test_an_absent_calibration_artifact_leaves_the_section_uncalibrated(tmp_path):
    from loop.validate import section_calibration

    assert section_calibration("compaction", analysis_path=tmp_path / "nothing.json") is None


def test_a_calibration_measured_by_a_different_verifier_is_not_a_calibration(tmp_path):
    """Stale = uncalibrated. `runner_sha` is the verifier's content identity: a bound
    measured by a different verifier is not a measurement of this one."""
    from loop.validate import section_calibration

    stale = _analysis(tmp_path, stamped_sha="0" * 64)
    assert json.loads(stale.read_text())["computed_at_runner_sha"] == "0" * 64
    assert section_calibration("compaction", FP, analysis_path=stale) is None


def test_a_fresh_calibration_loads_exactly_what_the_artifact_measured(tmp_path):
    from loop.validate import _SECTION_CONFIRM_GUARDS, section_calibration

    path = _analysis(tmp_path, stamped_sha=FP["runner_sha"])
    cal = section_calibration("compaction", FP, analysis_path=path)
    analysis = json.loads(path.read_text())
    assert cal is not None
    assert cal.section == "compaction"
    assert cal.supported == frozenset(analysis["per_task"])
    assert cal.noise_in == analysis["section_noise"]["held_in"]
    assert cal.noise_ho == analysis["section_noise"]["held_out"]
    assert cal.guards == _SECTION_CONFIRM_GUARDS["compaction"] == frozenset({"A1", "G2", "G5"})


def test_compaction_enters_the_rule_only_through_a_fresh_calibration(tmp_path, monkeypatch):
    """The gate the whole phase turns on, in its three states."""
    import loop.validate as validate_mod
    from loop.validate import rule_disposition

    base, cand = _cmp_results(), _cmp_results({"G2": 14, "X1": 9})
    candidate = _cmp_candidate(["compaction"])

    monkeypatch.setitem(validate_mod._SECTION_ANALYSIS, "compaction", tmp_path / "absent.json")
    missing = rule_disposition(candidate, base, cand, _COVERAGE)
    assert missing["applied"] is False
    assert "not calibrated" in missing["why"]

    monkeypatch.setitem(
        validate_mod._SECTION_ANALYSIS,
        "compaction",
        _analysis(tmp_path, stamped_sha="0" * 64, name="stale.json"),
    )
    stale = rule_disposition(candidate, base, cand, _COVERAGE)
    assert stale["applied"] is False
    assert "not calibrated" in stale["why"] and "stale" in stale["why"].lower()

    monkeypatch.setitem(
        validate_mod._SECTION_ANALYSIS,
        "compaction",
        _analysis(tmp_path, stamped_sha=FP["runner_sha"], name="fresh.json"),
    )
    fresh = rule_disposition(candidate, base, cand, _COVERAGE)
    assert fresh["applied"] is True
    assert fresh["outcome"] == "CONFIRM"
    assert fresh["improved_tasks"] == ["G2"]
    assert fresh["calibration"]["section"] == "compaction"
    assert set(fresh["calibration"]["guards"]) <= set(fresh["confirm_tasks"])
    # The evidence-grade exclusion is context, never a subtraction.
    assert fresh["raw"]["unreachable_probable"] == ["X1"]
    assert fresh["raw"]["full_split_delta_in"] < 0


def test_compaction_prompt_reaches_the_rule_through_the_same_section(tmp_path, monkeypatch):
    import loop.validate as validate_mod
    from loop.validate import _FIELD_SECTION, rule_disposition

    assert _FIELD_SECTION["compaction_prompt"] == _FIELD_SECTION["compaction"] == "compaction"
    monkeypatch.setitem(
        validate_mod._SECTION_ANALYSIS,
        "compaction",
        _analysis(tmp_path, stamped_sha=FP["runner_sha"]),
    )
    out = rule_disposition(
        _cmp_candidate(["compaction_prompt"]), _cmp_results(), _cmp_results({"G2": 14}), _COVERAGE
    )
    assert out["applied"] is True and out["calibration"]["section"] == "compaction"


def test_an_edit_spanning_two_sections_gets_no_rule_even_when_both_are_calibrated(
    tmp_path, monkeypatch
):
    """Two sections' evidence in one measurement belongs to neither section."""
    import loop.validate as validate_mod
    from loop.validate import rule_disposition

    monkeypatch.setitem(
        validate_mod._SECTION_ANALYSIS,
        "compaction",
        _analysis(tmp_path, stamped_sha=FP["runner_sha"]),
    )
    out = rule_disposition(
        _cmp_candidate(["compaction", "tool_output"]),
        _cmp_results(),
        _cmp_results({"G2": 14}),
        _COVERAGE,
    )
    assert out["applied"] is False
    assert "compaction" in out["why"] and "tool_output" in out["why"]


def test_tool_output_decides_without_consulting_any_calibration_artifact(tmp_path, monkeypatch):
    """The cardinal constraint of this change, pinned: the calibrated regime must be
    unreachable from tool_output. No artifact exists for it, none is consulted, and
    its Decision carries none of the calibrated regime's fields."""
    import loop.validate as validate_mod
    from loop.validate import rule_disposition, section_calibration

    assert section_calibration("tool_output") is None
    monkeypatch.setitem(validate_mod._SECTION_ANALYSIS, "compaction", tmp_path / "absent.json")
    out = rule_disposition(
        _cmp_candidate(["tool_output"]), _cmp_results(), _cmp_results({"G2": 14}), _COVERAGE
    )
    assert out["applied"] is True
    assert "calibration" not in out
    assert set(out["raw"]) == {"delta_in", "delta_ho"}


def test_the_dead_max_item_chars_mapping_is_gone_and_says_why(tmp_path):
    """carbon locked `max_item_chars` out of the editable surface (config v3), so the
    field can never appear in a candidate — its `tool_output` mapping was dead code
    pointing at the one calibrated section. The removal keeps a comment: a mapping
    deleted with no reason recorded is a mapping someone restores."""
    from pathlib import Path

    import loop.validate as validate_mod
    from loop.validate import _FIELD_SECTION

    assert "max_item_chars" not in _FIELD_SECTION
    assert "max_item_chars" in Path(validate_mod.__file__).read_text(), (
        "the removal must leave a comment saying why"
    )


# --- freshness against the measurements' own provenance (§4 amendment) ------------


def test_freshness_is_judged_against_the_measurements_not_the_live_process(tmp_path):
    """Three ways an artifact can fail to be a bound for THESE measurements, and each
    one names the field that mismatched.

    `runner_sha` alone was the old test, and it is the weakest of the three: a noise
    floor measured under a different MODEL, or a different carbon config version, is
    not a floor for this comparison either — those are exactly the two things the null
    protocol (§2) makes every arm share, so an artifact whose arms disagree with the
    results being judged is measuring a different world.
    """
    from loop.validate import calibration_status

    fresh, why = calibration_status(
        "compaction", FP, analysis_path=_analysis(tmp_path, stamped_sha=FP["runner_sha"])
    )
    assert fresh is not None and why == ""

    for kwargs, field in (
        ({"stamped_sha": "0" * 64, "name": "sha.json"}, "runner_sha"),
        ({"stamped_sha": FP["runner_sha"], "name": "cfg.json", "config_version": 99}, "config_"),
        ({"stamped_sha": FP["runner_sha"], "name": "mdl.json", "model": "other"}, "model"),
    ):
        cal, reason = calibration_status(
            "compaction", FP, analysis_path=_analysis(tmp_path, **kwargs)
        )
        assert cal is None, f"{field} mismatch must not be judged fresh"
        assert field in reason and "not calibrated" in reason


def test_no_fingerprint_means_uncalibrated_never_unchecked(tmp_path):
    """Fail closed. A caller with no measurements in hand cannot be told a bound is
    fresh — there is nothing for it to be fresh FOR."""
    from loop.validate import calibration_status

    path = _analysis(tmp_path, stamped_sha=FP["runner_sha"])
    cal, why = calibration_status("compaction", None, analysis_path=path)
    assert cal is None and "fingerprint" in why


def test_rule_disposition_judges_freshness_against_the_baseline_it_was_handed(
    tmp_path, monkeypatch
):
    import loop.validate as validate_mod
    from loop.validate import rule_disposition

    monkeypatch.setitem(
        validate_mod._SECTION_ANALYSIS,
        "compaction",
        _analysis(tmp_path, stamped_sha=FP["runner_sha"]),
    )
    other = dict(FP, model="a-model-the-arms-never-ran")
    base = {**_cmp_results(), "fingerprint": other}
    cand = {**_cmp_results({"G2": 14}), "fingerprint": other}
    out = rule_disposition(_cmp_candidate(["compaction"]), base, cand, _COVERAGE)
    assert out["applied"] is False
    assert "model" in out["why"]


def test_a_supported_task_no_arm_measured_is_not_in_the_supported_set(tmp_path):
    """`loop.calibrate` writes a `per_task` entry for every task it was ASKED about,
    with an empty `passes_by_arm` when no arm carried it. Reading that as a supported
    task would put a name in the denominator that nothing was ever measured on — and
    then refuse every comparison for missing it."""
    from loop.validate import section_calibration

    path = _analysis(tmp_path, stamped_sha=FP["runner_sha"], supported_extra={"NEVER_RAN"})
    analysis = json.loads(path.read_text())
    assert analysis["per_task"]["NEVER_RAN"]["passes_by_arm"] == {}

    cal = section_calibration("compaction", FP, analysis_path=path)
    assert cal is not None
    assert "NEVER_RAN" not in cal.supported
    assert cal.supported == {"A1", "G2", "G4", "G5"}


# --- the confirm CLI passes the calibration through (contract §5 amendment) -------


def _calibrated_first_record(it_dir, candidate_id, cal):
    """A validation record whose rule verdict is a CALIBRATED first CONFIRM, exactly
    as `rule_disposition` writes one — this is what `loop.cli confirm` reads back."""
    from loop.acceptance import evaluate

    decision = evaluate(
        _cmp_results(), _cmp_results({"G2": 14}), calibration=cal, always_confirm=cal.guards
    )
    assert decision.outcome == "CONFIRM" and decision.raw["regime"] == "section_calibration"
    (it_dir / f"validation-{candidate_id}.json").write_text(
        json.dumps({"candidate_id": candidate_id, "rule": {"applied": True, **decision.to_json()}})
    )
    return decision


def test_confirm_cli_judges_a_calibrated_claim_against_its_measured_bound(
    tmp_path, monkeypatch, fake_carbon
):
    """Contract §5 amendment. Without the wiring the CLI hands `confirmed()` no
    calibration, which now REFUSES a calibrated first decision — so the same run that
    would silently have been judged against the weaker one-attempt bar instead fails
    loudly with nothing written."""
    import loop.cli as cli_mod
    import loop.validate as validate_mod
    from loop.cli import run_confirmation

    artifact = _analysis(tmp_path, stamped_sha=FP["runner_sha"])
    cal = validate_mod.section_calibration("compaction", FP, analysis_path=artifact)
    it_dir = tmp_path / "iter-cmp"
    it_dir.mkdir()
    candidate = _cmp_candidate(["compaction"])
    _calibrated_first_record(it_dir, candidate.id, cal)

    monkeypatch.setattr(cli_mod, "apply_candidate", lambda *a, **k: {"version": 9})
    monkeypatch.setattr(cli_mod, "require_clean_tree", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "revert_config", lambda *a, **k: None)

    def arms(gain: int):
        def run(label, only, attempts):
            passes = {"A1": 40, "G4": 40, "G5": 40, "G2": 25 if label == "b" else gain}
            return {
                "fingerprint": dict(FP),
                "tasks": {
                    n: {
                        "split": "held_out" if n == "G2" else "held_in",
                        "attempts": 50,
                        "passes": passes[n],
                        "pass_fraction": round(passes[n] / 50, 4),
                        "outcomes": ["pass"] * passes[n] + ["fail"] * (50 - passes[n]),
                    }
                    for n in only
                },
            }

        return run

    # Unwired (no artifact reachable): the calibrated claim cannot be judged at all.
    monkeypatch.setitem(validate_mod._SECTION_ANALYSIS, "compaction", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="confirmation could not be measured"):
        run_confirmation(
            candidate,
            it_dir,
            "b",
            "c",
            50,
            carbon_root=fake_carbon,
            run_runner=arms(34),
            log=lambda *_: None,
        )
    assert not list(it_dir.glob("confirmation-*.json"))

    # Wired: the repeat is judged against the MEASURED bound, not a one-attempt grain.
    monkeypatch.setitem(validate_mod._SECTION_ANALYSIS, "compaction", artifact)
    weak = run_confirmation(
        candidate,
        it_dir,
        "b",
        "c",
        50,
        carbon_root=fake_carbon,
        run_runner=arms(28),
        log=lambda *_: None,
    )
    assert 0 < 28 / 50 - 25 / 50 < cal.noise_ho
    assert weak.confirmation["outcome"] == "REJECT", "0.06 is inside the measured noise"

    strong = run_confirmation(
        candidate,
        it_dir,
        "b",
        "c",
        50,
        carbon_root=fake_carbon,
        run_runner=arms(34),
        log=lambda *_: None,
    )
    assert strong.confirmation["outcome"] == "ACCEPT"
    assert strong.confirmation["raw"]["regime"] == "section_calibration"
