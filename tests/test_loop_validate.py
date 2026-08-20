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


# --- the null-model loader and the rule's section gate (contract §1/§2) -----------


_SPLIT_OF = {"A1": "held_in", "G4": "held_in", "G5": "held_in", "G2": "held_out"}
_STANDARD_ATTEMPTS = {t: (5 if s == "held_out" else 3) for t, s in _SPLIT_OF.items()}

# Contract §3's arm protocol: three full-suite arms at standard attempts, four
# `--only A1 G2 G4 G5 --attempts 10` subset arms, nothing changed between any of them.
_FULL_ARMS = {
    "r2-null-full-a": {"A1": 2, "G4": 1, "G5": 2, "G2": 3},
    "r2-null-full-b": {"A1": 2, "G4": 1, "G5": 2, "G2": 2},
    "r2-null-full-c": {"A1": 1, "G4": 0, "G5": 3, "G2": 3},
}
_SUBSET_ARMS = {
    "r2-null-cmp-a": {"A1": 5, "G4": 2, "G5": 6, "G2": 5},
    "r2-null-cmp-b": {"A1": 6, "G4": 1, "G5": 7, "G2": 6},
    "r2-null-cmp-c": {"A1": 5, "G4": 2, "G5": 6, "G2": 4},
    "r2-null-cmp-d": {"A1": 6, "G4": 1, "G5": 5, "G2": 7},
}


def _model(
    tmp_path,
    *,
    stamped_sha,
    name="model-r2.json",
    config_version=None,
    model=None,
    carbon_sha=None,
    dirty_sha=None,
    force_passes=None,
    saturate=frozenset(),
    mutate=None,
):
    """A round-2 null-model artifact, MEASURED by `loop.calibrate.calibrate_model`
    from the seven fabricated null arms above — nothing here is a hand-written rate.

    The arms' provenance is what freshness is judged against (contract §1): it defaults
    to what `FP` carries, so an artifact built with no overrides is fresh for results
    stamped `FP`. Override any field to build one measured in a different world;
    `mutate` edits the computed artifact before it is written, which is how the unfit
    and malformed cases are built without hand-authoring an artifact.
    """
    from loop.calibrate import SUPPORTED, calibrate_model

    results_dir = tmp_path / f"arms-{name}"
    results_dir.mkdir()
    fp = {
        "runner_sha": stamped_sha,
        "config_version": FP["config_version"] if config_version is None else config_version,
        "model": FP["model"] if model is None else model,
        "gemma_sha": FP["gemma_sha"] if carbon_sha is None else carbon_sha,
        "dirty_sha": FP["dirty_sha"] if dirty_sha is None else dirty_sha,
    }
    for arms, subset in ((_FULL_ARMS, False), (_SUBSET_ARMS, True)):
        for label, passes in arms.items():
            tasks = {}
            for task, p in {**passes, **(force_passes or {})}.items():
                attempts = 10 if subset else _STANDARD_ATTEMPTS[task]
                # `saturate` means "passed every attempt in this arm", which depends on
                # the arm's own attempt count -- a literal pass count would be illegal
                # (passes > attempts) on the 3-attempt full-suite arms and would only
                # ever exercise `calibrate_model` on data the runner cannot produce.
                if task in saturate:
                    p = attempts
                tasks[task] = {
                    "split": _SPLIT_OF[task],
                    "attempts": attempts,
                    "passes": p,
                    "pass_fraction": round(p / attempts, 4),
                }
            arm = {"fingerprint": dict(fp), "tasks": tasks}
            if subset:
                arm["filter"] = sorted(passes)
            (results_dir / f"{label}.json").write_text(json.dumps(arm))
    labels = sorted({**_FULL_ARMS, **_SUBSET_ARMS})
    artifact = calibrate_model(labels, results_dir, SUPPORTED)
    if mutate is not None:
        mutate(artifact)
    path = tmp_path / name
    path.write_text(json.dumps(artifact))
    return path


# The supported set (A1/G4/G5 held-in, G2 held-out) at confirmation-shaped attempt
# counts, plus ballast the null model was never measured on. X1 moves in the candidate
# and is the evidence-grade exclusion.
_CMP = {
    "A1": ("held_in", 50, 27),
    "G4": ("held_in", 50, 8),
    "G5": ("held_in", 50, 31),
    "G2": ("held_out", 50, 27),
    "X1": ("held_in", 3, 2),
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

    assert section_calibration("compaction", model_path=tmp_path / "nothing.json") is None


def test_a_calibration_measured_by_a_different_verifier_is_not_a_calibration(tmp_path):
    """Stale = uncalibrated. `runner_sha` is the verifier's content identity: a null
    model measured by a different verifier is not a model of this one."""
    from loop.validate import section_calibration

    stale = _model(tmp_path, stamped_sha="0" * 64)
    assert json.loads(stale.read_text())["computed_at_runner_sha"] == "0" * 64
    assert section_calibration("compaction", FP, model_path=stale) is None


def test_a_fresh_calibration_loads_exactly_the_null_model_the_artifact_measured(tmp_path):
    from fractions import Fraction

    from loop.validate import _SECTION_CONFIRM_GUARDS, section_calibration

    path = _model(tmp_path, stamped_sha=FP["runner_sha"])
    cal = section_calibration("compaction", FP, model_path=path)
    artifact = json.loads(path.read_text())
    assert cal is not None
    assert cal.section == "compaction"
    assert cal.supported == frozenset(artifact["null_model"])
    assert cal.null_rates == {
        task: Fraction(*(int(x) for x in row["null_rate"].split("/")))
        for task, row in artifact["null_model"].items()
    }
    assert cal.coverage_level == Fraction(artifact["coverage_level"])
    assert cal.guards == _SECTION_CONFIRM_GUARDS["compaction"] == frozenset({"A1", "G2", "G5"})


def test_an_unfit_artifact_refuses_to_install_itself_and_says_which_check_failed(tmp_path):
    """Contract §4: "ALL must pass or the artifact is written with `fit: false` and the
    loader refuses it". The refusal must name the failed check — an artifact that is
    merely "not calibrated" tells nobody whether to re-run the arms or fix the tool."""
    from loop.validate import calibration_status

    def fail_grain(artifact):
        artifact["fitness"]["grain"]["pass"] = False
        artifact["fitness"]["fit"] = False

    path = _model(tmp_path, stamped_sha=FP["runner_sha"], mutate=fail_grain)
    cal, why = calibration_status("compaction", FP, model_path=path)
    assert cal is None
    assert "not calibrated" in why and "grain" in why
    assert "goodness" not in why, "only the checks that actually failed get named"


def test_a_missing_provenance_value_is_a_mismatch_never_a_match(tmp_path):
    """The None==None hole, closed. `arm.get(field) != fingerprint.get(field)` calls an
    artifact with NO recorded model fresh for measurements with no recorded model —
    two absences comparing equal and passing a freshness check neither side could
    answer. Absence on either side is a refusal."""
    from loop.validate import calibration_status

    def blank_model(artifact):
        for entry in artifact["provenance"]:
            entry["model"] = None

    path = _model(tmp_path, stamped_sha=FP["runner_sha"], mutate=blank_model)
    blind_fp = {k: v for k, v in FP.items() if k != "model"}
    cal, why = calibration_status("compaction", blind_fp, model_path=path)
    assert cal is None, "two absences must not be read as a match"
    assert "model" in why


def test_freshness_is_judged_against_the_measurements_on_every_provenance_field(tmp_path):
    """Five ways an artifact can fail to be a null model for THESE measurements, each
    naming the field that mismatched.

    `runner_sha` alone was the old test and is the weakest of the five: a null model
    measured under a different MODEL, a different carbon revision, a different config
    version, or against a dirty tree is not a model for this comparison either — those
    are exactly the fields the protocol (§1) makes every arm share.
    """
    from loop.validate import calibration_status

    fresh, why = calibration_status(
        "compaction", FP, model_path=_model(tmp_path, stamped_sha=FP["runner_sha"])
    )
    assert fresh is not None and why == ""

    fine = {"stamped_sha": FP["runner_sha"]}
    for kwargs, field in (
        ({"stamped_sha": "0" * 64, "name": "sha.json"}, "runner_sha"),
        ({**fine, "name": "cfg.json", "config_version": 99}, "config_version"),
        ({**fine, "name": "mdl.json", "model": "other"}, "model"),
        ({**fine, "name": "carbon.json", "carbon_sha": "another-carbon"}, "carbon_sha"),
        ({**fine, "name": "dirty.json", "dirty_sha": "uncommitted"}, "dirty_sha"),
    ):
        cal, reason = calibration_status("compaction", FP, model_path=_model(tmp_path, **kwargs))
        assert cal is None, f"{field} mismatch must not be judged fresh"
        assert field in reason and "not calibrated" in reason


def test_the_loader_refuses_a_supported_set_other_than_the_pinned_four(tmp_path):
    """Contract §1 pins {A1, G2, G4, G5} and says the loader refuses any other set. A
    model covering a different set is a model of a different section's evidence, and
    silently judging against it would put an unmeasured name in the denominator."""
    from loop.validate import calibration_status

    def drop_g4(artifact):
        del artifact["null_model"]["G4"]

    path = _model(tmp_path, stamped_sha=FP["runner_sha"], mutate=drop_g4)
    cal, why = calibration_status("compaction", FP, model_path=path)
    assert cal is None
    assert "G4" in why and "not calibrated" in why


def test_a_malformed_null_rate_refuses_instead_of_falling_back(tmp_path):
    """Round 1's loader fell back to a lossy float when an exact field was malformed.
    There is nothing to fall back TO here: a rate that cannot be parsed exactly has no
    null distribution, so the artifact is refused rather than approximated."""
    from loop.validate import calibration_status

    def corrupt(artifact):
        artifact["null_model"]["A1"]["null_rate"] = "27 over 49"

    path = _model(tmp_path, stamped_sha=FP["runner_sha"], mutate=corrupt)
    cal, why = calibration_status("compaction", FP, model_path=path)
    assert cal is None
    assert "A1" in why and "null_rate" in why


def test_compaction_enters_the_rule_only_through_a_fresh_fit_calibration(tmp_path, monkeypatch):
    """ATTACK CASE (f): the gate the whole phase turns on, in its four states. Every
    failure falls back to the CAUSAL verdict, loudly — `applied: False` with a reason
    that names the cause, never a silent re-decision on a weaker bar."""
    import loop.validate as validate_mod
    from loop.validate import rule_disposition

    base, cand = _cmp_results(), _cmp_results({"G2": 38, "X1": 1})
    candidate = _cmp_candidate(["compaction"])

    monkeypatch.setitem(validate_mod._SECTION_MODEL, "compaction", tmp_path / "absent.json")
    missing = rule_disposition(candidate, base, cand, _COVERAGE)
    assert missing["applied"] is False
    assert "not calibrated" in missing["why"]

    monkeypatch.setitem(
        validate_mod._SECTION_MODEL,
        "compaction",
        _model(tmp_path, stamped_sha="0" * 64, name="stale.json"),
    )
    stale = rule_disposition(candidate, base, cand, _COVERAGE)
    assert stale["applied"] is False
    assert "not calibrated" in stale["why"] and "stale" in stale["why"].lower()

    def unfit(artifact):
        artifact["fitness"]["stability"]["pass"] = False
        artifact["fitness"]["fit"] = False

    monkeypatch.setitem(
        validate_mod._SECTION_MODEL,
        "compaction",
        _model(tmp_path, stamped_sha=FP["runner_sha"], name="unfit.json", mutate=unfit),
    )
    refused = rule_disposition(candidate, base, cand, _COVERAGE)
    assert refused["applied"] is False
    assert "stability" in refused["why"]

    monkeypatch.setitem(
        validate_mod._SECTION_MODEL,
        "compaction",
        _model(tmp_path, stamped_sha=FP["runner_sha"], name="fresh.json"),
    )
    fresh = rule_disposition(candidate, base, cand, _COVERAGE)
    assert fresh["applied"] is True
    assert fresh["outcome"] == "CONFIRM"
    assert fresh["improved_tasks"] == ["G2"]
    assert fresh["calibration"]["section"] == "compaction"
    assert fresh["calibration"]["null_rates"]["A1"] == "27/49"
    assert fresh["calibration"]["coverage_level"] == "39/40"
    assert set(fresh["calibration"]["guards"]) <= set(fresh["confirm_tasks"])
    assert fresh["raw"]["null_quantiles"]["held_out"]["quantile"] == "1/5"
    # The evidence-grade exclusion is context, never a subtraction.
    assert fresh["raw"]["unreachable_probable"] == ["X1"]
    assert fresh["raw"]["full_split_delta_in"] < 0


def test_compaction_prompt_reaches_the_rule_through_the_same_section(tmp_path, monkeypatch):
    import loop.validate as validate_mod
    from loop.validate import _FIELD_SECTION, rule_disposition

    assert _FIELD_SECTION["compaction_prompt"] == _FIELD_SECTION["compaction"] == "compaction"
    monkeypatch.setitem(
        validate_mod._SECTION_MODEL,
        "compaction",
        _model(tmp_path, stamped_sha=FP["runner_sha"]),
    )
    out = rule_disposition(
        _cmp_candidate(["compaction_prompt"]), _cmp_results(), _cmp_results({"G2": 38}), _COVERAGE
    )
    assert out["applied"] is True and out["calibration"]["section"] == "compaction"


def test_an_edit_spanning_two_sections_gets_no_rule_even_when_both_are_calibrated(
    tmp_path, monkeypatch
):
    """Two sections' evidence in one measurement belongs to neither section."""
    import loop.validate as validate_mod
    from loop.validate import rule_disposition

    monkeypatch.setitem(
        validate_mod._SECTION_MODEL,
        "compaction",
        _model(tmp_path, stamped_sha=FP["runner_sha"]),
    )
    out = rule_disposition(
        _cmp_candidate(["compaction", "tool_output"]),
        _cmp_results(),
        _cmp_results({"G2": 38}),
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
    monkeypatch.setitem(validate_mod._SECTION_MODEL, "compaction", tmp_path / "absent.json")
    out = rule_disposition(
        _cmp_candidate(["tool_output"]), _cmp_results(), _cmp_results({"G2": 38}), _COVERAGE
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


def test_module_docstring_names_the_two_actual_decision_paths():
    """The docstring used to describe the pre-three-outcome world (a single
    Self-Harness rule). It must now name both paths a candidate can actually take:
    the calibrated rule for a section in `RULE_SECTIONS`, and the causal verdict
    everywhere else."""
    import loop.validate as validate_mod

    doc = validate_mod.__doc__
    assert "causal verdict" in doc.lower()
    assert "calibrated" in doc.lower() and "rule" in doc.lower()
    assert "CONFIRM" in doc and "REJECT" in doc


def test_an_unreadable_calibration_artifact_scrubs_the_machine_path(tmp_path):
    """`str(OSError)` embeds the absolute path it failed on (e.g. `[Errno 21] Is a
    directory: '/private/var/.../model-r2.json'`) — a machine path landing in a
    recorded reason is exactly the leak AGENTS.md calls out, and this reason reaches
    `rule_disposition()`'s `why` and from there a committed iteration record. A
    directory in place of the file reproduces an unreadable path portably (no
    permission-bit dance, no root-bypasses-chmod surprise)."""
    import os

    from loop.validate import calibration_status

    if os.getuid() == 0:
        pytest.skip("root bypasses the permission bits this fixture relies on")
    bogus = tmp_path / "some" / "nested" / "model-r2.json"
    bogus.parent.mkdir(parents=True)
    bogus.write_text("{}")
    bogus.chmod(0o000)
    try:
        cal, why = calibration_status("compaction", FP, model_path=bogus)
    finally:
        bogus.chmod(0o644)  # tmp_path cleanup needs to be able to remove it

    assert cal is None
    assert "unreadable" in why
    assert "model-r2.json" in why
    assert str(tmp_path) not in why
    for leak in ("/Users/", "/home/", "/private/var", "/tmp"):
        assert leak not in why, f"{leak!r} leaked into the recorded reason: {why!r}"


def test_no_fingerprint_means_uncalibrated_never_unchecked(tmp_path):
    """Fail closed. A caller with no measurements in hand cannot be told a null model
    is fresh — there is nothing for it to be fresh FOR."""
    from loop.validate import calibration_status

    path = _model(tmp_path, stamped_sha=FP["runner_sha"])
    cal, why = calibration_status("compaction", None, model_path=path)
    assert cal is None and "fingerprint" in why


def test_rule_disposition_judges_freshness_against_the_baseline_it_was_handed(
    tmp_path, monkeypatch
):
    import loop.validate as validate_mod
    from loop.validate import rule_disposition

    monkeypatch.setitem(
        validate_mod._SECTION_MODEL,
        "compaction",
        _model(tmp_path, stamped_sha=FP["runner_sha"]),
    )
    other = dict(FP, model="a-model-the-arms-never-ran")
    base = {**_cmp_results(), "fingerprint": other}
    cand = {**_cmp_results({"G2": 38}), "fingerprint": other}
    out = rule_disposition(_cmp_candidate(["compaction"]), base, cand, _COVERAGE)
    assert out["applied"] is False
    assert "model" in out["why"]


# --- the confirm CLI passes the calibration through (contract §5) -----------------


def _calibrated_first_record(it_dir, candidate_id, cal):
    """A validation record whose rule verdict is a CALIBRATED first CONFIRM, exactly
    as `rule_disposition` writes one — this is what `loop.cli confirm` reads back."""
    from loop.acceptance import evaluate

    decision = evaluate(
        _cmp_results(), _cmp_results({"G2": 38}), calibration=cal, always_confirm=cal.guards
    )
    assert decision.outcome == "CONFIRM" and decision.raw["regime"] == "section_calibration"
    (it_dir / f"validation-{candidate_id}.json").write_text(
        json.dumps({"candidate_id": candidate_id, "rule": {"applied": True, **decision.to_json()}})
    )
    return decision


def test_confirm_cli_judges_a_calibrated_claim_against_its_computed_quantile(
    tmp_path, monkeypatch, fake_carbon
):
    """Contract §5. Without the wiring the CLI hands `confirmed()` no calibration,
    which REFUSES a calibrated first decision — so the same run that would silently
    have been judged against the weaker one-attempt bar instead fails loudly with
    nothing written."""
    import loop.cli as cli_mod
    import loop.validate as validate_mod
    from loop.cli import run_confirmation

    artifact = _model(tmp_path, stamped_sha=FP["runner_sha"])
    cal = validate_mod.section_calibration("compaction", FP, model_path=artifact)
    it_dir = tmp_path / "iter-cmp"
    it_dir.mkdir()
    candidate = _cmp_candidate(["compaction"])
    _calibrated_first_record(it_dir, candidate.id, cal)

    monkeypatch.setattr(cli_mod, "apply_candidate", lambda *a, **k: {"version": 9})
    monkeypatch.setattr(cli_mod, "require_clean_tree", lambda *a, **k: None)
    monkeypatch.setattr(cli_mod, "revert_config", lambda *a, **k: None)

    def arms(gain: int):
        def run(label, only, attempts):
            passes = {"A1": 27, "G4": 8, "G5": 31, "G2": 27 if label == "b" else gain}
            return {
                "fingerprint": dict(FP),
                "tasks": {
                    n: {
                        "split": _SPLIT_OF[n],
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
    monkeypatch.setitem(validate_mod._SECTION_MODEL, "compaction", tmp_path / "absent.json")
    with pytest.raises(SystemExit, match="confirmation could not be measured"):
        run_confirmation(
            candidate,
            it_dir,
            "b",
            "c",
            50,
            carbon_root=fake_carbon,
            run_runner=arms(38),
            log=lambda *_: None,
        )
    # `run_confirmation` itself never writes a file on ANY path, success or refusal
    # alike — only `main()`'s `confirm` branch does, after a successful return. A
    # glob check here would be true unconditionally and pin nothing; the property
    # that matters (main()'s writer only fires when run_confirmation actually
    # returns) is exercised directly in
    # `test_main_confirm_writes_the_record_only_when_run_confirmation_succeeds`.

    # Wired: the repeat is judged against the quantile computed at 50v50, not a grain.
    monkeypatch.setitem(validate_mod._SECTION_MODEL, "compaction", artifact)
    weak = run_confirmation(
        candidate,
        it_dir,
        "b",
        "c",
        50,
        carbon_root=fake_carbon,
        run_runner=arms(30),
        log=lambda *_: None,
    )
    assert weak.confirmation["outcome"] == "REJECT", "+0.06 is inside G2's own null band"

    strong = run_confirmation(
        candidate,
        it_dir,
        "b",
        "c",
        50,
        carbon_root=fake_carbon,
        run_runner=arms(38),
        log=lambda *_: None,
    )
    assert strong.confirmation["outcome"] == "ACCEPT"
    assert strong.confirmation["raw"]["regime"] == "section_calibration"
    assert strong.confirmation["raw"]["carrier_quantiles"]["G2"]["quantile"] == "1/5"


def test_main_confirm_writes_the_record_only_when_run_confirmation_succeeds(tmp_path, monkeypatch):
    """The actual no-artifact-on-refusal property, exercised for real this time.

    `run_confirmation` never writes anything itself (see the comment two tests up) —
    the ONE write site is `main()`'s `confirm` branch, which calls
    `write_confirmation_record` only after `run_confirmation` returns. Driving
    `main()` with `run_confirmation` faked (the module-level name `main()` calls by,
    looked up at call time — unlike a default-argument seam, monkeypatching it here
    genuinely takes effect) pins the wiring itself: a `SystemExit` from
    `run_confirmation` must leave the write call unreached, and a normal return must
    reach it.
    """
    import sys

    import loop.cli as cli_mod
    from loop.artifacts import ConfirmationRecord

    it_dir = tmp_path / "iter-01"
    it_dir.mkdir()
    (it_dir / "candidates.json").write_text(
        json.dumps(
            [
                {
                    "id": "cand-x",
                    "cluster_id": "CL-1",
                    "proposer": "Fable",
                    "proposer_detail": "test",
                    "fields": {"max_tokens": {"old": 1, "new": 2}},
                    "rationale": "r",
                    "expected_effect": "e",
                    "regression_risk": "g",
                }
            ]
        )
    )
    monkeypatch.setattr(cli_mod, "ITERATIONS_DIR", tmp_path)
    argv = [
        "loop",
        "confirm",
        "--iteration",
        "iter-01",
        "--candidate",
        "cand-x",
        "--baseline-label",
        "b",
        "--candidate-label",
        "c",
        "--attempts",
        "10",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    def refuse(*a, **k):
        raise SystemExit("confirmation could not be measured: boom")

    monkeypatch.setattr(cli_mod, "run_confirmation", refuse)
    with pytest.raises(SystemExit):
        cli_mod.main()
    assert list(it_dir.glob("confirmation-*.json")) == []

    def succeed(*a, **k):
        return ConfirmationRecord(
            candidate_id="cand-x",
            baseline_label="b",
            candidate_label="c",
            attempts_per_task_per_arm=10,
            confirm_set=("E4",),
            first_decision={},
            confirmation={"outcome": "ACCEPT"},
            per_task={},
            finding="f",
        )

    monkeypatch.setattr(cli_mod, "run_confirmation", succeed)
    cli_mod.main()
    written = list(it_dir.glob("confirmation-*.json"))
    assert len(written) == 1


# --- what must not install: degenerate rates and unchecked per-arm provenance -----


def test_a_degenerate_pooled_rate_produces_a_fit_artifact_the_loader_refuses(tmp_path):
    """Contract §4.1 amendment, end to end and honestly.

    G4 never passes in ANY arm, so its pooled rate is 0/49. Every fitness check still
    passes — grain reads the split MEAN (A1 and G5 still vary), goodness sees each arm
    agreeing perfectly with a rate of zero, and stability sees nothing move — so
    `calibrate_model` writes `fit: true`. Nothing at the artifact level catches it, and
    G4's per-task quantile is 0 at every attempt count: the repeat and guard gates on
    that task could not be failed.

    The refusal therefore has to live at the load boundary, and it does. The situation
    a human then has to look at is real: a task that never passes across seven arms is
    not calibrated evidence, it is a broken or impossible task.
    """
    from fractions import Fraction

    from loop.calibrate import COVERAGE_LEVEL, null_task_quantile
    from loop.validate import calibration_status

    path = _model(tmp_path, stamped_sha=FP["runner_sha"], force_passes={"G4": 0})
    artifact = json.loads(path.read_text())
    assert artifact["fitness"]["fit"] is True, (
        "the artifact certifies itself — this is exactly why the loader must not trust "
        "fitness alone"
    )
    assert artifact["null_model"]["G4"]["null_rate"] == "0/49"
    assert null_task_quantile(Fraction(0, 49), 10, 10, COVERAGE_LEVEL) == 0

    cal, why = calibration_status("compaction", FP, model_path=path)
    assert cal is None
    assert "G4" in why and "0/49" in why
    assert "0 < rate < 1" in why


def test_a_pooled_rate_of_one_is_refused_the_same_way(tmp_path):
    """The mirror case: a task that never FAILS pins the same zero quantile."""
    from loop.validate import calibration_status

    path = _model(
        tmp_path,
        stamped_sha=FP["runner_sha"],
        name="allpass.json",
        saturate={"G4"},
    )
    artifact = json.loads(path.read_text())
    assert artifact["null_model"]["G4"]["null_rate"] == "49/49", (
        "fixture precondition: G4 passed every attempt of every arm"
    )
    cal, why = calibration_status("compaction", FP, model_path=path)
    assert cal is None and "G4" in why and "0 < rate < 1" in why


def test_a_single_arm_disagreeing_on_runner_sha_is_stale(tmp_path):
    """`computed_at_runner_sha` is ONE arm's value (the first), and the artifact records
    every arm's own provenance beside it. Checking only the summary field leaves the
    other six arms unchecked against the measurements — an artifact pooled across two
    verifier versions would pass, which is precisely the mix the null protocol exists to
    forbid. `runner_sha` belongs in the per-arm loop with the rest."""
    from loop.validate import calibration_status

    def drift_one_arm(artifact):
        assert artifact["provenance"][0]["runner_sha"] == FP["runner_sha"]
        artifact["provenance"][-1]["runner_sha"] = "0" * 64

    path = _model(tmp_path, stamped_sha=FP["runner_sha"], mutate=drift_one_arm)
    artifact = json.loads(path.read_text())
    assert artifact["computed_at_runner_sha"] == FP["runner_sha"], (
        "the summary field still matches — only a per-arm row disagrees"
    )
    cal, why = calibration_status("compaction", FP, model_path=path)
    assert cal is None
    assert "runner_sha" in why and "STALE" in why
