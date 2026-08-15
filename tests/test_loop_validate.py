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
