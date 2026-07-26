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


CANDIDATE = Candidate(
    id="cand-x",
    cluster_id="CL-1",
    proposer="Fable",
    proposer_detail="test",
    fields={"max_tokens": {"old": 4096, "new": 8192}},
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


def results_json(fractions: dict[str, float], fingerprint=FP) -> dict:
    tasks = {}
    for name, frac in fractions.items():
        split = "held_out" if name in {"A3", "A4"} else "held_in"
        n = 5 if split == "held_out" else 3
        tasks[name] = {"split": split, "attempts": n, "pass_fraction": frac}
    return {"fingerprint": dict(fingerprint), "tasks": tasks, "summary": {}}


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
        results_dir=results_dir,
        log=lambda *_: None,
    )
    assert seen["config"]["max_tokens"] == 8192 and seen["config"]["version"] == _bumped()
    # reverted afterwards, tree clean
    assert json.loads(config_path(fake_carbon).read_text())["max_tokens"] == 4096
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
            results_dir=tmp_path,
            log=lambda *_: None,
        )
    assert json.loads(config_path(fake_carbon).read_text())["max_tokens"] == 4096
    require_clean_tree(fake_carbon)


def test_dirty_tree_refused(fake_carbon, baseline, tmp_path):
    (fake_carbon / "stray.txt").write_text("x")
    with pytest.raises(RuntimeError, match="not clean"):
        validate_candidate(
            CANDIDATE,
            baseline_path=baseline,
            carbon_root=fake_carbon,
            run_runner=lambda *a: None,
            results_dir=tmp_path,
            log=lambda *_: None,
        )
