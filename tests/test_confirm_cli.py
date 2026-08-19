"""The coded confirmation pathway: `ConfirmationRecord` + the `confirm` CLI seam.

Mirrors `tests/test_loop_validate.py`'s pattern — a real (throwaway) git repo for
`apply_candidate`/`revert_config` to act on, a fake runner seam so no test spawns a
real suite run or touches `results/`. `confirmed()` is the real, unmocked function
from `loop.acceptance`; only the runner is faked.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from loop.acceptance import ACCEPT, CONFIRM, REJECT, Decision
from loop.artifacts import Candidate, ConfirmationRecord, write_confirmation_record
from loop.cli import load_first_decision, run_confirmation
from runner.carbon_env import CARBON_ROOT

REAL_CONFIG = CARBON_ROOT / "harness" / "harness_config.json"


def _live(field: str):
    return json.loads(REAL_CONFIG.read_text())[field]


OLD_MT = _live("max_tokens")
NEW_MT = OLD_MT * 2  # legal (positive int, no ceiling) and always distinct

CANDIDATE = Candidate(
    id="cand-confirm-x",
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

# The first decision every test in this file starts from: a real CONFIRM, carried by
# E4 on held-out, whose confirmation pair must rerun E4 and G5 — mirrors iter-06's
# actual shape (see `iterations/iter-06/confirmation-tool-output-offload-r3.json`).
FIRST_CONFIRM = Decision(
    outcome=CONFIRM,
    reasons=("gain beyond one attempt on held-out (carried by E4)",),
    delta_in=0.0,
    delta_ho=0.1,
    threshold_in=0.02,
    threshold_ho=0.02,
    evidence_split="held_out",
    improved_tasks=("E4",),
    confirm_tasks=("E4", "G5"),
)


def _write_validation_record(it_dir, candidate_id: str, decision: Decision) -> None:
    rec = {"candidate_id": candidate_id, "rule": {"applied": True, **decision.to_json()}}
    (it_dir / f"validation-{candidate_id}.json").write_text(json.dumps(rec))


def _arm(counts: dict[str, tuple[int, int, str]], fingerprint=FP) -> dict:
    """counts: task -> (passes, attempts, split). Mirrors the runner's real per-task
    result shape (`passes`/`attempts`/`pass_fraction`/`outcomes`)."""
    tasks = {}
    for name, (p, a, split) in counts.items():
        tasks[name] = {
            "split": split,
            "attempts": a,
            "passes": p,
            "pass_fraction": round(p / a, 4),
            "outcomes": ["pass"] * p + ["fail"] * (a - p),
        }
    return {"fingerprint": dict(fingerprint), "tasks": tasks}


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
def it_dir(tmp_path):
    d = tmp_path / "iter-01"
    d.mkdir()
    return d


# --- record production, via the fake seam ---------------------------------------


def test_confirm_produces_a_record_with_decision_json_and_stage(fake_carbon, it_dir):
    _write_validation_record(it_dir, CANDIDATE.id, FIRST_CONFIRM)

    base_arm = _arm({"E4": (0, 10, "held_out"), "G5": (4, 10, "held_in")})
    cand_arm = _arm({"E4": (10, 10, "held_out"), "G5": (8, 10, "held_in")})
    calls = []

    def fake_runner(label, only, attempts):
        calls.append((label, tuple(only), attempts))
        return base_arm if label == "confirm-base" else cand_arm

    record = run_confirmation(
        CANDIDATE,
        it_dir,
        baseline_label="confirm-base",
        candidate_label="confirm-cand",
        attempts=10,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        log=lambda *_: None,
    )

    assert calls == [
        ("confirm-base", ("E4", "G5"), 10),
        ("confirm-cand", ("E4", "G5"), 10),
    ]
    assert record.stage == "paired_confirmation"
    assert record.candidate_id == CANDIDATE.id
    assert record.baseline_label == "confirm-base"
    assert record.candidate_label == "confirm-cand"
    assert record.attempts_per_task_per_arm == 10
    assert record.confirm_set == ("E4", "G5")
    assert record.first_decision == FIRST_CONFIRM.to_json()
    assert record.confirmation["outcome"] == ACCEPT
    assert record.finding  # non-empty, human-readable


def test_per_task_matches_the_canned_pass_counts(fake_carbon, it_dir):
    _write_validation_record(it_dir, CANDIDATE.id, FIRST_CONFIRM)
    base_arm = _arm({"E4": (0, 10, "held_out"), "G5": (4, 10, "held_in")})
    cand_arm = _arm({"E4": (10, 10, "held_out"), "G5": (8, 10, "held_in")})

    def fake_runner(label, only, attempts):
        return base_arm if label == "confirm-base" else cand_arm

    record = run_confirmation(
        CANDIDATE,
        it_dir,
        baseline_label="confirm-base",
        candidate_label="confirm-cand",
        attempts=10,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        log=lambda *_: None,
    )

    assert record.per_task == {
        "E4": {"base": [0, 10], "cand": [10, 10]},
        "G5": {"base": [4, 10], "cand": [8, 10]},
    }


def test_the_config_is_applied_for_the_candidate_arm_and_reverted_after(fake_carbon, it_dir):
    """The candidate arm must actually run against the EDITED config and the baseline
    arm against the UNMODIFIED one, or a confirmation would just rerun the same state
    twice under two different labels — this is the property the whole apply/revert
    design exists for, pinned on both arms, not just the edited one."""
    _write_validation_record(it_dir, CANDIDATE.id, FIRST_CONFIRM)
    base_arm = _arm({"E4": (0, 10, "held_out"), "G5": (4, 10, "held_in")})
    cand_arm = _arm({"E4": (10, 10, "held_out"), "G5": (8, 10, "held_in")})
    seen_config = {}

    def fake_runner(label, only, attempts):
        live = json.loads((fake_carbon / "harness" / "harness_config.json").read_text())
        if label == "confirm-cand":
            seen_config["max_tokens"] = live["max_tokens"]
            return cand_arm
        seen_config["base_max_tokens"] = live["max_tokens"]
        return base_arm

    run_confirmation(
        CANDIDATE,
        it_dir,
        baseline_label="confirm-base",
        candidate_label="confirm-cand",
        attempts=10,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        log=lambda *_: None,
    )

    # baseline arm ran on the config AS RECORDED — unmodified, not yet touched.
    assert seen_config["base_max_tokens"] == OLD_MT
    # candidate arm ran on the config WITH THE EDIT APPLIED.
    assert seen_config["max_tokens"] == NEW_MT
    # reverted afterwards
    assert (
        json.loads((fake_carbon / "harness" / "harness_config.json").read_text())["max_tokens"]
        == OLD_MT
    )


# --- from_json round trip --------------------------------------------------------


def test_confirmation_record_round_trips_through_json():
    rec = ConfirmationRecord(
        candidate_id="cand-x",
        baseline_label="b",
        candidate_label="c",
        attempts_per_task_per_arm=10,
        confirm_set=("E4", "G5"),
        first_decision=FIRST_CONFIRM.to_json(),
        confirmation={"outcome": "ACCEPT"},
        per_task={"E4": {"base": [0, 10], "cand": [10, 10]}},
        finding="text",
    )
    assert ConfirmationRecord.from_json(rec.to_json()) == rec


def test_write_confirmation_record_round_trips_through_disk(tmp_path):
    rec = ConfirmationRecord(
        candidate_id="cand-x",
        baseline_label="b",
        candidate_label="c",
        attempts_per_task_per_arm=10,
        confirm_set=("E4",),
        first_decision=FIRST_CONFIRM.to_json(),
        confirmation={"outcome": "ACCEPT"},
        per_task={"E4": {"base": [0, 10], "cand": [10, 10]}},
        finding="text",
    )
    out = write_confirmation_record(rec, tmp_path / "confirmation-cand-x.json")
    written = json.loads(out.read_text())
    assert ConfirmationRecord.from_json(written) == rec


def test_stage_defaults_to_paired_confirmation():
    rec = ConfirmationRecord(
        candidate_id="c",
        baseline_label="b",
        candidate_label="c2",
        attempts_per_task_per_arm=1,
        confirm_set=(),
        first_decision={},
        confirmation={},
        per_task={},
        finding="",
    )
    assert rec.stage == "paired_confirmation"


# --- confirm-set / parity mismatch fails loudly, never as a recorded REJECT -----


def test_arm_mismatch_fails_loudly_with_no_artifact_written(fake_carbon, it_dir):
    """The two fresh arms disagreeing on which tasks they measured is exactly the
    hole `acceptance.confirmed()` raises `ValueError` on (`_parity`) — the pair was
    never actually MEASURED. Refusal is the feature here (mirrors `runner.delta`'s
    refusal to compare filtered/mismatched results, per AGENTS.md): a written
    `outcome: REJECT` is reserved for a genuine `confirmed()` verdict, and stamping
    one on an unmeasured pair would be a false measurement claim. The failure must be
    loud — `SystemExit` naming the cause — and must leave no `confirmation-*.json`
    behind to be mistaken for a real decision."""
    _write_validation_record(it_dir, CANDIDATE.id, FIRST_CONFIRM)
    base_arm = _arm({"E4": (0, 10, "held_out"), "G5": (4, 10, "held_in")})
    cand_arm = _arm({"E4": (10, 10, "held_out")})  # G5 missing entirely

    def fake_runner(label, only, attempts):
        return base_arm if label == "confirm-base" else cand_arm

    with pytest.raises(SystemExit) as excinfo:
        run_confirmation(
            CANDIDATE,
            it_dir,
            baseline_label="confirm-base",
            candidate_label="confirm-cand",
            attempts=10,
            carbon_root=fake_carbon,
            run_runner=fake_runner,
            log=lambda *_: None,
        )

    message = str(excinfo.value)
    assert "confirmation could not be measured" in message
    assert "task sets differ" in message  # the underlying ValueError's cause, named
    assert list(it_dir.glob("confirmation-*.json")) == []


# --- the refusal case: no first CONFIRM -------------------------------------------


def test_refuses_when_the_first_decision_was_a_reject(it_dir):
    rejected = Decision(
        outcome=REJECT,
        reasons=("no gain beyond one attempt on either split",),
        delta_in=0.0,
        delta_ho=0.0,
        threshold_in=0.01,
        threshold_ho=0.01,
    )
    _write_validation_record(it_dir, CANDIDATE.id, rejected)
    with pytest.raises(SystemExit, match="no first CONFIRM"):
        load_first_decision(it_dir, CANDIDATE.id)


def test_refuses_when_the_rule_was_not_applied(it_dir):
    (it_dir / f"validation-{CANDIDATE.id}.json").write_text(
        json.dumps({"rule": {"applied": False, "why": "section not calibrated"}})
    )
    with pytest.raises(SystemExit, match="no first CONFIRM"):
        load_first_decision(it_dir, CANDIDATE.id)


def test_refuses_when_there_is_no_validation_record_at_all(it_dir):
    with pytest.raises(SystemExit, match="no validation record"):
        load_first_decision(it_dir, "never-validated")


def test_run_confirmation_itself_refuses_without_a_first_confirm(fake_carbon, it_dir):
    """The refusal must fire before either arm is run — a confirmation with nothing
    to confirm must not spend a single suite attempt."""
    rejected = Decision(
        outcome=REJECT,
        reasons=("no gain",),
        delta_in=0.0,
        delta_ho=0.0,
        threshold_in=0.01,
        threshold_ho=0.01,
    )
    _write_validation_record(it_dir, CANDIDATE.id, rejected)
    ran = []
    with pytest.raises(SystemExit, match="no first CONFIRM"):
        run_confirmation(
            CANDIDATE,
            it_dir,
            baseline_label="confirm-base",
            candidate_label="confirm-cand",
            attempts=10,
            carbon_root=fake_carbon,
            run_runner=lambda label, *_: ran.append(label),
            log=lambda *_: None,
        )
    assert ran == []
