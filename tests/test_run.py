import json

import pytest

from runner.run import load_done, run_task
from runner.spec import Attempt, TaskSpec
from runner.suite import run_suite

FP = {"gemma_sha": "abc123", "gemma_dirty": False, "config_version": 1, "model": "gemma"}


def _spec(name="A1", split="held_in", passed=True):
    return TaskSpec(
        name=name,
        split=split,
        cluster="A",
        expected_baseline="pass",
        run=lambda: Attempt(passed, "pass" if passed else "fail", "detail"),
    )


def _record(task="A1", attempt=0, **overrides):
    rec = {
        "task": task,
        "split": "held_in",
        "cluster": "A",
        "expected_baseline": "pass",
        "attempt": attempt,
        "passed": True,
        "outcome": "pass",
        "detail": "d",
        "approvals": [],
        "turns": 1,
        "duration_s": 0.1,
        **FP,
    }
    rec.update(overrides)
    return rec


def test_resume_refuses_stale_fingerprint(tmp_path):
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record(gemma_sha="OLDSHA")) + "\n")
    with pytest.raises(RuntimeError, match="resume mismatch"):
        run_task(_spec(), FP, jsonl, log=lambda *a: None)


def test_resume_accepts_matching_fingerprint(tmp_path):
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record()) + "\n")
    tr = run_task(_spec(), FP, jsonl, log=lambda *a: None)
    assert len(tr.records) == 3  # 1 resumed + 2 fresh
    assert tr.pass_fraction == 1.0


def test_load_done_tolerates_malformed_final_line(tmp_path):
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record()) + "\n" + '{"task": "A1", "attempt"')
    done = load_done(jsonl, log=lambda *a: None)
    assert set(done) == {("A1", 0)}


def test_load_done_raises_on_malformed_nonfinal_line(tmp_path):
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text('{"broken\n' + json.dumps(_record()) + "\n")
    with pytest.raises(ValueError, match="line 1"):
        load_done(jsonl, log=lambda *a: None)


def test_orphaned_attempts_logged_on_shrunk_attempts(tmp_path):
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text("".join(json.dumps(_record(attempt=i)) + "\n" for i in range(3)))
    lines: list[str] = []
    tr = run_task(_spec(), FP, jsonl, attempts=1, log=lines.append)
    assert len(tr.records) == 1
    assert any("ignor" in line for line in lines)  # "ignored"/"ignoring"


def test_run_suite_refuses_partial_overwrite_of_full_results(tmp_path):
    (tmp_path / "full.json").write_text(json.dumps({"fingerprint": FP, "tasks": {}, "summary": {}}))
    with pytest.raises(RuntimeError, match="label"):
        run_suite(
            [_spec()],
            label="full",
            only={"A1"},
            fingerprint=FP,
            results_dir=tmp_path,
            log=lambda *a: None,
        )


def test_run_suite_filtered_run_stamps_filter_and_can_self_overwrite(tmp_path):
    results = run_suite(
        [_spec(), _spec(name="B1")],
        label="part",
        only={"A1"},
        fingerprint=FP,
        results_dir=tmp_path,
        log=lambda *a: None,
    )
    assert results["filter"] == ["A1"]
    assert set(results["tasks"]) == {"A1"}
    on_disk = json.loads((tmp_path / "part.json").read_text())
    assert on_disk["filter"] == ["A1"]
    # a filtered label overwriting its own filtered json is fine
    run_suite(
        [_spec(), _spec(name="B1")],
        label="part",
        only={"A1"},
        fingerprint=FP,
        results_dir=tmp_path,
        log=lambda *a: None,
    )


def test_run_suite_unfiltered_writes_results(tmp_path):
    results = run_suite(
        [_spec()],
        label="base",
        fingerprint=FP,
        results_dir=tmp_path,
        log=lambda *a: None,
    )
    assert "filter" not in results
    assert results["summary"]["held_in_rate"] == 1.0
    assert (tmp_path / "base.json").is_file()
