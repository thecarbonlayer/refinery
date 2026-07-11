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


def test_resume_refuses_different_model(tmp_path):
    """Same gemma sha + config but a different model is still a different
    harness state — resuming across a model swap would blend fractions."""
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record(model="other-model")) + "\n")
    with pytest.raises(RuntimeError, match="resume mismatch.*other-model"):
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


def test_load_done_truncates_torn_final_line_so_appends_stay_well_formed(tmp_path):
    jsonl = tmp_path / "r.jsonl"
    good = json.dumps(_record()) + "\n"
    fragment = '{"task": "A1", "attempt"'
    jsonl.write_text(good + fragment)  # torn write: no trailing newline
    lines: list[str] = []
    done = load_done(jsonl, log=lines.append)
    assert set(done) == {("A1", 0)}
    assert any("dropped" in line for line in lines)
    # the fragment must be gone from the file, which now ends in a clean newline
    content = jsonl.read_text()
    assert fragment not in content
    assert content == good
    # simulate run_task's append of a fresh record — must not fuse with anything
    with jsonl.open("a") as f:
        f.write(json.dumps(_record(attempt=1)) + "\n")
    done2 = load_done(jsonl, log=lambda *a: None)  # no ValueError
    assert set(done2) == {("A1", 0), ("A1", 1)}


def test_load_done_restores_newline_on_valid_final_line_missing_it(tmp_path):
    jsonl = tmp_path / "r.jsonl"
    # write torn exactly at the newline boundary: record is valid, newline gone
    jsonl.write_text(json.dumps(_record()))
    done = load_done(jsonl, log=lambda *a: None)
    assert set(done) == {("A1", 0)}
    assert jsonl.read_text().endswith("\n")
    with jsonl.open("a") as f:
        f.write(json.dumps(_record(attempt=1)) + "\n")
    assert set(load_done(jsonl, log=lambda *a: None)) == {("A1", 0), ("A1", 1)}


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
