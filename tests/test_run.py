import json

import pytest

from runner import guard
from runner.guard import StaleBaseline
from runner.run import load_done, run_task
from runner.spec import Attempt, TaskSpec
from runner.suite import run_suite

# The behavior-relevant fingerprint fields. behavior_key folds all of these EXCEPT
# the committed gemma_sha (which is provenance, not behavior); see runner/guard.py.
BASE_FP = {
    "gemma_sha": "abc123",
    "gemma_dirty": False,
    "dirty_sha": None,
    "config_version": 1,
    "model": "carbon",
    "runner_sha": "runnersha1",
}


def _fp(**over):
    """A self-consistent fingerprint: behavior_key derived from the (possibly
    overridden) behavior-relevant fields, exactly as carbon_fingerprint stamps it."""
    fp = {**BASE_FP, **over}
    fp["behavior_key"] = guard.fingerprint_behavior_key(fp)
    return fp


FP = _fp()


def _spec(name="A1", split="held_in", passed=True, metrics=None):
    return TaskSpec(
        name=name,
        split=split,
        cluster="A",
        expected_baseline="pass",
        run=lambda: Attempt(
            passed,
            "pass" if passed else "fail",
            "detail",
            metrics=metrics or {},
        ),
    )


def _record(task="A1", attempt=0, **overrides):
    """A recorded attempt. Fingerprint-field overrides (gemma_sha, model, ...) flow
    through _fp so behavior_key stays consistent with them; other keys override the
    attempt payload."""
    fp_over = {k: overrides.pop(k) for k in list(overrides) if k in BASE_FP}
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
        **_fp(**fp_over),
    }
    rec.update(overrides)
    return rec


def test_resume_accepts_additive_gemma_sha_bump(tmp_path):
    """The whole point: a record from a different COMMITTED gemma_sha but the same
    behavior_key (an additive, default-neutral release) resumes instead of forcing an
    empty re-baseline."""
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record(gemma_sha="OLDSHA")) + "\n")
    tr = run_task(_spec(), FP, jsonl, log=lambda *a: None)
    assert len(tr.records) == 3  # 1 resumed + 2 fresh — no refusal
    assert tr.records[0]["gemma_sha"] == "OLDSHA"  # the resumed record kept its provenance


def test_resume_refuses_different_model(tmp_path):
    """A different model is a different behavior — resuming across a model swap would
    blend fractions, so behavior_key differs and resume is refused."""
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record(model="other-model")) + "\n")
    with pytest.raises(StaleBaseline, match="resume mismatch.*other-model"):
        run_task(_spec(), FP, jsonl, log=lambda *a: None)


def test_resume_refuses_different_config_version(tmp_path):
    """A config bump is a real behavior change — behavior_key moves, resume refused."""
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record(config_version=2)) + "\n")
    with pytest.raises(StaleBaseline, match="resume mismatch"):
        run_task(_spec(), FP, jsonl, log=lambda *a: None)


def test_resume_refuses_different_dirty_state(tmp_path):
    """A different dirty-tree content identity is uncommitted carbon behavior that no
    version counter attests — behavior_key includes dirty_sha, so resume is refused."""
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record(gemma_dirty=True, dirty_sha="deadbeef")) + "\n")
    with pytest.raises(StaleBaseline, match="resume mismatch"):
        run_task(_spec(), FP, jsonl, log=lambda *a: None)


def test_resume_accepts_matching_dirty_sha(tmp_path):
    """The SAME uncommitted state (identical dirty_sha) is resumable."""
    fp = _fp(gemma_dirty=True, dirty_sha="deadbeef")
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record(gemma_dirty=True, dirty_sha="deadbeef")) + "\n")
    tr = run_task(_spec(), fp, jsonl, log=lambda *a: None)
    assert len(tr.records) == 3  # 1 resumed + 2 fresh


def test_resume_refuses_different_runner_sha(tmp_path):
    """Records measured by a different runner (verifier) version must not blend with
    fresh attempts — runner_sha stays in the behavior_key, so resume is refused."""
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record(runner_sha="other-runner")) + "\n")
    with pytest.raises(StaleBaseline, match="resume mismatch"):
        run_task(_spec(), FP, jsonl, log=lambda *a: None)


def test_resume_refuses_legacy_record_without_behavior_key(tmp_path):
    """A record predating this guard carries no behavior_key and cannot attest which
    behavior it measured (the old recorded baseline is exactly this) — stale by
    definition, so it is refused and re-recorded under a fresh label or with --force."""
    jsonl = tmp_path / "r.jsonl"
    rec = _record()
    del rec["behavior_key"]
    jsonl.write_text(json.dumps(rec) + "\n")
    with pytest.raises(StaleBaseline, match="resume mismatch"):
        run_task(_spec(), FP, jsonl, log=lambda *a: None)


def test_resume_accepts_matching_fingerprint(tmp_path):
    jsonl = tmp_path / "r.jsonl"
    jsonl.write_text(json.dumps(_record()) + "\n")
    tr = run_task(_spec(), FP, jsonl, log=lambda *a: None)
    assert len(tr.records) == 3  # 1 resumed + 2 fresh
    assert tr.pass_fraction == 1.0


def test_attempt_metrics_are_persisted(tmp_path):
    jsonl = tmp_path / "metrics.jsonl"
    tr = run_task(
        _spec(metrics={"tokens": 123.0, "tool_calls": 4.0}),
        FP,
        jsonl,
        attempts=1,
        log=lambda *a: None,
    )
    assert tr.records[0]["metrics"] == {"tokens": 123.0, "tool_calls": 4.0}
    assert json.loads(jsonl.read_text())["metrics"]["tokens"] == 123.0


def test_suite_aggregates_task_equal_weight_metrics(tmp_path):
    results = run_suite(
        [
            _spec(name="A1", metrics={"tokens": 100.0}),
            _spec(name="B1", metrics={"tokens": 300.0}),
        ],
        label="metrics",
        attempts=1,
        fingerprint=FP,
        results_dir=tmp_path,
        log=lambda *a: None,
    )
    assert results["tasks"]["A1"]["metrics"]["tokens"] == 100.0
    assert results["summary"]["metrics"]["tokens"] == 200.0


def test_task_metric_mean_covers_only_reporting_attempts(tmp_path):
    """Every other metrics test uses ``attempts=1``, where zero-filling and
    skip-absent coincide — so the attempt-level denominator was never exercised.

    ``run.py`` records ``metrics={}`` for an attempt that raises, so a task with one
    good attempt out of three must report the ONE-attempt mean and publish that
    count. Otherwise a reader sees ``attempts: 3`` beside a 1-attempt figure and
    reads it as a 3-attempt mean.
    """
    calls = {"n": 0}

    def run():
        calls["n"] += 1
        if calls["n"] == 1:
            return Attempt(True, "pass", "ok", metrics={"tokens": 900.0})
        return Attempt(False, "error", "boom", metrics={})

    spec = TaskSpec(name="A1", split="held_in", cluster="A", expected_baseline="pass", run=run)
    results = run_suite(
        [spec],
        label="attempts",
        attempts=3,
        fingerprint=FP,
        results_dir=tmp_path,
        log=lambda *a: None,
    )
    task = results["tasks"]["A1"]
    assert task["attempts"] == 3
    assert task["metrics"]["tokens"] == 900.0  # not 300.0
    assert task["metric_attempt_counts"] == {"tokens": 1}


def test_suite_to_delta_carries_attempt_drift_end_to_end(tmp_path):
    """The only test that runs run_suite -> delta with REAL summaries.

    Every other attempt-level test hand-builds the summary dict, so
    `_total_attempt_counts` could return `{}` — or `max()` instead of a sum — and
    the whole attempt-level feature would be silently dead while 154 tests passed.
    This is the seam: the suite writes the counts, delta reads them.
    """
    from runner.delta import delta

    def flaky(good_attempts: int):
        calls = {"n": 0}

        def run():
            calls["n"] += 1
            if calls["n"] <= good_attempts:
                return Attempt(True, "pass", "ok", metrics={"tokens": 900.0})
            return Attempt(False, "error", "boom", metrics={})

        return run

    def suite(label, good):
        return run_suite(
            [
                TaskSpec("A1", "held_in", "A", "pass", flaky(good)),
                TaskSpec("B1", "held_in", "B", "pass", flaky(good)),
            ],
            label=label,
            attempts=3,
            fingerprint=FP,
            results_dir=tmp_path,
            log=lambda *a: None,
        )

    base = suite("base", good=3)  # 2 tasks x 3 reporting attempts
    cand = suite("cand", good=1)  # 2 tasks x 1 reporting attempt
    assert base["summary"]["metric_attempt_counts"] == {"tokens": 6}
    assert cand["summary"]["metric_attempt_counts"] == {"tokens": 2}
    d = delta(base, cand)
    # Same task count on both sides, so ONLY the attempt counts reveal the change.
    assert d["metric_task_counts"]["tokens"] == {"baseline": 2, "candidate": 2}
    assert d["metric_attempt_counts"]["tokens"] == {"baseline": 6, "candidate": 2}
    assert d["metric_denominator_drift"] == ["tokens"]


def test_suite_metric_mean_excludes_tasks_that_cannot_report(tmp_path):
    """A task with no cost telemetry must be ABSENT from that metric's mean, not
    averaged in as a measured zero.

    Both tasks carrying every metric is the easy case; this is the one that was
    wrong. Scripted-provider diagnostics report no tokens at all, so counting
    them as zeros silently deflated suite-wide cost as the suite grew. The
    contributing-task count is published so a mean over some tasks cannot be
    read as a mean over all of them.
    """
    results = run_suite(
        [
            _spec(name="A1", metrics={"tokens": 100.0, "retries": 1.0}),
            _spec(name="B1", metrics={"retries": 3.0}),  # no cost telemetry at all
        ],
        label="sparse",
        attempts=1,
        fingerprint=FP,
        results_dir=tmp_path,
        log=lambda *a: None,
    )
    assert results["summary"]["metrics"]["tokens"] == 100.0  # over 1 task, not 50.0 over 2
    assert results["summary"]["metrics"]["retries"] == 2.0
    assert results["summary"]["metric_task_counts"] == {"retries": 2, "tokens": 1}


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


def test_run_suite_aborts_when_carbon_state_changes_mid_suite(tmp_path, monkeypatch):
    """A carbon edit between tasks would stamp later attempts with the
    stale suite-start fingerprint — the per-task recompute must catch it."""
    import runner.suite as suite_mod

    fps = iter([FP, FP, {**FP, "gemma_sha": "MUTATED"}])
    monkeypatch.setattr(suite_mod, "carbon_fingerprint", lambda: next(fps))
    with pytest.raises(RuntimeError, match="changed mid-suite"):
        run_suite(
            [_spec(), _spec(name="B1")],
            label="drift",
            results_dir=tmp_path,
            log=lambda *a: None,
        )


def test_run_suite_injected_fingerprint_skips_mid_suite_recompute(tmp_path, monkeypatch):
    """Injection is a test seam that bypasses the live carbon checkout — the
    mid-suite guard must not fire (or even call carbon_fingerprint)."""
    import runner.suite as suite_mod

    def boom():
        raise AssertionError("carbon_fingerprint must not be called when injected")

    monkeypatch.setattr(suite_mod, "carbon_fingerprint", boom)
    results = run_suite(
        [_spec(), _spec(name="B1")],
        label="injected",
        fingerprint=FP,
        results_dir=tmp_path,
        log=lambda *a: None,
    )
    assert set(results["tasks"]) == {"A1", "B1"}


def test_run_suite_resumes_after_additive_carbon_bump(tmp_path):
    """Acceptance test: a complete baseline recorded at one gemma_sha resumes (re-runs
    nothing) when only the committed gemma_sha has moved — the additive-release case."""
    jsonl = tmp_path / "base.jsonl"
    jsonl.write_text(
        "".join(json.dumps(_record(gemma_sha="OLDSHA", attempt=i)) + "\n" for i in range(3))
    )
    ran = {"n": 0}

    def counting_spec():
        def run():
            ran["n"] += 1
            return Attempt(True, "pass", "d")

        return TaskSpec("A1", "held_in", "A", "pass", run=run)

    results = run_suite(
        [counting_spec()],
        label="base",
        fingerprint=_fp(gemma_sha="NEWSHA"),  # additive move
        results_dir=tmp_path,
        log=lambda *a: None,
    )
    assert ran["n"] == 0  # every attempt resumed, nothing re-run
    assert results["tasks"]["A1"]["attempts"] == 3


def test_run_suite_refuses_stale_baseline_up_front(tmp_path):
    """A stale baseline (real behavior change) fails LOUDLY before any attempt runs."""
    jsonl = tmp_path / "base.jsonl"
    jsonl.write_text(json.dumps(_record()) + "\n")
    (tmp_path / "base.json").write_text(json.dumps({"fingerprint": FP, "tasks": {}, "summary": {}}))
    ran = {"n": 0}

    def counting_spec():
        def run():
            ran["n"] += 1
            return Attempt(True, "pass", "d")

        return TaskSpec("A1", "held_in", "A", "pass", run=run)

    with pytest.raises(StaleBaseline, match="behavior_key"):
        run_suite(
            [counting_spec()],
            label="base",
            fingerprint=_fp(config_version=2),
            results_dir=tmp_path,
            log=lambda *a: None,
        )
    assert ran["n"] == 0  # refused before spending a single attempt


def test_run_suite_force_overwrites_stale_baseline(tmp_path):
    """--force discards a stale baseline and re-runs from scratch instead of refusing."""
    jsonl = tmp_path / "base.jsonl"
    jsonl.write_text(json.dumps(_record(config_version=99)) + "\n")  # incompatible prior
    results = run_suite(
        [_spec()],
        label="base",
        fingerprint=FP,
        results_dir=tmp_path,
        force=True,
        log=lambda *a: None,
    )
    assert results["tasks"]["A1"]["attempts"] == 3
    # the stale record was discarded, not blended in
    on_disk = load_done(tmp_path / "base.jsonl", log=lambda *a: None)
    assert all(rec["behavior_key"] == FP["behavior_key"] for rec in on_disk.values())


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


def test_summary_carries_security_classes_and_session_env(tmp_path):
    """security_classes rides beside outcomes per task (Attempt.security_class ->
    the record -> the summary), and results["session_env"] stamps carbon's local
    environment metadata at the top level — built through run_suite's real path (the
    injected-fingerprint test seam), not a hand-assembled summary dict.

    Two attempts with different security_class values (one "behavioral", one None)
    prove alignment rather than a length-1 coincidence.
    """
    from harness.session_env import LOCAL_METADATA

    calls = {"n": 0}

    def run():
        calls["n"] += 1
        if calls["n"] == 1:
            return Attempt(False, "critical_failure", "leak", security_class="behavioral")
        return Attempt(True, "pass", "ok")

    spec = TaskSpec(name="C1", split="held_in", cluster="C", expected_baseline="pass", run=run)
    results = run_suite(
        [spec],
        label="secclass",
        fingerprint=FP,
        results_dir=tmp_path,
        log=lambda *a: None,
    )
    task = results["tasks"]["C1"]
    assert task["outcomes"] == ["critical_failure", "pass", "pass"]
    assert task["security_classes"] == ["behavioral", None, None]
    assert results["session_env"] == dict(LOCAL_METADATA)
    # session_env is a manifest stamp, not a behavior input — the resume guard
    # compares fingerprints, so it must stay out of that dict.
    assert "session_env" not in results["fingerprint"]
