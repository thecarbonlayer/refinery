"""Pins for the derived coverage map.

Several cases read the REAL recorded runs rather than a fixture. The module exists
because of one specific misreading of those runs, and a synthetic fixture would let the
module keep passing while the fact it was built on stopped being true.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loop.knob_coverage import DELIBERATE_NON_OBSERVERS, KNOB_COVERAGE
from loop.observed_coverage import (
    REQUIRES,
    activity_from_rows,
    contradicted_observers,
    observed_activity,
    partition_deltas,
    select_attempts,
    unlisted_with_activity,
    unreachable,
)

# The exact pair iteration 3 compared. NOT `result_files()`: pooling every log mixes
# runner versions, config versions and partial runs into one claim, and under the
# now-strict "fully recorded" rule that mixture leaves almost everything unknown. A
# coverage claim is only meaningful about a cohort, so the fixture names one.
COHORT = ["results/baseline-r5-v7.jsonl", "results/cand-tool-output-offload.jsonl"]
# `coverage_note` takes (attempt log, result JSON) pairs, because the log may hold rows
# the summary excludes and only the summarized attempts are the cohort `delta` compared.
COHORT_PAIRS = [(Path(f), Path(f).with_suffix(".json")) for f in COHORT]


@pytest.fixture(scope="module")
def activity():
    from pathlib import Path

    files = [Path(f) for f in COHORT]
    assert all(f.exists() for f in files), f"missing cohort files: {COHORT}"
    return observed_activity(files)


def test_the_tasks_that_sank_iteration_3_cannot_be_reached_by_tool_output(activity):
    """The incident this module exists for, asserted against the real runs.

    Iteration 3's offload candidate was rejected on Δ_in −0.118, assembled from A1
    (−1.00), G5 (−0.67), G4 (−0.33) and B2 (−0.33). A1 and G4 build agents with no tool
    registry and carbon gates execution on `self.tools is not None`, so no value of
    `tool_output` reaches them. Those two contributed −1.33 of the −2.00 that made the
    mean negative.

    G5 is deliberately NOT asserted here: it makes real `write_file` calls, so this
    module cannot exclude it, and the reason it is unreachable in fact — ~35-char
    results against a 4,000 budget — is a size argument no metric carries. Claiming it
    would be exactly the decorative coverage `knob_coverage` warns about.
    """
    dead = unreachable("tool_output", activity)
    assert {"A1", "G4"} <= dead
    assert "G5" not in dead


def test_partition_splits_iteration_3s_own_deltas(activity):
    per_task = {"A1": -1.0, "B2": -1 / 3, "G1": 1 / 3, "G4": -1 / 3, "G5": -2 / 3}
    split = partition_deltas("tool_output", per_task, activity)

    assert set(split["unreachable_proven"]) == {"A1", "G1", "G4"}
    # G1 sits in the unreachable half too, and that matters more than it looks: it moved
    # UP. An earlier pass at this analysis dropped only the tasks that hurt the
    # candidate and got Δ_in = 0.0000, which accepts. Dropping symmetrically leaves
    # B2 and G5 and a negative sum, which rejects. A partition that filtered by sign
    # would be a way to fit any verdict you wanted.
    assert split["unreachable_proven"]["G1"] > 0
    assert set(split["evidence"]) == {"B2", "G5"}


def test_a_knob_with_no_necessary_condition_excludes_nothing(activity):
    """Silence, not a guess. `default_context_limit` has no metric that can rule a task
    out — a task that never compacted at the shipped window compacts as soon as the
    window is lowered — so the map must decline rather than reuse `compactions`."""
    assert "default_context_limit" not in REQUIRES
    for knob in ("default_context_limit", "system_prompt", "temperature", "file_injection"):
        assert unreachable(knob, activity) == set()


def test_airtight_only_drops_the_evidence_grade_exclusions(activity):
    """The two grades must stay separable by callers. `compaction` is evidence-grade
    because `trigger_fraction` belongs to that knob and can create the very activity
    its absence is being read as proof of."""
    assert not REQUIRES["compaction"].airtight
    # `compaction_prompt` looked airtight and is not: `compact()` sends the prompt to the
    # provider, that call can raise, and carbon increments `compaction_count` only after
    # `compact()` RETURNS — so a legal wording that makes the summarizer request fail
    # moves the verdict while the recorded count stays zero. Proving the prompt is not
    # read before a compaction ATTEMPT is not proving every effect of it is counted.
    assert not REQUIRES["compaction_prompt"].airtight
    assert REQUIRES["tool_output"].airtight

    assert unreachable("compaction", activity)
    assert unreachable("compaction", activity, airtight_only=True) == set()
    assert unreachable("compaction_prompt", activity, airtight_only=True) == set()
    assert unreachable("tool_output", activity, airtight_only=True)


def test_peak_not_mean_decides_whether_something_ever_happened(tmp_path):
    """A task with one active attempt among many idle ones is NOT inert.

    The results JSON stores metric MEANS. One tool call in six attempts is a mean of
    0.17, which rounds to nothing in a report and would file a reachable task as
    unreachable — turning this map from a noise filter into a way to discard real
    evidence.
    """
    path = tmp_path / "run.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"task": "X", "metrics": {"tool_calls": n}}) for n in (0, 0, 0, 0, 0, 1)
        )
        + "\n"
    )
    act = observed_activity([path])
    assert act["X"]["tool_calls"] == 1.0
    assert unreachable("tool_output", act) == set()


def test_a_false_observer_row_is_caught(activity):
    """The check has to be able to fire. `knob_coverage` says an audit once found six
    of its rows false at once, and that no test in this repo could detect it."""
    assert contradicted_observers(activity) == {}, "real table should be consistent"

    fabricated = dict(KNOB_COVERAGE["tool_output"])
    fabricated["observers"] = (*fabricated["observers"], "A1")  # A1 has no tools at all
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(KNOB_COVERAGE, "tool_output", fabricated)
        # PROVEN, not merely probable: `tool_output` is the airtight grade, so calling
        # the row false is warranted. On an evidence-grade knob the same absence lands
        # in `probable`, because the knob could create the activity it is missing.
        assert contradicted_observers(activity)["tool_output"]["proven"] == ["A1"]


def test_the_review_queue_is_not_silenced_by_a_claim_known_to_be_false(activity):
    """A suppression entry is executable, so a wrong one silences its own warning.

    `unlisted_with_activity()` drops every task named in `DELIBERATE_NON_OBSERVERS`.
    A2, C1, C2 and C3 were all in there on `tool_output` and all four rationales were
    shown false or half-argued: A2 states a prior the registry contradicts and reasons
    only about budget on a knob that also carries strategy; C1/C2 protect their leak
    predicate but not the functional reply carbon sends the truncated result to; C3
    scans files but the model reads tool results before deciding what to write. They
    are removed, so the queue raises them again.

    Removing them does NOT promote them into `KNOB_COVERAGE`. That distinction is the
    whole point: suppressing an unresolved warning is a defect, promoting a task to
    observer is a decision about what the loop may tune.
    """
    queue = unlisted_with_activity(activity)
    # A2 and C3 have since been CLASSIFIED (guards), so they leave the queue by being
    # answered rather than by being silenced — which is the distinction. C1/C2 remain,
    # held out pending a counterfactual sweep rather than on a settled argument.
    assert {"C1", "C2"} <= set(queue["tool_output"])
    assert {"A2", "C3"}.isdisjoint(queue["tool_output"])
    # G5 is suppressed again, and legitimately: its verdict reads the file list off the
    # tool CALLS, so no budget moves it. Mechanism, not size.
    assert "G5" in DELIBERATE_NON_OBSERVERS["tool_output"]

    # The remaining entries are argued from a mechanism carbon's code makes true, and
    # must still suppress — a queue that raises settled decisions every run is one
    # nobody reads.
    for knob, tasks in DELIBERATE_NON_OBSERVERS.items():
        for task in tasks:
            assert task not in queue.get(knob, []), f"{knob}/{task} argued but still queued"
    assert "H2" in DELIBERATE_NON_OBSERVERS["compaction"]


def test_every_argued_exclusion_names_a_knob_that_exists_and_gives_a_reason():
    """An exclusion without an argument is a way to silence the queue by editing a set.
    The bar is the same one `GUARD_ONLY_KNOBS` sets: state why, in the file."""
    for knob, tasks in DELIBERATE_NON_OBSERVERS.items():
        assert knob in KNOB_COVERAGE, f"{knob} is not a knob"
        assert knob in REQUIRES, f"{knob} has no derived activity, so nothing to exclude from"
        for task, why in tasks.items():
            assert len(why) > 40, f"{knob}/{task} exclusion has no real argument"


def test_a_candidate_editing_two_knobs_is_unreachable_only_where_BOTH_are(activity):
    """A task reached by either edited knob is reachable, so the sets intersect.

    The accumulator starts at None, not {}. An empty first set means "this knob reaches
    everything" and must swallow the result; treating it as "nothing seen yet" made the
    second knob's exclusions the answer, which would file real evidence as noise
    whenever a candidate paired a narrow knob with a broad one.
    """
    from loop.artifacts import Candidate
    from loop.validate import coverage_note

    def candidate(*fields):
        return Candidate(
            id="c",
            cluster_id="CL",
            proposer="p",
            proposer_detail="d",
            fields={f: {"old": 0, "new": 1} for f in fields},
            rationale="r",
            expected_effect="e",
            regression_risk="g",
        )

    paths = COHORT_PAIRS
    per_task = {"A1": -1.0, "B2": -0.5}
    # `tool_output` alone cannot reach A1 (no tool registry at all).
    note = coverage_note(candidate("tool_output"), per_task, paths)
    assert set(note["unreachable_proven"]) == {"A1"}
    # `system_prompt` reaches every live task, so pairing it removes every exclusion.
    both = coverage_note(candidate("tool_output", "system_prompt"), per_task, paths)
    assert both["unreachable_proven"] == {}
    assert both["unreachable_probable"] == {}
    assert set(both["evidence"]) == {"A1", "B2"}


def test_every_measured_task_is_classified_rather_than_excluded_by_a_floor():
    """The floor is gone; the rows carry the tasks instead.

    A plausibility floor was written to keep this row small, then marked NOT ENFORCED
    because nothing in the proposal path honoured it. The decision was to abandon it
    rather than enforce it: enforcing adds a real constraint on what the loop may
    propose in order to fix a bookkeeping inconsistency, and every previous constraint
    added for tidiness in this program turned into a false veto.

    So every task measured as breakable by a legal `tool_output` value is now listed,
    and `MEASURED_BREAK_BUDGETS` is the evidence for each. C1/C2 are the only tasks
    held out, and held out PENDING a sweep rather than on a settled argument.
    """
    from loop.knob_coverage import (
        _TOOL_RESULT_READERS,
        KNOB_COVERAGE,
        MEASURED_BREAK_BUDGETS,
    )

    assert not hasattr(
        __import__("loop.knob_coverage", fromlist=["x"]), "TOOL_OUTPUT_TUNING_FLOOR"
    ), "the floor was abandoned; a constant left behind would invite it back unenforced"

    missing = set(MEASURED_BREAK_BUDGETS) - set(_TOOL_RESULT_READERS)
    assert not missing, f"measured as breakable but not listed: {sorted(missing)}"

    # A2 is listed on a STRATEGY argument, not a budget one — `keep_head` drops the tail
    # its sentinel sits in — so it would belong under any floor and has no measurement.
    assert "A2" in _TOOL_RESULT_READERS and "A2" not in MEASURED_BREAK_BUDGETS
    assert "A2" in KNOB_COVERAGE["tool_output"]["guards"]

    # C1/C2 are the only held-out tasks, and are held out pending evidence.
    assert {"C1", "C2"}.isdisjoint(_TOOL_RESULT_READERS)


def test_the_written_artifact_carries_the_gate_and_coverage_fields():
    """The record on DISK, not the object in memory. This is where the bug was.

    `gates` and `coverage` were both attached to `ValidationRecord`, both described in
    their own docstrings as part of the record, and both dropped by `to_json()` — the
    only path to disk, since `loop/cli.py` writes `to_json()` and nothing else. So the
    harness-gate outcome was never persisted for any candidate, and neither was the
    coverage split, while the code read as though the evidence existed.

    Every test written for either feature exercised the computation. None read the
    artifact, which is exactly how a serializer omission survives: the thing under test
    was one call short of the thing being claimed.
    """
    import json as _json
    import pathlib

    from loop.artifacts import ValidationRecord

    record = ValidationRecord(
        candidate_id="c",
        label="l",
        accepted=False,
        delta_in=-0.1,
        delta_ho=0.2,
        gates={"passed": True, "checks": {"carbon_verify": {"passed": True}}},
        coverage={"knobs": ["tool_output"], "unreachable_proven": {"A1": -1.0}},
    )
    # Through the REAL write path, to a REAL file. An in-memory round trip of
    # `to_json()` checks the serializer and stops there: a caller that drops a key just
    # before writing survives it, which is one layer away from the omission this test
    # exists for. `loop/cli.py` now routes through the same helper.
    import tempfile

    from loop.artifacts import write_validation_record

    with tempfile.TemporaryDirectory() as d:
        out = write_validation_record(record, pathlib.Path(d) / "validation-c.json")
        written = _json.loads(out.read_text())

    assert written["gates"]["passed"] is True
    assert written["coverage"]["unreachable_proven"] == {"A1": -1.0}
    # And every declared field must survive the round trip, so the next one added is
    # not silently dropped the same way.
    from dataclasses import fields

    assert set(written) == {f.name for f in fields(ValidationRecord)}, (
        "to_json() and the dataclass have diverged — a field that exists on the record "
        "but never reaches disk is worse than one that was never added"
    )


def test_a_task_measured_for_other_metrics_but_not_this_one_is_not_excluded(tmp_path):
    """Absence of evidence must not be spelled "zero".

    `metrics.get(metric, 0.0)` read a metric that was never recorded as a measured
    zero, so a task became PROVABLY unreachable on the strength of never having been
    measured for the thing in question. 298 attempt rows in `results/` carry no metrics
    at all — older runs, plus the error path that records a raised task without agent
    metrics — and the failure direction is the dangerous one: it discards real evidence
    rather than keeping noise.
    """
    path = tmp_path / "run.jsonl"
    path.write_text(json.dumps({"task": "X", "metrics": {"compactions": 2.0}}) + "\n")
    act = observed_activity([path])

    assert "tool_calls" not in act["X"], "fixture must not record the metric at all"
    assert unreachable("tool_output", act) == set(), "never measured is not measured zero"
    # …while a task measured AT zero for that same metric is still excluded, or the fix
    # would have thrown the mechanism out with the bug.
    path.write_text(json.dumps({"task": "X", "metrics": {"tool_calls": 0.0}}) + "\n")
    assert unreachable("tool_output", observed_activity([path])) == {"X"}


def test_an_evidence_grade_exclusion_lands_in_probable_never_proven(activity):
    """The two grades must stay apart at the point a caller reads them.

    `compaction` exclusions are evidence only: `trigger_fraction` belongs to that knob,
    so lowering it makes a task compact that never has. Collapsing them into one bucket
    is what let the report claim "no legal value can affect this task" about a knob
    whose own `REQUIRES` entry says the opposite.
    """
    # B1 and D1 never compact in any recorded run; A1 does, so it stays evidence.
    per_task = {"B1": -0.5, "D1": 0.25, "A1": -0.25}
    split = partition_deltas("compaction", per_task, activity)

    assert split["unreachable_proven"] == {}, "compaction can never yield a proof"
    assert set(split["unreachable_probable"]) == {"B1", "D1"}
    assert set(split["evidence"]) == {"A1"}
    # And the airtight knob must still produce proofs, so the grades are not simply
    # both routed to the weaker bucket.
    assert partition_deltas("tool_output", per_task, activity)["unreachable_proven"]


def test_a_task_in_both_grades_is_reported_once_as_proven(activity):
    """`proven` and `probable` must stay disjoint, whatever a caller is handed.

    An overlap would double-count a movement and let a reader add the buckets to a
    number larger than the split it came from.
    """
    for knob in ("tool_output", "compaction", "compaction_prompt", "retry"):
        per_task = dict.fromkeys(activity, 1.0)
        split = partition_deltas(knob, per_task, activity)
        proven, probable = set(split["unreachable_proven"]), set(split["unreachable_probable"])
        assert not (proven & probable), f"{knob}: task in both grades"
        assert not (proven | probable) & set(split["evidence"]), f"{knob}: task in two buckets"
        assert proven | probable | set(split["evidence"]) == set(per_task)


def test_a_probable_contradiction_is_never_reported_as_a_false_row(activity):
    """Calling a row FALSE on evidence-grade absence is the same overclaim the grades
    exist to stop, one level up. `main()` must exit non-zero only on a proof."""
    from loop.observed_coverage import main

    fabricated = dict(KNOB_COVERAGE["compaction"])
    # B1 never compacts in this cohort, but `trigger_fraction` belongs to this knob and
    # could make it — so this is `probable`, not a false row.
    fabricated["observers"] = (*fabricated["observers"], "B1")
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(KNOB_COVERAGE, "compaction", fabricated)
        graded = contradicted_observers(activity)["compaction"]
        assert graded["probable"] == ["B1"]
        assert graded["proven"] == []
        mp.setattr("loop.observed_coverage.result_files", lambda *a, **k: [Path(f) for f in COHORT])
        assert main([]) == 0, "a probable contradiction must not fail the check"


def test_partial_metric_recording_is_unknown_not_zero(tmp_path):
    """Recorded on some attempts and absent on others is UNKNOWN.

    Keeping the surviving zeros and discarding the misses made a task with the metric
    absent from 19 attempts and present in 31 read as measured-at-zero, and it was
    reported "PROVABLY unreachable" on telemetry that had proven nothing.
    """
    path = tmp_path / "run.jsonl"
    path.write_text(
        json.dumps({"task": "X", "metrics": {"tool_calls": 0.0}})
        + "\n"
        + json.dumps({"task": "X", "metrics": {}})
        + "\n"
    )
    act = observed_activity([path])
    assert "tool_calls" not in act["X"], "partial recording must not survive as a value"
    assert unreachable("tool_output", act) == set()


def test_the_cohort_is_recorded_so_the_claim_can_be_reproduced():
    """A coverage claim is about a cohort or it is about nothing.

    `delta` refuses to compare results across runner versions. A coverage claim
    assembled across them, stated with no record of which files it used, is the same
    mistake with none of the refusal.
    """
    from loop.observed_coverage import cohort

    files = cohort([Path(f) for f in COHORT])["files"]
    assert [f["file"] for f in files] == sorted(Path(f).name for f in COHORT)
    assert all(f["rows"] > 0 and f["runner_sha"] and f["config_version"] for f in files)
    # The pair this validation compared shares one runner version, which is the property
    # that makes pooling them legitimate at all.
    assert len({sha for f in files for sha in f["runner_sha"]}) == 1


def test_a_task_proven_for_one_knob_and_probable_for_another_is_not_evidence():
    """The mixed-grade case, which intersecting the two grades separately gets wrong.

    If `tool_output` PROVES X unreachable and `compaction` finds X only PROBABLY so, X
    appears in neither intersection and lands in `evidence` — asserting the candidate
    can reach a task no edited knob shows any route to. The correct algebra intersects
    the UNION of the two grades and then subtracts the proven intersection:

        proven_all   = ∩ proven_i
        excluded_all = ∩ (proven_i ∪ probable_i)
        probable_all = excluded_all − proven_all

    The earlier multi-knob test cannot see this: it pairs `tool_output` with
    `system_prompt`, which has no necessary condition at all, so both algebras agree.
    """
    import loop.validate as validate_mod
    from loop.artifacts import Candidate

    with pytest.MonkeyPatch.context() as mp:
        # X: no tool calls (PROVEN for tool_output) and no compactions (PROBABLE for
        # compaction, since trigger_fraction could create them). Y is reached by both.
        mp.setattr(
            validate_mod,
            "activity_from_rows",
            lambda rows: {
                "X": {"tool_calls": 0.0, "compactions": 0.0},
                "Y": {"tool_calls": 4.0, "compactions": 2.0},
            },
        )
        cand = Candidate(
            id="c",
            cluster_id="CL",
            proposer="p",
            proposer_detail="d",
            fields={f: {"old": 0, "new": 1} for f in ("tool_output", "compaction")},
            rationale="r",
            expected_effect="e",
            regression_risk="g",
        )
        note = validate_mod.coverage_note(cand, {"X": -1.0, "Y": 0.5}, COHORT_PAIRS)

    assert note["evidence"] == {"Y": 0.5}, "X reached neither knob and must not be evidence"
    assert note["unreachable_proven"] == {}, "only ONE knob proves X, so the pair cannot"
    assert note["unreachable_probable"] == {"X": -1.0}


def test_validate_candidate_writes_a_record_carrying_the_cohort_and_the_exclusion(tmp_path):
    """End to end: validate → record → disk → read back.

    Every earlier test stopped short of one seam or another, and each gap was real. The
    serializer dropped two fields for as long as they existed. The "written artifact"
    test round-tripped in memory. Dropping the candidate log from the cohort wiring, or
    setting `coverage={}` on the record, both survived the suite. This asserts the whole
    path: that the cohort names BOTH logs, that the exclusion computed from them lands in
    the record, and that it reaches the file.
    """
    import json as _json

    from loop.artifacts import Candidate, write_validation_record
    from loop.validate import validate_candidate

    tasks = {"A1": {"split": "held_in"}, "E2": {"split": "held_out"}}

    def write_run(stem: str, tool_calls: dict[str, float]) -> None:
        summary = {
            "fingerprint": {"runner_sha": "deadbeef", "model": "m"},
            "tasks": {
                name: {**meta, "attempts": 1, "passes": 1, "pass_fraction": 1.0}
                for name, meta in tasks.items()
            },
        }
        (tmp_path / f"{stem}.json").write_text(_json.dumps(summary))
        (tmp_path / f"{stem}.jsonl").write_text(
            "\n".join(
                _json.dumps(
                    {
                        "task": name,
                        "attempt": 0,
                        "runner_sha": "deadbeef",
                        "config_version": 7,
                        "model": "m",
                        "metrics": {"tool_calls": tool_calls[name]},
                    }
                )
                for name in tasks
            )
            + "\n"
        )

    write_run("base", {"A1": 0.0, "E2": 2.0})
    write_run("cand", {"A1": 0.0, "E2": 2.0})

    candidate = Candidate(
        id="x",
        cluster_id="CL",
        proposer="p",
        proposer_detail="d",
        fields={"tool_output": {"old": 0, "new": 1}},
        rationale="r",
        expected_effect="e",
        regression_risk="g",
    )
    # The carbon-side steps have no injection seam and need a real git tree; they are
    # not what this test is about, so they are stubbed at the module boundary.
    import loop.validate as validate_mod

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(validate_mod, "require_clean_tree", lambda *a, **k: None)
        mp.setattr(validate_mod, "revert_config", lambda *a, **k: None)
        mp.setattr(validate_mod, "apply_candidate", lambda *a, **k: {"version": 8})
        record = validate_candidate(
            candidate,
            baseline_path=tmp_path / "base.json",
            label="cand",
            run_runner=lambda *a, **k: None,
            run_gates=lambda *a, **k: {"passed": True, "checks": {}},
            results_dir=tmp_path,
            carbon_root=tmp_path,
            log=lambda *_: None,
        )

    written = _json.loads(write_validation_record(record, tmp_path / "out.json").read_text())
    files = {f["file"].split("/")[-1] for f in written["coverage"]["cohort"]["files"]}
    assert files == {"base.jsonl", "cand.jsonl"}, "both logs must be named, not just one"
    # A1 makes no tool calls in either arm, so `tool_output` provably cannot reach it.
    assert "A1" in written["coverage"]["unreachable_proven"]
    assert "E2" in written["coverage"]["evidence"]
    assert all(f["rows_selected"] == 2 for f in written["coverage"]["cohort"]["files"])


def _run_files(tmp_path, stem, attempts, rows):
    import json as _json

    (tmp_path / f"{stem}.json").write_text(
        _json.dumps({"tasks": {"T": {"split": "held_in", "attempts": attempts}}})
    )
    (tmp_path / f"{stem}.jsonl").write_text("\n".join(_json.dumps(r) for r in rows) + "\n")
    return tmp_path / f"{stem}.jsonl", tmp_path / f"{stem}.json"


def test_an_orphaned_attempt_the_summary_excludes_is_not_read(tmp_path):
    """The log is not the cohort. The runner deliberately leaves higher-numbered
    attempts in place and ignores them when a later run uses a smaller `--attempts`
    (`runner/run.py` logs "prior record(s) with attempt index >= n ignored"). `delta`
    compares the SUMMARIES, so a row it never saw must not decide a coverage claim.

    Both failure directions matter. An orphan with activity makes an inactive
    comparison look reachable; an orphan with no telemetry voids an otherwise complete
    metric. This pins the first, which is the one that discards real evidence.
    """
    jsonl, result = _run_files(
        tmp_path,
        "r",
        attempts=2,
        rows=[
            {"task": "T", "attempt": 0, "metrics": {"tool_calls": 0.0}},
            {"task": "T", "attempt": 1, "metrics": {"tool_calls": 0.0}},
            # Orphan: a leftover from an earlier, longer run. It shows activity.
            {"task": "T", "attempt": 2, "metrics": {"tool_calls": 9.0}},
        ],
    )
    rows, note = select_attempts(jsonl, result)

    assert note["rows_selected"] == 2 and note["rows_in_file"] == 3
    assert unreachable("tool_output", activity_from_rows(rows)) == {"T"}, (
        "the orphan's activity must not make a task the summary saw as inactive look reachable"
    )


def test_a_duplicated_attempt_takes_the_LAST_row_as_the_runner_does(tmp_path):
    """Match `runner.run.load_done`, do not out-think it.

    An earlier version refused duplicates, reasoning that keeping either copy picks a
    winner with no rule behind it. There is a rule: `load_done` assigns each key into a
    dict in file order, so where a log already holds a duplicate the LAST row is the one
    the result JSON summarizes. Refusing therefore rejected a log shape the runner
    accepts, and reading the first would read an attempt `delta` never saw.

    Normal resume does not produce this shape — existing keys are resumed and skipped,
    never appended again. This is a rule for reading an already-duplicated log, not for
    one ordinary execution creates. Both copies here disagree on exactly the metric an
    exclusion turns on, which is the case where picking the wrong one is invisible.
    """
    jsonl, result = _run_files(
        tmp_path,
        "d",
        attempts=1,
        rows=[
            {"task": "T", "attempt": 0, "metrics": {"tool_calls": 0.0}},
            {"task": "T", "attempt": 0, "metrics": {"tool_calls": 7.0}},
        ],
    )
    rows, note = select_attempts(jsonl, result)

    assert note["rows_selected"] == 1 and note["rows_in_file"] == 2
    assert activity_from_rows(rows)["T"]["tool_calls"] == 7.0, "the LAST row is the run"
    assert unreachable("tool_output", activity_from_rows(rows)) == set()

    # PARITY, not a restatement of the same constant. Asserting `== 7.0` alone pins this
    # module's behaviour to a literal, so changing the RUNNER to first-wins would leave
    # it green while the two selectors silently disagreed about which row the summary
    # describes. Compare the two selectors over the same file instead.
    from runner.run import load_done

    assert load_done(jsonl, log=lambda *_: None)[("T", 0)] == rows[0], (
        "select_attempts and load_done must pick the same record, or coverage explains "
        "a different attempt than the one delta summarized"
    )


def test_a_summarized_attempt_missing_from_the_log_is_refused(tmp_path):
    """The other direction: a summary that claims attempts the log does not carry.
    Deriving from the rows that happen to be present would quietly answer a question
    about a different, smaller cohort than the one `delta` compared."""
    jsonl, result = _run_files(
        tmp_path,
        "m",
        attempts=3,
        rows=[{"task": "T", "attempt": 0, "metrics": {"tool_calls": 0.0}}],
    )
    with pytest.raises(ValueError, match="summarized but not logged"):
        select_attempts(jsonl, result)


def test_both_arms_are_required_and_both_are_read(tmp_path):
    """A cohort missing an arm is unusable, not smaller — and both arms must be read.

    Filtering the supplied pairs to the ones that happen to exist derived coverage from
    one arm: with the baseline log gone it read the candidate alone, reported NO error,
    and still emitted `unreachable_proven` — a claim about a comparison made from half
    of it. The activity here is deliberately ASYMMETRIC, because a fixture where both
    arms agree cannot tell a two-arm read from a last-arm-only one.
    """
    import json as _json

    from loop.artifacts import Candidate
    from loop.validate import coverage_note

    def write(stem, tool_calls, cfg):
        (tmp_path / f"{stem}.json").write_text(
            _json.dumps({"tasks": {"T": {"split": "held_in", "attempts": 1}}})
        )

        def row(attempt, calls):
            return _json.dumps(
                {
                    "task": "T",
                    "attempt": attempt,
                    "runner_sha": "abc123",
                    "config_version": cfg,
                    "model": f"model-{cfg}",
                    "metrics": {"tool_calls": calls},
                }
            )

        # An ORPHAN at attempt 1, left by an earlier longer run and excluded by the
        # summary's `attempts: 1`. Without it raw and selected rows are identical, and a
        # hash taken over the raw file is indistinguishable from one over the selected
        # rows — the fixture, not the code, would be deciding the test.
        (tmp_path / f"{stem}.jsonl").write_text(row(0, tool_calls) + "\n" + row(1, 99.0) + "\n")
        return (tmp_path / f"{stem}.jsonl", tmp_path / f"{stem}.json")

    # Asymmetric on BOTH axes. Symmetric fixtures let a mutant hard-code the provenance
    # values and stay green, because every arm reports the same thing.
    base, cand = write("base", 1.0, cfg=7), write("cand", 0.0, cfg=8)
    candidate = Candidate(
        id="x",
        cluster_id="CL",
        proposer="p",
        proposer_detail="d",
        fields={"tool_output": {"old": 0, "new": 1}},
        rationale="r",
        expected_effect="e",
        regression_risk="g",
    )

    both = coverage_note(candidate, {"T": -1.0}, [base, cand])
    # T made a tool call in the baseline arm, so the pair does NOT prove it unreachable.
    # Reading only the candidate arm would wrongly call it proven.
    assert both["unreachable_proven"] == {}, "a single active arm is enough to reach T"
    assert both["evidence"] == {"T": -1.0}
    assert {f["file"].split("/")[-1] for f in both["cohort"]["files"]} == {
        "base.jsonl",
        "cand.jsonl",
    }
    # Provenance derived from each arm, not restated as a literal, and the hash
    # recomputed over the SELECTED rows — hashing the raw file including ignored
    # orphans produces a 64-char value that differs between arms and would pass a
    # length-and-difference check while binding the wrong content.
    import hashlib

    by_name = {f["file"].split("/")[-1]: f for f in both["cohort"]["files"]}
    for (jsonl, result_json), cfg in ((base, 7), (cand, 8)):
        selected, _ = select_attempts(jsonl, result_json)
        expected = hashlib.sha256(
            _json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        got = by_name[jsonl.name]
        assert got["selected_sha256"] == expected, "hash must bind the SELECTED rows"
        assert got["config_version"] == [cfg] and got["model"] == [f"model-{cfg}"]
        assert got["runner_sha"] == ["abc123"]

    (tmp_path / "base.jsonl").unlink()
    missing = coverage_note(candidate, {"T": -1.0}, [base, cand])
    assert "cohort incomplete" in missing["error"]
    assert "unreachable_proven" not in missing, "a partial cohort must yield no claim"


def test_the_recorded_path_never_carries_an_absolute_location(tmp_path, monkeypatch):
    """Two failure modes, one on each side, and both were shipped.

    A path computed relative to the file's GRANDPARENT turned an external
    `/archive/results/base.jsonl` into `results/base.jsonl` — reads as a repo file,
    points at the wrong one. Recording the absolute path instead fixed that and broke a
    harder rule: this record is written under `iterations/` in a public repo, and
    AGENTS.md forbids absolute paths there because they carry a real machine's home
    directory. The source-file grep that enforces that rule cannot see a value produced
    at runtime, so only a test can.
    """
    import json as _json

    import loop.observed_coverage as oc

    def build(root, stem):
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{stem}.json").write_text(_json.dumps({"tasks": {"T": {"attempts": 1}}}))
        (root / f"{stem}.jsonl").write_text(
            _json.dumps({"task": "T", "attempt": 0, "metrics": {}}) + "\n"
        )
        return root / f"{stem}.jsonl", root / f"{stem}.json"

    inside = build(tmp_path / "repo" / "results", "in")
    outside = build(tmp_path / "elsewhere", "out")
    monkeypatch.setattr(oc, "REPO_ROOT", (tmp_path / "repo").resolve())

    _, in_note = oc.select_attempts(*inside)
    assert in_note["file"] == "results/in.jsonl", "under the root, a stable relative path"
    assert "external" not in in_note

    _, out_note = oc.select_attempts(*outside)
    assert out_note == {**out_note, "file": "out.jsonl", "external": True}
    blob = _json.dumps(out_note)
    assert str(tmp_path) not in blob and "/Users/" not in blob and "/home/" not in blob
    assert not out_note["file"].startswith("results/"), "must not claim to be a repo file"


def test_the_causal_verdict_on_iteration_3_itself():
    """The case this rule exists for, against the real recorded data.

    It does NOT rescue iteration 3, and that is the honest outcome rather than a
    shortfall. Removing the impossible attributions halves Δ_in (−0.1176 → −0.0588) and
    clears the catastrophic veto entirely — A1 was its only entry, and A1 builds an
    agent with no tool registry against a candidate that edited `tool_output`. What
    remains is G5 (−0.67) and B2 (−0.33), the two held-in tasks that DO make tool calls
    and therefore cannot be proven unreachable from telemetry.

    So the rejection now rests on evidence instead of on movements the knob could not
    have caused. That is the whole claim; anything more would mean trusting a
    hand-authored exclusion to decide acceptance, which is what the derived map exists
    to avoid — an audit found six such rows false at the same time.
    """
    import json as _json

    from loop.artifacts import Candidate
    from loop.validate import causal_verdict, coverage_note
    from runner.delta import delta

    base = _json.loads(Path("results/baseline-r5-v7.json").read_text())
    cand = _json.loads(Path("results/cand-tool-output-offload.json").read_text())
    d = delta(base, cand)
    candidate = Candidate(
        id="tool-output-offload",
        cluster_id="CL",
        proposer="p",
        proposer_detail="d",
        fields={"tool_output": {"old": 0, "new": 1}},
        rationale="r",
        expected_effect="e",
        regression_risk="g",
    )
    coverage = coverage_note(candidate, d["per_task"], COHORT_PAIRS)
    v = causal_verdict(d, coverage, base)

    assert {"A1", "G4"} <= set(v["excluded"]), "the two no-tool-registry tasks"
    assert v["delta_in"] > d["delta_in"], "removing impossible attributions must help"
    assert round(v["delta_in"], 4) == -0.0588 and round(d["delta_in"], 4) == -0.1176
    # The spurious hard veto is gone: A1 was the only catastrophic entry.
    assert d["catastrophic_regressions"] and not v["catastrophic_regressions"]
    # Still rejected, on the two tasks that genuinely make tool calls.
    assert not v["accepted"] and not v["raw"]["accepted"]
    assert {"G5", "B2"}.isdisjoint(v["excluded"])
    # The raw verdict is kept, never discarded.
    assert v["raw"]["delta_in"] == d["delta_in"]


def test_only_proof_grade_exclusions_reach_the_verdict():
    """An evidence-grade exclusion must NOT zero a movement.

    `compaction`'s absence of activity can be created by the same knob's
    `trigger_fraction`, so treating it as impossible would discard real movement. And
    the zeroing must preserve the DENOMINATOR — dropping the task instead changes the
    split mean for a second, unrelated reason and makes two candidates that exclude
    different tasks incomparable.
    """
    from loop.validate import causal_verdict

    base = {"tasks": {n: {"split": "held_in"} for n in ("X", "Y", "Z")}}
    d = {
        "per_task": {"X": -1.0, "Y": -1.0, "Z": 0.0},
        "delta_in": -2 / 3,
        "delta_ho": 0.0,
        "accepted": False,
        "catastrophic_regressions": {"X": -1.0, "Y": -1.0},
    }
    v = causal_verdict(
        d, {"unreachable_proven": {"X": -1.0}, "unreachable_probable": {"Y": -1.0}}, base
    )

    assert v["per_task"] == {"X": 0.0, "Y": -1.0, "Z": 0.0}, "probable must survive"
    assert v["delta_in"] == -1 / 3, "denominator stays 3, not 2"
    assert list(v["catastrophic_regressions"]) == ["Y"], "veto skips proven only"


def test_the_record_takes_the_CAUSAL_verdict_when_the_two_disagree(tmp_path):
    """The decision itself, end to end, on a case where raw and causal differ.

    This is the whole change: the raw verdict was applied correctly to real
    measurements and still rejected iteration 3 on movements the edited knob could not
    have caused. Recording the split beside a verdict the noise still decided left that
    in place, so the causal verdict decides and the raw one is kept as evidence.

    Constructed so the two genuinely disagree — a task with no tool calls collapses,
    which sinks raw Δ_in, while a held-out task the knob CAN reach improves.
    """
    import json as _json

    from loop.artifacts import Candidate, write_validation_record
    from loop.validate import validate_candidate

    def write(stem, fracs, calls):
        (tmp_path / f"{stem}.json").write_text(
            _json.dumps(
                {
                    "fingerprint": {"runner_sha": "deadbeef", "model": "m", "gemma_sha": "c0ffee"},
                    "tasks": {
                        "NOTOOLS": {"split": "held_in", "attempts": 1, "pass_fraction": fracs[0]},
                        "REACHED": {"split": "held_in", "attempts": 1, "pass_fraction": fracs[1]},
                        "GAIN": {"split": "held_out", "attempts": 1, "pass_fraction": fracs[2]},
                    },
                }
            )
        )
        (tmp_path / f"{stem}.jsonl").write_text(
            "\n".join(
                _json.dumps({"task": n, "attempt": 0, "metrics": {"tool_calls": calls[n]}})
                for n in ("NOTOOLS", "REACHED", "GAIN")
            )
            + "\n"
        )
        return (tmp_path / f"{stem}.jsonl", tmp_path / f"{stem}.json")

    calls = {"NOTOOLS": 0.0, "REACHED": 3.0, "GAIN": 3.0}
    write("base", (1.0, 1.0, 0.0), calls)
    write("cand", (0.0, 1.0, 1.0), calls)  # NOTOOLS collapses; GAIN improves

    import loop.validate as validate_mod

    candidate = Candidate(
        id="x",
        cluster_id="CL",
        proposer="p",
        proposer_detail="d",
        fields={"tool_output": {"old": 0, "new": 1}},
        rationale="r",
        expected_effect="e",
        regression_risk="g",
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(validate_mod, "require_clean_tree", lambda *a, **k: None)
        mp.setattr(validate_mod, "revert_config", lambda *a, **k: None)
        mp.setattr(validate_mod, "apply_candidate", lambda *a, **k: {"version": 8})
        record = validate_candidate(
            candidate,
            baseline_path=tmp_path / "base.json",
            label="cand",
            run_runner=lambda *a, **k: None,
            run_gates=lambda *a, **k: {"passed": True, "checks": {}},
            results_dir=tmp_path,
            carbon_root=tmp_path,
            log=lambda *_: None,
        )

    written = _json.loads(write_validation_record(record, tmp_path / "out.json").read_text())
    assert written["causal"]["raw"]["accepted"] is False, "raw sinks on the unreachable task"
    assert written["causal"]["accepted"] is True, "causal zeroes it and the gain carries"
    assert written["accepted"] is True, "the RECORD must take the causal verdict"
    assert "NOTOOLS" in written["causal"]["excluded"]
    # The raw numbers survive as evidence rather than being overwritten.
    assert written["causal"]["raw"]["delta_in"] < written["causal"]["delta_in"]
