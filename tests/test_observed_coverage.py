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
    contradicted_observers,
    observed_activity,
    partition_deltas,
    unlisted_with_activity,
    unreachable,
)

# The exact pair iteration 3 compared. NOT `result_files()`: pooling every log mixes
# runner versions, config versions and partial runs into one claim, and under the
# now-strict "fully recorded" rule that mixture leaves almost everything unknown. A
# coverage claim is only meaningful about a cohort, so the fixture names one.
COHORT = ["results/baseline-r5-v7.jsonl", "results/cand-tool-output-offload.jsonl"]


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


def test_the_review_queue_honours_the_argued_exclusions(activity):
    """A queue that re-raises settled decisions every run is one nobody reads, and the
    single genuinely unreviewed entry hides among them. C1/C2 are the clearest case:
    they read the RAW tool result on purpose, so no truncation value can touch them."""
    queue = unlisted_with_activity(activity)
    for task in ("A2", "C1", "C2", "G5"):
        assert task in DELIBERATE_NON_OBSERVERS["tool_output"]
        assert task not in queue.get("tool_output", [])

    # G5 is the entry this queue was built to find: added as "the observer that made
    # compaction-v4 measurable" and never entered the compaction rows. It was queued,
    # it was argued, and it is now IN those rows — so it must have left the queue.
    # Asserting it is still there would pin the bug rather than the mechanism.
    assert "G5" in KNOB_COVERAGE["compaction"]["observers"]
    assert "G5" not in queue.get("compaction", [])
    # GUARD, not miner, on both compaction rows. Carbon's checkpoint lifts file paths out
    # of `tool_calls` deterministically and reattaches them independently of the summary
    # prose — the same fact that makes G5 a good STRATEGY observer means better wording
    # cannot mine it. It is also 3/3 in the v7 baseline, so there is nothing to turn
    # green. It was first added as a miner; that was wrong on both counts.
    for row in ("compaction", "compaction_prompt"):
        assert "G5" in KNOB_COVERAGE[row]["guards"]
        assert "G5" not in KNOB_COVERAGE[row]["miners"]

    # The mechanism must still be able to fire, or emptying the queue would look
    # identical to breaking it.
    stripped = {
        role: tuple(t for t in tasks if t != "G4")
        for role, tasks in KNOB_COVERAGE["compaction"].items()
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(KNOB_COVERAGE, "compaction", stripped)
        assert "G4" in unlisted_with_activity(activity)["compaction"]


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

    from pathlib import Path

    paths = [Path(f) for f in COHORT]
    per_task = {"A1": -1.0, "B2": -0.5}
    # `tool_output` alone cannot reach A1 (no tool registry at all).
    note = coverage_note(candidate("tool_output"), per_task, paths)
    assert set(note["unreachable_proven"]) == {"A1"}
    # `system_prompt` reaches every live task, so pairing it removes every exclusion.
    both = coverage_note(candidate("tool_output", "system_prompt"), per_task, paths)
    assert both["unreachable_proven"] == {}
    assert both["unreachable_probable"] == {}
    assert set(both["evidence"]) == {"A1", "B2"}


def test_the_tuning_floor_does_not_pretend_to_be_enforced():
    """The floor constrains nothing, and the test says so rather than blessing it.

    A previous version asserted F1 AS a permitted exception — one listed observer whose
    listing rests on a budget below the floor. That froze an incoherent contract instead
    of detecting one, which is the failure mode this whole module was built to catch.

    What is actually true: nothing imports the constant outside this test,
    `proposal_surface()` still publishes carbon's `positive: true`, and
    `apply_candidate()` still delegates to carbon, which accepts any positive budget. So
    the loop can propose 20 today. Until one of the two documented routes is taken —
    enforce it in the proposal path, or abandon it — this asserts only that the number
    has no teeth, so that giving it teeth is what turns the test red.
    """
    import loop.config_edit as config_edit
    from loop.knob_coverage import MEASURED_BREAK_BUDGETS, TOOL_OUTPUT_TUNING_FLOOR

    surface = config_edit.proposal_surface()["editable"]["tool_output"]
    budget = surface["parameters"]["budget"]
    assert budget == {"type": "int", "positive": True}, (
        "the published proposal surface now carries a bound. If the floor was enforced, "
        "delete this test and add ones covering proposal_surface() and apply_candidate(), "
        "then settle F1's row — its listing rests on a budget of 20."
    )
    # Every measured task, F1 included, breaks below the floor. Under the domain the
    # pipeline actually enforces they are therefore all alike, and F1 being listed while
    # the other five are not is an inconsistency the floor does not currently resolve.
    assert all(b < TOOL_OUTPUT_TUNING_FLOOR for b in MEASURED_BREAK_BUDGETS.values())


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
            "observed_activity",
            lambda paths: {
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
        note = validate_mod.coverage_note(cand, {"X": -1.0, "Y": 0.5}, [Path(COHORT[0])])

    assert note["evidence"] == {"Y": 0.5}, "X reached neither knob and must not be evidence"
    assert note["unreachable_proven"] == {}, "only ONE knob proves X, so the pair cannot"
    assert note["unreachable_probable"] == {"X": -1.0}
