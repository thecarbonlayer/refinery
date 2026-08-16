"""Pins for the derived coverage map.

Several cases read the REAL recorded runs rather than a fixture. The module exists
because of one specific misreading of those runs, and a synthetic fixture would let the
module keep passing while the fact it was built on stopped being true.
"""

from __future__ import annotations

import json

import pytest

from loop.knob_coverage import DELIBERATE_NON_OBSERVERS, KNOB_COVERAGE
from loop.observed_coverage import (
    REQUIRES,
    contradicted_observers,
    observed_activity,
    partition_deltas,
    result_files,
    unlisted_with_activity,
    unreachable,
)


@pytest.fixture(scope="module")
def activity():
    files = result_files()
    assert files, "no recorded runs to derive coverage from"
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

    per_task = {"A1": -1.0, "B2": -0.5}
    # `tool_output` alone cannot reach A1 (no tool registry at all).
    note = coverage_note(candidate("tool_output"), per_task)
    assert set(note["unreachable_proven"]) == {"A1"}
    # `system_prompt` reaches every live task, so pairing it removes every exclusion.
    both = coverage_note(candidate("tool_output", "system_prompt"), per_task)
    assert both["unreachable_proven"] == {}
    assert both["unreachable_probable"] == {}
    assert set(both["evidence"]) == {"A1", "B2"}


def test_the_plausibility_floor_is_derived_and_applied_consistently():
    """The floor exists because two criteria were in use, one written and one not.

    Written: an observer is a task "whose verdict some legal value can move". Taken
    literally that admits nearly every tool-using task, since `tool_output.budget` is
    any positive integer — a budget of 6 breaks D1. Unwritten, in prose: "only E1 and
    E2 are sensitive at PLAUSIBLE budgets". The second one chose the rows.

    The floor is a POLICY number, not a derivation. A first version anchored it to
    carbon's `SHRINK_MIN_BUDGET` and claimed carbon refuses budgets below it; that was
    false — the constant belongs to overflow RECOVERY, and normal truncation uses the
    configured budget directly. Deliberately NOT asserted against any carbon constant
    now, because tracking one would restate the same false derivation as a test.

    What is asserted is that every measured task sits on the side of the floor its
    listing implies. F1 is the one exception and is asserted AS an exception, so it
    stays visible until someone decides it — a floor that quietly tolerated the
    contradiction would be decoration.
    """
    from loop.knob_coverage import (
        _TOOL_RESULT_READERS,
        MEASURED_BREAK_BUDGETS,
    )
    from loop.knob_coverage import (
        TOOL_OUTPUT_TUNING_FLOOR as floor,
    )

    below = {t for t, b in MEASURED_BREAK_BUDGETS.items() if b < floor}
    listed_but_below = below & set(_TOOL_RESULT_READERS)
    assert listed_but_below == {"F1"}, (
        "exactly one listed observer is known to break only below the floor. If this "
        "set grew, a new row was added on the criterion the floor rejects; if it "
        "emptied, F1 was resolved and this assertion should go with it. Either way it "
        f"is a decision, not a test fix. Currently: {sorted(listed_but_below)}"
    )
    # And the ones we measured and did NOT list must all be below it — otherwise the
    # floor is not what kept them out and the real reason is unrecorded.
    for task in ("B1", "B2", "B3", "D1", "D2"):
        assert task not in _TOOL_RESULT_READERS
        assert MEASURED_BREAK_BUDGETS[task] < floor


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
    written = _json.loads(_json.dumps(record.to_json()))

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
