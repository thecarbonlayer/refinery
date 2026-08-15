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

    assert set(split["unreachable"]) == {"A1", "G1", "G4"}
    # G1 sits in the unreachable half too, and that matters more than it looks: it moved
    # UP. An earlier pass at this analysis dropped only the tasks that hurt the
    # candidate and got Δ_in = 0.0000, which accepts. Dropping symmetrically leaves
    # B2 and G5 and a negative sum, which rejects. A partition that filtered by sign
    # would be a way to fit any verdict you wanted.
    assert split["unreachable"]["G1"] > 0
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
    assert REQUIRES["compaction_prompt"].airtight

    assert unreachable("compaction", activity)
    assert unreachable("compaction", activity, airtight_only=True) == set()
    assert unreachable("compaction_prompt", activity, airtight_only=True)


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
        assert contradicted_observers(activity)["tool_output"] == ["A1"]


def test_the_review_queue_honours_the_argued_exclusions(activity):
    """A queue that re-raises settled decisions every run is one nobody reads, and the
    single genuinely unreviewed entry hides among them. C1/C2 are the clearest case:
    they read the RAW tool result on purpose, so no truncation value can touch them."""
    queue = unlisted_with_activity(activity)
    for task in ("A2", "C1", "C2"):
        assert task in DELIBERATE_NON_OBSERVERS["tool_output"]
        assert task not in queue.get("tool_output", [])

    # And it must still surface the one an audit found by hand: G5 was added as "the
    # observer that made compaction-v4 measurable" and never entered the compaction rows.
    assert "G5" in queue["compaction"]
    assert "G5" in queue["compaction_prompt"]


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
    assert set(coverage_note(candidate("tool_output"), per_task)["unreachable"]) == {"A1"}
    # `system_prompt` reaches every live task, so pairing it removes every exclusion.
    both = coverage_note(candidate("tool_output", "system_prompt"), per_task)
    assert both["unreachable"] == {}
    assert set(both["evidence"]) == {"A1", "B2"}
