"""PR eligibility (contract §5 amendment): today `loop.cli pr` reads only the
validation record, so a rule-section candidate whose FIRST decision is CONFIRM can
never reach a PR even after a fresh confirmation ACCEPT (`validate_candidate` only
ever sets `accepted=True` for a rule outcome of ACCEPT, which `evaluate()` never
returns — CONFIRM candidates leave `accepted=False` on the validation record
forever). `_pr_eligible_record` closes that: a candidate is PR-eligible when its
validation record already says accepted, OR when its rule outcome is CONFIRM and a
confirmation record for the SAME candidate (matching id and stage) has outcome
ACCEPT. Every other case must refuse exactly as loudly as before, unweakened.
"""

from __future__ import annotations

import json

import pytest

from loop.acceptance import ACCEPT, CONFIRM, REJECT
from loop.artifacts import (
    Candidate,
    ConfirmationRecord,
    ValidationRecord,
    write_confirmation_record,
)

CANDIDATE = Candidate(
    id="cmp-1",
    cluster_id="CL-1",
    proposer="Fable",
    proposer_detail="test",
    fields={"compaction": {"old": 1, "new": 2}},
    rationale="r",
    expected_effect="e",
    regression_risk="g",
)


def _record(**over) -> ValidationRecord:
    base = dict(
        candidate_id=CANDIDATE.id,
        label="cand-x",
        accepted=False,
        delta_in=0.1,
        delta_ho=0.1,
        rule={},
    )
    base.update(over)
    return ValidationRecord(**base)


def _confirmation(**over) -> ConfirmationRecord:
    base = dict(
        candidate_id=CANDIDATE.id,
        baseline_label="b",
        candidate_label="c",
        attempts_per_task_per_arm=10,
        confirm_set=("A1",),
        first_decision={},
        confirmation={"outcome": ACCEPT},
        per_task={},
        finding="f",
    )
    base.update(over)
    return ConfirmationRecord(**base)


def test_an_already_accepted_record_passes_through_unchanged(tmp_path):
    from loop.cli import _pr_eligible_record

    record = _record(accepted=True, rule={"applied": True, "outcome": ACCEPT})
    out = _pr_eligible_record(tmp_path, CANDIDATE, record)
    assert out is record


def test_a_plain_reject_with_no_confirm_falls_through_unweakened(tmp_path):
    """A candidate that was simply REJECTED (or never reached the calibrated rule —
    `rule == {}`) never had anything to confirm. `_pr_eligible_record` must not
    invent a refusal here; it returns the record as-is so `open_pr`'s own
    `if not record.accepted: raise ValueError(...)` fires exactly as it always has."""
    from loop.cli import _pr_eligible_record

    record = _record(accepted=False, rule={"applied": True, "outcome": REJECT})
    out = _pr_eligible_record(tmp_path, CANDIDATE, record)
    assert out is record
    assert out.accepted is False


def test_confirm_outcome_with_no_confirmation_record_refuses_naming_whats_missing(tmp_path):
    from loop.cli import _pr_eligible_record

    record = _record(accepted=False, rule={"applied": True, "outcome": CONFIRM})
    with pytest.raises(SystemExit, match="CONFIRM") as excinfo:
        _pr_eligible_record(tmp_path, CANDIDATE, record)
    assert "confirmation" in str(excinfo.value).lower()


def test_confirm_outcome_with_a_confirmed_accept_promotes_to_eligible(tmp_path):
    from loop.cli import _pr_eligible_record

    record = _record(accepted=False, rule={"applied": True, "outcome": CONFIRM})
    write_confirmation_record(
        _confirmation(confirmation={"outcome": ACCEPT}),
        tmp_path / f"confirmation-{CANDIDATE.id}.json",
    )
    out = _pr_eligible_record(tmp_path, CANDIDATE, record)
    assert out.accepted is True
    # nothing else about the record is disturbed by the promotion
    assert out.delta_in == record.delta_in and out.rule == record.rule


def test_confirm_outcome_with_a_confirmed_reject_refuses(tmp_path):
    from loop.cli import _pr_eligible_record

    record = _record(accepted=False, rule={"applied": True, "outcome": CONFIRM})
    write_confirmation_record(
        _confirmation(confirmation={"outcome": REJECT}),
        tmp_path / f"confirmation-{CANDIDATE.id}.json",
    )
    with pytest.raises(SystemExit, match=REJECT):
        _pr_eligible_record(tmp_path, CANDIDATE, record)


def test_a_confirmation_recorded_for_a_different_candidate_is_refused(tmp_path):
    """The confirmation record is read by a fixed filename (`confirmation-<id>.json`)
    that already names the candidate, but the record's OWN `candidate_id` field is
    the thing actually trusted here — a copy-pasted or hand-edited file must not
    silently promote the wrong candidate."""
    from loop.cli import _pr_eligible_record

    record = _record(accepted=False, rule={"applied": True, "outcome": CONFIRM})
    write_confirmation_record(
        _confirmation(candidate_id="some-other-candidate", confirmation={"outcome": ACCEPT}),
        tmp_path / f"confirmation-{CANDIDATE.id}.json",
    )
    with pytest.raises(SystemExit, match="some-other-candidate"):
        _pr_eligible_record(tmp_path, CANDIDATE, record)


def test_a_confirmation_of_the_wrong_stage_is_refused(tmp_path):
    from loop.cli import _pr_eligible_record

    record = _record(accepted=False, rule={"applied": True, "outcome": CONFIRM})
    path = tmp_path / f"confirmation-{CANDIDATE.id}.json"
    raw = _confirmation(confirmation={"outcome": ACCEPT}).to_json()
    raw["stage"] = "some_other_stage"
    path.write_text(json.dumps(raw))
    with pytest.raises(SystemExit, match="some_other_stage"):
        _pr_eligible_record(tmp_path, CANDIDATE, record)
