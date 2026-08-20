"""The final verification batch: digests are mandatory, power is like-for-like, headers true.

Three findings from the verification pass on the closing wave, each reproduced here as
the attack or the contradiction that found it:

N1 (blocking). The stage-binding digests were checked ONLY when the record said they
   should be. Both bindings keyed on `first.raw["regime"]`, and the decision-digest
   check additionally skipped when `raw["decision_digest"]` was absent. A record is a
   JSON file on disk, so "check this if the file asks to be checked" is not a check:
   deleting a key turned the gate off. The auditor got two end-to-end ACCEPTs out of a
   pair that honestly REJECTs, by shrinking `improved_tasks` and then deleting first
   the digest and then the regime. Both bypasses are pinned below. The fix is that both
   digests are MANDATORY whenever a calibration is in hand — an absent digest refuses
   exactly like a mismatched one — and the trigger is the calibration object, which
   comes from the artifact on disk and not from the record being judged.

N2. `fitness.power.end_to_end` published detection rates marginal over baselines, while
   `fitness.false_confirm` published its rate BOTH marginally and conditional on the
   designated baseline arm. A reader comparing the two was comparing a conditional
   number against a marginal one. Each end-to-end row now carries both.

N3. A record promoted by a confirmation ACCEPT still rendered `PENDING_CONFIRMATION` in
   its own PR header, directly above the confirmation section stating the ACCEPT.
"""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction
from pathlib import Path

import pytest

from loop.acceptance import ACCEPT, CONFIRM, REJECT, Decision, confirmed
from loop.artifacts import ConfirmationRecord, ValidationRecord, write_confirmation_record
from tests.test_p2b_closing import (
    CANDIDATE,
    CLUSTER,
    REAL_MODEL,
    SUPPORTED,
    _calibrated_record,
    _first_decision,
    _results,
    installed_model,
    load_calibration,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTEMPTS = dict.fromkeys(SUPPORTED, 10)

# A confirmation pair on which the ORIGINAL three-carrier claim honestly fails: A1
# repeats beyond its own 2/5 quantile at 10v10, G4 and G5 do not move at all. Every
# other gate holds — both supported-set means are non-negative, no guard drops, nothing
# collapses — so the ONLY thing standing between this pair and an ACCEPT is the rule
# that every named carrier must repeat. That makes it the exact pair a record-edit
# attack wants: shrink the claim to the one carrier that did repeat.
HONEST_BASE = {"A1": 2, "G4": 5, "G5": 8, "G2": 5}
HONEST_CAND = {"A1": 9, "G4": 5, "G5": 8, "G2": 5}


def _honest_first(tmp_path) -> tuple[Decision, dict, dict]:
    record, _, _ = _calibrated_record(tmp_path)
    first = _first_decision(record)
    assert first.outcome == CONFIRM
    assert first.improved_tasks == ("A1", "G4", "G5"), first.improved_tasks
    assert first.raw["decision_digest"] and first.raw["calibration_digest"]
    return first, _results(HONEST_BASE, ATTEMPTS), _results(HONEST_CAND, ATTEMPTS)


def _edited(first: Decision, *, carriers=("A1",), drop=()) -> Decision:
    """The record as a tamperer leaves it: fewer carriers, and any binding key they
    chose to delete on the way out."""
    raw = {k: v for k, v in (first.raw or {}).items() if k not in drop}
    return dataclasses.replace(first, improved_tasks=tuple(carriers), raw=raw)


# ---------------------------------------------------------------------------
# N1: the two auditor bypasses
# ---------------------------------------------------------------------------


def test_the_honest_record_still_reaches_a_verdict_on_this_pair(tmp_path):
    """The control, and it has to come first: the pair is judged, not refused, and the
    honest answer is REJECT because two of the three named carriers did not repeat. A
    gate that refused everything would pass both attack tests below while proving
    nothing."""
    cal = load_calibration(REAL_MODEL)
    first, base, cand = _honest_first(tmp_path)

    decision = confirmed(first, base, cand, calibration=cal)

    assert decision.outcome == REJECT
    assert any("did not repeat on G4" in r for r in decision.reasons), decision.reasons
    assert any("did not repeat on G5" in r for r in decision.reasons), decision.reasons


def test_shrinking_the_carriers_and_deleting_the_digest_refuses(tmp_path):
    """AUDITOR BYPASS A. Three carriers become one — the one that did repeat — and the
    digest that would have caught it is deleted in the same edit. Before the fix the
    check read `if recorded is not None`, so deleting the key skipped it, and this pair
    came back ACCEPT: an end-to-end acceptance of a claim nobody ever made.

    An absent digest is not "nothing to check". It is a calibrated record that never
    bound itself to its own claim, which is precisely the record this check exists to
    stop being trusted."""
    cal = load_calibration(REAL_MODEL)
    first, base, cand = _honest_first(tmp_path)
    attacked = _edited(first, drop=("decision_digest",))

    with pytest.raises(ValueError, match="digest"):
        confirmed(attacked, base, cand, calibration=cal)


def test_deleting_the_regime_as_well_still_refuses(tmp_path):
    """AUDITOR BYPASS B. The same edit, plus `regime` deleted. Both bindings used to key
    on `raw["regime"] == "section_calibration"`, so removing it turned BOTH of them off
    while `loop.cli.run_confirmation` went on loading the calibration from the artifact
    and judging against it — the calibration still used, and never checked.

    The trigger is the calibration in hand now, never a field of the record. A record
    cannot switch off a check by forgetting to ask for it — and deleting the regime hits
    a check of its own: a calibration in hand beside a record that does not claim to
    have been judged under one is a contradiction, and the contradiction is the
    finding."""
    cal = load_calibration(REAL_MODEL)
    first, base, cand = _honest_first(tmp_path)
    attacked = _edited(first, drop=("decision_digest", "regime"))

    with pytest.raises(ValueError, match="does not record that it was judged under one"):
        confirmed(attacked, base, cand, calibration=cal)


def test_deleting_the_calibration_digest_alone_refuses(tmp_path):
    """The same principle on the other binding: the artifact-swap check must not be
    switchable off either, by deleting its digest or by deleting the regime."""
    cal = load_calibration(REAL_MODEL)
    first, base, cand = _honest_first(tmp_path)

    with pytest.raises(ValueError, match="calibration digest"):
        confirmed(
            _edited(first, carriers=first.improved_tasks, drop=("calibration_digest",)),
            base,
            cand,
            calibration=cal,
        )
    with pytest.raises(ValueError, match="does not record that it was judged under one"):
        confirmed(
            _edited(first, carriers=first.improved_tasks, drop=("calibration_digest", "regime")),
            base,
            cand,
            calibration=cal,
        )


def test_the_cli_reload_refuses_both_bypasses_too(tmp_path):
    """The same two attacks through the real reload path, where a tamperer actually
    works: edit the committed JSON, then run `confirm`.

    `require_binding` is how the caller says "this section is decided by a null model,
    so its records must be bound". `run_confirmation` derives it from the CANDIDATE's
    section, which comes from code and candidates.json — never from the record, which
    is the thing under suspicion."""
    from loop.cli import load_first_decision
    from loop.validate import CALIBRATION_REQUIRED, candidate_section

    assert candidate_section(CANDIDATE) in CALIBRATION_REQUIRED, (
        "fixture precondition: this candidate's section is the one that requires binding"
    )
    record, _, _ = _calibrated_record(tmp_path)
    it_dir = tmp_path / "iter"
    it_dir.mkdir()
    path = it_dir / f"validation-{CANDIDATE.id}.json"
    raw = record.to_json()
    path.write_text(json.dumps(raw))
    honest = load_first_decision(it_dir, CANDIDATE.id, require_binding=True)
    assert honest.improved_tasks == ("A1", "G4", "G5")

    for drop in (["decision_digest"], ["decision_digest", "regime"]):
        attacked = json.loads(json.dumps(raw))
        attacked["rule"]["improved_tasks"] = ["A1"]
        for key in drop:
            attacked["rule"]["raw"].pop(key, None)
        path.write_text(json.dumps(attacked))
        with pytest.raises(SystemExit, match="digest"):
            load_first_decision(it_dir, CANDIDATE.id, require_binding=True)


def test_a_calibration_required_section_cannot_be_confirmed_uncalibrated(tmp_path, monkeypatch):
    """The residual half of bypass B: with `regime` deleted, the old guard that refused
    an uncalibrated confirmation of a calibrated claim also stopped firing. The section
    is the thing that requires a calibration, and the section comes from the CANDIDATE,
    not from the record — so that is what the refusal keys on."""
    import loop.validate as validate_mod
    from loop.cli import run_confirmation

    record, _, _ = _calibrated_record(tmp_path)
    it_dir = tmp_path / "iter"
    it_dir.mkdir()
    raw = record.to_json()
    raw["rule"]["raw"].pop("regime", None)
    raw["candidate_fields"] = dict(CANDIDATE.fields)
    (it_dir / f"validation-{CANDIDATE.id}.json").write_text(json.dumps(raw))
    monkeypatch.setitem(validate_mod._SECTION_MODEL, "compaction", tmp_path / "gone.json")

    def fake_arm(label, only, attempts):
        return _results(HONEST_BASE, ATTEMPTS)

    with pytest.raises(SystemExit, match="calibrat"):
        run_confirmation(
            CANDIDATE,
            it_dir,
            baseline_label="cb",
            candidate_label="cc",
            attempts=10,
            carbon_root=tmp_path,
            run_runner=fake_arm,
            results_dir=tmp_path / "no-results",
            log=lambda *_: None,
        )


def test_an_honest_calibrated_confirmation_still_accepts(tmp_path):
    """Mandatory is not the same as impossible. A record written by `evaluate()` and
    left alone carries both digests, and a pair on which every carrier really does
    repeat still reaches ACCEPT."""
    cal = load_calibration(REAL_MODEL)
    record, _, _ = _calibrated_record(tmp_path)
    first = _first_decision(record)
    base = _results({"A1": 3, "G4": 1, "G5": 3, "G2": 3}, ATTEMPTS)
    cand = _results({"A1": 10, "G4": 7, "G5": 10, "G2": 9}, ATTEMPTS)

    assert confirmed(first, base, cand, calibration=cal).outcome == ACCEPT


def test_an_uncalibrated_confirmation_needs_no_digests_at_all(tmp_path):
    """Byte-identity, restated for this change: the digests are a property of the
    CALIBRATED regime. An uncalibrated first decision has none, is asked for none, and
    is confirmed exactly as it always was."""
    from loop.acceptance import evaluate

    base = _results({"A1": 4, "G4": 1, "G5": 4, "G2": 3}, ATTEMPTS)
    cand = _results({"A1": 9, "G4": 5, "G5": 9, "G2": 8}, ATTEMPTS)
    first = evaluate(base, cand)
    assert first.outcome == CONFIRM
    assert "decision_digest" not in first.raw and "calibration_digest" not in first.raw

    assert confirmed(first, base, cand).outcome in {ACCEPT, REJECT}


# ---------------------------------------------------------------------------
# N2: conditional power, like-for-like with the conditional false-CONFIRM rate
# ---------------------------------------------------------------------------

# The verifier's own independent enumeration, quoted to four/five places. These are the
# numbers this artifact has to reproduce, computed by someone else's code from the same
# definition: stage 1 against the DESIGNATED baseline arm's recorded counts, stage 2
# fresh on both sides.
VERIFIER_CONDITIONAL = {
    ("single_carrier", "G2"): 0.02128,
    ("single_carrier", "A1"): 0.00105,
    ("uniform", "3/10"): 0.11791,
    ("uniform", "1/2"): 0.46685,
}


def test_every_end_to_end_row_carries_a_conditional_value_beside_the_marginal():
    rows = installed_model()["fitness"]["power"]["end_to_end"]["rows"]
    baseline = installed_model()["fitness"]["false_confirm"]["baseline_arm"]
    assert installed_model()["fitness"]["power"]["end_to_end"]["baseline_arm"] == baseline
    for row in rows:
        assert Fraction(row["conditional_joint"]) <= Fraction(row["conditional_stage1_confirm"])
        assert 0 <= row["conditional_joint_float"] <= 1
        total = sum(Fraction(v["conditional_joint"]) for v in row["by_evidence_split"].values())
        assert total == Fraction(row["conditional_joint"])


def test_the_conditional_power_matches_the_verifiers_independent_enumeration():
    """Two implementations, one definition. A number this artifact publishes about its
    own weakness is exactly the number nobody re-derives, so it is re-derived here
    against figures computed outside this repo."""
    rows = installed_model()["fitness"]["power"]["end_to_end"]["rows"]
    by_key = {
        (row["kind"], row["carrier"] if row["kind"] == "single_carrier" else row["offset"]): row
        for row in rows
    }
    for key, expected in VERIFIER_CONDITIONAL.items():
        got = by_key[key]["conditional_joint_float"]
        assert got == pytest.approx(expected, abs=5e-6), (key, got, expected)


def test_the_conditional_rows_use_the_same_baseline_the_false_confirm_block_names():
    """Like-for-like or it is not a comparison: the conditional detection rate and the
    conditional false-CONFIRM rate must be conditional on the SAME recorded arm."""
    fitness = installed_model()["fitness"]
    assert (
        fitness["power"]["end_to_end"]["baseline_counts"]
        == fitness["false_confirm"]["baseline_counts"]
    )


# ---------------------------------------------------------------------------
# N3: a promoted record's header
# ---------------------------------------------------------------------------


def _promoted(tmp_path) -> tuple[ValidationRecord, dict, dict]:
    from loop.cli import _pr_eligible_record

    cal = load_calibration(REAL_MODEL)
    record, base, cand = _calibrated_record(tmp_path)
    first = _first_decision(record)
    cb = _results({"A1": 3, "G4": 1, "G5": 3, "G2": 3}, ATTEMPTS)
    cc = _results({"A1": 10, "G4": 7, "G5": 10, "G2": 9}, ATTEMPTS)
    decision = confirmed(first, cb, cc, calibration=cal)
    assert decision.outcome == ACCEPT
    write_confirmation_record(
        ConfirmationRecord(
            candidate_id=CANDIDATE.id,
            baseline_label="conf-base",
            candidate_label="conf-cand",
            attempts_per_task_per_arm=10,
            confirm_set=tuple(sorted(SUPPORTED)),
            first_decision=first.to_json(),
            confirmation=decision.to_json(),
            per_task={},
            finding="f",
        ),
        tmp_path / f"confirmation-{CANDIDATE.id}.json",
    )
    return _pr_eligible_record(tmp_path, CANDIDATE, record), base, cand


def test_a_confirmed_accept_makes_the_disposition_accepted(tmp_path):
    """`disposition` reported the FIRST decision's state forever. Once a confirmation
    has accepted the candidate, "pending confirmation" is not a cautious answer, it is
    a false one — and it printed directly above the confirmation section saying ACCEPT."""
    record, _, _ = _promoted(tmp_path)
    assert record.accepted is True
    assert record.disposition == "ACCEPTED"


def test_an_unconfirmed_confirm_is_still_pending(tmp_path):
    """The distinction that made `disposition` worth having is untouched: a CONFIRM with
    no confirmation on the record is PENDING_CONFIRMATION, not REJECTED."""
    record, _, _ = _calibrated_record(tmp_path)
    assert record.disposition == "PENDING_CONFIRMATION"


def test_a_confirmed_reject_does_not_promote_the_disposition(tmp_path):
    """Only an ACCEPT moves it. A confirmation that REJECTED leaves the record where it
    was, and `loop.cli` refuses to promote it at all."""
    record, _, _ = _calibrated_record(tmp_path)
    rejected = dataclasses.replace(
        record, confirmation={"confirmation": {"outcome": REJECT}, "stage": "paired_confirmation"}
    )
    assert rejected.disposition == "PENDING_CONFIRMATION"


def test_the_rendered_pr_header_states_accepted_above_its_own_confirmation(tmp_path):
    from loop.prpipe import pr_body

    record, base, cand = _promoted(tmp_path)
    body = pr_body(CANDIDATE, record, CLUSTER, base, cand)

    header = body[body.index("## Validation") : body.index("| task | split")]
    assert "-> ACCEPTED" in header, header
    assert "PENDING_CONFIRMATION" not in body
    assert "**Disposition: ACCEPTED**" in body
    assert body.index("-> ACCEPTED") < body.index("## Confirmation")
