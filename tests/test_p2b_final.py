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
    _UNMOVED as UNMOVED,
)
from tests.test_p2b_closing import (
    CANDIDATE,
    CLUSTER,
    COVERED,
    SPLIT_OF,
    SUPPORTED,
    _calibrated_record,
    _committed_pooling,
    _first_decision,
    _independent_diff_pmf,
    _results,
    committed_model,
    installed_calibration,
    installed_model,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
# Every COVERED task, because a confirmation reruns the guards too.
ATTEMPTS = dict.fromkeys(COVERED, 10)

# A confirmation pair on which the ORIGINAL three-carrier claim honestly fails: A1
# repeats beyond its own 2/5 quantile at 10v10, G4 and G5 do not move at all. Every
# other gate holds — both supported-set means are non-negative, no guard drops, nothing
# collapses — so the ONLY thing standing between this pair and an ACCEPT is the rule
# that every named carrier must repeat. That makes it the exact pair a record-edit
# attack wants: shrink the claim to the one carrier that did repeat.
HONEST_BASE = {"A1": 2, "G4": 5, "G5": 8, "G2": 5, **UNMOVED}
HONEST_CAND = {"A1": 9, "G4": 5, "G5": 8, "G2": 5, **UNMOVED}


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
    cal = installed_calibration()
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
    cal = installed_calibration()
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
    cal = installed_calibration()
    first, base, cand = _honest_first(tmp_path)
    attacked = _edited(first, drop=("decision_digest", "regime"))

    with pytest.raises(ValueError, match="does not record that it was judged under one"):
        confirmed(attacked, base, cand, calibration=cal)


def test_deleting_the_calibration_digest_alone_refuses(tmp_path):
    """The same principle on the other binding: the artifact-swap check must not be
    switchable off either, by deleting its digest or by deleting the regime."""
    cal = installed_calibration()
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
    cal = installed_calibration()
    record, _, _ = _calibrated_record(tmp_path)
    first = _first_decision(record)
    base = _results({"A1": 3, "G4": 1, "G5": 3, "G2": 3, **UNMOVED}, ATTEMPTS)
    cand = _results({"A1": 10, "G4": 7, "G5": 10, "G2": 9, **UNMOVED}, ATTEMPTS)

    assert confirmed(first, base, cand, calibration=cal).outcome == ACCEPT


def test_an_uncalibrated_confirmation_needs_no_digests_at_all(tmp_path):
    """Byte-identity, restated for this change: the digests are a property of the
    CALIBRATED regime. An uncalibrated first decision has none, is asked for none, and
    is confirmed exactly as it always was."""
    from loop.acceptance import evaluate

    # UNCALIBRATED: no calibration means no guards ride in, so the confirmation
    # reruns exactly what moved.
    base = _results({"A1": 4, "G4": 1, "G5": 4, "G2": 3}, ATTEMPTS)
    cand = _results({"A1": 9, "G4": 5, "G5": 9, "G2": 8}, ATTEMPTS)
    first = evaluate(base, cand)
    assert first.outcome == CONFIRM
    assert "decision_digest" not in first.raw and "calibration_digest" not in first.raw

    assert confirmed(first, base, cand).outcome in {ACCEPT, REJECT}


# ---------------------------------------------------------------------------
# N2: conditional power, like-for-like with the conditional false-CONFIRM rate
# ---------------------------------------------------------------------------

# The MARGINAL joint detection rates the committed ten-arm pooling publishes, quoted
# here to fifteen places. POST-HOC, and labelled as such: these came out of the
# re-derivation below rather than being predicted before it, so on their own they pin a
# number rather than check one. They earn their place beside the re-derivation, not
# instead of it — the enumeration reads the same artifact it is checking, so if someone
# re-pools over other arms both sides move together and only a literal notices.
VERIFIER_MARGINAL_JOINT = {
    ("single_carrier", "A1"): 4.3369886712965204e-05,
    ("single_carrier", "G2"): 0.0025255628384418718,
    ("single_carrier", "G4"): 4.453046685799452e-05,
    ("single_carrier", "G5"): 4.329961240224884e-05,
    ("uniform", "3/10"): 0.020169107295546457,
    ("uniform", "1/2"): 0.13821038958896642,
}


def _rows_by_key() -> dict[tuple[str, str], dict]:
    return {
        (row["kind"], row["carrier"] if row["kind"] == "single_carrier" else row["offset"]): row
        for row in committed_model()["fitness"]["power"]["end_to_end"]["rows"]
    }


def _stage2_quantiles(rates: dict[str, Fraction], level: Fraction) -> dict[str, Fraction]:
    """Each split's supported-set-mean quantile at the CONFIRMATION's ten attempts."""
    from loop.calibrate import null_gain_quantile

    return {
        split: null_gain_quantile(
            {t: rates[t] for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            {t: 10 for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            {t: 10 for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            level,
        )
        for split in ("held_in", "held_out")
    }


def _carrier_sets(tasks: list[str]) -> list[tuple[str, ...]]:
    """Every carrier set a first CONFIRM could name: a nonempty subset of ONE split's
    supported tasks, since `evaluate()` only ever names the evidence split's movers."""
    from itertools import combinations

    out = []
    for split in ("held_in", "held_out"):
        on_split = [t for t in tasks if SPLIT_OF[t] == split]
        for size in range(1, len(on_split) + 1):
            out.extend(combinations(on_split, size))
    return out


def test_every_end_to_end_row_publishes_a_null_conditional_and_says_why():
    """N2's rule, at a pooling that cannot satisfy it: each row still reports the
    conditional beside the marginal, and here the conditional is NULL.

    The designated baseline arm is not in this pooling, so there is no run to be
    conditional on. Folding that empty mass into a fraction published `0/1` and `0.0` —
    "this pipeline ships a real +0.2 improvement with probability exactly zero" — a
    strong false claim sitting beside a real marginal number a reader would compare it
    against. Absent is now written as absent, on the block AND on every row, with the
    resolver's own sentence saying which arm is missing.
    """
    e2e = committed_model()["fitness"]["power"]["end_to_end"]
    fc = committed_model()["fitness"]["false_confirm"]
    assert e2e["baseline_arm"] == fc["baseline_arm"] is None
    assert e2e["baseline_counts"] is None
    assert e2e["baseline_note"] == fc["baseline_note"], (
        "one resolver, one reason: the detection rates and the false-CONFIRM rate go "
        "missing for the same cause and must not offer two accounts of it"
    )
    assert "r2-null-full-a" in e2e["baseline_note"]

    for row in e2e["rows"]:
        assert row["conditional_stage1_confirm"] is None, row["carrier"]
        assert row["conditional_stage1_confirm_float"] is None, row["carrier"]
        assert row["conditional_joint"] is None, row["carrier"]
        assert row["conditional_joint_float"] is None, row["carrier"]
        assert row["baseline_note"] == e2e["baseline_note"]
        for split_row in row["by_evidence_split"].values():
            assert split_row["conditional_stage1_confirm"] is None
            assert split_row["conditional_joint"] is None
            assert split_row["conditional_joint_float"] is None
        # The marginal half is whole. This is one absent number, not a dropped block.
        assert Fraction(row["joint"]) <= Fraction(row["stage1_confirm"])
        assert 0 <= row["joint_float"] <= 1
        total = sum(Fraction(v["joint"]) for v in row["by_evidence_split"].values())
        assert total == Fraction(row["joint"])


def test_the_stage2_predicate_factors_into_common_gates_and_per_carrier_thresholds():
    """The premise the enumeration below leans on, pinned before it leans on it.

    `_stage2_accepts(diffs, carriers)` is claimed to be `common(diffs) AND every carrier
    strictly beats its own quantile` — carrier-independent gates times a per-carrier
    test. The re-derivation uses that shape to ask the real predicate once per count
    vector instead of once per carrier set, which is the difference between ten seconds
    and a minute. If the predicate ever stops factoring this way, the re-derivation
    would go quietly wrong; this makes it go loudly wrong instead.
    """
    import random

    from loop.calibrate import CONFIRMATION_GUARDS, _stage2_accepts, null_task_quantile

    rates, _attempts, _q1 = _committed_pooling()
    tasks = sorted(SUPPORTED)
    level = Fraction(committed_model()["coverage_level"])
    q2 = _stage2_quantiles(rates, level)
    tq = {t: null_task_quantile(rates[t], 10, 10, level) for t in tasks}
    sets = _carrier_sets(tasks)

    rng = random.Random(20260821)
    for _ in range(400):
        diffs = {t: Fraction(rng.randint(-10, 10), 10) for t in tasks}
        passing = {t for t in tasks if diffs[t] > tq[t]}
        common = [
            _stage2_accepts(diffs, SPLIT_OF, q2, tq, cs, CONFIRMATION_GUARDS)
            for cs in sets
            if set(cs) <= passing
        ]
        assert len(set(common)) <= 1, (diffs, "carrier-independent gates are not independent")
        for cs in sets:
            got = _stage2_accepts(diffs, SPLIT_OF, q2, tq, cs, CONFIRMATION_GUARDS)
            expected = bool(common and common[0]) and set(cs) <= passing
            assert got == expected, (diffs, cs)


def test_the_marginal_end_to_end_power_matches_a_direct_enumeration():
    """Two implementations, one definition — re-keyed to the pooling that exists.

    This test used to check the CONDITIONAL rates against figures a verifier computed
    outside this repo. Those rates are gone (no designated baseline in this pooling) and
    the marginal ones are what the artifact still publishes, so the marginal ones are
    what get a second implementation rather than the test being skipped.

    Independent where it counts. `loop.calibrate` computes each row by folding stage-1
    mass by evidence split and multiplying in a stage-2 acceptance probability FACTORED
    across the two splits — the factoring is the clever step and therefore the one worth
    checking. This enumeration does not factor: it walks the joint four-task difference
    vector at the confirmation's own ten attempts, all 21^4 of them, and convolves its
    own difference distributions from `_binom_pmf` rather than reusing the cached one.
    Only the two verdict predicates are shared, and those are pinned against the real
    `evaluate()`/`confirmed()` separately in `test_p2b_closing.py`.

    Exact fractions, not floats: an approximate agreement between two exact
    enumerations would only be hiding something.
    """
    from itertools import product

    from loop.calibrate import (
        CONFIRMATION_GUARDS,
        _stage1_verdict,
        _stage2_accepts,
        null_task_quantile,
    )

    rates, attempts, q1 = _committed_pooling()
    tasks = sorted(SUPPORTED)
    level = Fraction(committed_model()["coverage_level"])
    q2 = _stage2_quantiles(rates, level)
    tq = {t: null_task_quantile(rates[t], 10, 10, level) for t in tasks}

    rows = _rows_by_key()
    assert set(rows) == set(VERIFIER_MARGINAL_JOINT), "every published row is re-derived"

    for key, row in sorted(rows.items()):
        offset = Fraction(row["offset"])
        carrier = row["carrier"]
        alt = {
            t: min(Fraction(1), rates[t] + (offset if carrier in (None, t) else Fraction(0)))
            for t in tasks
        }

        # STAGE 1, at the suite's standard attempt counts: which carrier sets CONFIRM,
        # and with how much mass.
        stage1: dict[tuple[str, ...], Fraction] = {}
        grid1 = [
            sorted(_independent_diff_pmf(attempts[t], rates[t], alt[t]).items()) for t in tasks
        ]
        for combo in product(*grid1):
            diffs = {t: d for t, (d, _) in zip(tasks, combo, strict=True)}
            ok, _split, carriers = _stage1_verdict(diffs, SPLIT_OF, q1)
            if not ok:
                continue
            p = Fraction(1)
            for _, prob in combo:
                p *= prob
            if p:
                stage1[carriers] = stage1.get(carriers, Fraction(0)) + p
        assert sum(stage1.values()) == Fraction(row["stage1_confirm"]), key

        # STAGE 2, at ten attempts on both sides, over the JOINT vector — no factoring.
        accept = dict.fromkeys(stage1, Fraction(0))
        grid2 = [sorted(_independent_diff_pmf(10, rates[t], alt[t]).items()) for t in tasks]
        for combo in product(*grid2):
            diffs = {t: d for t, (d, _) in zip(tasks, combo, strict=True)}
            passing = {t for t in tasks if diffs[t] > tq[t]}
            eligible = [cs for cs in stage1 if set(cs) <= passing]
            if not eligible:
                continue
            if not _stage2_accepts(diffs, SPLIT_OF, q2, tq, eligible[0], CONFIRMATION_GUARDS):
                continue
            p = Fraction(1)
            for _, prob in combo:
                p *= prob
            for cs in eligible:
                accept[cs] += p

        joint = sum((stage1[cs] * accept[cs] for cs in stage1), Fraction(0))
        assert joint == Fraction(row["joint"]), key
        assert float(joint) == pytest.approx(VERIFIER_MARGINAL_JOINT[key], abs=5e-12), key


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

    cal = installed_calibration()
    record, base, cand = _calibrated_record(tmp_path)
    first = _first_decision(record)
    cb = _results({"A1": 3, "G4": 1, "G5": 3, "G2": 3, **UNMOVED}, ATTEMPTS)
    cc = _results({"A1": 10, "G4": 7, "G5": 10, "G2": 9, **UNMOVED}, ATTEMPTS)
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
