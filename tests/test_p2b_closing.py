"""The closing wave of the round-2 calibration: fail closed, verify at load, report honestly.

Eight separate holes, each reproduced here as the attack that found it, in the order
the closing audit found them:

1. FAIL CLOSED. `compaction` is a calibration-REQUIRED section. With no fresh, fit
   calibration the rule cannot run — and the old wiring then fell through to the
   CAUSAL verdict, which the null data itself proves accepts two runs with nothing
   changed between them. The attack below is exactly that: two real committed
   no-change arms, no artifact on disk, and a candidate that came back accepted.
2. LOAD-TIME RECOMPUTATION. The artifact recorded its own fitness verdict and the
   loader read it off the file being checked. Editing the rates and leaving the
   verdict alone installed a model nothing had ever checked.
3. END-TO-END POWER. The artifact published stage-1 power only, so a reader could
   not tell what fraction of a real improvement survives BOTH gates.
5. PR RENDERING. A calibrated decision's PR body still stated the legacy one-number
   rule, and a promoted CONFIRM rendered stage 1's numbers under the word ACCEPTED.
6. CONDITIONAL FPR. The false-CONFIRM probability was published marginally only,
   while every real judgment is made against ONE designated baseline arm.
8. STAGE BINDING. Nothing tied the confirmation to the same calibration or the same
   carrier set the first decision was judged on.
9. PROVENANCE KEYS. An arm fingerprint MISSING `dirty_sha` read as clean.
10. SELF-CONTAINED DOCS. Refinery code cited documents that do not live in this repo.

The fixtures here prefer the REAL committed evidence — the eight `r2-null-*` result
files and the installed `model-r2.json` — over fabricated arms wherever the point of
the test survives it. A tamper test copies the real artifact and edits one field, so
what it demonstrates is a tamper of the thing actually installed.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from loop.artifacts import Candidate, Cluster, ValidationRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
REAL_MODEL = REPO_ROOT / "iterations" / "calibration-compaction" / "model-r2.json"

# The eight arms of the round-2 protocol, as committed. Named explicitly, never
# globbed: an arm appended later must force this list (and the sweep in
# tests/test_round2_attack.py) to be updated deliberately.
FULL_ARMS = ("r2-null-full-a", "r2-null-full-b", "r2-null-full-c")
SUBSET_ARMS = (
    "r2-null-cmp-a",
    "r2-null-cmp-b",
    "r2-null-cmp-c",
    "r2-null-cmp-d",
    "r2-null-cmp-e",
)
SUPPORTED = frozenset({"A1", "G2", "G4", "G5"})
SPLIT_OF = {"A1": "held_in", "G4": "held_in", "G5": "held_in", "G2": "held_out"}
STANDARD_ATTEMPTS = {"A1": 3, "G4": 3, "G5": 3, "G2": 5}


def arm(label: str) -> dict:
    return json.loads((RESULTS / f"{label}.json").read_text())


def real_fingerprint() -> dict:
    return arm("r2-null-full-a")["fingerprint"]


def installed_model() -> dict:
    return json.loads(REAL_MODEL.read_text())


def copy_model(tmp_path: Path, mutate=None, name="model-r2.json") -> Path:
    """The REAL installed artifact, copied, optionally with one field edited.

    A tamper built by editing the committed artifact demonstrates something a
    fabricated one cannot: that the loader refuses the file the pipeline actually
    installs once a single number in it stops being true.
    """
    model = installed_model()
    if mutate is not None:
        mutate(model)
    path = tmp_path / name
    path.write_text(json.dumps(model, indent=2) + "\n")
    return path


def load_calibration(path: Path):
    from loop.validate import section_calibration

    return section_calibration("compaction", real_fingerprint(), model_path=path)


def why_not(path: Path) -> str:
    from loop.validate import calibration_status

    cal, why = calibration_status("compaction", real_fingerprint(), model_path=path)
    assert cal is None, "expected the loader to refuse this artifact"
    return why


# ---------------------------------------------------------------------------
# 1. FAIL CLOSED (S1)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_carbon(tmp_path):
    from runner.carbon_env import CARBON_ROOT

    root = tmp_path / "carbon"
    (root / "harness").mkdir(parents=True)
    shutil.copy(CARBON_ROOT / "harness" / "harness_config.json", root / "harness")
    for args in (
        ["init", "-b", "main"],
        ["config", "user.email", "t@t"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-m", "seed"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root


def _compaction_candidate() -> Candidate:
    live = json.loads(
        (Path(__import__("runner.carbon_env", fromlist=["CARBON_ROOT"]).CARBON_ROOT))
        .joinpath("harness/harness_config.json")
        .read_text()
    )
    old = live["compaction"]
    return Candidate(
        id="cmp-fail-closed",
        cluster_id="CL-1",
        proposer="Fable",
        proposer_detail="test",
        fields={"compaction": {"old": old, "new": old}},
        rationale="r",
        expected_effect="e",
        regression_risk="g",
    )


def test_an_uncalibrated_compaction_candidate_is_refused_not_handed_to_the_causal_rule(
    fake_carbon, tmp_path, monkeypatch
):
    """The closing audit's attack, reproduced on real committed evidence.

    `r2-null-full-b` and `r2-null-full-c` are two no-change arms: the same harness,
    the same config, the same carbon revision, nothing edited between them. Their
    whole-split deltas are +0.0370 held-in and +0.0200 held-out, which satisfies the
    causal rule (`Δ_in >= 0, Δ_ho >= 0, max > 0`) outright. With `model-r2.json`
    absent, `compaction` has no calibration — and the old wiring answered that by
    falling back to exactly the rule this data defeats, so a candidate that changed
    nothing came back ACCEPTED.

    The section is calibration-REQUIRED. Missing calibration is a refusal with the
    reason attached, never a downgrade to a weaker rule.
    """
    from loop.validate import validate_candidate

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    shutil.copy(RESULTS / "r2-null-full-b.json", baseline_path)
    shutil.copy(RESULTS / "r2-null-full-b.jsonl", baseline_path.with_suffix(".jsonl"))
    monkeypatch.setitem(
        __import__("loop.validate", fromlist=["_SECTION_MODEL"])._SECTION_MODEL,
        "compaction",
        tmp_path / "no-such-model-r2.json",
    )

    def fake_runner(label, only, attempts):
        shutil.copy(RESULTS / "r2-null-full-c.json", results_dir / f"{label}.json")
        shutil.copy(RESULTS / "r2-null-full-c.jsonl", results_dir / f"{label}.jsonl")

    record = validate_candidate(
        _compaction_candidate(),
        baseline_path=baseline_path,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        run_gates=lambda _root: {"passed": True, "checks": {}},
        results_dir=results_dir,
        log=lambda *_: None,
    )

    assert record.causal["accepted"] is True, (
        "fixture precondition: the causal rule accepts these two no-change arms — "
        "that is the whole reason falling back to it is unsafe"
    )
    assert record.rule["applied"] is False
    assert record.accepted is False, "an uncalibrated compaction candidate must not accept"
    assert record.disposition == "REJECTED"
    assert "not calibrated" in record.rule["why"]
    assert record.rule.get("calibration_required") is True


def test_the_refusal_names_the_missing_artifact_and_survives_serialization(
    fake_carbon, tmp_path, monkeypatch
):
    """The record has to SAY it refused for want of a calibration, on disk, or the
    next reader sees a bare `accepted: false` and cannot tell a measured rejection
    from an unmeasurable one."""
    from loop.artifacts import write_validation_record
    from loop.validate import validate_candidate

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    shutil.copy(RESULTS / "r2-null-full-b.json", baseline_path)
    shutil.copy(RESULTS / "r2-null-full-b.jsonl", baseline_path.with_suffix(".jsonl"))
    monkeypatch.setitem(
        __import__("loop.validate", fromlist=["_SECTION_MODEL"])._SECTION_MODEL,
        "compaction",
        tmp_path / "gone.json",
    )

    def fake_runner(label, only, attempts):
        shutil.copy(RESULTS / "r2-null-full-c.json", results_dir / f"{label}.json")
        shutil.copy(RESULTS / "r2-null-full-c.jsonl", results_dir / f"{label}.jsonl")

    record = validate_candidate(
        _compaction_candidate(),
        baseline_path=baseline_path,
        carbon_root=fake_carbon,
        run_runner=fake_runner,
        run_gates=lambda _root: {"passed": True, "checks": {}},
        results_dir=results_dir,
        log=lambda *_: None,
    )
    out = write_validation_record(record, tmp_path / "validation.json")
    raw = json.loads(out.read_text())
    assert raw["accepted"] is False
    assert raw["rule"]["calibration_required"] is True
    assert "gone.json" in raw["rule"]["why"]


def test_an_uncalibrated_section_that_is_not_calibration_required_still_uses_causal():
    """`tool_output` and every unmapped edit keep the behavior they had. Fail-closed
    is a property of the sections that DECLARE they need a null model, not a blanket
    refusal that would stop the loop from judging anything else."""
    from loop.validate import disposition_accepted

    causal = {"accepted": True}
    unmapped = {"applied": False, "why": "edited section is not calibrated"}
    assert disposition_accepted(unmapped, causal) is True
    required = {"applied": False, "why": "no artifact", "calibration_required": True}
    assert disposition_accepted(required, causal) is False
    applied = {"applied": True, "outcome": "ACCEPT"}
    assert disposition_accepted(applied, causal) is True
    confirm = {"applied": True, "outcome": "CONFIRM"}
    assert disposition_accepted(confirm, causal) is False


# ---------------------------------------------------------------------------
# 2. LOAD-TIME RECOMPUTATION (S2)
# ---------------------------------------------------------------------------


def test_the_installed_artifact_still_loads_after_recomputation(tmp_path):
    """Precondition for every tamper below: the real, untampered artifact passes the
    loader's own recomputation. A checker that refuses everything proves nothing."""
    cal = load_calibration(REAL_MODEL)
    assert cal is not None
    assert set(cal.supported) == SUPPORTED
    assert cal.coverage_level == Fraction(39, 40)


def test_editing_only_the_null_rates_is_caught_by_recomputing_them_from_per_arm(tmp_path):
    """The audit's tamper: copy the real artifact, edit ONLY the `null_rate` strings
    to `1/1000`, leave `fitness` untouched. Every recorded check still says it passed,
    because nothing re-ran them. The rates are what every quantile is computed from,
    so this installs a model whose gates are set by a number no measurement produced.

    The loader re-derives each rate from the artifact's own `per_arm` counts, which
    the tamper did not touch, and refuses on the disagreement.
    """

    def tamper(model):
        for task in model["null_model"]:
            model["null_model"][task]["null_rate"] = "1/1000"

    reason = why_not(copy_model(tmp_path, tamper))
    assert "recomput" in reason.lower(), reason
    assert "1/1000" in reason
    assert "A1" in reason or "G2" in reason or "G4" in reason or "G5" in reason


def test_a_recorded_pass_over_recomputed_failing_data_refuses(tmp_path):
    """`fitness` recorded true, the data underneath it no longer agrees.

    One arm's `per_arm` counts are replaced with a saturated run (every attempt
    passing on G4, a task pooled near 1/10) — enough to fail goodness against the
    pooled rate. The recorded verdict still says `pass: true` everywhere, because
    the tamper edited data and not verdicts. The loader re-runs the checks against
    the loaded data and refuses the disagreement.
    """

    def tamper(model):
        model["null_model"]["G4"]["per_arm"]["r2-null-cmp-a"] = [10, 10]

    reason = why_not(copy_model(tmp_path, tamper))
    assert "recomput" in reason.lower(), reason
    assert "goodness" in reason.lower() or "null_rate" in reason.lower(), reason


def test_a_recomputed_unfit_verdict_refuses_even_with_fit_recorded_true(tmp_path):
    """The same shape from the other side: make the recomputed STABILITY verdict
    disagree with the recorded one by editing the recorded verdict itself. `fit`
    stays true and every rate is untouched, so nothing but a re-run of the check can
    see it."""

    def tamper(model):
        for split, row in model["fitness"]["stability"].items():
            if split != "pass" and isinstance(row, dict):
                row["pass"] = False
                row["moved_excluding"] = {"r2-null-cmp-a": "1/2"}

    reason = why_not(copy_model(tmp_path, tamper))
    assert "recomput" in reason.lower(), reason
    assert "stability" in reason.lower(), reason


def test_the_loader_pins_the_coverage_level_to_exactly_thirty_nine_fortieths(tmp_path):
    """A coverage level is not a free parameter of an installed artifact. `0.5` is a
    perfectly readable, perfectly exact Fraction, and it would halve every bound in
    the pipeline with nothing in the record looking wrong."""

    def tamper(model):
        model["coverage_level"] = "1/2"

    reason = why_not(copy_model(tmp_path, tamper))
    assert "39/40" in reason, reason
    assert "coverage" in reason.lower()


def test_a_grain_row_edited_to_a_different_quantile_refuses(tmp_path):
    """The recorded quantile is the number the fitness verdict was reached on. If it
    can be edited without the loader noticing, the recorded verdict means nothing."""

    def tamper(model):
        model["fitness"]["grain"]["held_in"]["quantile"] = "9/10"

    reason = why_not(copy_model(tmp_path, tamper))
    assert "recomput" in reason.lower() and "grain" in reason.lower(), reason


# ---------------------------------------------------------------------------
# 3. END-TO-END POWER (S3) + amendment 4's published leave-one-out margins
# ---------------------------------------------------------------------------


def test_power_publishes_stage1_only_rows_under_that_name(tmp_path):
    """The stage-1 numbers stay — they are the gain gate's own power — but they are
    no longer the only rows, so they must say which stage they describe."""
    power = installed_model()["fitness"]["power"]
    assert set(power["stage1_only"]) == {"per_task", "gain_gate"}
    assert set(power["stage1_only"]["per_task"]) == SUPPORTED
    assert set(power["stage1_only"]["gain_gate"]) == {"held_in", "held_out"}


def test_power_publishes_joint_stage1_x_stage2_rows_for_every_declared_alternative():
    """Contract amendment 3: +0.2 on a single carrier (each supported task, one row
    each) and +0.3/+0.5 uniform across the supported set, each reported per evidence
    split as well as in total. Exact fractions, floats beside them for reading."""
    rows = installed_model()["fitness"]["power"]["end_to_end"]["rows"]
    single = {r["carrier"] for r in rows if r["kind"] == "single_carrier"}
    assert single == SUPPORTED
    uniform = {r["offset"] for r in rows if r["kind"] == "uniform"}
    assert uniform == {"3/10", "1/2"}
    for row in rows:
        assert Fraction(row["joint"]) <= Fraction(row["stage1_confirm"]), (
            "a joint detection rate can never exceed its own stage-1 factor"
        )
        assert 0 <= row["joint_float"] <= 1
        assert set(row["by_evidence_split"]) == {"held_in", "held_out"}
        total = sum(Fraction(v["joint"]) for v in row["by_evidence_split"].values())
        assert total == Fraction(row["joint"]), "the split rows must sum to the total"


def test_the_joint_stage1_predicate_agrees_with_the_real_evaluate_on_sampled_vectors(tmp_path):
    """The enumeration is only worth publishing if its predicate is the rule.

    Two hundred pseudo-random count vectors at the suite's own standard attempt
    counts are handed BOTH to the enumeration's stage-1 predicate and to the real
    `loop.acceptance.evaluate()` under the installed calibration, and the outcome,
    the evidence split, and the carrier set must agree every time.
    """
    from loop.acceptance import evaluate
    from loop.calibrate import _stage1_verdict, null_gain_quantile

    cal = load_calibration(REAL_MODEL)
    quantiles = {
        split: null_gain_quantile(
            {t: cal.null_rates[t] for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            {t: STANDARD_ATTEMPTS[t] for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            {t: STANDARD_ATTEMPTS[t] for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            cal.coverage_level,
        )
        for split in ("held_in", "held_out")
    }
    rng = random.Random(20260820)
    for _ in range(200):
        base = {t: rng.randint(0, STANDARD_ATTEMPTS[t]) for t in sorted(SUPPORTED)}
        cand = {t: rng.randint(0, STANDARD_ATTEMPTS[t]) for t in sorted(SUPPORTED)}
        b = _results(base, STANDARD_ATTEMPTS)
        c = _results(cand, STANDARD_ATTEMPTS)
        diffs = {
            t: Fraction(cand[t], STANDARD_ATTEMPTS[t]) - Fraction(base[t], STANDARD_ATTEMPTS[t])
            for t in sorted(SUPPORTED)
        }
        got = _stage1_verdict(diffs, SPLIT_OF, quantiles)
        real = evaluate(b, c, calibration=cal)
        assert got[0] == (real.outcome == "CONFIRM"), (base, cand)
        if got[0]:
            assert got[1] == real.evidence_split, (base, cand)
            assert tuple(got[2]) == real.improved_tasks, (base, cand)


def test_the_joint_stage2_predicate_agrees_with_the_real_confirmed_on_sampled_vectors(tmp_path):
    """The same check for stage 2, at the confirmation's own ten-attempt counts, with
    the carrier set varied across the sample so the per-carrier gate, the guard gate
    and the positivity gate all get exercised against the real function."""
    import dataclasses

    from loop.acceptance import (
        CONFIRM,
        Decision,
        calibration_digest,
        confirmed,
        decision_digest,
    )
    from loop.calibrate import (
        CONFIRMATION_GUARDS,
        _stage2_accepts,
        null_gain_quantile,
        null_task_quantile,
    )

    cal = load_calibration(REAL_MODEL)
    attempts = {t: 10 for t in SUPPORTED}
    quantiles = {
        split: null_gain_quantile(
            {t: cal.null_rates[t] for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            {t: 10 for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            {t: 10 for t in sorted(SUPPORTED) if SPLIT_OF[t] == split},
            cal.coverage_level,
        )
        for split in ("held_in", "held_out")
    }
    task_quantiles = {
        t: null_task_quantile(cal.null_rates[t], 10, 10, cal.coverage_level) for t in SUPPORTED
    }
    carrier_sets = [("A1",), ("G2",), ("A1", "G5"), ("A1", "G4", "G5"), ("G4",)]
    rng = random.Random(2026821)
    for i in range(200):
        improved = carrier_sets[i % len(carrier_sets)]
        base = {t: rng.randint(0, 10) for t in sorted(SUPPORTED)}
        cand = {t: rng.randint(0, 10) for t in sorted(SUPPORTED)}
        b, c = _results(base, attempts), _results(cand, attempts)
        diffs = {t: Fraction(cand[t] - base[t], 10) for t in sorted(SUPPORTED)}
        first = Decision(
            outcome=CONFIRM,
            reasons=(),
            delta_in=0.0,
            delta_ho=0.0,
            threshold_in=0.0,
            threshold_ho=0.0,
            evidence_split=SPLIT_OF[improved[0]],
            improved_tasks=improved,
            confirm_tasks=tuple(sorted(SUPPORTED)),
            raw={"regime": "section_calibration", "calibration_digest": calibration_digest(cal)},
        )
        # `confirmed()` requires the decision digest too, whenever a calibration is in
        # hand — an absent one refuses exactly like a wrong one, so a fabricated first
        # decision binds itself the same way `evaluate()` binds a real one.
        first = dataclasses.replace(
            first, raw={**first.raw, "decision_digest": decision_digest(first)}
        )
        got = _stage2_accepts(
            diffs, SPLIT_OF, quantiles, task_quantiles, improved, CONFIRMATION_GUARDS
        )
        real = confirmed(first, b, c, calibration=cal)
        assert got == (real.outcome == "ACCEPT"), (improved, base, cand, real.reasons)


def test_the_guard_set_the_power_model_uses_is_the_one_the_pipeline_confirms_with():
    """The end-to-end enumeration models the guard gate, so it has to model the SAME
    guards the confirmation actually applies. One constant, imported, not two."""
    from loop.calibrate import CONFIRMATION_GUARDS
    from loop.validate import _SECTION_CONFIRM_GUARDS

    assert _SECTION_CONFIRM_GUARDS["compaction"] is CONFIRMATION_GUARDS


def test_stability_publishes_a_leave_one_out_margin_for_every_arm():
    """Contract amendment 4: the leave-one-out margins are published, not merely
    consulted. Each removed arm gets the pooled quantile without it, the grain bucket
    that quantile lands in, and the coverage the FULL pool's threshold still holds
    under that reduced pool — the slack that says how close the verdict came."""
    stability = installed_model()["fitness"]["stability"]
    arms = set(FULL_ARMS) | set(SUBSET_ARMS)
    for split in ("held_in", "held_out"):
        margins = stability[split]["leave_one_out"]
        assert set(margins) == arms, split
        for label, row in margins.items():
            assert Fraction(row["quantile"]) >= -1
            assert isinstance(row["bucket"], int)
            cov = Fraction(row["coverage_at_full_quantile"])
            assert 0 <= cov <= 1, (split, label)
            assert Fraction(row["slack"]) == cov - Fraction(39, 40)


# ---------------------------------------------------------------------------
# 6. CONDITIONAL FALSE-CONFIRM RATE (S6)
# ---------------------------------------------------------------------------


def test_the_artifact_publishes_the_false_confirm_rate_conditional_on_the_designated_baseline():
    """Every real judgment compares a candidate against ONE recorded baseline arm,
    not against a fresh draw from the null. The marginal false-CONFIRM probability
    averages over baselines that will never be used again; the conditional one is the
    number that describes the comparison the pipeline actually makes."""
    from loop.calibrate import DESIGNATED_BASELINE

    fc = installed_model()["fitness"]["false_confirm"]
    assert fc["baseline_arm"] == DESIGNATED_BASELINE == "r2-null-full-a"
    assert Fraction(fc["marginal"]) >= 0
    assert Fraction(fc["conditional"]) >= 0
    assert fc["marginal_float"] == pytest.approx(float(Fraction(fc["marginal"])))
    assert fc["conditional_float"] == pytest.approx(float(Fraction(fc["conditional"])))
    assert set(fc["by_evidence_split"]) == {"held_in", "held_out"}
    assert fc["baseline_counts"]["A1"] == [2, 3]


def test_the_conditional_false_confirm_rate_matches_a_direct_enumeration(tmp_path):
    """Independently re-derived: fix the designated baseline arm's OWN counts, draw
    the candidate side at the pooled null rates, and count the mass that CONFIRMs."""
    from itertools import product

    from loop.calibrate import _binom_pmf, _stage1_verdict, null_gain_quantile

    cal = load_calibration(REAL_MODEL)
    tasks = sorted(SUPPORTED)
    baseline_counts = {
        t: installed_model()["null_model"][t]["per_arm"]["r2-null-full-a"][0] for t in tasks
    }
    quantiles = {
        split: null_gain_quantile(
            {t: cal.null_rates[t] for t in tasks if SPLIT_OF[t] == split},
            {t: STANDARD_ATTEMPTS[t] for t in tasks if SPLIT_OF[t] == split},
            {t: STANDARD_ATTEMPTS[t] for t in tasks if SPLIT_OF[t] == split},
            cal.coverage_level,
        )
        for split in ("held_in", "held_out")
    }
    pmfs = {t: _binom_pmf(STANDARD_ATTEMPTS[t], cal.null_rates[t]) for t in tasks}
    total = Fraction(0)
    for combo in product(*(range(STANDARD_ATTEMPTS[t] + 1) for t in tasks)):
        p = Fraction(1)
        diffs = {}
        for t, k in zip(tasks, combo, strict=True):
            p *= pmfs[t][k]
            diffs[t] = Fraction(k - baseline_counts[t], STANDARD_ATTEMPTS[t])
        if p and _stage1_verdict(diffs, SPLIT_OF, quantiles)[0]:
            total += p
    assert Fraction(installed_model()["fitness"]["false_confirm"]["conditional"]) == total


# ---------------------------------------------------------------------------
# 5. PR RENDERING (S5)
# ---------------------------------------------------------------------------

LEGACY_RULE_TEXT = "## Validation — acceptance rule `Δ_in ≥ 0, Δ_ho ≥ 0, max(Δ_in, Δ_ho) > 0`"


def _results(passes: dict[str, int], attempts: dict[str, int], fingerprint=None) -> dict:
    return {
        "fingerprint": dict(fingerprint or real_fingerprint()),
        "tasks": {
            t: {
                "split": SPLIT_OF[t],
                "attempts": attempts[t],
                "passes": p,
                "pass_fraction": round(p / attempts[t], 4),
            }
            for t, p in passes.items()
        },
    }


def _calibrated_record(tmp_path) -> tuple[ValidationRecord, dict, dict]:
    """A ValidationRecord whose `rule` really was produced by the calibrated rule."""
    from loop.acceptance import evaluate

    cal = load_calibration(REAL_MODEL)
    attempts = {t: 10 for t in SUPPORTED}
    base = _results({"A1": 4, "G4": 1, "G5": 4, "G2": 3}, attempts)
    cand = _results({"A1": 10, "G4": 6, "G5": 10, "G2": 9}, attempts)
    decision = evaluate(base, cand, calibration=cal)
    assert decision.outcome == "CONFIRM", decision.reasons
    record = ValidationRecord(
        candidate_id="cmp-1",
        label="cand-cmp-1",
        accepted=False,
        delta_in=decision.delta_in,
        delta_ho=decision.delta_ho,
        per_task={t: 0.1 for t in sorted(SUPPORTED)},
        baseline_fingerprint=base["fingerprint"],
        candidate_fingerprint=cand["fingerprint"],
        rule={"applied": True, **decision.to_json(), "calibration": cal.to_json()},
    )
    return record, base, cand


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
CLUSTER = Cluster(
    id="CL-1",
    mechanism="m",
    tasks=("G4",),
    hypothesis="h",
    evidence=("e",),
)


def test_a_calibrated_decisions_pr_body_states_the_rule_that_actually_ran(tmp_path):
    """The audit's reproduction: render the PR body for a decision judged against the
    null model and the body announced the one-number acceptance rule instead — the
    rule that decision was never judged by. A human approving that merge is reading a
    statement of a comparison nobody made."""
    from loop.prpipe import pr_body

    record, base, cand = _calibrated_record(tmp_path)
    body = pr_body(CANDIDATE, record, CLUSTER, base, cand)

    assert LEGACY_RULE_TEXT not in body, "the legacy rule text must not render here"
    assert "null model" in body.lower()
    assert "39/40" in body, "the coverage the quantiles were computed at"
    quantile = record.rule["raw"]["null_quantiles"]["held_in"]["quantile"]
    assert quantile in body, "the supported-set quantile actually gated on"
    assert record.rule["raw"]["null_quantiles"]["held_out"]["quantile"] in body
    assert "iterations/calibration-compaction" in body, "the artifact's own path"
    sha = real_fingerprint()["runner_sha"]
    assert sha[:12] in body, "the artifact's computed_at_runner_sha"
    assert "A1 10v10" in body, "the counts the quantile was computed at"


def test_a_legacy_decisions_pr_body_is_unchanged(tmp_path):
    """The legacy text renders for legacy decisions, byte for byte. The calibrated
    body is an addition, never a replacement of what uncalibrated records say."""
    from loop.prpipe import pr_body

    record = ValidationRecord(
        candidate_id="x",
        label="cand-x",
        accepted=True,
        delta_in=0.125,
        delta_ho=0.2,
        per_task={"A1": 1.0},
        baseline_fingerprint=real_fingerprint(),
        candidate_fingerprint=real_fingerprint(),
    )
    attempts = {t: 10 for t in SUPPORTED}
    base = _results({"A1": 4, "G4": 1, "G5": 4, "G2": 3}, attempts)
    cand = _results({"A1": 6, "G4": 1, "G5": 4, "G2": 3}, attempts)
    body = pr_body(CANDIDATE, record, CLUSTER, base, cand)
    assert LEGACY_RULE_TEXT in body
    assert "null model" not in body.lower()


def test_the_pr_body_renders_the_confirmed_stage_when_a_confirmation_promoted_it(tmp_path):
    """A promoted CONFIRM used to render stage 1's deltas under the word ACCEPTED.
    Stage 1 is not what accepted the candidate; the confirmation is, at its own
    attempt counts, against its own quantiles. Both belong in the body, labeled."""
    from loop.acceptance import ACCEPT, confirmed
    from loop.artifacts import ConfirmationRecord, write_confirmation_record
    from loop.cli import _pr_eligible_record
    from loop.prpipe import pr_body

    record, base, cand = _calibrated_record(tmp_path)
    cal = load_calibration(REAL_MODEL)
    first = _first_decision(record)
    # A DIFFERENT pair for the confirmation, deliberately: reusing the first pass's
    # numbers would let this test pass on a body that rendered stage 1 and called it
    # the confirmation. The two stages' deltas must be distinguishable.
    attempts = {t: 10 for t in SUPPORTED}
    conf_base = _results({"A1": 3, "G4": 1, "G5": 3, "G2": 3}, attempts)
    conf_cand = _results({"A1": 10, "G4": 7, "G5": 10, "G2": 9}, attempts)
    decision = confirmed(first, conf_base, conf_cand, calibration=cal)
    assert decision.outcome == ACCEPT, decision.reasons
    assert f"{decision.delta_in:+.4f}" != f"{record.rule['delta_in']:+.4f}", (
        "fixture precondition: the two stages must report different held-in means"
    )
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
    promoted = _pr_eligible_record(tmp_path, CANDIDATE, record)
    assert promoted.accepted is True
    assert promoted.confirmation, "the promotion must carry the confirmation onto the record"

    body = pr_body(CANDIDATE, promoted, CLUSTER, base, cand)
    assert "conf-base" in body and "conf-cand" in body
    assert "confirmation" in body.lower()
    assert f"{decision.delta_in:+.4f}" in body, "the CONFIRMED stage's own supported-set mean"


def _first_decision(record: ValidationRecord):
    import dataclasses

    from loop.acceptance import Decision

    tuple_fields = {"reasons", "excluded", "improved_tasks", "confirm_tasks", "targeted_rerun"}
    names = {f.name for f in dataclasses.fields(Decision)}
    return Decision(
        **{
            k: (tuple(v) if k in tuple_fields and isinstance(v, list) else v)
            for k, v in record.rule.items()
            if k in names
        }
    )


# ---------------------------------------------------------------------------
# 8. STAGE BINDING (S8/S9)
# ---------------------------------------------------------------------------


def test_a_calibrated_first_decision_records_both_digests(tmp_path):
    from loop.acceptance import calibration_digest, decision_digest

    record, _, _ = _calibrated_record(tmp_path)
    raw = record.rule["raw"]
    cal = load_calibration(REAL_MODEL)
    assert raw["calibration_digest"] == calibration_digest(cal)
    assert len(raw["calibration_digest"]) == 64
    first = _first_decision(record)
    assert raw["decision_digest"] == decision_digest(first)


def test_swapping_the_artifact_between_the_two_stages_refuses(tmp_path):
    """The attack the digest exists for: judge the first pass under one null model,
    then confirm the same claim under a DIFFERENT one. Both artifacts are fit, fresh
    and pinned to the right supported set, so every other check passes — only a
    digest of the rates the first decision was actually judged against can see it."""
    from loop.acceptance import confirmed
    from loop.calibrate import SUPPORTED as MODEL_SUPPORTED
    from loop.calibrate import calibrate_model

    record, base, cand = _calibrated_record(tmp_path)
    first = _first_decision(record)

    # A SECOND null model, measured the same way from the same protocol, differing by
    # one attempt on one arm (r2-null-cmp-a's G4 goes 2/10 -> 1/10, pooling G4 at 5/59
    # instead of 6/59). It is fit, fresh, and pinned to the right supported set, so
    # every check the loader has ever had passes on it: it installs cleanly. Only the
    # digest of the rates the FIRST decision was judged against can tell that this is
    # not that model.
    arms_dir = tmp_path / "swapped-arms"
    arms_dir.mkdir()
    for label in FULL_ARMS + SUBSET_ARMS:
        data = arm(label)
        if label == "r2-null-cmp-a":
            data["tasks"]["G4"]["passes"] = 1
            data["tasks"]["G4"]["pass_fraction"] = 0.1
        (arms_dir / f"{label}.json").write_text(json.dumps(data))
    swapped = calibrate_model(list(FULL_ARMS + SUBSET_ARMS), arms_dir, MODEL_SUPPORTED)
    assert swapped["fitness"]["fit"] is True, "a swap only proves something if it installs"
    assert swapped["null_model"]["G4"]["null_rate"] == "5/59"
    path = tmp_path / "other-model.json"
    path.write_text(json.dumps(swapped))
    other = load_calibration(path)
    assert other is not None and other.null_rates != _first_rates(record)

    with pytest.raises(ValueError, match="digest"):
        confirmed(first, base, cand, calibration=other)


def _first_rates(record: ValidationRecord) -> dict:
    return {
        task: Fraction(rate)
        for task, rate in (record.rule["raw"]["calibration"]["null_rates"]).items()
    }


def test_editing_the_records_carrier_set_refuses_when_the_cli_reloads_it(tmp_path):
    """The reviewer's edit: a record whose first decision named three carriers, saved
    with one. Every carrier must repeat beyond its OWN quantile, so dropping two of
    them makes the confirmation a strictly easier test than the one that was
    promised. The digest is over the record's own claim, so the edit cannot survive
    a reload."""
    from loop.cli import load_first_decision

    record, _, _ = _calibrated_record(tmp_path)
    assert len(record.rule["improved_tasks"]) >= 3, "fixture precondition: several carriers"
    it_dir = tmp_path / "iter"
    it_dir.mkdir()
    raw = record.to_json()
    path = it_dir / f"validation-{CANDIDATE.id}.json"
    path.write_text(json.dumps(raw))
    assert load_first_decision(it_dir, CANDIDATE.id).improved_tasks == tuple(
        record.rule["improved_tasks"]
    )

    raw["rule"]["improved_tasks"] = [record.rule["improved_tasks"][0]]
    path.write_text(json.dumps(raw))
    with pytest.raises(SystemExit, match="digest"):
        load_first_decision(it_dir, CANDIDATE.id)


def test_an_uncalibrated_decision_carries_no_digests_and_is_confirmed_as_before(tmp_path):
    """Byte-identity: the uncalibrated regime gains nothing, not even a new key."""
    from loop.acceptance import evaluate

    attempts = {t: 10 for t in SUPPORTED}
    base = _results({"A1": 4, "G4": 1, "G5": 4, "G2": 3}, attempts)
    cand = _results({"A1": 9, "G4": 5, "G5": 9, "G2": 8}, attempts)
    decision = evaluate(base, cand)
    assert "calibration_digest" not in decision.raw
    assert "decision_digest" not in decision.raw


# ---------------------------------------------------------------------------
# 9. PROVENANCE KEYS (Codex)
# ---------------------------------------------------------------------------


def test_an_arm_whose_fingerprint_lacks_dirty_sha_is_refused_at_calibrate_time(tmp_path):
    """`fp.get("dirty_sha")` reads an ABSENT key as None, and None is what a clean
    tree records — so an arm that never recorded its tree state compared equal to one
    that recorded a clean one. Absent is not clean; it is unknown, and the null
    protocol's whole claim is that nothing differed between the arms."""
    from loop.calibrate import SUPPORTED as MODEL_SUPPORTED
    from loop.calibrate import calibrate_model

    results_dir = tmp_path / "arms"
    results_dir.mkdir()
    for label in FULL_ARMS + SUBSET_ARMS:
        data = arm(label)
        if label == "r2-null-cmp-b":
            data["fingerprint"] = {k: v for k, v in data["fingerprint"].items() if k != "dirty_sha"}
        (results_dir / f"{label}.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="dirty_sha"):
        calibrate_model(list(FULL_ARMS + SUBSET_ARMS), results_dir, MODEL_SUPPORTED)


def test_the_real_arms_all_record_a_dirty_sha_key(tmp_path):
    """Precondition for the refusal above being about the tamper and not about the
    committed evidence."""
    for label in FULL_ARMS + SUBSET_ARMS:
        assert "dirty_sha" in arm(label)["fingerprint"], label


# ---------------------------------------------------------------------------
# 10. SELF-CONTAINED DOCS (Codex)
# ---------------------------------------------------------------------------


def test_no_refinery_module_cites_a_document_that_does_not_live_in_this_repo():
    """AGENTS.md: "Do not cite a document that does not live in this repo; a dangling
    citation is worse than none." The calibration contract lives in a private
    program repo, so every fact this code depends on has to be stated where the code
    is, not pointed at."""
    # Assembled from halves so THIS file does not become its own first offender —
    # a literal here would match every scan, including the one below.
    markers = ("contracts/" + "phase2-calibration-contract", "contracts/" + "phase2b-calibration")
    offenders = []
    for path in sorted((REPO_ROOT / "loop").glob("*.py")) + sorted(
        (REPO_ROOT / "tests").glob("*.py")
    ):
        text = path.read_text()
        for marker in markers:
            if marker in text:
                offenders.append(f"{path.name}: {marker}")
    assert not offenders, offenders


def test_the_validation_record_comment_no_longer_says_tool_output_today():
    """`compaction` has been a calibrated section since the round-2 artifact
    installed. A comment naming `tool_output` as the only one is a fact that stopped
    being true and would be read as current."""
    text = (REPO_ROOT / "loop" / "artifacts.py").read_text()
    assert "tool_output today" not in text
