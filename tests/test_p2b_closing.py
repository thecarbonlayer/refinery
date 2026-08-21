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

The fixtures here prefer the REAL committed evidence — the `r2-null-*` result files
and the committed `model-r2.json` — wherever the point of the test survives it, and
`committed_model()` is that file: every claim about what `calibrate_model` PUBLISHES
(the leave-one-out margins, the false-CONFIRM block, the end-to-end power rows) still
reads it directly.

What changed in Phase 2c: the committed artifact pools four tasks and the loader now
pins seven, so it no longer INSTALLS — correctly, because the campaign that rates
CMP-5/6/7 has not run. Tests that need a loadable calibration therefore build one with
the real `calibrate_model` over synthetic null arms (`installed_model()`), carrying the
real arms' provenance. The tamper mechanics are unchanged; the artifact they tamper
with moved because the committed one is superseded, not because a fabricated fixture
was easier. `tests/test_loop_validate.py` pins the committed artifact's own refusal.
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
    "r2-null-cmp-f",
    "r2-null-cmp-g",
    "r2-null-cmp-h",
)
# The COMMITTED artifact's own arms — the Phase 2c campaign's ten, in pooling order.
# Distinct from the two lists above, which name the synthetic fixture's arms: the
# fixture borrows the round-2 LABELS so its shape matches, while every claim about the
# committed record is keyed to the arms that record was actually pooled from.
P2C_ARMS = (
    "p2c-null-full-a",
    "p2c-null-full-b",
    "p2c-null-full-c",
    "p2c-null-cmp-a",
    "p2c-null-cmp-b",
    "p2c-null-cmp-c",
    "p2c-null-cmp-d",
    "p2c-null-cmp-e",
    "p2c-null-cmp-f",
    "p2c-null-cmp-g",
)
# The GAIN set — what the split means average over — and the wider COVERED set the
# model must rate. They coincided through Phase 2b; the Phase 2c scenario guards made
# them different sets (contract amendment 2).
SUPPORTED = frozenset({"A1", "G2", "G4", "G5"})
COVERED = SUPPORTED | frozenset({"CMP-5", "CMP-6", "CMP-7"})
SPLIT_OF = {
    "A1": "held_in",
    "G4": "held_in",
    "G5": "held_in",
    "G2": "held_out",
    "CMP-5": "held_in",
    "CMP-6": "held_out",
    "CMP-7": "held_in",
}
STANDARD_ATTEMPTS = {t: (5 if SPLIT_OF[t] == "held_out" else 3) for t in SPLIT_OF}


# Null-arm counts for a seven-task artifact of the shape the loader now pins. The arm
# LABELS are the real protocol's, so the artifact this builds is shape-identical to the
# installed one — same eleven arms, same designated baseline — and only the three guard
# rows (and the counts) are synthetic.
def _null_counts(label: str, subset: bool) -> dict[str, int]:
    """Deterministic per-arm counts: null variation, never a hopeful pattern."""
    seed = sum(ord(c) for c in label)
    base = {"A1": 0.55, "G4": 0.30, "G5": 0.62, "G2": 0.45, "CMP-5": 0.48, "CMP-6": 0.52,
            "CMP-7": 0.58}  # fmt: skip
    counts = {}
    for i, (task, rate) in enumerate(sorted(base.items())):
        attempts = 10 if subset else STANDARD_ATTEMPTS[task]
        wobble = ((seed + 7 * i) % 3) - 1  # -1, 0 or +1 attempt of null movement
        counts[task] = max(1, min(attempts - 1, round(rate * attempts) + wobble))
    return counts


def arm(label: str) -> dict:
    return json.loads((RESULTS / f"{label}.json").read_text())


def real_fingerprint() -> dict:
    return arm("r2-null-full-a")["fingerprint"]


def committed_model() -> dict:
    """The artifact as COMMITTED — `iterations/calibration-compaction/model-r2.json`.

    Now the Phase 2c pooling: ten arms, seven rated tasks, `fitness.fit = false`
    because stability refused. It does not install, but it is still the published
    record, and the tests that assert on what `calibrate_model` PUBLISHES — the
    leave-one-out margins, the false-CONFIRM block, the end-to-end power rows — are
    claims about that record. They read it here rather than through the loader fixture,
    so they keep their original subject.

    Re-keyed to those ten arms rather than skipped. These were the artifact's
    independent verifiers; a verifier pointed at a file that no longer exists verifies
    nothing, and one that is skipped verifies nothing more loudly.
    """
    return json.loads(REAL_MODEL.read_text())


def installed_model(mutate_arms=None) -> dict:
    """A null model of the shape the loader installs TODAY: seven covered tasks.

    Through Phase 2b this returned the committed `model-r2.json` itself, and the tamper
    tests below edited that file so their demonstration was about the artifact the
    pipeline actually installs. That artifact pools four tasks and no longer installs at
    all — the Phase 2c guards need rates it does not carry — so the fixture is MEASURED
    here instead, by running the real `calibrate_model` over synthetic null arms, with
    the real arms' provenance so freshness still resolves against real measurements.
    The tamper mechanics below are unchanged; what moved is the artifact they tamper
    with, and it moved because the committed one is superseded rather than because a
    fabricated fixture was more convenient. `tests/test_loop_validate.py` pins the
    committed artifact's own refusal separately.
    """
    import tempfile

    from loop.calibrate import MODEL_TASKS, calibrate_model
    from loop.calibrate import SUPPORTED as GAIN_SET

    fp = dict(real_fingerprint())
    results_dir = Path(tempfile.mkdtemp(prefix="p2b-closing-arms-"))
    for labels_group, subset in ((FULL_ARMS, False), (SUBSET_ARMS, True)):
        for label in labels_group:
            passes = _null_counts(label, subset)
            tasks = {}
            for task, count in passes.items():
                attempts = 10 if subset else STANDARD_ATTEMPTS[task]
                tasks[task] = {
                    "split": SPLIT_OF[task],
                    "attempts": attempts,
                    "passes": count,
                    "pass_fraction": round(count / attempts, 4),
                }
            if mutate_arms is not None:
                mutate_arms(label, tasks)
            record = {"fingerprint": dict(fp), "tasks": tasks}
            if subset:
                record["filter"] = sorted(passes)
            (results_dir / f"{label}.json").write_text(json.dumps(record))
    labels = sorted(FULL_ARMS + SUBSET_ARMS)
    model = calibrate_model(labels, results_dir, GAIN_SET, coverage=MODEL_TASKS)
    assert model["fitness"]["fit"] is True, "fixture precondition: these arms must be fit"
    return model


def copy_model(tmp_path: Path, mutate=None, name="model-r2.json") -> Path:
    """An installable artifact, optionally with one field edited.

    A tamper test is about the loader's checks, so what matters is that the artifact is
    one the loader would otherwise install — see `installed_model` for why that is no
    longer the committed file.
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


_PINNED_MODEL_PATH: Path | None = None


def pinned_model_path() -> Path:
    """The installable artifact, written once per session and reused.

    Built rather than committed: an artifact carrying rates for CMP-5/6/7 must come
    from a campaign that measured them, and that campaign has not run. Writing one into
    `iterations/` would install fabricated guard rates into the real pipeline, which is
    the one thing this fixture must never do — so it lives in a temp dir and never
    leaves the test session.
    """
    global _PINNED_MODEL_PATH
    if _PINNED_MODEL_PATH is None:
        import tempfile

        d = Path(tempfile.mkdtemp(prefix="p2b-closing-model-"))
        path = d / "model-r2.json"
        path.write_text(json.dumps(installed_model(), indent=2) + "\n")
        _PINNED_MODEL_PATH = path
    return _PINNED_MODEL_PATH


def installed_calibration():
    """The calibration the loader builds from an artifact of the pinned shape."""
    return load_calibration(pinned_model_path())


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
    cal = installed_calibration()
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
    power = committed_model()["fitness"]["power"]
    assert set(power["stage1_only"]) == {"per_task", "gain_gate"}
    assert set(power["stage1_only"]["per_task"]) == SUPPORTED
    assert set(power["stage1_only"]["gain_gate"]) == {"held_in", "held_out"}


def test_power_publishes_joint_stage1_x_stage2_rows_for_every_declared_alternative():
    """Contract amendment 3: +0.2 on a single carrier (each supported task, one row
    each) and +0.3/+0.5 uniform across the supported set, each reported per evidence
    split as well as in total. Exact fractions, floats beside them for reading."""
    rows = committed_model()["fitness"]["power"]["end_to_end"]["rows"]
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

    cal = installed_calibration()
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
        # Only the GAIN set is sampled, and the guards are held equal on both sides.
        # The predicate models the supported-set mean and nothing else, while
        # `evaluate()` also applies whole-suite vetoes (a collapse on ANY task, guards
        # included) — sampling the guards too would compare the predicate against
        # behavior it never claimed to reproduce.
        base = {t: rng.randint(0, STANDARD_ATTEMPTS[t]) for t in sorted(SUPPORTED)} | _HELD
        cand = {t: rng.randint(0, STANDARD_ATTEMPTS[t]) for t in sorted(SUPPORTED)} | _HELD
        b = _results(base, STANDARD_ATTEMPTS)
        c = _results(cand, STANDARD_ATTEMPTS)
        # The predicate reads the GAIN set alone — that is the mean the rule judges —
        # even though the results above carry every covered task.
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

    cal = installed_calibration()
    attempts = {t: 10 for t in COVERED}
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
        # Gain set only; the guards are held equal on both sides (see the stage-1 sweep
        # above for why).
        base = {t: rng.randint(0, 10) for t in sorted(SUPPORTED)} | _UNMOVED
        cand = {t: rng.randint(0, 10) for t in sorted(SUPPORTED)} | _UNMOVED
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
            confirm_tasks=tuple(sorted(COVERED)),
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
    under that reduced pool — the slack that says how close the verdict came.

    Re-keyed to the Phase 2c pooling, and the margins now carry the refusal itself, so
    this test reads the reason `fitness.fit` is false rather than only the shape of the
    rows: dropping ONE subset arm moves the held-in quantile a whole grain bucket, from
    4/9 to 1/3. That is what "not stable" means here, said in numbers.
    """
    stability = committed_model()["fitness"]["stability"]
    arms = set(P2C_ARMS)
    for split in ("held_in", "held_out"):
        margins = stability[split]["leave_one_out"]
        assert set(margins) == arms, split
        for label, row in margins.items():
            assert Fraction(row["quantile"]) >= -1
            assert isinstance(row["bucket"], int)
            cov = Fraction(row["coverage_at_full_quantile"])
            assert 0 <= cov <= 1, (split, label)
            assert Fraction(row["slack"]) == cov - Fraction(39, 40)

    held_in = stability["held_in"]
    moved = held_in["moved_excluding"]
    assert set(moved) == {"p2c-null-cmp-d"}, (
        "one arm carries the refusal; if that has become more or fewer arms, the "
        "recorded reason for fit=false has changed and the docs saying so are stale"
    )
    assert Fraction(held_in["full_quantile"]) == Fraction(4, 9)
    assert Fraction(moved["p2c-null-cmp-d"]) == Fraction(1, 3)
    # And the margin published for that arm agrees with the summary line above — the
    # two are written by different code paths and could disagree.
    assert Fraction(margins["p2c-null-cmp-d"]["quantile"]) == Fraction(3, 5), "held_out is stable"
    assert Fraction(held_in["leave_one_out"]["p2c-null-cmp-d"]["quantile"]) == Fraction(1, 3)
    assert held_in["pass"] is False and stability["held_out"]["pass"] is True


# ---------------------------------------------------------------------------
# 6. CONDITIONAL FALSE-CONFIRM RATE (S6)
# ---------------------------------------------------------------------------


def test_the_false_confirm_block_publishes_a_marginal_rate_and_a_null_conditional():
    """Every real judgment compares a candidate against ONE recorded baseline arm, not
    against a fresh draw from the null. The marginal false-CONFIRM probability averages
    over baselines that will never be used again; the conditional one is the number that
    describes the comparison the pipeline actually makes.

    The Phase 2c pooling has no designated baseline in it — `r2-null-full-a` belongs to
    the superseded round-2 protocol — so that number does not exist here, and the
    artifact says so with `null` and a `baseline_note` rather than inventing one. What
    is asserted is exactly that: a real marginal, an absent conditional, and a stated
    reason. `installed_model()` covers the other branch, where the arm IS present and
    every conditional field is filled in.
    """
    from loop.calibrate import DESIGNATED_BASELINE

    fc = committed_model()["fitness"]["false_confirm"]
    assert fc["baseline_arm"] is None
    assert fc["conditional"] is None and fc["conditional_float"] is None
    assert "baseline_counts" not in fc, "no baseline, so no counts to publish"
    assert DESIGNATED_BASELINE in fc["baseline_note"]
    assert "no designated baseline" in fc["baseline_note"]

    assert Fraction(fc["marginal"]) >= 0
    assert fc["marginal_float"] == pytest.approx(float(Fraction(fc["marginal"])))
    assert set(fc["by_evidence_split"]) == {"held_in", "held_out"}
    total = sum(Fraction(v["marginal"]) for v in fc["by_evidence_split"].values())
    assert total == Fraction(fc["marginal"]), "the split rows must sum to the total"
    for row in fc["by_evidence_split"].values():
        assert row["conditional"] is None


def _committed_pooling() -> tuple[dict[str, Fraction], dict[str, int], dict[str, Fraction]]:
    """The committed artifact's own rates, standard attempt counts and stage-1 quantiles.

    Read from the artifact, never hardcoded: the attempt counts come from the GRAIN rows
    (which record what they were computed over), so a re-derivation cannot silently use
    a different denominator from the one the number was published at.
    """
    from loop.calibrate import null_gain_quantile

    model = committed_model()
    tasks = sorted(SUPPORTED)
    rates = {t: Fraction(model["null_model"][t]["null_rate"]) for t in tasks}
    attempts = {
        t: int(n)
        for split in ("held_in", "held_out")
        for t, n in model["fitness"]["grain"][split]["attempts"].items()
    }
    assert set(attempts) == set(tasks), attempts
    level = Fraction(model["coverage_level"])
    quantiles = {
        split: null_gain_quantile(
            {t: rates[t] for t in tasks if SPLIT_OF[t] == split},
            {t: attempts[t] for t in tasks if SPLIT_OF[t] == split},
            {t: attempts[t] for t in tasks if SPLIT_OF[t] == split},
            level,
        )
        for split in ("held_in", "held_out")
    }
    return rates, attempts, quantiles


def _independent_diff_pmf(
    n: int, p_base: Fraction, p_cand: Fraction | None = None
) -> dict[Fraction, Fraction]:
    """P((cand - base)/n) for independent Binomial(n, p_base) and Binomial(n, p_cand)
    draws, convolved here. `p_cand` defaults to `p_base` — the null-versus-null case.

    `loop.calibrate` has `_task_diff_pmf` for this and caches it. This convolution is
    written out again on purpose: the difference distribution is the one step where a
    quiet error (an off-by-one in the sign, a reused cache keyed on the wrong argument)
    would move every published number at once and look like nothing.
    """
    from loop.calibrate import _binom_pmf

    pmf_base = _binom_pmf(n, p_base)
    pmf_cand = _binom_pmf(n, p_base if p_cand is None else p_cand)
    out: dict[Fraction, Fraction] = {}
    for a in range(n + 1):
        for b in range(n + 1):
            d = Fraction(b - a, n)
            out[d] = out.get(d, Fraction(0)) + pmf_base[a] * pmf_cand[b]
    return out


def test_the_committed_artifact_is_exactly_what_the_writer_produces_from_its_own_arms():
    """The artifact is reproducible from the record, byte for byte.

    Two things this pins at once. A hand-edit to `model-r2.json` fails here even if it
    survives the loader's own recomputation, because this compares the WHOLE file and
    not just the blocks `recompute_model` re-runs. And a change to the writer that moves
    a published number fails here too, which is the point: the committed artifact and
    the code that writes it are one claim, so they get regenerated together or not at
    all.

    It is also the assertion that the null-conditional fix changed nothing else. That
    edit rewrote six rows' conditional fields; every marginal number in this file is
    unchanged, and "unchanged" is only worth saying if something checks it.
    """
    from loop.calibrate import MODEL_TASKS, SUPPORTED, calibrate_model

    regenerated = calibrate_model(list(P2C_ARMS), RESULTS, SUPPORTED, coverage=MODEL_TASKS)
    assert regenerated == committed_model()


def test_the_marginal_false_confirm_rate_matches_a_direct_enumeration():
    """Independently re-derived: draw BOTH arms from the pooled null rates and count
    the mass that CONFIRMs, over the ten-arm pooling the artifact actually publishes.

    This was the conditional rate's verifier while a designated baseline existed in the
    pool. It is not retired now that one does not — the marginal is the number the
    artifact still publishes, so the marginal is what gets a second implementation. The
    conditional is asserted ABSENT in the same breath, because "nobody computed it" and
    "it came out zero" are the two readings this artifact must never blur.
    """
    from itertools import product

    from loop.calibrate import _stage1_verdict

    rates, attempts, quantiles = _committed_pooling()
    tasks = sorted(SUPPORTED)
    per_task = [sorted(_independent_diff_pmf(attempts[t], rates[t]).items()) for t in tasks]

    total = Fraction(0)
    for combo in product(*per_task):
        p = Fraction(1)
        for _, prob in combo:
            p *= prob
        if not p:
            continue
        diffs = {t: d for t, (d, _) in zip(tasks, combo, strict=True)}
        if _stage1_verdict(diffs, SPLIT_OF, quantiles)[0]:
            total += p

    fc = committed_model()["fitness"]["false_confirm"]
    assert Fraction(fc["marginal"]) == total
    assert fc["conditional"] is None, "absent, not zero"


# ---------------------------------------------------------------------------
# 5. PR RENDERING (S5)
# ---------------------------------------------------------------------------

LEGACY_RULE_TEXT = "## Validation — acceptance rule `Δ_in ≥ 0, Δ_ho ≥ 0, max(Δ_in, Δ_ho) > 0`"


# The covered guards, unmoved, at the confirmation-shaped attempt count every fixture
# below uses. They are required in both arms (`_require_supported` reads the COVERED
# set) and identical on both sides, so they enter no split mean and move no verdict.
_UNMOVED = {"CMP-5": 5, "CMP-6": 5, "CMP-7": 6}
# The same idea at the suite's standard attempt counts (3 held-in / 5 held-out).
_HELD = {"CMP-5": 1, "CMP-6": 2, "CMP-7": 2}


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

    cal = installed_calibration()
    attempts = {t: 10 for t in COVERED}
    base = _results({"A1": 4, "G4": 1, "G5": 4, "G2": 3, **_UNMOVED}, attempts)
    cand = _results({"A1": 10, "G4": 6, "G5": 10, "G2": 9, **_UNMOVED}, attempts)
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
    # The artifact's OWN source string, read off the record rather than spelled out:
    # the fixture's artifact lives in a temp dir (see `installed_model`), and pinning a
    # literal path here would test the fixture instead of the rendering.
    assert record.rule["calibration"]["source"] in body, "the artifact's own source"
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
    attempts = {t: 10 for t in COVERED}
    base = _results({"A1": 4, "G4": 1, "G5": 4, "G2": 3, **_UNMOVED}, attempts)
    cand = _results({"A1": 6, "G4": 1, "G5": 4, "G2": 3, **_UNMOVED}, attempts)
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
    cal = installed_calibration()
    first = _first_decision(record)
    # A DIFFERENT pair for the confirmation, deliberately: reusing the first pass's
    # numbers would let this test pass on a body that rendered stage 1 and called it
    # the confirmation. The two stages' deltas must be distinguishable.
    attempts = {t: 10 for t in COVERED}
    conf_base = _results({"A1": 3, "G4": 1, "G5": 3, "G2": 3, **_UNMOVED}, attempts)
    conf_cand = _results({"A1": 10, "G4": 7, "G5": 10, "G2": 9, **_UNMOVED}, attempts)
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
    cal = installed_calibration()
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

    record, base, cand = _calibrated_record(tmp_path)
    first = _first_decision(record)

    # A SECOND null model, MEASURED the same way from the same protocol and differing by
    # one attempt on one arm (G4 loses a pass on the first subset arm, so the pool
    # moves). It is fit, fresh, and pinned to the right covered set, so every check the
    # loader has ever had passes on it: it installs cleanly. Only the digest of the
    # rates the FIRST decision was judged against can tell that this is not that model.
    def one_fewer_g4_pass(label, tasks):
        if label == SUBSET_ARMS[0] and tasks["G4"]["passes"] > 1:
            tasks["G4"]["passes"] -= 1
            tasks["G4"]["pass_fraction"] = round(tasks["G4"]["passes"] / tasks["G4"]["attempts"], 4)

    swapped = installed_model(one_fewer_g4_pass)
    assert swapped["fitness"]["fit"] is True, "a swap only proves something if it installs"
    assert (
        swapped["null_model"]["G4"]["null_rate"]
        != installed_model()["null_model"]["G4"]["null_rate"]
    ), "fixture precondition: the second model really did pool differently"
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

    attempts = {t: 10 for t in COVERED}
    base = _results({"A1": 4, "G4": 1, "G5": 4, "G2": 3, **_UNMOVED}, attempts)
    cand = _results({"A1": 9, "G4": 5, "G5": 9, "G2": 8, **_UNMOVED}, attempts)
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
