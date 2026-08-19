"""Validation wiring: candidate -> suite run -> Δ -> acceptance rule.

The candidate is applied to the carbon WORKING TREE (never committed —
rejected candidates must leave no trace in that repo, and the runner's
fingerprint records the dirty state honestly via ``dirty_sha``). The suite
runs in a FRESH SUBPROCESS: carbon's config values bind at import time, so an
in-process ``run_suite`` after editing the file would measure the old config.
The edit is reverted in a ``finally`` — pass, fail, or crash.

Acceptance is the Self-Harness rule as implemented in ``runner.delta``:
``Δ_in >= 0, Δ_ho >= 0, max(Δ_in, Δ_ho) > 0`` — no partial credit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from loop.acceptance import ACCEPT, SectionCalibration
from loop.acceptance import evaluate as rule_evaluate
from loop.artifacts import Candidate, ValidationRecord
from loop.calibrate import ANALYSIS_PATH as COMPACTION_ANALYSIS_PATH
from loop.config_edit import CONFIG_REL, apply_candidate
from loop.observed_coverage import activity_from_rows, partition_deltas, select_attempts
from loop.surface_sweep import sweep as run_sweep
from runner.carbon_env import CARBON_ROOT, _git
from runner.delta import delta
from runner.suite import RESULTS_DIR

EDITOR_ROOT = Path(__file__).resolve().parents[1]


def require_clean_tree(carbon_root: Path = CARBON_ROOT) -> None:
    """A candidate must be measured against exactly one harness state; a tree
    that is already dirty would entangle the candidate with unknown edits."""
    status = _git(carbon_root, "status", "--porcelain").strip()
    if status:
        raise RuntimeError(
            f"carbon working tree is not clean — refusing to apply a candidate "
            f"on top of unrelated changes:\n{status}"
        )


def revert_config(carbon_root: Path = CARBON_ROOT) -> None:
    _git(carbon_root, "checkout", "--", str(CONFIG_REL))


def run_harness_gates(carbon_root: Path = CARBON_ROOT, editor_root: Path = EDITOR_ROOT) -> dict:
    """Both repos' own test suites, run against the candidate as applied.

    The task suite answers "does the agent do better work?". It cannot answer "is the
    harness still sound?" — a config value can turn either repo red without moving a
    single task score, because no task asserts on carbon's invariants or on this
    repo's fixtures. Three such breakages reached a merged branch before this existed:
    a checked-in-defaults test pinned to a strategy the loop is allowed to change, a
    fault-injection guard whose recovery rule assumed one strategy's call ordering,
    and two premise probes built on a tail_fraction the surface never permitted.

    Run with the candidate APPLIED, so what is gated is the state the loop proposes to
    ship — and run BEFORE the suite, because a broken harness does not deserve forty
    minutes of model time.

    Three checks, and the third is a different question from the first two. Those ask
    "is the harness sound at the point this candidate proposes?"; the sweep asks "is it
    sound at every point the surface publishes?". A no to the first is the candidate's
    fault. A no to the third never is — it means a test pinned whatever value happened
    to ship, and the loop is about to be blocked by its own instrument. The sweep costs
    roughly a fifth of a suite run, against a suite run of about forty minutes.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # This shells out to `pytest`, so calling it from inside a test spawns a nested
        # full run — minutes of it, once per call. Fail loudly instead: a test that
        # wants a verdict passes one through `run_gates=`.
        raise RuntimeError(
            "run_harness_gates() shells out to pytest and must not run inside pytest; "
            "inject a stub via validate_candidate(run_gates=...)"
        )
    checks = (
        ("carbon_verify", carbon_root, ["uv", "run", "verify"]),
        ("refinery_pytest", editor_root, ["uv", "run", "pytest", "-q"]),
    )
    out: dict = {"passed": True, "checks": {}}
    for name, cwd, cmd in checks:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        ok = proc.returncode == 0
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-12:]
        out["checks"][name] = {"passed": ok, "exit_code": proc.returncode}
        if not ok:
            out["passed"] = False
            out["checks"][name]["tail"] = tail
    if not out["passed"]:
        # A red suite at the shipped point makes every swept point red too, so the
        # sweep would spend minutes restating one failure.
        return out
    # Both suites are green HERE — but "here" is one point on a surface the loop may
    # move anywhere. The two checks above pass a candidate that is legal; the sweep
    # asks whether the checks themselves survive every OTHER legal value, which is
    # the question that has been wrong four times. Without it a fragile test turns
    # into a candidate veto, reported as a harness break rather than as a worse agent.
    report = run_sweep(carbon_root, editor_root, log=lambda line: None)
    reds = [label for label, point in report["points"].items() if not point["passed"]]
    out["checks"]["surface_sweep"] = {
        "passed": report["passed"],
        "probed": report["probed"],
        "red_points": reds,
    }
    if not report["passed"]:
        out["passed"] = False
        out["checks"]["surface_sweep"]["tail"] = [
            f"legal value {label} turns a suite red — fix the test, not the candidate"
            for label in reds
        ]
    return out


def coverage_note(
    candidate: Candidate,
    per_task: dict[str, float],
    cohort_paths: list[tuple[Path, Path]] | None = None,
) -> dict:
    """Split this candidate's per-task movements by whether the edited knobs reach them.

    Derived from every recorded run, not from an authored table: `loop.knob_coverage`
    says of itself that it is "a human-audited claim with mechanical guardrails, not a
    proof", and an audit found six of its rows false at once. What a task actually did
    is in the attempt logs.

    Reported, never applied. Dropping the unreachable half and re-deciding would be a
    second acceptance rule hidden inside the first — and the direction it moves the
    verdict depends entirely on which tasks happen to be unreachable that day. On
    iteration 3, dropping only the tasks that HURT the candidate flips it to accepted;
    dropping symmetrically leaves it rejected. A rule you can point either way is not a
    rule. So the record carries the split and a person reads it.
    """
    knobs = sorted(candidate.fields)
    # The EXACT pair this validation compared, not every log on disk. Pooling all of
    # them mixed runner versions, config versions and partial runs into one claim —
    # `delta` refuses to compare results across runner versions, and a coverage claim
    # assembled across them is the same mistake with none of the refusal. It also let a
    # months-old candidate permanently seed activity for later validations.
    pairs = list(cohort_paths or [])
    if not pairs:
        return {"knobs": knobs, "error": "no cohort supplied; coverage not derived"}
    # EVERY supplied arm must be readable. Filtering to the ones that happen to exist
    # silently derived coverage from one arm: with the baseline log missing it read the
    # candidate alone, reported no error, and still emitted `unreachable_proven` — a
    # claim about a comparison from half of it. A partial cohort is unusable, not smaller.
    absent = [str(p) for pair in pairs for p in pair if not p.exists()]
    if absent:
        return {"knobs": knobs, "error": f"cohort incomplete, missing: {', '.join(absent)}"}
    rows: list[dict] = []
    provenance: list[dict] = []
    try:
        # Bound to the attempts each result JSON SUMMARIZES, not to whatever the log
        # holds. `delta` compares the summaries; a coverage claim derived from rows it
        # never saw describes a different cohort under the same filename.
        for jsonl, result_json in pairs:
            selected, note = select_attempts(jsonl, result_json)
            rows += selected
            provenance.append(note)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {"knobs": knobs, "error": f"cohort unusable: {exc}"}
    activity = activity_from_rows(rows)
    # Unreachable by EVERY edited knob. A candidate editing two knobs reaches a task if
    # EITHER does, so the sets intersect. The accumulator starts at None rather than {}
    # because an empty first set is meaningful — it means that knob reaches everything,
    # which must swallow the whole result, not be mistaken for "nothing seen yet".
    # Set algebra, and the naive version is wrong. Intersecting the two grades
    # SEPARATELY loses a task that one knob proves unreachable and another only
    # probably does: it appears in neither intersection and lands in `evidence`,
    # asserting the candidate CAN reach a task no edited knob shows any route to.
    #   proven_all   = ∩ proven_i
    #   excluded_all = ∩ (proven_i ∪ probable_i)
    #   probable_all = excluded_all − proven_all
    proven_sets: list[set[str]] = []
    excluded_sets: list[set[str]] = []
    for knob in knobs:
        split = partition_deltas(knob, per_task, activity)
        proven_sets.append(set(split["unreachable_proven"]))
        excluded_sets.append(set(split["unreachable_proven"]) | set(split["unreachable_probable"]))
    proven_all = set.intersection(*proven_sets) if proven_sets else set()
    excluded_all = set.intersection(*excluded_sets) if excluded_sets else set()
    probable_all = excluded_all - proven_all
    proven = {t: v for t, v in per_task.items() if t in proven_all}
    probable = {t: v for t, v in per_task.items() if t in probable_all}
    evidence = {t: v for t, v in per_task.items() if t not in excluded_all}
    return {
        "knobs": knobs,
        "evidence": evidence,
        "unreachable_proven": proven,
        "unreachable_probable": probable,
        "cohort": {"files": provenance},
        "note": (
            "`unreachable_proven` movements are on tasks NO value of the edited knob(s) "
            "can affect. `unreachable_probable` is weaker: the knob showed no activity "
            "to act on, but could CREATE it (lowering compaction.trigger_fraction makes "
            "a task compact that never has), so absence there is a prompt to re-measure "
            "rather than a verdict. Neither is subtracted from the delta."
        ),
    }


def causal_verdict(d: dict, coverage: dict, baseline: dict) -> dict:
    """Acceptance recomputed with impossible attributions removed.

    Iteration 3 was rejected on a Δ_in of −0.118 built from four tasks, two of which
    build agents with no tool registry at all against a candidate that edited
    `tool_output`. The raw rule was applied correctly to real measurements; the
    measurements just did not mean what the rule read them as. Recording that beside the
    verdict — the previous state of this code — leaves the noisy verdict deciding, so
    the incident stayed unsolved. The causal verdict decides now; the raw one is kept as
    audit evidence, never discarded.

    Five choices, each load-bearing:

    - Only PROOF-grade exclusions are removed. Evidence-grade ones stay fully eligible:
      `compaction`'s absence can be created by the same knob's `trigger_fraction`, so
      treating it as impossible would discard real movement.
    - A movement is replaced with ZERO, not dropped. Dropping shrinks the split's
      denominator, which changes the mean for a second, unrelated reason and makes two
      candidates excluding different tasks incomparable.
    - Multi-knob exclusion is the intersection, already computed in `coverage_note`: a
      task reached by any edited knob is reached.
    - The catastrophic per-task veto skips the same tasks. Leaving it alone would let a
      1.00 → 0.00 swing on a task the knob cannot touch veto by itself, which is the
      original failure wearing a different hat.
    - Lives in `loop/`, not `runner/`. `runner_sha` is a content hash of that package
      and every recorded baseline is stamped with it; a governance rule must not cost a
      re-measurement.
    """
    from runner.delta import acceptance

    excluded = set(coverage.get("unreachable_proven", {}))
    splits = {name: meta["split"] for name, meta in baseline["tasks"].items()}
    per_task = {n: (0.0 if n in excluded else v) for n, v in d["per_task"].items()}

    def mean(split: str) -> float:
        vals = [v for n, v in per_task.items() if splits.get(n) == split]
        return sum(vals) / len(vals) if vals else 0.0

    d_in, d_ho = mean("held_in"), mean("held_out")
    catastrophic = {n: c for n, c in d["catastrophic_regressions"].items() if n not in excluded}
    verdict = acceptance(d_in, d_ho)
    return {
        "accepted": verdict["accepted"] and not catastrophic,
        "delta_in": d_in,
        "delta_ho": d_ho,
        "excluded": sorted(excluded),
        "per_task": per_task,
        "catastrophic_regressions": catastrophic,
        "raw": {
            "accepted": d["accepted"],
            "delta_in": d["delta_in"],
            "delta_ho": d["delta_ho"],
            "catastrophic_regressions": d["catastrophic_regressions"],
        },
    }


# The sections the three-outcome rule may DECIDE, and on what footing.
#
# `tool_output` earned its place from the six unchanged runs: proof-grade exclusions
# (a task with no tool registry cannot be reached by any value of the knob) and a null
# measurement the one-attempt allowance fits. It consults NO artifact and is unchanged
# by everything below.
#
# `compaction` is the opposite case and enters on the opposite footing. Those same six
# runs showed two-attempt held-in swings on it, which a one-attempt allowance would
# false-reject, and its exclusions can never be proof-grade — `trigger_fraction`
# belongs to the knob, so the knob can CREATE the activity whose absence would be the
# exclusion. It therefore enters ONLY through a fresh `SectionCalibration`: a supported
# task set and bounds measured by null arms whose PROVENANCE — runner_sha, config
# version, model — matches the measurements being judged. No artifact, or one measured
# in a different world, and the section falls back to the causal verdict exactly as
# before (contract §4 and its 2026-08-19 amendment).
RULE_SECTIONS = frozenset({"tool_output", "compaction"})

# Sections whose entry into the rule is CONDITIONAL on that fresh calibration. A
# section outside this set decides on its own footing and never reads an artifact —
# which is what keeps `tool_output`'s behavior identical to the day it was calibrated.
_CALIBRATION_REQUIRED = frozenset({"compaction"})

# field -> section. `compaction` and `compaction_prompt` are two fields of ONE section:
# they change the same mechanism (when the harness compacts, and what it says while
# doing it) and the calibration measured that mechanism, not either field alone.
#
# `max_item_chars` used to map to `tool_output` here and is deliberately GONE: carbon
# locked it out of the editable surface at config v3 (it survives in the schema for
# chapter/API compatibility — see `locked_fields` in carbon's surface), so no candidate
# can ever carry it and the mapping was dead code pointing at the one section the rule
# decides. Recorded rather than silently deleted: a mapping removed with no reason is a
# mapping someone restores.
_FIELD_SECTION = {
    "tool_output": "tool_output",
    "compaction": "compaction",
    "compaction_prompt": "compaction",
}

# Where each calibrated section's artifact lives. `loop.calibrate` WRITES this path;
# importing the constant rather than re-spelling it means the writer and the reader
# cannot drift apart — a divergence would silently leave the section uncalibrated
# forever, with a perfectly good artifact sitting on disk.
_SECTION_ANALYSIS = {"compaction": COMPACTION_ANALYSIS_PATH}

# Tasks a confirmation pair must rerun for a section EVEN IF UNMOVED — the section's
# known trade-off guards and its reachable security checks. Movement-only selection
# lets a candidate confirm its gain without re-testing what the gain might cost:
#   E1 — retrieval economy, the priced-out direction for every tool_output change
#        (a budget large enough to flood the window passes E3/E4's letter and fails
#        E1; iteration 3's 16k control was rejected by exactly this guard).
#   C1, C2, C3 — the security conjuncts that emit `critical_failure`. Their leak
#        predicates read RAW results, so a tool_output edit cannot HIDE a leak, but
#        it changes what the model sees and therefore what it does next — and a leak
#        whose critical outcome appears only under the confirmation's higher attempt
#        counts must block there, not slip past a movement filter.
#   A1, G2, G5 — compaction's guards (contract §1). G4 is the MINER and is deliberately
#        absent: the task a candidate is mined from cannot also be the task that
#        vouches for it. G2 is the held-out member, so the guard set spans both splits.
_SECTION_CONFIRM_GUARDS = {
    "tool_output": frozenset({"E1", "C1", "C2", "C3"}),
    "compaction": frozenset({"A1", "G2", "G5"}),
}


def _repo_relative(path: Path) -> str:
    """A path fit to sit in a committed record: repo-relative, or the bare filename.

    This string reaches `iterations/` through the decision's `raw` and its reasons.
    This repo is public and machine paths have leaked into it twice; nothing here may
    carry `/Users/...` even when the artifact is a test's temp file.
    """
    try:
        return str(path.resolve().relative_to(EDITOR_ROOT))
    except ValueError:
        return path.name


def _provenance_mismatch(analysis: dict, fingerprint: dict, where: str) -> str:
    """The first way this artifact's provenance differs from the measurements', or "".

    Freshness is a question about the RESULTS being judged, not about the process doing
    the judging (contract §4, amendment 2026-08-19). The null protocol (§2) forces every
    arm to share `runner_sha`, `config_version` and `model` precisely because a Δ across
    any of them is not a measurement of noise — so a noise floor whose arms disagree
    with the baseline being judged was measured in a different world. Checking the LIVE
    process instead would pass an artifact that matches today's checkout while the
    results predate it, and fail one that matches the results perfectly after an
    unrelated `runner/` edit.
    """
    computed = analysis.get("computed_at_runner_sha")
    if computed != fingerprint.get("runner_sha"):
        return (
            f"{where} is STALE on runner_sha — computed at {str(computed)[:12]}, the "
            f"measurements were recorded at {str(fingerprint.get('runner_sha'))[:12]}"
        )
    arms = analysis.get("arms") or []
    if not arms:
        return f"{where} records no arms — there is no provenance to check it against"
    for arm in arms:
        for field in ("config_version", "model"):
            if arm.get(field) != fingerprint.get(field):
                return (
                    f"{where} is STALE on {field} — arm {arm.get('label')!r} measured at "
                    f"{arm.get(field)!r}, the measurements were recorded at "
                    f"{fingerprint.get(field)!r}"
                )
    return ""


def calibration_status(
    section: str, fingerprint: dict | None, *, analysis_path: Path | None = None
) -> tuple[SectionCalibration | None, str]:
    """`section_calibration()` plus the reason it came back empty.

    Public, and returning the reason, because "not calibrated" is a fact every caller
    must be able to EXPLAIN: missing artifact, artifact from other provenance, and
    unmeasured bound are three different states of the world, only one of which is
    fixed by re-running the null protocol. A bare None makes them one mystery.
    """
    path = analysis_path or _SECTION_ANALYSIS.get(section)
    if path is None:
        return None, f"section {section!r} has no calibration artifact to load"
    where = _repo_relative(path)
    if not path.is_file():
        return None, (
            f"section {section!r} is not calibrated: no calibration artifact at {where} — "
            "run the null-arm protocol and `python -m loop.calibrate <arm labels>` first"
        )
    if not fingerprint:
        # Fail CLOSED. Without the measurements' provenance there is nothing for the
        # artifact to be fresh for, and "could not check" must never read as "checked".
        return None, (
            f"section {section!r} is not calibrated: no results fingerprint was supplied, "
            "and freshness is judged against the provenance of the measurements being "
            "compared, never against the live process alone"
        )
    try:
        analysis = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"section {section!r} is not calibrated: {where} is unreadable ({exc})"
    drift = _provenance_mismatch(analysis, fingerprint, where)
    if drift:
        return None, (
            f"section {section!r} is not calibrated: {drift}. A bound measured under "
            "other provenance is not a measurement of this comparison, so the section "
            "falls back to the causal verdict until the arms are re-run."
        )
    noise = analysis.get("section_noise") or {}
    # A split named in `section_noise` with fewer than two arms behind it carries a
    # bound of 0.0 by construction — no pair, no spread. That is an ABSENCE of
    # measurement, and installing it as a threshold would give the section a floor of
    # zero, where any movement at all reads as evidence. `section_noise_arms` is what
    # tells the two apart; a genuine measured zero names its arms.
    arms = analysis.get("section_noise_arms") or {}
    # A `per_task` entry with no arm behind it is a task the tool was ASKED about and
    # no arm carried — `loop.calibrate` still writes the row, with an empty
    # `passes_by_arm`. Reading it as supported would put a name in the denominator
    # nothing was ever measured on, and then refuse every comparison for not having it.
    per_task = {
        name: row
        for name, row in (analysis.get("per_task") or {}).items()
        if (row or {}).get("passes_by_arm")
    }
    unmeasured = [s for s in ("held_in", "held_out") if s not in noise or len(arms.get(s, ())) < 2]
    if unmeasured or not per_task:
        return None, (
            f"section {section!r} is not calibrated: {where} carries no measured bound for "
            f"{', '.join(unmeasured) or 'any supported task'} — fewer than two arms covered "
            "it, and a bound no pair of arms produced is an absence, not a measurement of zero"
        )
    return (
        SectionCalibration(
            section=section,
            supported=frozenset(per_task),
            noise_in=float(noise["held_in"]),
            noise_ho=float(noise["held_out"]),
            guards=_SECTION_CONFIRM_GUARDS.get(section, frozenset()),
            source=where,
        ),
        "",
    )


def section_calibration(
    section: str, fingerprint: dict | None = None, *, analysis_path: Path | None = None
) -> SectionCalibration | None:
    """The section's measured bounds for THESE measurements, or None.

    None means the section stays on the causal verdict. It is returned for a section
    with no artifact at all (`tool_output`, which does not want one), for an artifact
    that has not been written yet, for one whose provenance differs from the results
    being judged, and for one whose bounds no pair of arms actually produced.

    `fingerprint` is the BASELINE results' fingerprint — the provenance freshness is
    judged against (contract §4, amendment 2026-08-19). Omitting it is not a way to
    skip the check: with nothing to be fresh FOR, the answer is None. Callers that
    need to know WHY should use `calibration_status()`, which returns the reason.

    `analysis_path` is a test seam, mirroring `run_runner=`/`run_gates=` elsewhere in
    this module; production reads `_SECTION_ANALYSIS`.
    """
    return calibration_status(section, fingerprint, analysis_path=analysis_path)[0]


def candidate_section(candidate: Candidate) -> str | None:
    """The ONE section this candidate edits, or None for unmapped/multi-section edits.

    Shared by `rule_disposition()` and `loop.cli confirm` so the two cannot disagree
    about which section's calibration a candidate belongs to — a disagreement would
    mean validating under a measured bound and confirming under a different one.
    """
    sections = {_FIELD_SECTION.get(f) for f in candidate.fields}
    if len(sections) != 1 or None in sections:
        return None
    return next(iter(sections))


def rule_disposition(candidate: Candidate, baseline: dict, results: dict, coverage: dict) -> dict:
    """Apply the three-outcome rule where it is calibrated; say why where it is not.

    The record carries this either way, so "the rule was not applied" is a stated fact
    with a reason rather than an absence someone reads as an oversight.

    ONE section at a time, always. An edit spanning two sections produces one Δ that
    belongs to neither section's evidence, and no calibration covers it — that was
    already true when `tool_output` was the only entry, and it stays true now that the
    set has two.
    """
    section = candidate_section(candidate)
    if section is None:
        sections = {_FIELD_SECTION.get(f) for f in candidate.fields}
        return {
            "applied": False,
            "why": (
                f"edited sections {sorted(s or 'unmapped' for s in sections)} are not "
                "calibrated for the three-outcome rule: it decides ONE calibrated section "
                f"at a time (calibrated: {', '.join(sorted(RULE_SECTIONS))}), and an "
                "unmapped field or an edit spanning two sections belongs to no section's "
                "evidence"
            ),
        }
    if section not in RULE_SECTIONS:
        return {
            "applied": False,
            "why": (
                f"edited section {section!r} is not calibrated for the three-outcome rule; "
                f"only {', '.join(sorted(RULE_SECTIONS))} carry the exclusions and measured "
                "limits the rule reads"
            ),
        }
    # Judged against the BASELINE's own provenance: these are the measurements the
    # bound has to be a bound for. `_parity` in `evaluate()` already refuses a
    # candidate arm recorded under a different runner, so one side is enough.
    calibration, why_not = calibration_status(section, baseline.get("fingerprint") or {})
    if calibration is None and section in _CALIBRATION_REQUIRED:
        return {"applied": False, "why": why_not}
    excluded = frozenset(coverage.get("unreachable_proven", ()))
    decision = rule_evaluate(
        baseline,
        results,
        excluded=excluded,
        always_confirm=_SECTION_CONFIRM_GUARDS.get(section, frozenset()),
        calibration=calibration,
        # Evidence-grade exclusions are CONTEXT on the decision and change no verdict.
        # Passed only for a calibrated section: `tool_output`'s decisions must stay
        # exactly what they were when the six null runs calibrated them, down to the
        # reason strings, and this adds one.
        unreachable_probable=(
            frozenset(coverage.get("unreachable_probable", ()))
            if calibration is not None
            else frozenset()
        ),
    )
    out = {"applied": True, **decision.to_json()}
    if calibration is not None:
        out["calibration"] = calibration.to_json()
    return out


def _run_runner(label: str, only: list[str] | None, attempts: int | None) -> None:
    """Run the suite in a fresh interpreter (same venv), cwd at the repo root."""
    cmd = [sys.executable, "-m", "runner.cli", "run", "--label", label]
    if only:
        cmd += ["--only", *only]
    if attempts:
        cmd += ["--attempts", str(attempts)]
    subprocess.run(cmd, cwd=EDITOR_ROOT, check=True)


def validate_candidate(
    candidate: Candidate,
    baseline_path: str | Path,
    label: str | None = None,
    carbon_root: Path = CARBON_ROOT,
    run_runner: Callable[[str, list[str] | None, int | None], None] = _run_runner,
    run_gates: Callable[[Path], dict] = run_harness_gates,
    results_dir: Path = RESULTS_DIR,
    log=print,
) -> ValidationRecord:
    """Full-suite validation of one candidate against a recorded baseline."""
    label = label or f"cand-{candidate.id}"
    baseline = json.loads(Path(baseline_path).read_text())
    require_clean_tree(carbon_root)
    new_config = apply_candidate(carbon_root, candidate)
    log(
        f"candidate {candidate.id}: applied "
        + ", ".join(f"{k}: {v['old']!r} -> {v['new']!r}" for k, v in candidate.fields.items())
        + f" (config v{new_config['version']})"
    )
    try:
        gates = run_gates(carbon_root)
        if not gates["passed"]:
            broken = [n for n, c in gates["checks"].items() if not c["passed"]]
            log(f"candidate {candidate.id}: HARNESS GATE FAILED ({', '.join(broken)}) — REJECTED")
            for name in broken:
                for line in gates["checks"][name].get("tail", []):
                    log(f"    {name}: {line}")
            # Vetoed before the suite runs, so there is no Δ to report and none is
            # invented: zeros here mean "not measured", which the gates field explains.
            return ValidationRecord(
                candidate_id=candidate.id,
                label=label,
                accepted=False,
                delta_in=0.0,
                delta_ho=0.0,
                gates=gates,
            )
        run_runner(label, None, None)
    finally:
        revert_config(carbon_root)
        require_clean_tree(carbon_root)  # the revert must actually have reverted
    results = json.loads((results_dir / f"{label}.json").read_text())
    d = delta(baseline, results)
    # Which of the movements are even ABOUT the candidate. A per-task delta on a task
    # the edited knob cannot reach is the grader's run-to-run variance wearing the
    # candidate's name. Iteration 3 was rejected on Δ_in −0.118 of which −1.33 came
    # from two tasks whose agents have no tool registry at all, while the candidate
    # edited `tool_output`. This does not change the verdict — the acceptance rule is
    # what it is, and quietly reweighting it here would be a second, hidden rule — it
    # records what the verdict was made of, next to the verdict.
    baseline_json = Path(baseline_path)
    coverage = coverage_note(
        candidate,
        d["per_task"],
        [
            (baseline_json.with_suffix(".jsonl"), baseline_json),
            (results_dir / f"{label}.jsonl", results_dir / f"{label}.json"),
        ],
    )
    # Count MOVEMENTS, not tasks. `per_task` carries every task including the unmoved
    # ones, so reporting its length overstated how many actually moved.
    moved = {t: v for t, v in d["per_task"].items() if v}
    flagged = {t for t in coverage.get("unreachable_proven", {}) if t in moved}
    unsure = {t for t in coverage.get("unreachable_probable", {}) if t in moved}
    if flagged or unsure:
        log(
            f"  note: of {len(moved)} tasks that moved, {len(flagged)} cannot be "
            f"reached by the edited knob(s) at any value and {len(unsure)} showed no "
            f"activity for it — see `coverage` in the record"
        )
    causal = causal_verdict(d, coverage, baseline)
    rule = rule_disposition(candidate, baseline, results, coverage)
    # Where the rule is calibrated it DECIDES, and `evaluate()` cannot return ACCEPT —
    # a CONFIRM candidate is promising, not accepted, until a fresh paired
    # confirmation run repeats its improvement. Elsewhere the causal verdict still
    # decides, and the record says which regime applied and why.
    accepted = (rule["outcome"] == ACCEPT) if rule["applied"] else causal["accepted"]
    record = ValidationRecord(
        candidate_id=candidate.id,
        label=label,
        accepted=accepted,
        delta_in=d["delta_in"],
        delta_ho=d["delta_ho"],
        per_task=d["per_task"],
        aggregate_accepted=d["aggregate_accepted"],
        regressions=d["regressions"],
        catastrophic_regressions=d["catastrophic_regressions"],
        baseline_metrics=d["baseline_metrics"],
        candidate_metrics=d["candidate_metrics"],
        metric_delta=d["metric_delta"],
        metric_not_compared=d["metric_not_compared"],
        metric_task_counts=d["metric_task_counts"],
        metric_attempt_counts=d["metric_attempt_counts"],
        metric_denominator_drift=d["metric_denominator_drift"],
        baseline_fingerprint=d["baseline_fingerprint"],
        candidate_fingerprint=d["candidate_fingerprint"],
        gates=gates,
        coverage=coverage,
        causal=causal,
        rule=rule,
    )
    if rule["applied"]:
        log(
            f"candidate {candidate.id}: rule outcome {rule['outcome']} "
            f"(Δ_in={rule['delta_in']:+.4f} Δ_ho={rule['delta_ho']:+.4f})"
        )
        if rule["outcome"] == "CONFIRM":
            log(
                "  eligible — needs a fresh paired confirmation on: "
                + ", ".join(rule["confirm_tasks"])
            )
        if rule.get("targeted_rerun"):
            log(
                "  behavioral security movement on "
                + ", ".join(rule["targeted_rerun"])
                + " — decided at confirmation by the predeclared Fisher comparison"
            )
        for reason in rule["reasons"]:
            log(f"    - {reason}")
    else:
        log(f"  rule not applied: {rule['why']}")
        log(
            f"candidate {candidate.id}: causal Δ_in={causal['delta_in']:+.4f} "
            f"Δ_ho={causal['delta_ho']:+.4f} -> "
            f"{'ACCEPTED' if causal['accepted'] else 'REJECTED'}"
        )
    if causal["accepted"] != d["accepted"]:
        log(
            f"  raw Δ_in={d['delta_in']:+.4f} Δ_ho={d['delta_ho']:+.4f} would have "
            f"{'ACCEPTED' if d['accepted'] else 'REJECTED'} — the difference is "
            f"{len(causal['excluded'])} task(s) the edited knob(s) cannot reach: "
            f"{', '.join(causal['excluded'])}"
        )
    if d["catastrophic_regressions"]:
        log(
            "  catastrophic per-task regression veto: "
            + ", ".join(
                f"{name} {change:+.4f}" for name, change in d["catastrophic_regressions"].items()
            )
        )
    return record


def dry_run(
    candidate: Candidate,
    only: list[str],
    attempts: int = 1,
    carbon_root: Path = CARBON_ROOT,
    run_runner: Callable[[str, list[str] | None, int | None], None] = _run_runner,
    log=print,
) -> None:
    """Exercise apply -> run -> revert on a small task subset, cheaply.

    Deliberately computes NO Δ and makes NO acceptance decision: partial
    (--only) results are refused by ``runner.delta`` by design. This exists to
    prove the pipeline mechanics before committing to a full validation run."""
    label = f"dryrun-{candidate.id}"
    require_clean_tree(carbon_root)
    apply_candidate(carbon_root, candidate)
    log(f"dry-run {candidate.id}: applied; running --only {' '.join(only)} x{attempts}")
    try:
        run_runner(label, only, attempts)
    finally:
        revert_config(carbon_root)
        require_clean_tree(carbon_root)
    log(f"dry-run {candidate.id}: done (results/{label}.json is partial — no Δ, by design)")
