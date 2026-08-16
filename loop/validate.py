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

from loop.artifacts import Candidate, ValidationRecord
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
    record = ValidationRecord(
        candidate_id=candidate.id,
        label=label,
        # The CAUSAL verdict. `causal["raw"]` keeps the unadjusted one as evidence.
        accepted=causal["accepted"],
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
    )
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
