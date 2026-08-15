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
    record = ValidationRecord(
        candidate_id=candidate.id,
        label=label,
        accepted=d["accepted"],
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
    )
    log(
        f"candidate {candidate.id}: Δ_in={d['delta_in']:+.4f} Δ_ho={d['delta_ho']:+.4f} "
        f"-> {'ACCEPTED' if d['accepted'] else 'REJECTED'}"
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
