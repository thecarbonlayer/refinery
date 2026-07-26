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
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from loop.artifacts import Candidate, ValidationRecord
from loop.config_edit import CONFIG_REL, apply_candidate
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


def _run_runner(label: str, only: list[str] | None, attempts: int | None) -> None:
    """Run the suite in a fresh interpreter (same venv), cwd at harness-editor."""
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
        baseline_fingerprint=d["baseline_fingerprint"],
        candidate_fingerprint=d["candidate_fingerprint"],
    )
    log(
        f"candidate {candidate.id}: Δ_in={d['delta_in']:+.4f} Δ_ho={d['delta_ho']:+.4f} "
        f"-> {'ACCEPTED' if d['accepted'] else 'REJECTED'}"
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
