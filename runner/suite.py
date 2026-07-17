"""Suite-level runner: all tasks, per-split aggregation, one results JSON."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from runner import guard
from runner.delta import split_rate
from runner.gemma_env import gemma_fingerprint
from runner.run import load_done, run_task
from runner.spec import TaskSpec

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def _prior_fingerprint(out_path: Path, jsonl_path: Path) -> dict | None:
    """The fingerprint a prior baseline under this label was stamped with, from the
    results JSON if present, else the first JSONL record. ``None`` when no prior run
    exists."""
    if out_path.is_file():
        return json.loads(out_path.read_text()).get("fingerprint")
    if jsonl_path.is_file():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                return json.loads(line)
    return None


def run_suite(
    tasks: list[TaskSpec],
    label: str,
    only: set[str] | None = None,
    attempts: int | None = None,
    fingerprint: dict | None = None,
    results_dir: Path = RESULTS_DIR,
    force: bool = False,
    log=print,
) -> dict:
    # injected is a TEST SEAM: tests pass a synthetic fingerprint precisely
    # because there is no live dist/gemma checkout to recompute against, so
    # the mid-suite drift guard below is skipped for injected fingerprints.
    injected = fingerprint is not None
    fingerprint = fingerprint if injected else gemma_fingerprint()
    jsonl_path = results_dir / f"{label}.jsonl"
    out_path = results_dir / f"{label}.json"
    # --force overwrites this label from scratch: discard prior records (so nothing
    # resumes) and skip the resume-guard + partial-overwrite guards entirely.
    if force:
        jsonl_path.unlink(missing_ok=True)
    else:
        # resume-guard: refuse a stale baseline LOUDLY and up front, before spending
        # any live-model attempts. Additive gemma bumps (behavior_key unchanged) pass
        # and resume; a config/model/verifier/working-tree change raises StaleBaseline.
        # gemma_sha alone moving is not a mismatch — that's the whole point.
        prior = _prior_fingerprint(out_path, jsonl_path)
        if prior is not None:
            guard.assert_resumable(prior, fingerprint)
        if only and out_path.is_file():
            existing = json.loads(out_path.read_text())
            if "filter" not in existing:
                raise RuntimeError(
                    f"{out_path} holds a FULL suite run; a --only run would overwrite it "
                    f"with a partial one — use a different label for partial runs"
                )
    done = load_done(jsonl_path, log=log)
    log(
        f"suite '{label}' against {fingerprint['gemma_sha'][:10]}"
        f"{'+dirty' if fingerprint['gemma_dirty'] else ''} "
        f"(config v{fingerprint['config_version']}, model {fingerprint['model']})"
    )
    results: dict = {"fingerprint": fingerprint, "tasks": {}, "summary": {}}
    if only:
        results["filter"] = sorted(only)
    for spec in tasks:
        if only and spec.name not in only:
            continue
        # drift guard: a mid-suite edit to dist/gemma would stamp every later
        # attempt with the stale suite-start fingerprint — recompute before
        # each task and abort on any difference.
        if not injected:
            current = gemma_fingerprint()
            if current != fingerprint:
                changed = ", ".join(
                    f"{key}: {fingerprint.get(key)!r} -> {current.get(key)!r}"
                    for key in sorted(set(fingerprint) | set(current))
                    if fingerprint.get(key) != current.get(key)
                )
                raise RuntimeError(
                    f"gemma state changed mid-suite ({changed}); aborting — "
                    f"restart with a fresh label or clean state"
                )
        log(f"task {spec.name} [{spec.split}] ...")
        tr = run_task(spec, fingerprint, jsonl_path, done=done, attempts=attempts, log=log)
        results["tasks"][spec.name] = {
            "split": spec.split,
            "cluster": spec.cluster,
            "expected_baseline": spec.expected_baseline,
            "attempts": len(tr.records),
            "passes": sum(1 for r in tr.records if r["passed"]),
            "pass_fraction": round(tr.pass_fraction, 4),
            "outcomes": [r["outcome"] for r in tr.records],
        }
    results["summary"] = {
        "held_in_rate": round(split_rate(results, "held_in"), 4),
        "held_out_rate": round(split_rate(results, "held_out"), 4),
    }
    fd, tmp_name = tempfile.mkstemp(dir=results_dir, prefix=f".{label}.", suffix=".json.tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(results, indent=2) + "\n")
    os.replace(tmp_name, out_path)
    log(f"wrote {out_path}")
    return results
