"""Suite-level runner: all tasks, per-split aggregation, one results JSON."""

from __future__ import annotations

import json
from pathlib import Path

from runner.delta import split_rate
from runner.gemma_env import gemma_fingerprint
from runner.run import load_done, run_task
from runner.spec import TaskSpec

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def run_suite(
    tasks: list[TaskSpec],
    label: str,
    only: set[str] | None = None,
    attempts: int | None = None,
    log=print,
) -> dict:
    fingerprint = gemma_fingerprint()
    jsonl_path = RESULTS_DIR / f"{label}.jsonl"
    out_path = RESULTS_DIR / f"{label}.json"
    done = load_done(jsonl_path)
    log(
        f"suite '{label}' against {fingerprint['gemma_sha'][:10]}"
        f"{'+dirty' if fingerprint['gemma_dirty'] else ''} "
        f"(config v{fingerprint['config_version']}, model {fingerprint['model']})"
    )
    results: dict = {"fingerprint": fingerprint, "tasks": {}, "summary": {}}
    for spec in tasks:
        if only and spec.name not in only:
            continue
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
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    log(f"wrote {out_path}")
    return results
