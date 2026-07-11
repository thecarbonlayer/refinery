"""Per-task runner: N attempts -> pass fraction, streamed to JSONL (resumable).

Attempts are expensive (each is a multi-turn live-model run, minutes each), so
every attempt is persisted the moment it finishes; a rerun with the same JSONL
skips (task, attempt_index) pairs already on disk. An attempt that raises is
recorded as outcome="error" and counts as a failure in the fraction — visible
in the record, never silently retried (that would be single-favorable-run
promotion by another name).
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from runner.spec import Attempt, TaskSpec


@dataclass
class TaskResult:
    spec: TaskSpec
    records: list[dict] = field(default_factory=list)

    @property
    def pass_fraction(self) -> float:
        n = len(self.records)
        return (sum(1 for r in self.records if r["passed"]) / n) if n else 0.0


def load_done(jsonl_path: Path) -> dict[tuple[str, int], dict]:
    done: dict[tuple[str, int], dict] = {}
    if jsonl_path.is_file():
        for line in jsonl_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[(rec["task"], rec["attempt"])] = rec
    return done


def run_task(
    spec: TaskSpec,
    fingerprint: dict,
    jsonl_path: Path,
    done: dict[tuple[str, int], dict] | None = None,
    attempts: int | None = None,
    log=print,
) -> TaskResult:
    done = done if done is not None else load_done(jsonl_path)
    n = attempts or spec.attempts
    result = TaskResult(spec)
    for i in range(n):
        if (spec.name, i) in done:
            result.records.append(done[(spec.name, i)])
            log(
                f"  {spec.name} attempt {i + 1}/{n}: (resumed) "
                f"{'PASS' if done[(spec.name, i)]['passed'] else 'FAIL'}"
            )
            continue
        t0 = time.monotonic()
        try:
            att = spec.run()
        except Exception:
            att = Attempt(False, "error", traceback.format_exc(limit=8))
        rec = {
            "task": spec.name,
            "split": spec.split,
            "cluster": spec.cluster,
            "expected_baseline": spec.expected_baseline,
            "attempt": i,
            "passed": att.passed,
            "outcome": att.outcome,
            "detail": att.detail[:2000],
            "approvals": att.approvals,
            "turns": att.turns,
            "duration_s": round(time.monotonic() - t0, 1),
            **fingerprint,
        }
        with jsonl_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        result.records.append(rec)
        log(
            f"  {spec.name} attempt {i + 1}/{n}: "
            f"{'PASS' if att.passed else 'FAIL'} ({att.outcome}, {rec['duration_s']}s)"
        )
    return result
