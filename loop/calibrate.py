"""Measured noise from NULL arms: nothing changed between them, so every |Δ| this
module reports is sampling noise, not signal. Where `runner.delta` computes a Δ
between a baseline and a candidate and refuses anything partial, this tool exists
to consume the null-run PROTOCOL itself (contracts/phase2-calibration-contract.md
§2): one full-suite arm plus several `--only <supported set>` subset arms, all at
the same harness state, and turn their spread into thresholds a later rule
extension can use instead of borrowing `one_attempt` wholesale.

Pure computation: no subprocess, no network, no writes except the one analysis
artifact `main()` produces. Every rate comparison stays exact (`Fraction` of the
integer pass/attempt counts, mirroring `runner.delta`'s style) until the very
last step, so a real spread of zero can never surface as a tiny nonzero float.

`pairwise_outcomes` records what the CURRENT, uncalibrated `evaluate()` rule
would have decided on every arm pair -- the honest baseline this whole exercise
calibrates against. Two arms drawn from the same protocol (both subset, same
task set, same attempts) evaluate cleanly. A full-suite arm paired with a subset
arm never can: their task sets differ, so `evaluate()` is first restricted to the
INTERSECTION, and even then their per-task attempt counts still differ (3 or 5 vs
10) -- `evaluate()`'s own parity gate refuses that, by design, the same way it
refuses a candidate re-measured at a different sample size. That refusal is not a
bug to route around: it is recorded as an honest `"outcome": "ERROR"` entry
rather than allowed to crash the whole analysis.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from loop.acceptance import evaluate
from runner.suite import RESULTS_DIR

EDITOR_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = EDITOR_ROOT / "iterations" / "calibration-compaction" / "analysis.json"

# Contract §1: A1, G2, G4, G5 (aliases CMP-1..CMP-4) -- the section this phase
# calibrates. Only `main()`'s CLI default depends on this; `calibrate()` itself
# always takes `supported` explicitly, so tests never need to touch this.
SUPPORTED = frozenset({"A1", "G2", "G4", "G5"})

# The three fields the null-run protocol requires every arm to share (contract
# §2): a Δ across runner versions, config versions, or models is not a
# measurement of noise, it is a measurement of the thing that changed.
_FINGERPRINT_FIELDS = ("runner_sha", "config_version", "model")


def _exact_pass_fraction(task: dict) -> Fraction:
    """A task's pass rate as an exact `Fraction` of its integer counts.

    Same fallback and the same reason as `runner.delta`'s and `loop.acceptance`'s
    private `_exact`: `pass_fraction` is stored rounded to 4 places for display,
    and comparing rounded values can manufacture a nonzero spread between two
    results with IDENTICAL integer counts. Falls back to the stored fraction only
    for a legacy row with no counts recorded.
    """
    if "passes" in task and task.get("attempts"):
        return Fraction(int(task["passes"]), int(task["attempts"]))
    return Fraction(task["pass_fraction"]).limit_denominator(10_000)


def _max_pairwise_abs_delta(values: list[Fraction]) -> Fraction:
    if len(values) < 2:
        return Fraction(0)
    return max(abs(a - b) for a, b in combinations(values, 2))


def _load_arm(results_dir: Path, label: str) -> dict:
    path = results_dir / f"{label}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no results for arm {label!r} at {path}")
    return json.loads(path.read_text())


def _check_fingerprints(arm_results: dict[str, dict], labels: list[str]) -> dict:
    """All arms must share runner_sha/config_version/model (contract §2); refuse
    loudly, naming the first field that differs, rather than silently comparing
    unattributed or mismatched measurements."""
    first = arm_results[labels[0]]["fingerprint"]
    for field in _FINGERPRINT_FIELDS:
        values = {label: arm_results[label]["fingerprint"].get(field) for label in labels}
        if len(set(values.values())) > 1:
            detail = ", ".join(f"{label}={v!r}" for label, v in values.items())
            raise ValueError(f"fingerprint mismatch on {field!r} across arms: {detail}")
    return {field: first.get(field) for field in _FINGERPRINT_FIELDS}


def _build_arms(arm_results: dict[str, dict], labels: list[str]) -> list[dict]:
    return [
        {
            "label": label,
            **{
                field: arm_results[label]["fingerprint"].get(field) for field in _FINGERPRINT_FIELDS
            },
        }
        for label in labels
    ]


def _per_task(arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]) -> dict:
    """Per supported task: pass counts per arm, and the max pairwise |Δ pass_fraction|.

    An arm missing the task entirely (should not happen under the protocol, but
    defended rather than assumed) simply does not contribute a count or a fraction
    for that task -- it is never imputed as a measured zero.
    """
    per_task: dict[str, dict] = {}
    for task in sorted(supported):
        present = [
            (label, arm_results[label]["tasks"][task])
            for label in labels
            if task in arm_results[label]["tasks"]
        ]
        per_task[task] = {
            "passes_by_arm": {
                label: {"passes": int(t["passes"]), "attempts": int(t["attempts"])}
                for label, t in present
            },
            "max_abs_delta": float(
                _max_pairwise_abs_delta([_exact_pass_fraction(t) for _, t in present])
            ),
        }
    return per_task


def _supported_split_mean(results: dict, split: str, supported: frozenset[str]) -> Fraction:
    fracs = [
        _exact_pass_fraction(t)
        for name, t in results["tasks"].items()
        if name in supported and t["split"] == split
    ]
    return sum(fracs) / len(fracs) if fracs else Fraction(0)


def _section_noise(
    arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]
) -> tuple[dict[str, float], dict[str, list[str]]]:
    """Per split, the max pairwise |Δ supported-set mean| -- the measured analog of
    `one_attempt`, from arms that actually cover the WHOLE supported set for that
    split at a SHARED attempt count.

    A full-suite arm (standard 3/5 attempts) and the subset arms (10 attempts,
    contract §2) never share an attempt count, so a bound mixing them would
    average fractions of unequal precision -- the same reason `evaluate()`'s own
    parity gate refuses mismatched attempts. Group arms by their (uniform, within
    the arm) attempt count for this split's supported tasks, and use the LARGEST
    such group; `section_noise_arms` names exactly which arms fed the bound, so an
    excluded arm is never silently dropped.
    """
    task_split: dict[str, str] = {}
    for label in labels:
        for name, t in arm_results[label]["tasks"].items():
            if name in supported:
                task_split.setdefault(name, t["split"])

    noise: dict[str, float] = {}
    noise_arms: dict[str, list[str]] = {}
    for split in sorted(set(task_split.values())):
        split_tasks = {name for name, sp in task_split.items() if sp == split}
        groups: dict[int, list[str]] = {}
        for label in labels:
            tasks = arm_results[label]["tasks"]
            if not split_tasks <= set(tasks):
                continue  # this arm does not cover the entire supported set here
            attempts_values = {int(tasks[name]["attempts"]) for name in split_tasks}
            if len(attempts_values) != 1:
                continue  # not uniform within the arm -- no single grain to group on
            (attempts,) = attempts_values
            groups.setdefault(attempts, []).append(label)
        if not groups:
            noise[split] = 0.0
            noise_arms[split] = []
            continue
        best = max(groups, key=lambda a: (len(groups[a]), a))
        chosen = groups[best]
        means = [_supported_split_mean(arm_results[label], split, supported) for label in chosen]
        noise[split] = float(_max_pairwise_abs_delta(means))
        noise_arms[split] = chosen
    return noise, noise_arms


def _restrict(results: dict, names: set[str]) -> dict:
    return {**results, "tasks": {n: results["tasks"][n] for n in names}}


def _pairwise_outcomes(arm_results: dict[str, dict], labels: list[str]) -> dict:
    """What `evaluate()` (the CURRENT, uncalibrated rule -- no `calibration=`, no
    named guards: `always_confirm=frozenset()`) returns for every arm pair.

    A full-suite arm and a subset arm never share a task set; restrict both sides
    to the INTERSECTION first and record it. `evaluate()` may still refuse (their
    attempt counts differ even on the shared tasks) or otherwise raise -- that is
    caught and recorded as an honest ERROR entry rather than allowed to crash the
    whole analysis, exactly as the design calls for.
    """
    outcomes: dict[str, dict] = {}
    for a, b in combinations(labels, 2):
        key = f"{a}::{b}"
        ra, rb = arm_results[a], arm_results[b]
        tasks_a, tasks_b = set(ra["tasks"]), set(rb["tasks"])
        entry: dict = {}
        if tasks_a != tasks_b:
            inter = sorted(tasks_a & tasks_b)
            entry["task_intersection"] = inter
            ra, rb = _restrict(ra, set(inter)), _restrict(rb, set(inter))
        try:
            decision = evaluate(ra, rb, always_confirm=frozenset())
            entry["outcome"] = decision.outcome
        except Exception as exc:  # honest record of whatever evaluate() raised, never a crash
            entry["outcome"] = "ERROR"
            entry["error"] = str(exc)
        outcomes[key] = entry
    return outcomes


def calibrate(labels: list[str], results_dir: Path, supported: frozenset[str]) -> dict:
    """The analysis artifact of contract §3, from the named arms' results JSONs
    under `results_dir`. Refuses (`ValueError` naming the field) on a fingerprint
    mismatch across arms; accepts filtered/subset arms outright -- that is the
    entire point of this tool, where `runner.delta` refuses them.
    """
    if not labels:
        raise ValueError("calibrate: at least one arm label is required")
    arm_results = {label: _load_arm(results_dir, label) for label in labels}
    shared_fp = _check_fingerprints(arm_results, labels)
    section_noise, section_noise_arms = _section_noise(arm_results, labels, supported)
    return {
        "arms": _build_arms(arm_results, labels),
        "per_task": _per_task(arm_results, labels, supported),
        "section_noise": section_noise,
        "section_noise_arms": section_noise_arms,
        "pairwise_outcomes": _pairwise_outcomes(arm_results, labels),
        "computed_at_runner_sha": shared_fp["runner_sha"],
    }


def main(argv: list[str] | None = None) -> None:
    """`python -m loop.calibrate <label> [<label> ...]` -- writes
    `iterations/calibration-compaction/analysis.json`."""
    labels = list(sys.argv[1:] if argv is None else argv)
    if not labels:
        raise SystemExit("usage: python -m loop.calibrate <label> [<label> ...]")
    analysis = calibrate(labels, RESULTS_DIR, SUPPORTED)
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"wrote {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
