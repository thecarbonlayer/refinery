"""What each task was OBSERVED to exercise, derived from recorded attempts.

``knob_coverage`` states which tasks can observe each knob. It says of itself that it
is "a human-audited claim with mechanical guardrails, not a proof", that an audit once
found six of its rows false at once, and that nothing reads it. This module is the
missing half: not what a person asserted, but what the runs actually show.

The motivating failure is concrete. Iteration 3's ``tool_output`` candidate was rejected
on a Δ_in of −0.118 assembled from A1 (−1.00), G5 (−0.67), G4 (−0.33) and B2 (−0.33).
A1 and G4 build their agents with NO tool registry, and carbon gates tool execution on
``self.tools is not None`` — so no value of ``tool_output`` can produce a byte either
task ever sees. Those movements were the grader's own sampling jitter, read as the
candidate's fault. Nothing in the pipeline noticed, because nothing related a per-task
delta to whether the edited knob could reach that task.

**Strength of the claim, stated per knob.** "No observed activity" is not one kind of
fact. For ``tool_output`` and ``max_tool_steps`` it is close to proof: a task that made
no tool call has no tool result, and truncation policy acts on nothing else. For the
rest it is evidence only, and for two distinct reasons — ``compaction`` because
``trigger_fraction`` belongs to that knob and lowering it makes a task compact that
never did, and ``compaction_prompt`` because a wording that makes the summarizer call
FAIL changes the verdict while the count stays zero.

``REQUIRES`` records which is which and every caller now honours it: ``partition_deltas``
reports the two grades in separate buckets, ``contradicted_observers`` grades its claim,
and only a proof-grade contradiction exits non-zero. An earlier version stored the grade
and then ignored it at every call site, which let the report say "no legal value can
affect this task" about a knob whose own entry said otherwise — the authored claim
wearing mechanical clothes, one level up from the table this module was built to check.

Lives in ``loop/`` for the reason ``knob_coverage`` gives: ``runner_sha`` is a content
hash of the runner package, so anything kept there invalidates every baseline when it
changes. This measures nothing and must never cost a re-measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from loop.knob_coverage import DELIBERATE_NON_OBSERVERS, KNOB_COVERAGE

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@dataclass(frozen=True)
class Requirement:
    """The activity a knob needs before it can move a task's verdict."""

    metric: str
    airtight: bool
    why: str


# Only knobs with a defensible necessary condition appear here. The rest are absent on
# purpose rather than mapped to a metric that nearly fits: `default_context_limit` is
# the clearest case — a task that never compacted at the shipped window compacts as soon
# as the window is lowered, so zero compactions excludes nothing at all.
REQUIRES: dict[str, Requirement] = {
    "tool_output": Requirement(
        metric="tool_calls",
        airtight=True,
        why=(
            "Truncation policy acts on tool RESULTS and nothing else. Carbon gates "
            "execution on `self.tools is not None` and takes no default registry, so a "
            "task whose agent was built without tools can never produce one."
        ),
    ),
    "max_tool_steps": Requirement(
        metric="tool_calls",
        airtight=True,
        why="A per-turn round budget binds only a task that takes tool rounds.",
    ),
    "retry": Requirement(
        metric="retries",
        airtight=False,
        why=(
            "Retry acts only on a transient provider error. Zero retries in a BACKOFF "
            "baseline means no such error arose, so no policy value changes anything. "
            "Not airtight: read from a `fail_fast` run the same zero would mean the "
            "error arose and was not retried."
        ),
    ),
    "compaction": Requirement(
        metric="compactions",
        airtight=False,
        why=(
            "`trigger_fraction` is part of this knob, so lowering it makes a task "
            "compact that never did. Zero compactions is evidence, not exclusion."
        ),
    ),
    "compaction_prompt": Requirement(
        metric="compactions",
        airtight=False,
        why=(
            "The prompt is read only inside `compact()`, which is why this looked "
            "airtight: unlike the `compaction` object it carries no field that can make "
            "compaction fire. But `compact()` sends the prompt to the provider, that "
            "call can raise, and carbon increments `compaction_count` only AFTER "
            "`compact()` returns — so a legal wording that makes the summarizer request "
            "fail changes the verdict while the recorded count stays zero. Proving the "
            "prompt is not read BEFORE a compaction attempt is not the same as proving "
            "every effect of it produces a counted, successful compaction."
        ),
    ),
}


def observed_activity(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Peak value of each metric per task, across every attempt in every run given.

    The PEAK, not the mean the results JSON records. The question is "did this ever
    happen", and a mean of 0.33 and a mean of 0.00 answer it differently while both
    round to nothing in a report.

    A metric appears here ONLY if it was fully recorded — present on every attempt that
    ran. A first version kept the surviving zeros and discarded the misses, so a task
    with the metric absent from 19 attempts and present in 31 read as measured-at-zero
    and was reported "PROVABLY unreachable" on telemetry that had proven nothing. Partial
    recording is now unknown, and unknown is never an exclusion.
    """
    peaks: dict[str, dict[str, float]] = {}
    attempts: dict[str, int] = {}
    present: dict[str, dict[str, int]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task")
            if not task:
                continue
            attempts[task] = attempts.get(task, 0) + 1
            seen, counts = peaks.setdefault(task, {}), present.setdefault(task, {})
            for metric, value in (row.get("metrics") or {}).items():
                if isinstance(value, int | float):
                    seen[metric] = max(seen.get(metric, 0.0), float(value))
                    counts[metric] = counts.get(metric, 0) + 1
    return {
        task: {m: v for m, v in metrics.items() if present[task].get(m, 0) == attempts[task]}
        for task, metrics in peaks.items()
    }


def cohort(paths: list[Path]) -> dict:
    """Provenance for a coverage claim: which files, how big, and what state they record.

    Without this the claim is unreproducible — a reader cannot tell whether "A1 never
    called a tool" was derived from the two runs this validation compared or from a
    two-month-old partial run against a different runner. `delta` refuses to compare
    results across runner versions; a coverage claim assembled across them, and stated
    with no record of which, is the same mistake with none of the refusal.
    """
    out = []
    for path in sorted(paths):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out.append(
            {
                "file": path.name,
                "rows": len(rows),
                "runner_sha": sorted({str(r.get("runner_sha"))[:12] for r in rows}),
                "config_version": sorted(
                    {r.get("config_version") for r in rows if r.get("config_version") is not None}
                ),
                "model": sorted({str(r.get("model")) for r in rows if r.get("model")}),
            }
        )
    return {"files": out}


def result_files(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Every recorded attempt log. Partial runs (`dryrun-*`, `--only`) are included:
    they cannot show activity that did not happen, and a task they never ran simply
    contributes nothing."""
    return sorted(results_dir.glob("*.jsonl"))


def unreachable(knob: str, activity: dict[str, dict[str, float]], *, airtight_only: bool = False):
    """Tasks the knob provably (or, when ``airtight_only`` is False, evidently) cannot move.

    Returns an empty set for any knob with no recorded necessary condition — the honest
    answer for `system_prompt`, `temperature`, `max_tokens`, `file_injection`,
    `verify_attempts` and `default_context_limit`, none of which a metric can exclude.
    """
    requirement = REQUIRES.get(knob)
    if requirement is None or (airtight_only and not requirement.airtight):
        return set()
    # The metric must be PRESENT and zero. `.get(metric, 0.0)` read a missing metric as a
    # measured zero, which fails in the dangerous direction: an attempt row that carries
    # no metrics at all — 298 of them exist in `results/`, written by older runs and by
    # the error path that reports a raised task without agent metrics — would make a task
    # look proven-unreachable on the strength of never having been measured. Absence of
    # evidence had to be spelled `not measured`, not `zero`.
    return {
        task
        for task, metrics in activity.items()
        if requirement.metric in metrics and metrics[requirement.metric] == 0.0
    }


def contradicted_observers(
    activity: dict[str, dict[str, float]],
) -> dict[str, dict[str, list[str]]]:
    """Rows of ``KNOB_COVERAGE`` the recorded runs deny.

    An observer is a claim that some legal value of the knob can move that task's
    verdict. If every recorded attempt shows the task never doing the thing the knob
    acts on, the claim is false and the coverage it buys is decorative — the exact
    failure the table warns about but no test in this repo could detect.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for knob, roles in KNOB_COVERAGE.items():
        proven = unreachable(knob, activity, airtight_only=True)
        probable = unreachable(knob, activity) - proven
        observers = roles.get("observers", ())
        graded = {
            "proven": sorted(t for t in observers if t in proven),
            "probable": sorted(t for t in observers if t in probable),
        }
        if graded["proven"] or graded["probable"]:
            out[knob] = graded
    return out


def unlisted_with_activity(activity: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """Tasks that DO the thing a knob acts on but are named in none of its roles.

    The opposite direction from ``contradicted_observers``, and a weaker signal on
    purpose: observed activity is necessary for a knob to reach a task, never
    sufficient. G5 makes two to three ``write_file`` calls whose results are about 35
    characters against a 4,000-character budget, so it genuinely observes nothing about
    ``tool_output`` — listing it would be the decorative padding ``knob_coverage``
    warns about. This is a review queue, not a defect list.

    It earns its place because the one real gap an audit found in that table was of
    exactly this shape: G5 was added as "the observer that made compaction-v4
    measurable" and never entered the compaction rows. A missing row understates
    coverage in PR evidence, and no subset relation can detect it.
    """
    out: dict[str, list[str]] = {}
    for knob, roles in KNOB_COVERAGE.items():
        if knob not in REQUIRES:
            continue
        named = {t for role in roles.values() for t in role}
        named |= set(DELIBERATE_NON_OBSERVERS.get(knob, {}))
        dead = unreachable(knob, activity)
        extra = [t for t in sorted(activity) if t not in named and t not in dead]
        if extra:
            out[knob] = extra
    return out


def partition_deltas(
    knob: str, per_task: dict[str, float], activity: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Split a candidate's per-task movements into evidence and noise.

    A movement on a task the edited knob cannot reach is not a small effect or a
    surprising one — it is the grader's run-to-run variance wearing the candidate's
    name. Reporting the two together produced a Δ_in of −0.118 for a candidate whose
    reachable tasks moved by −0.33 in total.

    THREE buckets, not two, because the exclusions are not one kind of fact and an
    earlier version flattened them. `unreachable_proven` holds only airtight
    exclusions — `tool_output` on a task with no tool registry. `unreachable_probable`
    holds the evidence-grade ones, where the knob could CREATE the activity whose
    absence is being read as exclusion: lowering `compaction.trigger_fraction` makes a
    task compact that never has. Collapsing the two let a caller say "no legal value
    can affect this task" about a knob whose own `REQUIRES` entry says otherwise.
    """
    proven = unreachable(knob, activity, airtight_only=True)
    probable = unreachable(knob, activity) - proven
    return {
        "evidence": {t: d for t, d in per_task.items() if t not in proven | probable},
        "unreachable_proven": {t: d for t, d in per_task.items() if t in proven},
        "unreachable_probable": {t: d for t, d in per_task.items() if t in probable},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--knob", help="restrict the report to one knob")
    args = parser.parse_args(argv)

    files = result_files()
    activity = observed_activity(files)
    print(f"{len(activity)} tasks across {len(files)} recorded runs\n")

    knobs = [args.knob] if args.knob else sorted(REQUIRES)
    for knob in knobs:
        requirement = REQUIRES.get(knob)
        if requirement is None:
            print(f"{knob}: no mechanical necessary condition — nothing excluded")
            continue
        dead = sorted(unreachable(knob, activity))
        kind = "PROVABLY unreachable" if requirement.airtight else "no observed activity"
        print(f"{knob}  (needs {requirement.metric} > 0; {kind})")
        print(f"  {', '.join(dead) if dead else '(none)'}")

    denied = contradicted_observers(activity)
    print("\nknob_coverage observer claims the runs deny:")
    if not denied:
        print("  (none)")
    for knob, graded in sorted(denied.items()):
        if graded["proven"]:
            print(f"  {knob}: FALSE ROW — {', '.join(graded['proven'])}")
        if graded["probable"]:
            # Not a false row. The knob could create the missing activity, so absence
            # is a prompt to re-measure, never a verdict on the claim.
            print(f"  {knob}: unmeasured, re-check — {', '.join(graded['probable'])}")

    # Advisory, and never a failure: activity is necessary for a knob to reach a task,
    # never sufficient. Exiting non-zero on this would make a review queue into a gate.
    print("\nTasks with the activity but named in no role (review, not a defect):")
    extra = unlisted_with_activity(activity)
    if not extra:
        print("  (none)")
    for knob, tasks in sorted(extra.items()):
        print(f"  {knob}: {', '.join(tasks)}")
    return 1 if any(g["proven"] for g in denied.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
