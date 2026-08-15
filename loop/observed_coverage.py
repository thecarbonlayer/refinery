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
fact. For ``tool_output`` it is close to proof: a task that made no tool call has no
tool result, and truncation policy acts on nothing else. For ``compaction`` it is
merely evidence, because ``trigger_fraction`` is part of that knob and lowering it can
make a task compact that never did. ``REQUIRES`` records which is which, and callers
that want only the airtight exclusions filter on it. A silent mix of the two would be
the same defect one level up — an authored claim wearing mechanical clothes.

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
        airtight=True,
        why=(
            "The prompt is read only inside `compact()`. Unlike the `compaction` object "
            "it carries no field that can cause compaction to fire, so a task that "
            "never compacts cannot be reached by any wording."
        ),
    ),
}


def observed_activity(paths: list[Path]) -> dict[str, dict[str, float]]:
    """Peak value of each metric per task, across every attempt in every run given.

    The PEAK, not the mean the results JSON records. The question is "did this ever
    happen", and a mean of 0.33 and a mean of 0.00 answer it differently while both
    round to nothing in a report. Pooling many runs is deliberate too: one run showing
    zero tool calls is a weaker claim than eight runs showing zero.
    """
    activity: dict[str, dict[str, float]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task = row.get("task")
            if not task:
                continue
            seen = activity.setdefault(task, {})
            for metric, value in (row.get("metrics") or {}).items():
                if isinstance(value, int | float):
                    seen[metric] = max(seen.get(metric, 0.0), float(value))
    return activity


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
    return {
        task for task, metrics in activity.items() if metrics.get(requirement.metric, 0.0) == 0.0
    }


def contradicted_observers(activity: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """Rows of ``KNOB_COVERAGE`` the recorded runs deny.

    An observer is a claim that some legal value of the knob can move that task's
    verdict. If every recorded attempt shows the task never doing the thing the knob
    acts on, the claim is false and the coverage it buys is decorative — the exact
    failure the table warns about but no test in this repo could detect.
    """
    out: dict[str, list[str]] = {}
    for knob, roles in KNOB_COVERAGE.items():
        dead = unreachable(knob, activity)
        named = [t for t in roles.get("observers", ()) if t in dead]
        if named:
            out[knob] = sorted(named)
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
    """
    dead = unreachable(knob, activity)
    return {
        "evidence": {t: d for t, d in per_task.items() if t not in dead},
        "unreachable": {t: d for t, d in per_task.items() if t in dead},
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
    print("\nknob_coverage observer claims the runs DENY (a false row):")
    if not denied:
        print("  (none)")
    for knob, tasks in sorted(denied.items()):
        print(f"  {knob}: {', '.join(tasks)}")

    # Advisory, and never a failure: activity is necessary for a knob to reach a task,
    # never sufficient. Exiting non-zero on this would make a review queue into a gate.
    print("\nTasks with the activity but named in no role (review, not a defect):")
    extra = unlisted_with_activity(activity)
    if not extra:
        print("  (none)")
    for knob, tasks in sorted(extra.items()):
        print(f"  {knob}: {', '.join(tasks)}")
    return 1 if denied else 0


if __name__ == "__main__":
    sys.exit(main())
