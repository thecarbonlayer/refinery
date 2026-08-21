"""Byte-identity replay: the UNCALIBRATED rule must decide exactly as it did before.

`loop.acceptance` promises, in its own module docstring, that "without a calibration
this module behaves exactly as it did before the mechanism existed, byte for byte,
including its reason strings" (with one stated exception, the confirmation collapse
veto). Every phase of this program has restated that promise while editing the file it
is about. A promise nobody re-checks is a claim, and the round-1 review's finding was
precisely that the replay backing it existed only as a throwaway snippet in someone's
shell history.

So it lives here, committed, and it runs against real data: every ordered pair of
recorded `results/*.json` files is judged twice — once by the working tree's
`loop/acceptance.py`, once by the SAME module as of a git ref — and the two
`Decision.to_json()` payloads must serialize identically. A pair that raises must raise
the same exception type with the same message on both sides; a refusal is part of the
behavior being pinned, not an excuse to skip a case. Every pair whose first decision is
CONFIRM is then re-judged through `confirmed()` as well, with the pair itself restricted
to `confirm_tasks` standing in for the confirmation rerun, so the ACCEPT gate is covered
by the same replay rather than assumed.

`calibration=` is never passed in that mode. The calibrated regime is NEW behavior and
has nothing to be identical to; what must not move is what the loop already decides
today for `tool_output` and for every uncalibrated section.

`--calibrated` is the OTHER mode, and it answers a different question: does the
calibrated path execute end to end on real recorded data, and does it produce a false
outcome on the pairs where a false outcome is identifiable? See `replay_calibrated` for
exactly what that covers and what it deliberately does not.

Both modes fail on a result file they could not read. A skipped file is a recorded
measurement the replay never covered, and a gate that silently narrows its own input
can pass by covering nothing.

Usage::

    python -m loop.replay_check                     # working tree vs HEAD
    python -m loop.replay_check --ref <git-ref>     # working tree vs any ref
    python -m loop.replay_check --results-dir <dir>
    python -m loop.replay_check --calibrated        # the calibrated path, on real data

Prints the pair and mismatch counts and exits nonzero if anything differs, so it can be
read as a gate rather than as prose. Run it BEFORE committing a change to
`loop/acceptance.py`: the default `--ref HEAD` compares the edit in your working tree
against the last committed state, which is the comparison the promise is about.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType

from runner.suite import RESULTS_DIR

EDITOR_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_REL = "loop/acceptance.py"


def load_source(source: str, name: str) -> ModuleType:
    """One version of `loop/acceptance.py`, compiled FROM SOURCE TEXT under `name`.

    Both sides of the comparison go through here, and that symmetry is the point.
    Importing the working-tree side normally (`import loop.acceptance`) looked
    equivalent and is not: CPython validates cached bytecode by the source's
    (mtime, size), both at ONE-SECOND resolution, so an edit and a revert inside the
    same second that leave the file the same length hand the importer a stale `.pyc`
    with no error. That is not a hypothetical — writing this script hit it, and the
    gate reported 28 mismatches against a working tree that did not contain them.
    A gate that can be silently wrong about its own inputs is worse than no gate.
    Compiling from text consults no cache on either side.

    The module name is deliberately outside `loop.*`, so `sys.modules["loop.acceptance"]`
    is never shadowed and neither loaded copy can be mistaken for the installed one.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:  # pragma: no cover -- defensive
            raise SystemExit(f"cannot compile {ACCEPTANCE_REL} as {name!r}")
        module = importlib.util.module_from_spec(spec)
        # Registered BEFORE execution: `@dataclass` resolves a field's declared type by
        # looking the defining class's `__module__` up in `sys.modules`, and a module
        # that is not there yet makes that lookup return None and crash inside
        # `dataclasses`.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def load_reference(ref: str = "HEAD", editor_root: Path = EDITOR_ROOT) -> ModuleType:
    """`loop/acceptance.py` as of `ref`."""
    proc = subprocess.run(
        ["git", "show", f"{ref}:{ACCEPTANCE_REL}"],
        cwd=editor_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SystemExit(f"cannot read {ACCEPTANCE_REL} at {ref!r}: {proc.stderr.strip()}")
    return load_source(proc.stdout, "acceptance_reference")


def load_current(editor_root: Path = EDITOR_ROOT) -> ModuleType:
    """`loop/acceptance.py` as it sits in the working tree, right now."""
    return load_source((editor_root / ACCEPTANCE_REL).read_text(), "acceptance_current")


def load_results(results_dir: Path) -> tuple[dict[str, dict], list[str]]:
    """Every readable, suite-shaped `*.json` under `results_dir`, AND what was skipped.

    A file that is unreadable, is not JSON, or carries no `tasks` map cannot be
    replayed. It used to be dropped silently, on the reasoning that this directory is
    written by live runs and a half-written file is an artifact of timing. That
    reasoning is fine and the silence was not: a gate that quietly narrows its own
    input can pass by covering nothing, and nobody reading "0 mismatches" would know
    how many recorded measurements never entered the comparison. The skipped names come
    back with the results, `main()` prints them, and any skip fails the run — a
    concurrent write is a reason to re-run the gate, not to trust a partial one.
    """
    out: dict[str, dict] = {}
    skipped: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            skipped.append(path.stem)
            continue
        if isinstance(data, dict) and isinstance(data.get("tasks"), dict) and data["tasks"]:
            out[path.stem] = data
        else:
            skipped.append(path.stem)
    return out, skipped


def _payload(judge, *args) -> str:
    """A decision as a stable string, or the refusal it raised as one.

    Sorted keys so dict ordering cannot masquerade as a behavior change, and the
    exception's type AND message so a refusal that starts refusing for a different
    reason still counts as a difference.
    """
    try:
        decision = judge(*args)
    except Exception as exc:  # every refusal is behavior this replay pins
        return f"RAISED {type(exc).__name__}: {exc}"
    return json.dumps(decision.to_json(), sort_keys=True, default=repr)


def _restrict(results: dict, names) -> dict:
    """The same results filtered to `names`, marked `filter` the way a real
    confirmation rerun is — `confirmed()` accepts filtered pairs by design."""
    kept = {n: results["tasks"][n] for n in sorted(names) if n in results["tasks"]}
    return {**results, "filter": sorted(kept), "tasks": kept}


def replay(
    reference: ModuleType, results: dict[str, dict], current: ModuleType | None = None
) -> dict:
    """Judge every ordered pair under both modules; collect the differences.

    Ordered, not unordered: `evaluate(a, b)` and `evaluate(b, a)` are different
    judgments (one is a gain where the other is a regression), and both are behavior.
    """
    current = current or load_current()

    report = {
        "results": len(results),
        "evaluate_pairs": 0,
        # Split out and printed, because it is the number that undercuts the headline:
        # most recorded pairs are not comparable at all (different task sets, different
        # attempt counts, different runner_sha) and `_parity` refuses them. Those pairs
        # still pin behavior — the refusal and its message must match — but a reader
        # comparing "1806 pairs" against "how much of the rule did this exercise?"
        # deserves to see how many actually reached a verdict.
        "evaluate_decided": 0,
        "evaluate_refused": 0,
        "evaluate_mismatches": [],
        "confirmed_pairs": 0,
        "confirmed_mismatches": [],
    }
    labels = sorted(results)
    for a in labels:
        for b in labels:
            if a == b:
                continue
            base, cand = results[a], results[b]
            report["evaluate_pairs"] += 1
            got = _payload(current.evaluate, base, cand)
            want = _payload(reference.evaluate, base, cand)
            if got != want:
                report["evaluate_mismatches"].append(
                    {"pair": f"{a}::{b}", "reference": want, "current": got}
                )
                continue
            if got.startswith("RAISED"):
                report["evaluate_refused"] += 1
                continue
            report["evaluate_decided"] += 1
            first_now = current.evaluate(base, cand)
            if first_now.outcome != current.CONFIRM:
                continue
            # The confirmation rerun stands in as the same pair restricted to the
            # selected tasks: same attempt counts on both sides, same fingerprint, a
            # `filter` key `confirmed()` is built to accept. Each module is handed its
            # OWN first decision, so the whole path is replayed, not just the tail.
            fb = _restrict(base, first_now.confirm_tasks)
            fc = _restrict(cand, first_now.confirm_tasks)
            report["confirmed_pairs"] += 1
            got_c = _payload(current.confirmed, first_now, fb, fc)
            want_c = _payload(reference.confirmed, reference.evaluate(base, cand), fb, fc)
            if got_c != want_c:
                report["confirmed_mismatches"].append(
                    {"pair": f"{a}::{b}", "reference": want_c, "current": got_c}
                )
    return report


# The label prefix of this program's round-2 NULL arms: runs with nothing changed
# between them. A CONFIRM or an ACCEPT on a pair of these is a false outcome by
# construction, which is what makes them the one population the calibrated mode can
# gate on rather than merely report.
NULL_ARM_PREFIX = "r2-null-"


def replay_calibrated(
    results_dir: Path, model_path: Path | None = None, section: str = "compaction"
) -> dict:
    """Exercise the CALIBRATED branch over every recorded pair, and gate on the arms
    whose right answer is known.

    This is NOT a byte-identity replay and cannot be: the calibrated regime is new
    behavior with nothing to be identical to. What it covers, stated plainly so nobody
    reads more into a green run than it earns:

    - Every ordered pair of recorded results whose BASELINE the installed artifact is
      fresh for is judged through `evaluate(calibration=...)`. Pairs the parity gates
      refuse (different task sets, different attempt counts, different runner_sha) are
      counted as refused, exactly as in the uncalibrated replay — the refusal is
      behavior, not a skipped case.
    - Every CONFIRM is carried into `confirmed()` with the pair restricted to the
      first decision's own `confirm_tasks`, so the ACCEPT gate runs too.
    - Pairs where both sides are round-2 NULL ARMS are the gate: nothing changed
      between those runs, so a CONFIRM or an ACCEPT there is a false outcome and the
      run fails, naming the pair.
    - Everything else is REPORTED, never gated: a real candidate pair reaching CONFIRM
      is the rule working, and this tool has no way to know which of those is right.

    So a green run means "the calibrated path executes on real recorded data end to
    end, and produces zero false outcomes on the pairs where a false outcome is
    identifiable". It does not mean the calibrated rule is correct in general.
    """
    from loop.acceptance import ACCEPT, CONFIRM, evaluate
    from loop.validate import calibration_status

    results, skipped = load_results(results_dir)
    report: dict = {
        "results": len(results),
        "skipped": skipped,
        "pairs": 0,
        "uncalibrated_pairs": 0,
        "refused": 0,
        "decided": 0,
        "confirms": 0,
        "accepts": 0,
        "null_arm_pairs": 0,
        "false_outcomes": [],
    }
    labels = sorted(results)
    for a in labels:
        for b in labels:
            if a == b:
                continue
            base, cand = results[a], results[b]
            calibration = calibration_status(
                section, base.get("fingerprint") or {}, model_path=model_path
            )[0]
            if calibration is None:
                report["uncalibrated_pairs"] += 1
                continue
            report["pairs"] += 1
            try:
                first = evaluate(base, cand, calibration=calibration)
            except Exception:  # a parity refusal is behavior, not a skipped case
                report["refused"] += 1
                continue
            report["decided"] += 1
            both_null = a.startswith(NULL_ARM_PREFIX) and b.startswith(NULL_ARM_PREFIX)
            if both_null:
                report["null_arm_pairs"] += 1
            outcome = first.outcome
            if outcome == CONFIRM:
                report["confirms"] += 1
                fb = _restrict(base, first.confirm_tasks)
                fc = _restrict(cand, first.confirm_tasks)
                decision = confirmed_or_error(first, fb, fc, calibration)
                if decision == ACCEPT:
                    report["accepts"] += 1
                if both_null:
                    report["false_outcomes"].append(f"{a}::{b} CONFIRM -> {decision}")
            elif both_null and outcome == ACCEPT:  # pragma: no cover -- evaluate cannot
                report["false_outcomes"].append(f"{a}::{b} {outcome}")
    return report


def confirmed_or_error(first, baseline: dict, candidate: dict, calibration) -> str:
    """`confirmed()`'s outcome, or the name of the refusal it raised.

    A refusal here is a legitimate answer (the restricted stand-in pair may not cover
    exactly what the first decision selected), and it is emphatically not an ACCEPT,
    so it is recorded as itself rather than allowed to end the sweep."""
    from loop.acceptance import confirmed

    try:
        return confirmed(first, baseline, candidate, calibration=calibration).outcome
    except Exception as exc:
        return f"RAISED {type(exc).__name__}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="HEAD", help="git ref to compare against (default HEAD)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--show", type=int, default=3, help="how many mismatches to print")
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="exercise the CALIBRATED branch instead of the byte-identity replay",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.calibrated:
        report = replay_calibrated(args.results_dir)
        print(
            f"calibrated replay over {report['results']} recorded results: "
            f"{report['pairs']} ordered pairs had a fresh calibration "
            f"({report['uncalibrated_pairs']} did not), {report['decided']} reached a "
            f"verdict, {report['refused']} were refused by the parity gates; "
            f"{report['confirms']} CONFIRM, {report['accepts']} ACCEPT"
        )
        print(
            f"null-arm pairs judged (nothing changed between those runs, so any CONFIRM "
            f"or ACCEPT is false): {report['null_arm_pairs']}; "
            f"false outcomes: {len(report['false_outcomes'])}"
        )
        for item in report["false_outcomes"][: args.show]:
            print(f"  FALSE OUTCOME {item}")
        skipped = _report_skipped(report["skipped"], args.results_dir)
        return 1 if report["false_outcomes"] or skipped else 0

    reference = load_reference(args.ref)
    results, skipped_files = load_results(args.results_dir)
    if not results:
        print(f"no readable results under {args.results_dir} — nothing to replay")
        return 1
    report = replay(reference, results)
    report["skipped"] = skipped_files
    print(
        f"replayed {report['results']} recorded results as "
        f"{report['evaluate_pairs']} ordered evaluate() pairs against {args.ref}: "
        f"{report['evaluate_decided']} reached a verdict, "
        f"{report['evaluate_refused']} were refused by the parity gates "
        f"(the refusal message is pinned too); "
        f"{report['confirmed_pairs']} of the verdicts were CONFIRM and were replayed "
        "through confirmed() as well"
    )
    print(
        f"evaluate mismatches: {len(report['evaluate_mismatches'])}; "
        f"confirmed mismatches: {len(report['confirmed_mismatches'])}"
    )
    for kind in ("evaluate_mismatches", "confirmed_mismatches"):
        for item in report[kind][: args.show]:
            print(f"\n{kind} {item['pair']}\n  reference: {item['reference']}")
            print(f"  current:   {item['current']}")
    skipped = _report_skipped(skipped_files, args.results_dir)
    mismatched = report["evaluate_mismatches"] or report["confirmed_mismatches"]
    return 1 if mismatched or skipped else 0


def _report_skipped(skipped: list[str], results_dir: Path) -> bool:
    """Print what the replay could not read, and say whether that is a failure.

    Always a failure when nonempty. A skipped result file is a recorded measurement
    this gate did not cover, and a gate that decides how much of its own input to
    ignore is not a gate. The usual cause is benign (a live run writing a file while
    the replay reads the directory), and the remedy is to run it again, not to accept
    a partial pass.
    """
    if not skipped:
        return False
    print(
        f"SKIPPED {len(skipped)} unreadable or non-suite-shaped file(s) under "
        f"{results_dir}: {', '.join(skipped)} — every recorded result must be replayable, "
        "so this run does not pass. If a live measurement was writing during the replay, "
        "re-run it."
    )
    return True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
