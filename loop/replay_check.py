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

`calibration=` is never passed. The calibrated regime is NEW behavior and has nothing to
be identical to; what must not move is what the loop already decides today for
`tool_output` and for every uncalibrated section.

Usage::

    python -m loop.replay_check                     # working tree vs HEAD
    python -m loop.replay_check --ref <git-ref>     # working tree vs any ref
    python -m loop.replay_check --results-dir <dir>

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


def load_results(results_dir: Path) -> dict[str, dict]:
    """Every readable, suite-shaped `*.json` under `results_dir`.

    A file that is unreadable, is not JSON, or carries no `tasks` map is SKIPPED rather
    than failed on: this directory is written by live runs, and a partially written
    result appearing mid-replay is an artifact of when the tool ran, not a difference
    between two versions of the rule.
    """
    out: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(data.get("tasks"), dict) and data["tasks"]:
            out[path.stem] = data
    return out


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ref", default="HEAD", help="git ref to compare against (default HEAD)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--show", type=int, default=3, help="how many mismatches to print")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    reference = load_reference(args.ref)
    results = load_results(args.results_dir)
    if not results:
        print(f"no readable results under {args.results_dir} — nothing to replay")
        return 1
    report = replay(reference, results)
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
    return 1 if report["evaluate_mismatches"] or report["confirmed_mismatches"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
