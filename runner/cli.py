"""CLI: `uv run python -m runner.cli run --label baseline-main` and
`... delta results/a.json results/b.json`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def positive_int(value: str) -> int:
    """argparse type for --attempts: `--attempts 0` would silently fall back
    to the per-split defaults (`attempts or spec.attempts`) and a negative
    would run zero attempts — both must fail at parse time."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"attempts must be >= 1, got {n}")
    return n


def validate_only(only: list[str], tasks) -> list[str]:
    """Unknown task names in --only, sorted. A typo must error up front —
    run_suite would otherwise happily run an empty filtered suite."""
    return sorted(set(only) - {t.name for t in tasks})


def check(label: str) -> int:
    """Report whether a recorded baseline still resumes under the current harness
    state, without running the model. Returns a process exit code: 0 = current
    (resumes; any gemma_sha move was additive), 1 = stale (re-baseline required) or
    no such baseline."""
    from runner import guard
    from runner.gemma_env import gemma_fingerprint
    from runner.suite import RESULTS_DIR

    out_path = RESULTS_DIR / f"{label}.json"
    if not out_path.is_file():
        print(f"no such baseline: {out_path}")
        return 1
    prior = json.loads(out_path.read_text()).get("fingerprint", {})
    current = gemma_fingerprint()
    status = guard.baseline_status(prior, current)
    print(f"baseline '{label}': {status.upper()}")
    print(
        f"  recorded behavior_key = {prior.get('behavior_key')}  "
        f"(gemma_sha {prior.get('gemma_sha')}, config_version {prior.get('config_version')})"
    )
    print(
        f"  current  behavior_key = {current['behavior_key']}  "
        f"(gemma_sha {current['gemma_sha']}, config_version {current['config_version']}, "
        f"model {current['model']})"
    )
    print(
        "  -> resumes; any gemma_sha move was additive"
        if status == "current"
        else "  -> re-baseline required; config_version / model / verifier / working tree changed"
    )
    return 0 if status == "current" else 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run the task suite against the current dist/gemma checkout")
    run_p.add_argument("--label", required=True, help="results file stem (results/<label>.json[l])")
    run_p.add_argument("--only", nargs="+", default=None, help="task names to run (default: all)")
    run_p.add_argument(
        "--attempts",
        type=positive_int,
        default=None,
        help="override attempt count (default: 3 held-in / 5 held-out)",
    )
    run_p.add_argument(
        "--force",
        action="store_true",
        help="bypass the resume-guard: re-run and overwrite this label from scratch, "
        "even a still-current or stale baseline",
    )

    check_p = sub.add_parser(
        "check",
        help="report whether a recorded baseline still resumes under the current "
        "harness state (no model run)",
    )
    check_p.add_argument("label", help="results file stem to check (results/<label>.json)")

    delta_p = sub.add_parser("delta", help="Δ between two results JSONs + acceptance rule")
    delta_p.add_argument("baseline")
    delta_p.add_argument("candidate")

    args = parser.parse_args()
    if args.cmd == "run":
        from runner.suite import run_suite
        from runner.tasks import TASKS

        if args.only:
            unknown = validate_only(args.only, TASKS)
            if unknown:
                parser.error(
                    f"unknown task name(s) in --only: {', '.join(unknown)} "
                    f"(valid: {', '.join(t.name for t in TASKS)})"
                )
        results = run_suite(
            TASKS,
            label=args.label,
            only=set(args.only) if args.only else None,
            attempts=args.attempts,
            force=args.force,
        )
        print(json.dumps(results["summary"], indent=2))
    elif args.cmd == "check":
        raise SystemExit(check(args.label))
    else:
        from runner.delta import delta

        d = delta(
            json.loads(Path(args.baseline).read_text()),
            json.loads(Path(args.candidate).read_text()),
        )
        print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
