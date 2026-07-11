"""CLI: `uv run python -m runner.cli run --label baseline-main` and
`... delta results/a.json results/b.json`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="runner")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run the task suite against the current dist/gemma checkout")
    run_p.add_argument("--label", required=True, help="results file stem (results/<label>.json[l])")
    run_p.add_argument("--only", nargs="+", default=None, help="task names to run (default: all)")
    run_p.add_argument(
        "--attempts",
        type=int,
        default=None,
        help="override attempt count (default: 3 held-in / 5 held-out)",
    )

    delta_p = sub.add_parser("delta", help="Δ between two results JSONs + acceptance rule")
    delta_p.add_argument("baseline")
    delta_p.add_argument("candidate")

    args = parser.parse_args()
    if args.cmd == "run":
        from runner.suite import run_suite
        from runner.tasks import TASKS

        results = run_suite(
            TASKS,
            label=args.label,
            only=set(args.only) if args.only else None,
            attempts=args.attempts,
        )
        print(json.dumps(results["summary"], indent=2))
    else:
        from runner.delta import delta

        d = delta(
            json.loads(Path(args.baseline).read_text()),
            json.loads(Path(args.candidate).read_text()),
        )
        print(json.dumps(d, indent=2))


if __name__ == "__main__":
    main()
