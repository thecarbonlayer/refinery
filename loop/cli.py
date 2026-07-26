"""CLI for the code half of the loop: validate candidates, dry-run, open PRs.

    uv run python -m loop.cli dry-run  --iteration iter-01 --candidate cand-x --tasks A2 D1
    uv run python -m loop.cli validate --iteration iter-01 [--candidate cand-x]
    uv run python -m loop.cli pr       --iteration iter-01 --candidate cand-x

Artifacts live in iterations/<iteration>/: clusters.json and candidates.json
are written by the proposer (reasoning, not code); validation-<id>.json records
are written here, accepted or rejected alike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loop.artifacts import Candidate, Cluster, load_candidates, load_clusters
from loop.validate import EDITOR_ROOT, dry_run, validate_candidate
from runner.suite import RESULTS_DIR

ITERATIONS_DIR = EDITOR_ROOT / "iterations"
DEFAULT_BASELINE = RESULTS_DIR / "baseline-main.json"


def iteration_dir(iteration: str) -> Path:
    d = ITERATIONS_DIR / iteration
    if not d.is_dir():
        raise SystemExit(f"no such iteration dir: {d}")
    return d


def pick_candidate(candidates: list[Candidate], cand_id: str) -> Candidate:
    by_id = {c.id: c for c in candidates}
    if cand_id not in by_id:
        raise SystemExit(f"unknown candidate {cand_id!r} (have: {', '.join(by_id)})")
    return by_id[cand_id]


def pick_cluster(clusters: list[Cluster], cluster_id: str) -> Cluster:
    by_id = {c.id: c for c in clusters}
    if cluster_id not in by_id:
        raise SystemExit(f"candidate names unknown cluster {cluster_id!r}")
    return by_id[cluster_id]


def main() -> None:
    parser = argparse.ArgumentParser(prog="loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "surface",
        help="print Carbon's editable strategy menu and explicit immutable boundaries",
    )

    dry_p = sub.add_parser("dry-run", help="apply -> run a task subset -> revert (no Δ)")
    dry_p.add_argument("--iteration", required=True)
    dry_p.add_argument("--candidate", required=True)
    dry_p.add_argument("--tasks", nargs="+", required=True)
    dry_p.add_argument("--attempts", type=int, default=1)

    val_p = sub.add_parser("validate", help="full-suite validation + acceptance rule")
    val_p.add_argument("--iteration", required=True)
    val_p.add_argument("--candidate", default=None, help="default: all candidates in order")
    val_p.add_argument("--baseline", default=str(DEFAULT_BASELINE))

    pr_p = sub.add_parser("pr", help="open the PR for an ACCEPTED candidate")
    pr_p.add_argument("--iteration", required=True)
    pr_p.add_argument("--candidate", required=True)
    pr_p.add_argument(
        "--baseline",
        default=str(DEFAULT_BASELINE),
        help="baseline results JSON the candidate was validated against",
    )

    args = parser.parse_args()
    if args.cmd == "surface":
        from loop.config_edit import proposal_surface

        print(json.dumps(proposal_surface(), indent=2, sort_keys=True))
        return
    it_dir = iteration_dir(args.iteration)
    candidates = load_candidates(it_dir / "candidates.json")

    if args.cmd == "dry-run":
        cand = pick_candidate(candidates, args.candidate)
        dry_run(cand, only=args.tasks, attempts=args.attempts)
        return

    if args.cmd == "validate":
        chosen = [pick_candidate(candidates, args.candidate)] if args.candidate else candidates
        outcomes = []
        for cand in chosen:
            record = validate_candidate(cand, baseline_path=args.baseline)
            out = it_dir / f"validation-{cand.id}.json"
            out.write_text(json.dumps(record.to_json(), indent=2) + "\n")
            print(f"wrote {out}")
            outcomes.append((cand.id, record.accepted, record.delta_in, record.delta_ho))
        print("\n=== validation summary ===")
        for cand_id, accepted, d_in, d_ho in outcomes:
            print(
                f"{cand_id}: Δ_in={d_in:+.4f} Δ_ho={d_ho:+.4f} "
                f"{'ACCEPTED' if accepted else 'REJECTED'}"
            )
        return

    # pr
    from loop.artifacts import ValidationRecord
    from loop.prpipe import open_pr

    cand = pick_candidate(candidates, args.candidate)
    clusters = load_clusters(it_dir / "clusters.json")
    cluster = pick_cluster(clusters, cand.cluster_id)
    rec_path = it_dir / f"validation-{cand.id}.json"
    if not rec_path.is_file():
        raise SystemExit(f"no validation record {rec_path} — run `validate` first")
    rec_raw = json.loads(rec_path.read_text())
    record = ValidationRecord(**rec_raw)
    baseline_results = json.loads(Path(args.baseline).read_text())
    if baseline_results.get("fingerprint") != record.baseline_fingerprint:
        raise SystemExit(
            f"{args.baseline} is not the baseline this record was validated against "
            f"— its fingerprint differs from the one in {rec_path}"
        )
    candidate_results = json.loads((RESULTS_DIR / f"{record.label}.json").read_text())
    url = open_pr(
        cand, record, cluster, baseline_results, candidate_results, iteration=args.iteration
    )
    print(url)


if __name__ == "__main__":
    sys.exit(main())
