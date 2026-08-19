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
import dataclasses
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from loop.acceptance import CONFIRM, REJECT, Decision, confirmed
from loop.artifacts import (
    Candidate,
    Cluster,
    ConfirmationRecord,
    load_candidates,
    load_clusters,
    write_confirmation_record,
    write_validation_record,
)
from loop.config_edit import apply_candidate
from loop.validate import (
    EDITOR_ROOT,
    dry_run,
    require_clean_tree,
    revert_config,
    validate_candidate,
)
from runner.carbon_env import CARBON_ROOT
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


def load_first_decision(it_dir: Path, candidate_id: str) -> Decision:
    """The candidate's first CONFIRM verdict, read back off its validation record.

    The only legitimate starting point for a paired confirmation — ``confirmed()``
    reruns exactly ``first.confirm_tasks`` and judges against ``first.improved_tasks``,
    so a confirmation without a first CONFIRM has nothing to confirm and nothing to
    compare against. Refuses loudly (``SystemExit``, matching this module's other
    refusals) rather than fabricating a decision or silently no-oping.
    """
    rec_path = it_dir / f"validation-{candidate_id}.json"
    if not rec_path.is_file():
        raise SystemExit(f"no validation record {rec_path} — run `validate` first")
    rule = json.loads(rec_path.read_text()).get("rule", {})
    if not rule.get("applied") or rule.get("outcome") != CONFIRM:
        raise SystemExit(
            f"{rec_path} carries no first CONFIRM decision (rule.applied="
            f"{rule.get('applied')!r}, outcome={rule.get('outcome')!r}) — a "
            "confirmation without a first CONFIRM is meaningless"
        )
    # `rule` is `{"applied": True, **Decision.to_json()}` (see `loop.acceptance.
    # rule_disposition`) — every JSON-list field on `Decision` is a tuple, so restore
    # that instead of handing the frozen dataclass a list it never declared.
    tuple_fields = {"reasons", "excluded", "improved_tasks", "confirm_tasks", "targeted_rerun"}
    decision_fields = {f.name for f in dataclasses.fields(Decision)}
    kwargs = {
        k: (tuple(v) if k in tuple_fields and isinstance(v, list) else v)
        for k, v in rule.items()
        if k in decision_fields
    }
    return Decision(**kwargs)


def _run_confirm_arm(
    label: str, only: list[str], attempts: int, results_dir: Path = RESULTS_DIR
) -> dict:
    """Run one confirmation arm via the runner CLI in a fresh subprocess.

    Mirrors ``loop.validate._run_runner``'s injectable-seam idiom — the same
    subprocess invocation — but returns the parsed results dict directly rather than
    leaving the caller to reload it from disk: a confirmation needs BOTH arms' results
    in hand before it can call ``acceptance.confirmed()``. Tests inject a fake here so
    they never spawn a real run or touch ``results/``.
    """
    cmd = [
        sys.executable,
        "-m",
        "runner.cli",
        "run",
        "--label",
        label,
        "--only",
        *only,
        "--attempts",
        str(attempts),
    ]
    subprocess.run(cmd, cwd=EDITOR_ROOT, check=True)
    return json.loads((results_dir / f"{label}.json").read_text())


def _per_task_counts(confirm_set: tuple[str, ...], baseline: dict, candidate: dict) -> dict:
    """``{task: {"base": [passes, attempts], "cand": [passes, attempts]}}`` for every
    task BOTH arms actually measured — the iter-06 shape, mirrored exactly.

    Deliberately defensive on a partial mismatch (see ``run_confirmation``'s
    ``ValueError`` handling): a task missing from one side is skipped here rather than
    raising a second time while the record is being assembled to explain the first.
    """
    out: dict[str, dict[str, list[int]]] = {}
    for name in confirm_set:
        b = baseline.get("tasks", {}).get(name)
        c = candidate.get("tasks", {}).get(name)
        if not b or not c or "passes" not in b or "passes" not in c:
            continue
        out[name] = {
            "base": [int(b["passes"]), int(b["attempts"])],
            "cand": [int(c["passes"]), int(c["attempts"])],
        }
    return out


def _finding(candidate: Candidate, first: Decision, decision: Decision) -> str:
    basis = ", ".join(first.improved_tasks) or "no improved tasks recorded"
    text = (
        f"paired confirmation of {candidate.id}: first CONFIRM carried by {basis} "
        f"(evidence_split={first.evidence_split!r}) -> {decision.outcome}"
    )
    if decision.reasons:
        text += " — " + "; ".join(decision.reasons)
    return text


def run_confirmation(
    candidate: Candidate,
    it_dir: Path,
    baseline_label: str,
    candidate_label: str,
    attempts: int,
    carbon_root: Path = CARBON_ROOT,
    run_runner: Callable[[str, list[str], int], dict] = _run_confirm_arm,
    log=print,
) -> ConfirmationRecord:
    """Fresh paired confirmation: a baseline arm (config unedited) against a candidate
    arm (candidate applied), both filtered to the first decision's ``confirm_tasks``
    at ``attempts`` each, judged by ``acceptance.confirmed()``. The only path from a
    CONFIRM to an ACCEPT (contract §5) — shared infrastructure, not section-specific.
    """
    first = load_first_decision(it_dir, candidate.id)
    confirm_set = tuple(first.confirm_tasks)
    only = list(confirm_set)
    require_clean_tree(carbon_root)
    baseline_results = run_runner(baseline_label, only, attempts)
    log(f"candidate {candidate.id}: confirmation baseline arm {baseline_label!r} done")
    apply_candidate(carbon_root, candidate)
    try:
        candidate_results = run_runner(candidate_label, only, attempts)
    finally:
        revert_config(carbon_root)
        require_clean_tree(carbon_root)  # the revert must actually have reverted
    log(f"candidate {candidate.id}: confirmation candidate arm {candidate_label!r} done")
    try:
        decision = confirmed(first, baseline_results, candidate_results)
    except ValueError as exc:
        # A parity failure between the two fresh arms (mismatched task sets, mismatched
        # attempt counts, or a set that does not cover exactly `first.confirm_tasks`)
        # is still a confirmation OUTCOME the record must carry, not an unhandled
        # crash that drops the run on the floor with no artifact to show for it.
        decision = Decision(
            outcome=REJECT,
            reasons=(f"confirmation arms did not match: {exc}",),
            delta_in=0.0,
            delta_ho=0.0,
            threshold_in=0.0,
            threshold_ho=0.0,
            excluded=first.excluded,
            evidence_split=first.evidence_split,
            improved_tasks=first.improved_tasks,
            confirm_tasks=first.confirm_tasks,
            raw={"stage": "confirmation", "error": str(exc)},
        )
    log(f"candidate {candidate.id}: confirmation outcome {decision.outcome}")
    return ConfirmationRecord(
        candidate_id=candidate.id,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        attempts_per_task_per_arm=attempts,
        confirm_set=confirm_set,
        first_decision=first.to_json(),
        confirmation=decision.to_json(),
        per_task=_per_task_counts(confirm_set, baseline_results, candidate_results),
        finding=_finding(candidate, first, decision),
    )


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

    conf_p = sub.add_parser(
        "confirm", help="fresh paired confirmation of a candidate's first CONFIRM"
    )
    conf_p.add_argument("--iteration", required=True)
    conf_p.add_argument("--candidate", required=True)
    conf_p.add_argument("--baseline-label", required=True)
    conf_p.add_argument("--candidate-label", required=True)
    conf_p.add_argument("--attempts", type=int, required=True)

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
            out = write_validation_record(record, it_dir / f"validation-{cand.id}.json")
            print(f"wrote {out}")
            outcomes.append((cand.id, record.disposition, record.delta_in, record.delta_ho))
        print("\n=== validation summary ===")
        for cand_id, disposition, d_in, d_ho in outcomes:
            print(f"{cand_id}: Δ_in={d_in:+.4f} Δ_ho={d_ho:+.4f} {disposition}")
        return

    if args.cmd == "confirm":
        cand = pick_candidate(candidates, args.candidate)
        record = run_confirmation(
            cand,
            it_dir,
            baseline_label=args.baseline_label,
            candidate_label=args.candidate_label,
            attempts=args.attempts,
        )
        out = write_confirmation_record(record, it_dir / f"confirmation-{cand.id}.json")
        print(f"wrote {out}")
        print(f"{cand.id}: {record.confirmation['outcome']}")
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
    # `to_json` also writes DERIVED values (`disposition`) that are not constructor
    # fields, so a bare `ValidationRecord(**rec_raw)` raises TypeError and `pr` dies
    # before it can open anything. Filter to declared fields: the record on disk is
    # the durable artifact and may legitimately carry more than the dataclass takes.
    fields = {f.name for f in dataclasses.fields(ValidationRecord)}
    record = ValidationRecord(**{k: v for k, v in rec_raw.items() if k in fields})
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
