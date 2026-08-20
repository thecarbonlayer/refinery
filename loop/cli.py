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

from loop.acceptance import ACCEPT, CONFIRM, Decision, confirmed, decision_digest
from loop.artifacts import (
    STAGE_PAIRED_CONFIRMATION,
    Candidate,
    Cluster,
    ConfirmationRecord,
    ValidationRecord,
    load_candidates,
    load_clusters,
    write_confirmation_record,
    write_validation_record,
)
from loop.config_edit import apply_candidate
from loop.validate import (
    CALIBRATION_REQUIRED,
    EDITOR_ROOT,
    calibration_status,
    candidate_section,
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


def _load_validation_json(it_dir: Path, candidate_id: str) -> dict:
    """The raw validation record dict for ``candidate_id``, or a loud refusal.

    Shared by ``load_first_decision`` (which reads ``rule``) and
    ``_check_candidate_identity`` (which reads ``candidate_fields``) so both refuse
    identically on a missing record instead of maintaining two copies of that check.
    """
    rec_path = it_dir / f"validation-{candidate_id}.json"
    if not rec_path.is_file():
        raise SystemExit(f"no validation record {rec_path} — run `validate` first")
    return json.loads(rec_path.read_text())


def _check_candidate_identity(it_dir: Path, candidate: Candidate) -> None:
    """The validation record must have been produced by validating THIS candidate's
    edit, not merely one that happens to share its id.

    ``ValidationRecord.candidate_fields`` (added additively — ``None`` on a record
    written before this existed, and on load for one that still is) carries the
    ``{field: {old, new}}`` the record was actually validated against. When present,
    a mismatch against the candidate's CURRENT fields means the candidates.json
    changed underfoot, or a stale in-memory object is reusing an id — either way, a
    confirmation run now would judge a different edit than the one whose first
    decision this is, silently. Refuse loudly instead, naming the drift. A record
    with no ``candidate_fields`` at all has nothing to compare and is not refused —
    matching by id was the only check that ever existed for those, and this fix must
    not retroactively invalidate every record written before it.
    """
    raw = _load_validation_json(it_dir, candidate.id)
    recorded = raw.get("candidate_fields")
    if recorded is None:
        return
    if recorded != candidate.fields:
        raise SystemExit(
            f"candidate {candidate.id!r}'s fields drifted from the validated record — "
            f"validated {recorded!r}, now {dict(candidate.fields)!r}: a confirmation "
            "would judge a different edit than the one whose first decision this is"
        )


def load_first_decision(
    it_dir: Path, candidate_id: str, *, require_binding: bool = False
) -> Decision:
    """The candidate's first CONFIRM verdict, read back off its validation record.

    The only legitimate starting point for a paired confirmation — ``confirmed()``
    reruns exactly ``first.confirm_tasks`` and judges against ``first.improved_tasks``,
    so a confirmation without a first CONFIRM has nothing to confirm and nothing to
    compare against. Refuses loudly (``SystemExit``, matching this module's other
    refusals) rather than fabricating a decision or silently no-oping.

    ``require_binding`` says the record MUST carry a matching decision digest — absent
    refuses exactly like wrong. The caller derives it from the candidate's SECTION
    (`loop.validate.CALIBRATION_REQUIRED`), which comes from code and from
    candidates.json, never from the record being loaded: a record that could switch off
    its own check by dropping a key is not checked. ``confirmed()`` enforces the same
    binding again once the calibration is actually in hand; this is the early, better-
    worded refusal, not the only one.
    """
    rec_path = it_dir / f"validation-{candidate_id}.json"
    rule = _load_validation_json(it_dir, candidate_id).get("rule", {})
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
    decision = Decision(**kwargs)
    # STAGE BINDING at the reload boundary. Everything above trusts a JSON file on
    # disk, and the two fields it hands `confirmed()` are the whole exam: every task
    # in `improved_tasks` must repeat beyond its own null quantile, and `confirm_tasks`
    # is what gets rerun. Editing three carriers down to one is not a smaller claim,
    # it is an easier test of a claim that was never made — and nothing else in this
    # path could see it, because a shorter list is perfectly well-formed.
    #
    # An ABSENT digest refuses whenever the section requires a binding. The first
    # version skipped that case, which meant the attack was "shrink the list, delete
    # the digest" — the check asked the record's permission to run.
    recorded = (decision.raw or {}).get("decision_digest")
    if recorded is not None or require_binding:
        current = decision_digest(decision)
        if recorded != current:
            raise SystemExit(
                f"{rec_path} does not match its own decision digest: it "
                + ("carries none" if recorded is None else f"records {recorded!r}")
                + f" and the improved_tasks/confirm_tasks it now carries digest to "
                f"{current!r}. improved_tasks={list(decision.improved_tasks)}, "
                f"confirm_tasks={list(decision.confirm_tasks)} — a confirmation run from "
                "this record would test a different claim than the one that was made. "
                "Re-validate the candidate rather than repairing the record."
            )
    return decision


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
    results_dir: Path = RESULTS_DIR,
    log=print,
) -> ConfirmationRecord:
    """Fresh paired confirmation: a baseline arm (config unedited) against a candidate
    arm (candidate applied), both filtered to the first decision's ``confirm_tasks``
    at ``attempts`` each, judged by ``acceptance.confirmed()``. The only path from a
    CONFIRM to an ACCEPT — shared infrastructure, not section-specific.
    """
    for label in (baseline_label, candidate_label):
        if (results_dir / f"{label}.json").is_file():
            raise SystemExit(
                f"{label!r} already has recorded results — confirmation arms must "
                "be fresh: `runner.cli run` RESUMES an existing label instead of "
                "re-measuring it, which would silently judge the candidate against "
                "evidence that was never actually re-run for this confirmation"
            )
    _check_candidate_identity(it_dir, candidate)
    # Whether this record MUST be bound to its own claim is a property of the section
    # the candidate edits, read from code and from candidates.json — not from the
    # record, which is the thing being checked.
    section = candidate_section(candidate)
    first = load_first_decision(
        it_dir, candidate.id, require_binding=section in CALIBRATION_REQUIRED
    )
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
    # The section's measured bounds, if it has any that are fresh for THIS pair
    # The freshness question is asked of the confirmation
    # BASELINE arm — the run that was just recorded, not the process asking — so a
    # calibration is used only where it is actually a bound for these measurements.
    # For an uncalibrated section (`tool_output`) this is None and nothing changes.
    # When the first decision WAS calibrated and this comes back None, `confirmed()`
    # refuses rather than quietly re-deciding on the weaker one-attempt bound, and the
    # refusal lands in the same no-artifact-written path as any other parity failure.
    calibration, why_not = (
        calibration_status(section, baseline_results.get("fingerprint") or {})
        if section
        else (None, f"candidate {candidate.id} edits no single mapped section")
    )
    if calibration is not None:
        log(f"candidate {candidate.id}: judging against {calibration.source}")
    elif section in CALIBRATION_REQUIRED:
        # FAIL CLOSED here too, and keyed on the SECTION rather than on the record's
        # own `regime`. The old guard lived in `confirmed()` and asked the first
        # decision whether it had been calibrated — so deleting that one key bought an
        # uncalibrated confirmation of a calibrated claim, judged against the weaker
        # one-attempt grain. What requires a null model is the section, and the section
        # comes from the candidate.
        raise SystemExit(
            f"candidate {candidate.id!r} edits {section!r}, which is decided by a "
            f"measured null model or not at all, and this confirmation has none: "
            f"{why_not}. Confirming it against the one-attempt bound would answer a "
            "different question under the same word. Re-run the null arms and rebuild "
            "the model, then re-validate."
        )
    elif (first.raw or {}).get("regime") == "section_calibration":
        log(f"candidate {candidate.id}: no calibration for this confirmation — {why_not}")
    try:
        decision = confirmed(first, baseline_results, candidate_results, calibration=calibration)
    except ValueError as exc:
        # A parity failure between the two fresh arms (mismatched task sets, mismatched
        # attempt counts, missing/mismatched fingerprint, or a set that does not cover
        # exactly `first.confirm_tasks`) means the pair was never actually MEASURED —
        # this is an infrastructure refusal, the same family as `runner.delta`'s own
        # refusal to compare filtered or mismatched results (see AGENTS.md: "the
        # refusal is the feature"). Writing a REJECT record here would claim a
        # measurement that never happened, so nothing is written: fail loud instead,
        # with the cause named, and leave no confirmation-*.json behind to be mistaken
        # for a real verdict.
        raise SystemExit(f"confirmation could not be measured: {exc}") from exc
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


def _pr_eligible_record(
    it_dir: Path, candidate: Candidate, record: ValidationRecord
) -> ValidationRecord:
    """PR eligibility: the validation record's own
    ``accepted``, OR — for a rule-section candidate whose FIRST decision was
    CONFIRM — a fresh paired confirmation for the SAME candidate whose own outcome
    is ACCEPT.

    ``validate_candidate`` only ever sets ``accepted=True`` for a rule outcome of
    ACCEPT, which ``evaluate()`` (the first-pass judgment) never returns — so a
    CONFIRM candidate's validation record carries ``accepted=False`` forever, even
    after a real confirmation ACCEPT is recorded separately (``loop.cli confirm``
    writes ``ConfirmationRecord``s, never touches the validation record). Without
    this, such a candidate could win a genuine confirmed ACCEPT and still never
    reach a PR.

    Refuses loudly, naming what's missing, for every CONFIRM candidate that is not
    demonstrably eligible — rather than silently falling through to ``open_pr``'s
    generic "was not accepted", which is right for a candidate that was plainly
    REJECTED and never had anything to confirm, but unhelpful for one still waiting
    on, or that failed, its confirmation. A candidate whose rule was never CONFIRM
    (a plain REJECT, or ``rule == {}``) is returned UNCHANGED — ``open_pr``'s own
    ``if not record.accepted`` refusal fires exactly as it always has; nothing here
    weakens it, this function only ever ADDS a narrowly-verified path around it.
    """
    if record.accepted:
        return record
    rule = record.rule or {}
    if not (rule.get("applied") and rule.get("outcome") == CONFIRM):
        return record
    conf_path = it_dir / f"confirmation-{candidate.id}.json"
    if not conf_path.is_file():
        raise SystemExit(
            f"candidate {candidate.id!r}'s rule outcome is CONFIRM but no confirmation "
            f"record exists at {conf_path} — run `confirm` first"
        )
    confirmation = ConfirmationRecord.from_json(json.loads(conf_path.read_text()))
    if confirmation.candidate_id != candidate.id:
        raise SystemExit(
            f"{conf_path} was recorded for candidate {confirmation.candidate_id!r}, not "
            f"{candidate.id!r} — refusing to promote a mismatched confirmation"
        )
    if confirmation.stage != STAGE_PAIRED_CONFIRMATION:
        raise SystemExit(
            f"{conf_path} carries stage {confirmation.stage!r}, not "
            f"{STAGE_PAIRED_CONFIRMATION!r} — not a recognized confirmation"
        )
    outcome = confirmation.confirmation.get("outcome")
    if outcome != ACCEPT:
        raise SystemExit(
            f"candidate {candidate.id!r}'s confirmation outcome is {outcome!r}, not "
            f"{ACCEPT!r} — no PR"
        )
    # The confirmation rides along onto the record. It is what ACCEPTED this candidate:
    # its deltas, its quantiles and its attempt counts are the evidence for the merge,
    # and stage 1's are not. Promoting `accepted` alone left `pr_body` rendering the
    # first pass's numbers under the word ACCEPTED — the verdict of one measurement
    # printed beside another one's figures.
    return dataclasses.replace(record, accepted=True, confirmation=confirmation.to_json())


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
    record = _pr_eligible_record(it_dir, cand, record)
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
