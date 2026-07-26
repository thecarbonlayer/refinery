"""Branch + PR pipeline for an ACCEPTED candidate.

Decisions locked in docs/research/self-evolving-harness/open-questions.md §5:
one PR per accepted edit (never a bundle); the PR targets the dedicated
``self-improvement`` branch, set explicitly (GitHub would silently default to
``main`` otherwise); the PR body carries the evidence (cluster, knobs old->new,
Δ_in/Δ_ho with per-task breakdown, rationale, provenance) — the diff alone
never gets a vote; and the pipeline NEVER merges. The PR is the deliverable;
a human merges it. Branch protection on ``self-improvement`` is a one-time
GitHub setup step for the owner, not enforced here.

Provenance is a field on the candidate, not a hardcoded name: the line renders
as "<proposer>-proposed, task-suite-validated" so it stays accurate whoever
plays the proposer role in a later run.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loop.artifacts import Candidate, Cluster, ValidationRecord
from loop.config_edit import CONFIG_REL, apply_candidate
from runner.carbon_env import CARBON_ROOT, _git

BASE_BRANCH = "self-improvement"
COMMIT_TRAILER = "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"


def provenance(candidate: Candidate) -> str:
    return f"{candidate.proposer}-proposed, task-suite-validated"


def ensure_base_branch(carbon_root: Path = CARBON_ROOT, log=print) -> None:
    """Create ``self-improvement`` at the current main tip if it doesn't exist,
    and make sure origin has it (gh needs the base branch on the remote)."""
    branches = _git(carbon_root, "branch", "--list", BASE_BRANCH).strip()
    if not branches:
        main_tip = _git(carbon_root, "rev-parse", "main").strip()
        _git(carbon_root, "branch", BASE_BRANCH, "main")
        log(f"created {BASE_BRANCH} at main tip {main_tip[:10]}")
    remote = _git(carbon_root, "ls-remote", "--heads", "origin", BASE_BRANCH).strip()
    if not remote:
        _git(carbon_root, "push", "-u", "origin", BASE_BRANCH)
        log(f"pushed {BASE_BRANCH} to origin")


def _fields_summary(candidate: Candidate) -> str:
    return ", ".join(f"{k} {v['old']!r} -> {v['new']!r}" for k, v in candidate.fields.items())


def commit_message(candidate: Candidate, record: ValidationRecord, iteration: str) -> str:
    return (
        f"evolve({iteration}): {_fields_summary(candidate)} [{candidate.cluster_id}]\n\n"
        f"{candidate.rationale}\n\n"
        f"Validated against the 13-task suite: Δ_in={record.delta_in:+.4f}, "
        f"Δ_ho={record.delta_ho:+.4f} (acceptance rule: Δ_in ≥ 0, Δ_ho ≥ 0, max > 0).\n"
        f"{provenance(candidate)}.\n\n"
        f"{COMMIT_TRAILER}"
    )


def pr_body(
    candidate: Candidate,
    record: ValidationRecord,
    cluster: Cluster,
    baseline_results: dict,
    candidate_results: dict,
) -> str:
    """The required PR template — evidence in front of the reviewer, not just a diff."""
    knobs = "\n".join(
        f"| `{name}` | `{diff['old']}` | `{diff['new']}` |"
        for name, diff in candidate.fields.items()
    )
    rows = []
    for name in sorted(record.per_task):
        b = baseline_results["tasks"][name]
        c = candidate_results["tasks"][name]
        d = record.per_task[name]
        rows.append(
            f"| {name} | {b['split']} | {b['pass_fraction']:.4f} | "
            f"{c['pass_fraction']:.4f} | {d:+.4f} |"
        )
    per_task = "\n".join(rows)
    bf, cf = record.baseline_fingerprint, record.candidate_fingerprint
    return f"""## Failure cluster targeted

**{cluster.id}** — {cluster.mechanism}

{cluster.hypothesis}

Observed in: {", ".join(cluster.tasks)}.

## Knob(s) changed

| field | old | new |
|---|---|---|
{knobs}

(`version` bumped {bf.get("config_version")} -> {cf.get("config_version")} by the pipeline.)

## Validation — acceptance rule `Δ_in ≥ 0, Δ_ho ≥ 0, max(Δ_in, Δ_ho) > 0`

**Δ_in = {record.delta_in:+.4f}, Δ_ho = {record.delta_ho:+.4f} -> ACCEPTED**

| task | split | baseline | candidate | Δ |
|---|---|---|---|---|
{per_task}

Baseline fingerprint: `{bf.get("gemma_sha", "")[:12]}` (config v{bf.get("config_version")}, \
model {bf.get("model")}).
Candidate fingerprint: `{cf.get("gemma_sha", "")[:12]}`{"+dirty" if cf.get("gemma_dirty") else ""} \
(config v{cf.get("config_version")}, model {cf.get("model")}) — the candidate was measured as a \
working-tree edit (`dirty_sha` `{str(cf.get("dirty_sha"))[:12]}`) whose config content is \
byte-identical to this PR's; rejected candidates never become commits.

Known limitation (disclosed by design): these Δ numbers were computed locally against LM Studio; \
hosted CI cannot re-verify them.

## Proposer's rationale

{candidate.rationale}

**Expected effect:** {candidate.expected_effect}

**Named regression risk:** {candidate.regression_risk}

## Provenance

**{provenance(candidate)}** ({candidate.proposer_detail}).
"""


def open_pr(
    candidate: Candidate,
    record: ValidationRecord,
    cluster: Cluster,
    baseline_results: dict,
    candidate_results: dict,
    iteration: str,
    carbon_root: Path = CARBON_ROOT,
    gh_run=subprocess.run,
    log=print,
) -> str:
    """Branch off the ``self-improvement`` tip, commit the edit, push, open the
    PR (explicit base). Returns the PR URL. Never merges."""
    if not record.accepted:
        raise ValueError(f"candidate {candidate.id!r} was not accepted — no PR")
    ensure_base_branch(carbon_root, log=log)
    branch = f"evolve/{iteration}-{candidate.id}"
    if _git(carbon_root, "branch", "--list", branch).strip():
        raise RuntimeError(f"branch {branch} already exists in {carbon_root}")
    original = _git(carbon_root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    _git(carbon_root, "switch", "-c", branch, BASE_BRANCH)
    try:
        apply_candidate(carbon_root, candidate)
        _git(carbon_root, "add", str(CONFIG_REL))
        _git(carbon_root, "commit", "-m", commit_message(candidate, record, iteration))
        _git(carbon_root, "push", "-u", "origin", branch)
        title = f"evolve({iteration}): {_fields_summary(candidate)}"
        body = pr_body(candidate, record, cluster, baseline_results, candidate_results)
        proc = gh_run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                BASE_BRANCH,
                "--head",
                branch,
                "--title",
                title,
                "--body",
                body,
            ],
            cwd=carbon_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gh pr create failed: {proc.stderr.strip()}")
        url = proc.stdout.strip().splitlines()[-1]
        log(f"opened PR for {candidate.id}: {url}")
        return url
    finally:
        _git(carbon_root, "switch", original)
