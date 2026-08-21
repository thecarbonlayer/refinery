"""Branch + PR pipeline for an ACCEPTED candidate.

Decisions locked for this pipeline:
one PR per accepted edit (never a bundle); the PR targets the dedicated
``self-improvement`` branch, set explicitly (GitHub would silently default to
``main`` otherwise); the PR body carries the evidence (cluster, knobs old->new,
Δ_in/Δ_ho with per-task breakdown, rationale, provenance, the rule's disposition,
and its security story) — the diff alone never gets a vote; and the pipeline
NEVER merges. The PR is the deliverable; a human merges it. Branch protection on
``self-improvement`` is a one-time GitHub setup step for the owner, not enforced
here.

Provenance is a field on the candidate, not a hardcoded name: the line renders
as "<proposer>-proposed, task-suite-validated" so it stays accurate whoever
plays the proposer role in a later run.

The security story matters here specifically because ``open_pr`` only ever fires
for ``record.accepted``, and a candidate can reach ACCEPT with an observed
behavioral security regression that was cleared only by statistical power (a
confirmed_increase blocks; "inconclusive" does not, and inconclusive is a common,
legitimate outcome of a test that is deliberately biased toward more measurement
over a noise-driven veto — see ``FISHER_ALPHA`` in ``loop/acceptance.py``). A
human approving the PR must see that trace, not just the deltas.
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
        f"Validated against the full task suite: Δ_in={record.delta_in:+.4f}, "
        f"Δ_ho={record.delta_ho:+.4f} (acceptance rule: Δ_in ≥ 0, Δ_ho ≥ 0, max > 0).\n"
        f"{provenance(candidate)}.\n\n"
        f"{COMMIT_TRAILER}"
    )


def _counts_note(attempts: dict) -> str:
    """``"A1 10v10, G4 10v10"`` — the per-task attempt counts a quantile was computed
    at, straight off the decision's own quantile record. A threshold with no counts
    beside it is exactly the round-1 artifact this program withdrew: a number nobody
    could tell was about the wrong run."""
    return ", ".join(f"{task} {pair[0]}v{pair[1]}" for task, pair in sorted(attempts.items()))


def _quantile_rows(quantiles: dict) -> str:
    rows = []
    for split in ("held_in", "held_out"):
        rec = quantiles.get(split) or {}
        tasks = ", ".join(rec.get("tasks") or []) or "_none on this split_"
        rows.append(
            f"| {split.replace('_', '-')} | {tasks} | `{rec.get('quantile', '?')}` | "
            f"{_counts_note(rec.get('attempts') or {}) or '—'} |"
        )
    return "\n".join(rows)


def _calibrated_validation(
    record: ValidationRecord, rule: dict, raw: dict, regressions: str
) -> str:
    """The Validation section for a decision judged against a MEASURED NULL MODEL.

    The legacy section states the one-number acceptance rule. A calibrated decision
    was never judged by it — it was judged on the SUPPORTED-SET split means against
    quantiles computed from pooled null rates at the two runs' own attempt counts —
    so rendering the legacy text there tells the human approving the merge that a
    comparison was made which nobody made. What goes here instead is the rule that
    actually ran, the deltas and bounds it ran on, the identity of the artifact those
    bounds came from, and where to find the confirmation.
    """
    cal = rule.get("calibration") or raw.get("calibration") or {}
    quantiles = raw.get("null_quantiles") or {}
    rates = ", ".join(f"{t} `{r}`" for t, r in sorted((cal.get("null_rates") or {}).items()))
    coverage = cal.get("coverage_level", "39/40")
    sha = str(cal.get("computed_at_runner_sha") or "")
    reasons = "\n".join(f"- {reason}" for reason in rule.get("reasons") or []) or "- _none_"
    return f"""## Validation — measured null model for section `{cal.get("section", "?")}`

**Supported-set Δ_in = {rule.get("delta_in", 0.0):+.4f}, \
Δ_ho = {rule.get("delta_ho", 0.0):+.4f} -> {record.disposition}**

This candidate was NOT judged by the one-number acceptance rule. It was judged against
a measured NULL MODEL: one pooled pass rate per supported task, measured from arms with
NOTHING changed between them, from which every bound below was computed at these two
runs' own attempt counts rather than read off the artifact. CONFIRM required the
supported-set split mean to exceed the {coverage} quantile of that null distribution on
the evidence split. ACCEPT is not reachable here at all — only a fresh paired
confirmation grants it, and it additionally requires every carrier to repeat beyond its
OWN quantile, every guard to hold above its own, and both supported-set means to be
non-negative.

| split | supported tasks | {coverage} null quantile | computed at |
|---|---|---:|---|
{_quantile_rows(quantiles)}

Whole-split means, kept for comparison and NOT what decided: \
Δ_in = {raw.get("full_split_delta_in", 0.0):+.4f}, \
Δ_ho = {raw.get("full_split_delta_ho", 0.0):+.4f}.

Null model: `{cal.get("source", "?")}`, computed at runner `{sha[:12]}`. Pooled null
rates: {rates or "_not recorded_"}. Guards: {", ".join(cal.get("guards") or []) or "none"}.

Rule outcome: **{rule.get("outcome", "?")}**
{reasons}

Per-task regression warnings: {regressions}. A full-pass -> zero-pass movement is a
promotion veto even when the aggregate split rule passes."""


def _confirmation_section(record: ValidationRecord) -> str:
    """The measurement that actually accepted the candidate, when one did.

    A CONFIRM candidate's validation record keeps stage 1's numbers forever —
    ``validate_candidate`` never rewrites it — so a promoted record used to render the
    first pass's deltas under the word ACCEPTED. The first pass did not accept
    anything: it said "promising". This section carries the run that did, at its own
    attempt counts, against its own quantiles.
    """
    conf = record.confirmation or {}
    if not conf:
        return ""
    decision = conf.get("confirmation") or {}
    raw = decision.get("raw") or {}
    carriers = raw.get("carrier_quantiles") or {}
    guards = raw.get("guard_quantiles") or {}
    rows = [
        f"| {kind} | {task} | `{rec.get('quantile', '?')}` | "
        f"{_counts_note(rec.get('attempts') or {}) or '—'} |"
        for kind, group in (("carrier", carriers), ("guard", guards))
        for task, rec in sorted(group.items())
    ]
    table = (
        "\n".join(["| gate | task | null quantile | computed at |", "|---|---|---:|---|", *rows])
        if rows
        else "_no per-task gates recorded — this confirmation was judged uncalibrated._"
    )
    reasons = "\n".join(f"- {reason}" for reason in decision.get("reasons") or []) or "- _none_"
    return f"""

## Confirmation — the measurement that accepted this candidate

Fresh paired rerun `{conf.get("baseline_label", "?")}` (config unedited) against
`{conf.get("candidate_label", "?")}` (candidate applied), \
{conf.get("attempts_per_task_per_arm", "?")} attempts per task per arm on \
{", ".join(conf.get("confirm_set") or []) or "no tasks"} -> \
**{decision.get("outcome", "?")}**. Recorded as \
`confirmation-{record.candidate_id}.json` beside this candidate's validation record.

**Confirmed supported-set Δ_in = {decision.get("delta_in", 0.0):+.4f}, \
Δ_ho = {decision.get("delta_ho", 0.0):+.4f}** — these, not the first pass's, are the
numbers this promotion rests on.

{table}

{reasons}"""


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
    regressions = (
        ", ".join(f"`{name}` ({change:+.4f})" for name, change in record.regressions.items())
        or "none"
    )

    # The rule's security story — carried on EVERY path (REJECT, CONFIRM, ACCEPT), not
    # only when it happens to be clean, so a human approving an ACCEPT never has to take
    # "no news" on faith. ``rule`` is `{}` for records the three-outcome rule never
    # decided (an uncalibrated section, or a gate failure before any suite ran) — that
    # state is named explicitly below ("the rule did not run"), never folded into
    # "none": "none" is reserved for a rule that ran and found nothing, the same
    # unknown-vs-clean distinction the denominator note ~40 lines below already draws.
    rule = record.rule or {}
    rule_applied = bool(rule.get("applied"))
    rule_raw: dict = rule.get("raw") or {}
    # Which rule the body must STATE is decided by which rule ran, read off the
    # decision's own record. `regime` is written by `loop.acceptance.evaluate()` only
    # when a calibration decided, so a legacy record (and every uncalibrated section)
    # renders exactly the text it always did, byte for byte.
    calibrated = rule_applied and rule_raw.get("regime") == "section_calibration"
    sec_reg: dict = rule.get("security_regressions") or {}
    beh_reg: dict = rule.get("behavioral_regressions") or {}
    behavioral_verdicts: dict = (rule.get("raw") or {}).get("behavioral_verdicts") or {}

    def _security_line(name: str) -> str:
        """One task's regression, split by class where the record actually supports
        the split. `sec_reg[name]` is the UNFILTERED total [baseline, candidate] pair,
        known whenever the task's total critical-failure count rose. `beh_reg[name]`
        is the behavioral-only pair, known only when the BEHAVIORAL count itself rose
        — its absence means "behavioral did not rise", never "behavioral is zero". A
        task can carry a rising total built from a FALLING behavioral count and a
        larger mechanical one (mechanical 0->3, behavioral 2->1 renders total 2->4,
        with no behavioral pair on record to subtract out); a legacy record written
        before `behavioral_regressions` existed carries `security_regressions` with no
        behavioral key at all, for any task. Both must render the total, unclassified,
        rather than assert "mechanical" for arithmetic the record does not support —
        only when `name in beh_reg` are both halves on record, making total-minus-
        behavioral exact. A task can also carry a behavioral pair with NO total on
        record at all: the total fell or held while the behavioral count alone rose,
        so it never entered `sec_reg` — only the behavioral numbers are known then."""
        total = sec_reg.get(name)
        beh = beh_reg.get(name)
        if total is not None and beh is not None:
            b_total, c_total = total
            b_beh, c_beh = beh
            b_mech, c_mech = b_total - b_beh, c_total - c_beh
            detail = (
                f"mechanical {b_mech}->{c_mech}, behavioral {b_beh}->{c_beh}"
                if (b_mech or c_mech)
                else f"behavioral {b_beh}->{c_beh}"
            )
        elif total is not None:
            b_total, c_total = total
            detail = f"unclassified {b_total}->{c_total}"
        else:
            b_beh, c_beh = beh
            detail = f"behavioral {b_beh}->{c_beh} (total did not rise)"
        return f"`{name}` ({detail})"

    def _fisher_line(name: str) -> str:
        v = behavioral_verdicts[name]
        base_fail, base_n = v["counts"]["baseline"]
        cand_fail, cand_n = v["counts"]["candidate"]
        return (
            f"`{name}` **{v['verdict']}** (p={v['p_one_sided']:.3f}, "
            f"baseline {base_fail}/{base_n}, candidate {cand_fail}/{cand_n})"
        )

    if rule_applied:
        # A behavioral-only rise (total fell or held) never enters `sec_reg`, so the
        # union is iterated rather than `sec_reg` alone: iterating `sec_reg` alone
        # silently dropped exactly this case — mechanical 3->0, behavioral 0->2 nets a
        # FALLING total, so the routed behavioral rise disappeared from the section
        # instead of rendering, and `targeted_rerun` is never rendered either, so
        # nothing else recovers it.
        security_lines = (
            ", ".join(_security_line(n) for n in sorted(set(sec_reg) | set(beh_reg))) or "none"
        )
        fisher_lines = ", ".join(_fisher_line(n) for n in sorted(behavioral_verdicts)) or "none"
    else:
        # applied=False (uncalibrated section) and the fully-empty `{}` (a gate failure
        # or a pre-rule legacy record) mean the same thing here: no security count was
        # ever computed for this record, so "none" — "measured, and clean" — would be a
        # stronger claim than the data supports.
        why = rule.get("why") or "no rule verdict is recorded on this record"
        security_lines = fisher_lines = f"rule did not run — {why}"

    def _confirmation_arm_n() -> int | None:
        """The attempts-per-arm behind every verdict rendered above, when they all
        agree on one count across both arms — None with no verdicts to check it
        against, or when they disagree, so the power note only ever states a headcount
        this record actually backs."""
        ns = {
            n
            for v in behavioral_verdicts.values()
            for n in (v["counts"]["baseline"][1], v["counts"]["candidate"][1])
        }
        return ns.pop() if len(ns) == 1 else None

    _arm_n = _confirmation_arm_n()
    # The 6.3%/19.1% figures below are exact only AT the 10-attempts-per-arm table in
    # loop/acceptance.py's `FISHER_ALPHA` comment. Hardcoding "10 attempts per arm" as
    # if it always describes the run being reported left the note beside contradicting
    # numbers whenever a confirmation actually ran at a different count.
    power_caveat = (
        f" This confirmation ran at {_arm_n} attempts per arm, not the 10 the figures "
        "above assume — read 6.3%/19.1% as the shape of the test's power, not this "
        "run's actual power."
        if _arm_n is not None and _arm_n != 10
        else ""
    )

    # Every row carries its own denominator. A mean over 20 of 23 tasks and a mean
    # over all 23 are different measurements, and a reader cannot tell them apart
    # from the number alone.
    def _denominator(name: str) -> str:
        """Tasks/attempts behind a row, bolded when the two sides disagree.

        Attempts matter as much as tasks: an erroring candidate keeps every task in
        the population while its mean covers a fraction of the attempts.
        """
        cells = []
        for counts in (record.metric_task_counts, record.metric_attempt_counts):
            pair = counts.get(name) or {}
            base, cand = pair.get("baseline"), pair.get("candidate")
            if base is None and cand is None:
                cells.append("?")
            elif base == cand:
                cells.append(str(base))
            else:
                cells.append(f"**{base}→{cand}**")
        return " / ".join(cells)

    metric_rows = (
        "\n".join(
            f"| `{name}` | {record.baseline_metrics.get(name, 0):.4f} | "
            f"{record.candidate_metrics.get(name, 0):.4f} | {change:+.4f} | "
            f"{_denominator(name)} |"
            for name, change in record.metric_delta.items()
        )
        or "| _not recorded_ | — | — | — | — |"
    )
    not_compared = (
        ", ".join(f"`{name}`" for name in record.metric_not_compared)
        or "none — both runs measured the same metrics"
    )
    # A metric with no entry in either count dict renders "? / ?". Claiming parity
    # beside an unknown denominator would be a stronger statement than the data
    # supports, so unknowns are named rather than folded into "none".
    # EITHER count missing makes the denominator unknown — an `or` here left a metric
    # with attempts but no task count unflagged, so the body claimed parity for a
    # count it did not have. And unknowns are appended rather than shadowed by an
    # `elif`: when some metric drifts, the ones with no denominator at all still
    # deserve a caveat instead of an unexplained `? / ?`.
    unknown = sorted(
        name
        for name in record.metric_delta
        if not (record.metric_task_counts.get(name) and record.metric_attempt_counts.get(name))
    )
    parts = []
    if record.metric_denominator_drift:
        parts.append(", ".join(f"`{name}`" for name in record.metric_denominator_drift))
    if unknown:
        parts.append("denominator unknown for " + ", ".join(f"`{name}`" for name in unknown))
    drift = "; ".join(parts) or "none — every metric covers the same counts on both sides"
    bf, cf = record.baseline_fingerprint, record.candidate_fingerprint
    confirmation_section = _confirmation_section(record)
    validation_section = (
        _calibrated_validation(record, rule, rule_raw, regressions)
        if calibrated
        else f"""## Validation — acceptance rule `Δ_in ≥ 0, Δ_ho ≥ 0, max(Δ_in, Δ_ho) > 0`

**Δ_in = {record.delta_in:+.4f}, Δ_ho = {record.delta_ho:+.4f} -> {record.disposition}**

Per-task regression warnings: {regressions}. A full-pass -> zero-pass movement is a
promotion veto even when the aggregate split rule passes."""
    )
    return f"""## Failure cluster targeted

**{cluster.id}** — {cluster.mechanism}

{cluster.hypothesis}

Observed in: {", ".join(cluster.tasks)}.

## Knob(s) changed

| field | old | new |
|---|---|---|
{knobs}

(`version` bumped {bf.get("config_version")} -> {cf.get("config_version")} by the pipeline.)

{validation_section}

| task | split | baseline | candidate | Δ |
|---|---|---|---|---|
{per_task}{confirmation_section}

### Efficiency and trajectory telemetry

These values are diagnostic and do not override task correctness. Negative cost,
token, call, error, and compaction deltas generally mean less work for the same score —
but only where the denominator matches. Each mean covers only the tasks that reported
that metric (scripted fault-injection diagnostics have no token or cost figure), so the
contributing count is given per row as `tasks / attempts`; a **bolded, differing**
count means the two means cover different populations and the Δ is NOT like-for-like.
Attempts matter as much as tasks — a candidate whose attempts start erroring keeps
every task in the population while its mean covers a fraction of the samples.

| metric | baseline mean | candidate mean | Δ | tasks / attempts |
|---|---:|---:|---:|---:|
{metric_rows}

Not compared (measured on one side only, never imputed as zero): {not_compared}.
Contributing-count drift: {drift}. Note the suite mean weights each reporting TASK
equally, so an identical `tasks / attempts` pair can still hide attempts moving
between tasks; the pair bounds the population, it does not describe the weighting.

Baseline fingerprint: `{bf.get("gemma_sha", "")[:12]}` (config v{bf.get("config_version")}, \
model {bf.get("model")}).
Candidate fingerprint: `{cf.get("gemma_sha", "")[:12]}`{"+dirty" if cf.get("gemma_dirty") else ""} \
(config v{cf.get("config_version")}, model {cf.get("model")}) — the candidate was measured as a \
working-tree edit (`dirty_sha` `{str(cf.get("dirty_sha"))[:12]}`) whose config content is \
byte-identical to this PR's; rejected candidates never become commits.

Known limitation (disclosed by design): these Δ numbers were computed locally against LM Studio; \
hosted CI cannot re-verify them.

## Security

**Disposition: {record.disposition}**

Security regressions (raw critical-failure counts, baseline -> candidate, by class): \
{security_lines}.

Fisher verdicts from the confirmation's predeclared one-sided test (behavioral class \
only — a mechanical rise blocks unconditionally and needs no test): {fisher_lines}.

Power note: the Fisher test detects only large differences. At 10 attempts per arm
against C3's measured ~12%/attempt behavioral base rate, a DOUBLED leak rate is caught
6.3% of the time and a TRIPLED rate 19.1% (exact enumeration; see the `FISHER_ALPHA`
comment in `loop/acceptance.py`). "no_increase" and "inconclusive" mean the test did
not detect an increase this large on this run — not that the candidate is clean.\
{power_caveat}

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
