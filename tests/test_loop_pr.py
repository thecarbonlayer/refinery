"""Branch + PR pipeline: base-branch creation, one branch per edit, evidence body."""

import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from loop.artifacts import Candidate, Cluster, ValidationRecord
from loop.prpipe import commit_message, ensure_base_branch, open_pr, pr_body, provenance
from runner.carbon_env import CARBON_ROOT, _git

REAL_CONFIG = CARBON_ROOT / "harness" / "harness_config.json"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _bumped() -> int:
    """The version an edit should produce: whatever the real config carries now,
    plus one. Hardcoding it pinned these tests to a config that has since been
    bumped in carbon, so they broke on a change that was not theirs."""
    return json.loads(REAL_CONFIG.read_text())["version"] + 1


def _live(field: str):
    """The value carbon's config carries right now.

    Hardcoding `old` pinned these tests to one config exactly as hardcoding
    `version` once did: a legal `max_tokens` candidate broke five tests that had
    nothing to do with it. `max_tokens` is a knob the loop exists to tune.
    """
    return json.loads(REAL_CONFIG.read_text())[field]


OLD_MT = _live("max_tokens")
NEW_MT = OLD_MT * 2  # legal (positive int, no ceiling) and always distinct


CANDIDATE = Candidate(
    id="cand-raise-output-budget",
    cluster_id="CL-1",
    proposer="Fable",
    proposer_detail="claude-fable-5, in-session",
    fields={"max_tokens": {"old": OLD_MT, "new": NEW_MT}},
    rationale="The response budget cuts a required long answer before its final receipt.",
    expected_effect="G1 recovers",
    regression_risk="higher token cost and latency",
)

CLUSTER = Cluster(
    id="CL-1",
    mechanism="completion budget cuts valid output",
    tasks=("G1",),
    hypothesis="the response budget ends the answer before the final receipt",
    evidence=("reply: 'incident handoff ends before line 400'",),
)

BASE_FP = {
    "gemma_sha": "a" * 40,
    "gemma_dirty": False,
    "dirty_sha": None,
    "config_version": 1,
    "model": "m",
    "runner_sha": "r1",
}
CAND_FP = {
    "gemma_sha": "a" * 40,
    "gemma_dirty": True,
    "dirty_sha": "d" * 40,
    "config_version": 2,
    "model": "m",
    "runner_sha": "r1",
}

RECORD = ValidationRecord(
    candidate_id=CANDIDATE.id,
    label="cand-x",
    accepted=True,
    delta_in=0.125,
    delta_ho=0.2,
    per_task={"A2": 1.0, "A4": 0.8, "D1": 0.0},
    baseline_fingerprint=BASE_FP,
    candidate_fingerprint=CAND_FP,
)


def results(fractions):
    return {
        "tasks": {
            name: {
                "split": "held_out" if name in {"A3", "A4"} else "held_in",
                "pass_fraction": frac,
            }
            for name, frac in fractions.items()
        }
    }


BASELINE_RESULTS = results({"A2": 0.0, "A4": 0.0, "D1": 1.0})
CANDIDATE_RESULTS = results({"A2": 1.0, "A4": 0.8, "D1": 1.0})


@pytest.fixture
def repos(tmp_path):
    """A fake carbon clone with a local bare 'origin' — push is exercised for real."""
    origin = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True
    )
    root = tmp_path / "carbon"
    (root / "harness").mkdir(parents=True)
    shutil.copy(REAL_CONFIG, root / "harness" / "harness_config.json")
    for args in (
        ["init", "-b", "main"],
        ["config", "user.email", "t@t"],
        ["config", "user.name", "t"],
        ["add", "-A"],
        ["commit", "-m", "seed"],
        ["remote", "add", "origin", str(origin)],
        ["push", "-u", "origin", "main"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return root, origin


def test_provenance_uses_candidate_field():
    assert provenance(CANDIDATE) == "Fable-proposed, task-suite-validated"
    sol = Candidate(**{**CANDIDATE.__dict__, "proposer": "Sol"})
    assert provenance(sol) == "Sol-proposed, task-suite-validated"


def test_commit_message_carries_evidence_and_trailer():
    msg = commit_message(CANDIDATE, RECORD, "iter-01")
    assert f"evolve(iter-01): max_tokens {OLD_MT} -> {NEW_MT} [CL-1]" in msg
    assert "Δ_in=+0.1250" in msg and "Δ_ho=+0.2000" in msg
    assert "Fable-proposed, task-suite-validated" in msg
    assert msg.endswith("Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>")


def test_pr_body_is_the_required_template():
    body = pr_body(CANDIDATE, RECORD, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "CL-1" in body and "completion budget" in body
    assert f"| `max_tokens` | `{OLD_MT}` | `{NEW_MT}` |" in body
    assert "Δ_in = +0.1250, Δ_ho = +0.2000" in body and "ACCEPTED" in body
    assert "| A2 | held_in | 0.0000 | 1.0000 | +1.0000 |" in body  # per-task, not aggregate
    assert "**Fable-proposed, task-suite-validated**" in body
    assert "computed locally against LM Studio" in body  # disclosed limitation


def _rule(outcome, **extra):
    """A minimal `rule` dict shaped exactly like `{"applied": True, **decision.to_json()}`
    — the contract `loop/validate.py` actually writes onto `ValidationRecord.rule`
    (see `rule_disposition()` in loop/validate.py)."""
    base = {
        "applied": True,
        "outcome": outcome,
        "reasons": (),
        "delta_in": 0.0,
        "delta_ho": 0.0,
        "threshold_in": 0.0,
        "threshold_ho": 0.0,
        "excluded": [],
        "evidence_split": "",
        "improved_tasks": [],
        "confirm_tasks": [],
        "targeted_rerun": [],
        "security_regressions": {},
        "behavioral_regressions": {},
        "raw": {},
    }
    base.update(extra)
    return base


def test_pr_body_renders_the_rules_disposition_word_not_a_hardcoded_one():
    """A human approving a PR must see the RULE's disposition, not a literal "ACCEPTED"
    baked into the template: a CONFIRM-outcome record (not yet shipped — only a fresh
    paired confirmation can accept it) must never read as accepted, and a REJECT-outcome
    record must never read as pending."""
    pending = replace(RECORD, accepted=False, rule=_rule("CONFIRM"))
    body = pr_body(CANDIDATE, pending, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "-> PENDING_CONFIRMATION" in body
    assert "-> ACCEPTED" not in body and "-> REJECTED" not in body
    assert "**Disposition: PENDING_CONFIRMATION**" in body

    rejected = replace(RECORD, accepted=False, rule=_rule("REJECT"))
    body = pr_body(CANDIDATE, rejected, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "-> REJECTED" in body
    assert "-> ACCEPTED" not in body and "-> PENDING_CONFIRMATION" not in body

    accepted = replace(RECORD, accepted=True, rule=_rule("ACCEPT"))
    body = pr_body(CANDIDATE, accepted, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "-> ACCEPTED" in body


def test_pr_body_renders_security_regressions_with_class():
    """`O1` regressed purely behaviorally — both its baseline and candidate behavioral
    counts are on record, so the pure-behavioral rendering is exact. `O0`'s TOTAL rose
    with no entry in `behavioral_regressions` at all: that shape does NOT prove the
    rise was mechanical (absence means "behavioral did not rise", not "there is no
    behavioral component" — see
    test_pr_body_renders_unclassified_when_the_total_rose_but_behavioral_fell for why
    asserting "mechanical" here can print numbers that are not the actual mechanical
    count), so it renders unclassified."""
    rule = _rule(
        "REJECT",
        security_regressions={"O0": [0, 1], "O1": [0, 1]},
        behavioral_regressions={"O1": [0, 1]},
    )
    rec = replace(RECORD, accepted=False, rule=rule)
    body = pr_body(CANDIDATE, rec, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "`O0` (unclassified 0->1)" in body
    assert "`O1` (behavioral 0->1)" in body


def test_pr_body_renders_a_mixed_class_security_regression():
    """A task regressing in BOTH classes at once (the same shape as
    test_dual_class_regression_reason_names_mechanical_record_carries_unfiltered_total
    in test_acceptance.py) must show both halves: total [0, 4], of which [0, 3] is
    behavioral (from `behavioral_regressions`) and the remaining [0, 1] is mechanical
    (the unfiltered total minus the behavioral component) — a reviewer reading only the
    total would not know one leak type dominates over the other."""
    rule = _rule(
        "REJECT",
        security_regressions={"O0": [0, 4]},
        behavioral_regressions={"O0": [0, 3]},
    )
    rec = replace(RECORD, accepted=False, rule=rule)
    body = pr_body(CANDIDATE, rec, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "`O0` (mechanical 0->1, behavioral 0->3)" in body


def test_pr_body_renders_unclassified_when_the_total_rose_but_behavioral_fell():
    """A task's TOTAL can rise while its behavioral count FALLS: mechanical 0->3,
    behavioral 2->1 nets a total of 2->4. `behavioral_regressions` only ever carries a
    task whose behavioral count itself rose, so this task is absent from it even though
    the total genuinely regressed — the record carries neither baseline's nor
    candidate's individual behavioral count, so `pr_body` cannot recover the mechanical
    split (0->3) from what it has (2->4 total, no behavioral pair). It must say
    "unclassified 2->4", never assert "mechanical 2->4": that would both mislabel the
    class and print numbers that are not the actual mechanical count."""
    rule = _rule("REJECT", security_regressions={"C3": [2, 4]})  # behavioral_regressions: {}
    rec = replace(RECORD, accepted=False, rule=rule)
    body = pr_body(CANDIDATE, rec, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "`C3` (unclassified 2->4)" in body
    # Scoped to the rendered regression line, not the whole section — the Fisher-verdict
    # sentence right below it legitimately uses the word "mechanical" in static prose.
    reg_line = next(line for line in body.splitlines() if line.startswith("Security regressions"))
    assert "mechanical" not in reg_line


def test_pr_body_renders_a_legacy_record_with_no_behavioral_regressions_key_at_all():
    """`iterations/iter-04/validation-tool-output-offload-r2.json` predates
    `behavioral_regressions` entirely: its `rule["security_regressions"]` is
    `{"C3": [0, 1]}` and `behavioral_regressions` is not a key in the dict at all —
    not even an empty one. That 0->1 was actually the BEHAVIORAL case, but nothing in
    the record can prove that to `pr_body`, so it must render unclassified, never
    "mechanical" — the module comment's old claim that such records "still render
    cleanly" was true only in the sense that they render without crashing."""
    legacy_path = REPO_ROOT / "iterations" / "iter-04" / "validation-tool-output-offload-r2.json"
    legacy_rule = json.loads(legacy_path.read_text())["rule"]
    assert "behavioral_regressions" not in legacy_rule  # the historical shape this guards
    rec = replace(RECORD, accepted=False, rule=legacy_rule)
    body = pr_body(CANDIDATE, rec, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "`C3` (unclassified 0->1)" in body
    assert "`C3` (mechanical 0->1)" not in body


def test_pr_body_recovers_a_behavioral_rise_hidden_by_a_falling_total():
    """mechanical 3->0, behavioral 0->2: the TOTAL falls (3->2), so the task never
    enters `security_regressions` at all (`_regressions()` only keeps a task whose
    candidate count exceeds baseline) — only `behavioral_regressions` carries it.
    Iterating `sorted(security_regressions)` alone silently drops the task from the
    section entirely, even though a real routed behavioral rise exists; the union of
    both dicts must be walked instead."""
    rule = _rule("REJECT", behavioral_regressions={"C3": [0, 2]})  # security_regressions: {}
    rec = replace(RECORD, accepted=False, rule=rule)
    body = pr_body(CANDIDATE, rec, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "`C3` (behavioral 0->2 (total did not rise))" in body


def test_pr_body_renders_fisher_verdicts_and_the_power_limitation():
    """A confirmed behavioral increase must show the task, the verdict word, and the
    p-value — and every render must carry the stated power limitation so "no_increase"/
    "inconclusive" is never read as "clean", quoted (not re-derived) from
    loop/acceptance.py's FISHER_ALPHA comment: 10 attempts/arm against C3's measured
    ~12%/attempt base rate catches a doubled rate 6.3% of the time, a tripled rate
    19.1% (exact enumeration)."""
    rule = _rule(
        "ACCEPT",
        raw={
            "stage": "confirmation",
            "behavioral_verdicts": {
                "O4": {
                    "verdict": "confirmed_increase",
                    "p_one_sided": 210 / 4845,
                    "alpha": 0.05,
                    "counts": {"baseline": [0, 10], "candidate": [4, 10]},
                }
            },
        },
    )
    rec = replace(RECORD, accepted=True, rule=rule)
    body = pr_body(CANDIDATE, rec, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "`O4` **confirmed_increase** (p=0.043, baseline 0/10, candidate 4/10)" in body
    assert "10 attempts per arm" in body
    assert "12%/attempt" in body
    assert "6.3%" in body
    assert "19.1%" in body
    assert "FISHER_ALPHA" in body
    # This fixture's confirmation ran at exactly the table's own 10-per-arm, so no
    # mismatch caveat belongs here — one would be noise beside numbers that agree.
    assert "This confirmation ran at" not in body


def test_pr_body_power_note_flags_a_confirmation_that_ran_at_a_different_arm_count():
    """The 6.3%/19.1% figures are exact only at 10 attempts per arm (the table in
    loop/acceptance.py's `FISHER_ALPHA` comment). A confirmation that actually ran at a
    different count must not leave that fixed table standing, uncontradicted, beside
    verdicts that show a different n — the note must derive and state the count this
    record actually used."""
    rule = _rule(
        "ACCEPT",
        raw={
            "stage": "confirmation",
            "behavioral_verdicts": {
                "O4": {
                    "verdict": "no_increase",
                    "p_one_sided": 1.0,
                    "alpha": 0.05,
                    "counts": {"baseline": [1, 20], "candidate": [1, 20]},
                }
            },
        },
    )
    rec = replace(RECORD, accepted=True, rule=rule)
    body = pr_body(CANDIDATE, rec, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "This confirmation ran at 20 attempts per arm, not the 10" in body


def test_pr_body_security_section_says_none_only_when_the_rule_actually_ran_clean():
    """ "none" must mean "the rule ran and found nothing" — not "nothing was computed".
    A record whose rule genuinely applied and found no regressions or verdicts still
    renders "none"; that is the ONE state the word may honestly describe."""
    clean = replace(RECORD, accepted=True, rule=_rule("ACCEPT"))
    body = pr_body(CANDIDATE, clean, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    security_section = body.split("## Security", 1)[1].split("\n## ", 1)[0]
    assert "none" in security_section
    assert "did not run" not in security_section


def test_pr_body_names_an_uncalibrated_sections_rule_as_not_run_not_none():
    """`rule_disposition()` returns `{"applied": False, "why": ...}` when the edited
    section is not `tool_output` — the ONLY shape today's CLI can hand `pr_body` besides
    an applied rule (a gate failure never reaches here: `open_pr` only ever fires for
    `record.accepted`, and a gate failure forces `accepted=False`). No security count is
    ever computed in this state, so "none" — indistinguishable from "measured, clean" —
    is a stronger claim than the data supports; the section must say the rule did not
    run, and why, echoing the same unknown-vs-clean standard the denominator note
    already applies to telemetry."""
    uncalibrated = replace(
        RECORD,
        accepted=True,  # the CAUSAL verdict can still accept outside RULE_SECTIONS
        rule={"applied": False, "why": "edited sections ['max_tokens'] are not calibrated"},
    )
    body = pr_body(CANDIDATE, uncalibrated, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    security_section = body.split("## Security", 1)[1].split("\n## ", 1)[0]
    assert "rule did not run" in security_section
    assert "edited sections ['max_tokens'] are not calibrated" in security_section
    assert "none" not in security_section


def test_pr_body_names_a_fully_empty_rule_as_not_run_too():
    """The default `ValidationRecord.rule` is `{}` (a gate failure before any suite
    ran, or a record written before the rule existed at all) — no "why" to quote, but
    it is still "the rule did not run", not "none": "the record carries the security
    story on every path" includes the path where there is none to tell, honestly."""
    body = pr_body(CANDIDATE, RECORD, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "## Security" in body
    security_section = body.split("## Security", 1)[1].split("\n## ", 1)[0]
    assert "rule did not run" in security_section
    assert "none" not in security_section


def test_pr_body_nests_efficiency_telemetry_under_validation_not_security():
    """`### Efficiency and trajectory telemetry` must stay a subsection of `##
    Validation` — inserting `## Security` directly above it nested cost/token telemetry
    under Security instead, changing what the heading hierarchy claims without changing
    a single number. Isolating a section by its heading must anchor on the next
    TOP-LEVEL (`##`) heading, not the next `###` — the latter only ever bounded the
    Security section by coincidence, while it happened to be followed immediately by a
    `###` subsection that belonged to Validation."""
    body = pr_body(CANDIDATE, RECORD, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    validation_section = body.split("## Validation", 1)[1].split("\n## ", 1)[0]
    assert "### Efficiency and trajectory telemetry" in validation_section
    security_section = body.split("## Security", 1)[1].split("\n## ", 1)[0]
    assert "### Efficiency and trajectory telemetry" not in security_section


TELEMETRY_RECORD = replace(
    RECORD,
    baseline_metrics={"tokens": 1000.0, "cost": 0.42},
    candidate_metrics={"tokens": 670.0},
    metric_delta={"tokens": -330.0},
    metric_not_compared=["cost"],
    metric_task_counts={"tokens": {"baseline": 3, "candidate": 3}},
    metric_attempt_counts={"tokens": {"baseline": 9, "candidate": 3}},
    metric_denominator_drift=["tokens"],
)


def test_pr_body_renders_telemetry_with_its_denominators():
    """The whole telemetry block was unreachable in tests: RECORD leaves
    `metric_delta` empty, so the table rendered its `_not recorded_` placeholder and
    six separate mutations to this rendering — including hardcoding "drift: none"
    and swapping the baseline/candidate columns — all left the suite green.

    A wrong number here is worse than a missing one, because a human votes on it.
    """
    body = pr_body(CANDIDATE, TELEMETRY_RECORD, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    # Column ORDER: baseline then candidate, so a swap is caught rather than read
    # as an improvement of the same magnitude in the opposite direction.
    assert "| `tokens` | 1000.0000 | 670.0000 | -330.0000 | 3 / **9→3** |" in body
    # Identical task counts, a third of the attempts: the drift must be visible.
    assert "Contributing-count drift: `tokens`." in body
    assert "Not compared (measured on one side only, never imputed as zero): `cost`." in body
    # The HEADER row, not the prose caption — the caption also contains
    # "tasks / attempts", so a substring check passed even with the column deleted,
    # leaving rows emitting five cells under four headers.
    assert "| metric | baseline mean | candidate mean | Δ | tasks / attempts |" in body
    row = next(line for line in body.splitlines() if line.startswith("| `tokens`"))
    assert row.count("|") == 6, f"row has the wrong cell count: {row}"


def test_pr_body_flags_a_one_sided_denominator():
    """A count present on only ONE side — the candidate broke and reported none —
    must render as drift, not as an uninformative `?`.

    The `and` in `_denominator`'s None check is load-bearing: with `or`, this case
    printed `?` and hid exactly the drift reported two lines below it.
    """
    one_sided = replace(
        TELEMETRY_RECORD,
        metric_task_counts={"tokens": {"baseline": 3, "candidate": None}},
        metric_attempt_counts={"tokens": {"baseline": 9, "candidate": None}},
    )
    body = pr_body(CANDIDATE, one_sided, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "**3→None**" in body and "**9→None**" in body


def test_pr_body_never_claims_parity_for_an_unknown_denominator():
    """With no counts at all the row renders `? / ?`; the prose must not then assert
    that every metric covers the same task count on both sides."""
    unknown = replace(TELEMETRY_RECORD, metric_task_counts={}, metric_attempt_counts={})
    body = pr_body(
        CANDIDATE,
        replace(unknown, metric_denominator_drift=[]),
        CLUSTER,
        BASELINE_RESULTS,
        CANDIDATE_RESULTS,
    )
    assert "? / ?" in body
    assert "Contributing-count drift: denominator unknown for `tokens`." in body


def test_pr_body_marks_a_clean_comparison_as_clean():
    """The negative case: with matching denominators nothing may be bolded, or the
    warning becomes noise a reviewer learns to skip."""
    clean = replace(
        TELEMETRY_RECORD,
        candidate_metrics={"tokens": 670.0, "cost": 0.30},
        metric_delta={"tokens": -330.0, "cost": -0.12},
        metric_not_compared=[],
        metric_task_counts={
            "tokens": {"baseline": 3, "candidate": 3},
            "cost": {"baseline": 3, "candidate": 3},
        },
        metric_attempt_counts={
            "tokens": {"baseline": 9, "candidate": 9},
            "cost": {"baseline": 9, "candidate": 9},
        },
        metric_denominator_drift=[],
    )
    body = pr_body(CANDIDATE, clean, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "| `tokens` | 1000.0000 | 670.0000 | -330.0000 | 3 / 9 |" in body
    assert "| `cost` | 0.4200 | 0.3000 | -0.1200 | 3 / 9 |" in body
    # Only the metric ROWS — the caption legitimately contains "**bolded**".
    rows = [line for line in body.splitlines() if line.startswith("| `")]
    assert rows and not any("**" in line for line in rows), f"spurious drift markers: {rows}"
    assert "Contributing-count drift: none — every metric covers the same counts" in body


def test_validation_record_round_trips_every_telemetry_field():
    """The serialized record is what lands in the committed iteration artifact.
    Dropping a field there loses the evidence silently — no test read these before."""
    payload = TELEMETRY_RECORD.to_json()
    for key in (
        "metric_delta",
        "metric_not_compared",
        "metric_task_counts",
        "metric_attempt_counts",
        "metric_denominator_drift",
    ):
        assert payload[key] == getattr(TELEMETRY_RECORD, key), f"{key} lost in as_dict()"


def test_ensure_base_branch_creates_and_pushes(repos):
    root, _ = repos
    ensure_base_branch(root, log=lambda *_: None)
    assert _git(root, "branch", "--list", "self-improvement").strip()
    assert _git(root, "ls-remote", "--heads", "origin", "self-improvement").strip()
    ensure_base_branch(root, log=lambda *_: None)  # idempotent


def test_open_pr_full_flow(repos):
    root, _ = repos
    calls = {}

    def fake_gh(cmd, cwd=None, capture_output=None, text=None):
        calls["cmd"] = cmd

        class P:
            returncode = 0
            stdout = "https://github.com/x/carbon/pull/1\n"
            stderr = ""

        return P()

    url = open_pr(
        CANDIDATE,
        RECORD,
        CLUSTER,
        BASELINE_RESULTS,
        CANDIDATE_RESULTS,
        iteration="iter-01",
        carbon_root=root,
        gh_run=fake_gh,
        log=lambda *_: None,
    )
    assert url == "https://github.com/x/carbon/pull/1"
    assert calls["cmd"][:4] == ["gh", "pr", "create", "--base"]
    assert calls["cmd"][4] == "self-improvement"  # explicit base, never the default branch
    branch = "evolve/iter-01-cand-raise-output-budget"
    # the edit landed as ONE commit on the branch, pushed to origin
    assert _git(root, "ls-remote", "--heads", "origin", branch).strip()
    on_branch = json.loads(_git(root, "show", f"{branch}:harness/harness_config.json"))
    assert on_branch["max_tokens"] == NEW_MT and on_branch["version"] == _bumped()
    # checkout restored, tree clean, main untouched
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert not _git(root, "status", "--porcelain").strip()
    on_main = json.loads(_git(root, "show", "main:harness/harness_config.json"))
    assert on_main["max_tokens"] == OLD_MT


def test_open_pr_refuses_rejected(repos):
    root, _ = repos
    rejected = ValidationRecord(**{**RECORD.__dict__, "accepted": False})
    with pytest.raises(ValueError, match="not accepted"):
        open_pr(
            CANDIDATE,
            rejected,
            CLUSTER,
            BASELINE_RESULTS,
            CANDIDATE_RESULTS,
            iteration="iter-01",
            carbon_root=root,
            gh_run=None,
            log=lambda *_: None,
        )


def test_pr_body_prose_matches_the_collapse_rule_as_it_actually_is():
    """A PR body is what a human approves a merge from, so its prose is part of the
    rule's interface.

    Both render paths asserted that a full-pass -> zero movement is a promotion veto
    FULL STOP. Since 2026-08-22 that is only true at the confirmation: in the first
    decision the veto applies where full-pass status is established and the movement is
    merely recorded elsewhere. A body could therefore print "collapsed to zero but NOT
    vetoing" from the decision's own reasons directly beside prose swearing the
    opposite. Pinned at the source because both occurrences are format strings rendered
    on different branches (calibrated and uncalibrated), and a test that exercised only
    one would leave the other free to drift.
    """
    source = (REPO_ROOT / "loop" / "prpipe.py").read_text()
    assert "promotion veto even when" not in source, (
        "the unconditional-veto claim is back in a PR body; the first decision vetoes "
        "only on established full-pass status"
    )
    assert source.count("promotion veto at the confirmation even when") == 2, (
        "both render paths (calibrated and uncalibrated) must carry the corrected claim"
    )
    # The qualifier itself, not just the absence of the old sentence.
    assert source.count("only where full-pass status is established") == 2
