"""Branch + PR pipeline: base-branch creation, one branch per edit, evidence body."""

import json
import shutil
import subprocess

import pytest

from loop.artifacts import Candidate, Cluster, ValidationRecord
from loop.prpipe import commit_message, ensure_base_branch, open_pr, pr_body, provenance
from runner.carbon_env import CARBON_ROOT, _git

REAL_CONFIG = CARBON_ROOT / "harness" / "harness_config.json"


def _bumped() -> int:
    """The version an edit should produce: whatever the real config carries now,
    plus one. Hardcoding it pinned these tests to a config that has since been
    bumped in carbon, so they broke on a change that was not theirs."""
    return json.loads(REAL_CONFIG.read_text())["version"] + 1


CANDIDATE = Candidate(
    id="cand-raise-clamp-12k",
    cluster_id="CL-1",
    proposer="Fable",
    proposer_detail="claude-fable-5, in-session",
    fields={"max_item_chars": {"old": 4000, "new": 12000}},
    rationale="The clamp drops the tail; observed needles sit past it.",
    expected_effect="A2/A4 recover",
    regression_risk="window pressure on A1/A3",
)

CLUSTER = Cluster(
    id="CL-1",
    mechanism="clamp suffix-drop at the door",
    tasks=("A2",),
    hypothesis="max_item_chars=4000 truncates below the needle offset",
    evidence=("reply: 'the log ends abruptly'",),
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
    assert "evolve(iter-01): max_item_chars 4000 -> 12000 [CL-1]" in msg
    assert "Δ_in=+0.1250" in msg and "Δ_ho=+0.2000" in msg
    assert "Fable-proposed, task-suite-validated" in msg
    assert msg.endswith("Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>")


def test_pr_body_is_the_required_template():
    body = pr_body(CANDIDATE, RECORD, CLUSTER, BASELINE_RESULTS, CANDIDATE_RESULTS)
    assert "CL-1" in body and "clamp suffix-drop" in body
    assert "| `max_item_chars` | `4000` | `12000` |" in body
    assert "Δ_in = +0.1250, Δ_ho = +0.2000" in body and "ACCEPTED" in body
    assert "| A2 | held_in | 0.0000 | 1.0000 | +1.0000 |" in body  # per-task, not aggregate
    assert "**Fable-proposed, task-suite-validated**" in body
    assert "computed locally against LM Studio" in body  # disclosed limitation


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
    branch = "evolve/iter-01-cand-raise-clamp-12k"
    # the edit landed as ONE commit on the branch, pushed to origin
    assert _git(root, "ls-remote", "--heads", "origin", branch).strip()
    on_branch = json.loads(_git(root, "show", f"{branch}:harness/harness_config.json"))
    assert on_branch["max_item_chars"] == 12000 and on_branch["version"] == _bumped()
    # checkout restored, tree clean, main untouched
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip() == "main"
    assert not _git(root, "status", "--porcelain").strip()
    on_main = json.loads(_git(root, "show", "main:harness/harness_config.json"))
    assert on_main["max_item_chars"] == 4000


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
