"""gemma_fingerprint content identity — offline, via the _git seam."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import runner.gemma_env as ge

ROOT = Path("/fake/gemma")


def _patch(monkeypatch, outputs: dict[tuple[str, ...], str]):
    monkeypatch.setattr(ge, "_git", lambda root, *args: outputs[args])
    monkeypatch.setattr(ge, "make_provider", lambda: SimpleNamespace(model="test-model"))


def test_fingerprint_clean_tree_has_no_dirty_sha(monkeypatch):
    _patch(
        monkeypatch,
        {
            ("rev-parse", "HEAD"): "abc123\n",
            ("status", "--porcelain"): "",
            ("diff", "HEAD"): "",
        },
    )
    fp = ge.gemma_fingerprint(ROOT)
    assert fp["gemma_sha"] == "abc123"
    assert fp["gemma_dirty"] is False
    assert fp["dirty_sha"] is None
    assert fp["model"] == "test-model"


def test_fingerprint_dirty_tree_hashes_status_plus_diff(monkeypatch):
    status = " M harness/agent.py\n?? scratch.py\n"
    diff = "diff --git a/harness/agent.py b/harness/agent.py\n+edit one\n"
    _patch(
        monkeypatch,
        {
            ("rev-parse", "HEAD"): "abc123\n",
            ("status", "--porcelain"): status,
            ("diff", "HEAD"): diff,
        },
    )
    fp = ge.gemma_fingerprint(ROOT)
    assert fp["gemma_dirty"] is True
    assert fp["dirty_sha"] == hashlib.sha256((status + diff).encode()).hexdigest()


def test_fingerprint_distinguishes_two_dirty_states_at_same_sha(monkeypatch):
    """The whole point: two different uncommitted edits at one SHA must not
    share a content identity."""
    common = {("rev-parse", "HEAD"): "abc123\n", ("status", "--porcelain"): " M f.py\n"}
    _patch(monkeypatch, {**common, ("diff", "HEAD"): "+edit one\n"})
    fp1 = ge.gemma_fingerprint(ROOT)
    _patch(monkeypatch, {**common, ("diff", "HEAD"): "+edit two\n"})
    fp2 = ge.gemma_fingerprint(ROOT)
    assert fp1["gemma_sha"] == fp2["gemma_sha"]
    assert fp1["dirty_sha"] != fp2["dirty_sha"]


def test_fingerprint_behavior_key_present_and_gemma_sha_independent(monkeypatch):
    """gemma_fingerprint stamps a behavior_key; two checkouts differing only in the
    committed sha share it (the additive-release resume path)."""
    from runner import guard

    def fp_for(sha):
        _patch(
            monkeypatch,
            {
                ("rev-parse", "HEAD"): f"{sha}\n",
                ("status", "--porcelain"): "",
                ("diff", "HEAD"): "",
            },
        )
        return ge.gemma_fingerprint(ROOT)

    fp_a, fp_b = fp_for("shaA"), fp_for("shaB")
    assert fp_a["behavior_key"] == guard.fingerprint_behavior_key(fp_a)
    assert fp_a["gemma_sha"] != fp_b["gemma_sha"]
    assert fp_a["behavior_key"] == fp_b["behavior_key"]


def test_fingerprint_includes_runner_sha(monkeypatch):
    _patch(
        monkeypatch,
        {
            ("rev-parse", "HEAD"): "abc123\n",
            ("status", "--porcelain"): "",
            ("diff", "HEAD"): "",
        },
    )
    fp = ge.gemma_fingerprint(ROOT)
    assert fp["runner_sha"] == ge.runner_sha()


def test_runner_sha_stable_across_calls():
    assert ge.runner_sha() == ge.runner_sha()


def test_runner_sha_changes_when_file_bytes_differ(tmp_path):
    """Content identity over a source tree: same layout, one byte different
    -> different sha. Tested against a temp tree, not the live package."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg" / "b.py").write_text("y = 2\n")
    sha1 = ge.runner_sha(tmp_path)
    assert sha1 == ge.runner_sha(tmp_path)  # stable
    (tmp_path / "pkg" / "a.py").write_text("x = 2\n")
    assert ge.runner_sha(tmp_path) != sha1


def test_runner_sha_changes_when_file_renamed(tmp_path):
    """The relative path is part of the hash — a rename with identical bytes
    is still a different runner version."""
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    sha1 = ge.runner_sha(tmp_path)
    f.rename(tmp_path / "b.py")
    assert ge.runner_sha(tmp_path) != sha1


def test_git_raises_on_failure(monkeypatch):
    """A failing git command must never be stamped as a clean/empty answer."""
    import subprocess
    from types import SimpleNamespace as NS

    def fake_run(cmd, capture_output, text):
        return NS(returncode=128, stdout="", stderr="fatal: not a git repository")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="not a git repository"):
        ge._git(ROOT, "status", "--porcelain")


def test_fingerprint_untracked_only_still_dirty(monkeypatch):
    """Untracked files show in status --porcelain but not in diff HEAD — the
    status text alone must perturb the hash."""
    common = {("rev-parse", "HEAD"): "abc123\n", ("diff", "HEAD"): ""}
    _patch(monkeypatch, {**common, ("status", "--porcelain"): "?? new_a.py\n"})
    fp1 = ge.gemma_fingerprint(ROOT)
    _patch(monkeypatch, {**common, ("status", "--porcelain"): "?? new_b.py\n"})
    fp2 = ge.gemma_fingerprint(ROOT)
    assert fp1["gemma_dirty"] is True
    assert fp1["dirty_sha"] is not None
    assert fp1["dirty_sha"] != fp2["dirty_sha"]
