"""gemma_fingerprint content identity — offline, via the _git seam."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

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
