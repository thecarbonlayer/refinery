"""carbon_fingerprint content identity — offline, via the _git seam."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import runner.carbon_env as ge

ROOT = Path("/fake/carbon")


def _patch(
    monkeypatch, outputs: dict[tuple[str, ...], str], provider: SimpleNamespace | None = None
):
    monkeypatch.setattr(ge, "_git", lambda root, *args: outputs[args])
    provider = provider or SimpleNamespace(
        model="test-model",
        base_url="http://localhost:1234/v1",
        reasoning_effort=None,
        provider_order=None,
        quantization=None,
    )
    monkeypatch.setattr(ge, "make_provider", lambda: provider)


_CLEAN = {
    ("rev-parse", "HEAD"): "abc123\n",
    ("status", "--porcelain"): "",
    ("diff", "HEAD"): "",
}


def _scripted_responder(messages, **kwargs):
    """Module-level stand-in for a scripted provider responder (cluster-H style)."""
    raise AssertionError("never called — only its identity is fingerprinted")


def _runner_tree_responder(messages, **kwargs):
    """A top-level function posing as runner-tree code (unit seam: the marker
    policy reads __module__/__qualname__, so faking the module string tests the
    policy without planting a real function in runner/)."""
    raise AssertionError("never called — only its identity is fingerprinted")


_runner_tree_responder.__module__ = "runner.tasks.cluster_h"


def _provider_with(responder):
    return SimpleNamespace(
        model="test-model",
        base_url="http://localhost:1234/v1",
        reasoning_effort=None,
        provider_order=None,
        quantization=None,
        responder=responder,
    )


def test_fingerprint_clean_tree_has_no_dirty_sha(monkeypatch):
    _patch(
        monkeypatch,
        {
            ("rev-parse", "HEAD"): "abc123\n",
            ("status", "--porcelain"): "",
            ("diff", "HEAD"): "",
        },
    )
    fp = ge.carbon_fingerprint(ROOT)
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
    fp = ge.carbon_fingerprint(ROOT)
    assert fp["gemma_dirty"] is True
    assert fp["dirty_sha"] == hashlib.sha256((status + diff).encode()).hexdigest()


def test_fingerprint_distinguishes_two_dirty_states_at_same_sha(monkeypatch):
    """The whole point: two different uncommitted edits at one SHA must not
    share a content identity."""
    common = {("rev-parse", "HEAD"): "abc123\n", ("status", "--porcelain"): " M f.py\n"}
    _patch(monkeypatch, {**common, ("diff", "HEAD"): "+edit one\n"})
    fp1 = ge.carbon_fingerprint(ROOT)
    _patch(monkeypatch, {**common, ("diff", "HEAD"): "+edit two\n"})
    fp2 = ge.carbon_fingerprint(ROOT)
    assert fp1["gemma_sha"] == fp2["gemma_sha"]
    assert fp1["dirty_sha"] != fp2["dirty_sha"]


def test_fingerprint_behavior_key_present_and_gemma_sha_independent(monkeypatch):
    """carbon_fingerprint stamps a behavior_key; two checkouts differing only in the
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
        return ge.carbon_fingerprint(ROOT)

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
    fp = ge.carbon_fingerprint(ROOT)
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


def test_fingerprint_records_the_serving_pin_keys_even_unpinned(monkeypatch):
    """Local unpinned is a real, recordable serving state — the keys are present
    and None, the same convention as dirty_sha on a clean tree (None is data,
    absent is unknown)."""
    _patch(monkeypatch, _CLEAN)
    fp = ge.carbon_fingerprint(ROOT)
    assert fp["provider_order"] is None
    assert fp["quantization"] is None


def test_fingerprint_records_the_serving_pin_values(monkeypatch):
    _patch(
        monkeypatch,
        _CLEAN,
        provider=SimpleNamespace(
            model="test-model",
            base_url="https://openrouter.ai/api/v1",
            provider_order="deepinfra",
            quantization="fp8",
        ),
    )
    fp = ge.carbon_fingerprint(ROOT)
    assert fp["provider_order"] == "deepinfra"
    assert fp["quantization"] == "fp8"
    from runner import guard

    assert fp["behavior_key"] == guard.fingerprint_behavior_key(fp)


def test_fingerprint_refuses_a_remote_base_with_no_pin(monkeypatch):
    """Fail closed at the fingerprint itself: an unpinned remote serving state is
    unattributable, so nothing downstream (recording, resume, drift check) can
    even name it. The refusal carries the remediation."""
    from runner import guard

    _patch(
        monkeypatch,
        _CLEAN,
        provider=SimpleNamespace(
            model="test-model",
            base_url="https://openrouter.ai/api/v1",
            provider_order=None,
            quantization=None,
        ),
    )
    with pytest.raises(guard.UnpinnedServing, match="LLM_PROVIDER_ORDER"):
        ge.carbon_fingerprint(ROOT)


def test_fingerprint_reads_a_pre_pin_carbon_as_unpinned(monkeypatch):
    """A carbon Provider that predates the pin fields sends no pin whatever the
    env says, so the truthful fingerprint is unpinned — fine locally, refused
    remotely by the same gate as an unpinned modern carbon."""
    from runner import guard

    local = SimpleNamespace(model="test-model", base_url="http://localhost:1234/v1")
    _patch(monkeypatch, _CLEAN, provider=local)
    fp = ge.carbon_fingerprint(ROOT)
    assert fp["provider_order"] is None
    assert fp["quantization"] is None

    remote = SimpleNamespace(model="test-model", base_url="https://openrouter.ai/api/v1")
    _patch(monkeypatch, _CLEAN, provider=remote)
    with pytest.raises(guard.UnpinnedServing):
        ge.carbon_fingerprint(ROOT)


def test_fingerprint_records_the_base_url_normalized(monkeypatch):
    """LM Studio on :1234 and Ollama on :11434 are two serving bases with the same
    model string — base_url is what tells them apart, so it is fingerprinted.
    Recorded normalized, so a trailing slash or host case cannot split keys."""
    _patch(
        monkeypatch,
        _CLEAN,
        provider=SimpleNamespace(
            model="test-model",
            base_url="HTTP://LocalHost:1234/v1/",
            reasoning_effort=None,
            provider_order=None,
            quantization=None,
        ),
    )
    fp = ge.carbon_fingerprint(ROOT)
    assert fp["base_url"] == "http://localhost:1234/v1"


def test_fingerprint_records_reasoning_effort(monkeypatch):
    """reasoning_effort changes the wire request (carbon forwards it verbatim), so
    two runs at different efforts are different measured behavior."""
    _patch(
        monkeypatch,
        _CLEAN,
        provider=SimpleNamespace(
            model="test-model",
            base_url="http://localhost:1234/v1",
            reasoning_effort="high",
            provider_order=None,
            quantization=None,
        ),
    )
    fp = ge.carbon_fingerprint(ROOT)
    assert fp["reasoning_effort"] == "high"

    _patch(monkeypatch, _CLEAN)
    assert ge.carbon_fingerprint(ROOT)["reasoning_effort"] is None


def test_every_provider_field_is_fingerprinted_or_named_excluded(monkeypatch):
    """The future-field guard: every field on carbon's Provider dataclass must be
    either fingerprinted or excluded BY NAME with a stated reason. A new Provider
    field that alters the wire request cannot land silently — this test refuses it
    until someone puts it in one of the two lists, and the fingerprint builds its
    serving section from the fingerprinted list, so the list cannot drift from
    what is actually recorded."""
    from dataclasses import fields as dc_fields

    from carbon import Provider

    from runner import guard

    provider_fields = {f.name for f in dc_fields(Provider)}
    covered = set(ge.PROVIDER_FIELDS_FINGERPRINTED) | set(ge.PROVIDER_FIELDS_EXCLUDED)
    assert provider_fields == covered, (
        f"Provider fields not dispositioned: {sorted(provider_fields - covered)}; "
        f"dispositioned but no longer on Provider: {sorted(covered - provider_fields)}"
    )
    # the two lists must not overlap — a field cannot be both recorded and excluded
    assert not set(ge.PROVIDER_FIELDS_FINGERPRINTED) & set(ge.PROVIDER_FIELDS_EXCLUDED)
    # every fingerprinted serving field must also fold into the behavior key,
    # otherwise it is recorded but cannot gate resume/freshness.
    assert set(ge.PROVIDER_FIELDS_FINGERPRINTED) <= set(guard._KEY_FIELDS)
    # every exclusion carries a written reason
    for field, reason in ge.PROVIDER_FIELDS_EXCLUDED.items():
        assert reason.strip(), f"exclusion of {field!r} states no reason"


def test_fingerprint_records_no_responder_as_none(monkeypatch):
    """The env-built provider carries no responder; None is its truthful record."""
    _patch(monkeypatch, _CLEAN)
    assert ge.carbon_fingerprint(ROOT)["responder"] is None


def test_fingerprint_records_a_marker_for_a_top_level_runner_responder(monkeypatch):
    """A provider serving from a responder is not a network serving base at all —
    if one ever reaches the suite-level provider, the fingerprint must say so.
    The callable itself is uncapturable; its qualified name is recorded, and the
    ONLY callables whose qualified name is a stable identity are top-level
    functions under runner/ — the tree runner_sha hashes."""
    _patch(monkeypatch, _CLEAN, provider=_provider_with(_runner_tree_responder))
    fp = ge.carbon_fingerprint(ROOT)
    assert fp["responder"] == "runner.tasks.cluster_h._runner_tree_responder"
    from runner import guard

    assert fp["behavior_key"] == guard.fingerprint_behavior_key(fp)


def test_fingerprint_refuses_a_closure_responder(monkeypatch):
    """Cluster H's real responders are closures (`run_h1.<locals>.responder`), and
    closures from one factory share a qualname whatever state they close over —
    factory('A') and factory('B') would record the SAME marker for different
    behavior. A marker that lies is worse than a refusal."""

    def factory(tag):
        def responder(messages, **kwargs):
            return tag

        responder.__module__ = "runner.tasks.cluster_h"  # runner-tree, still refused
        return responder

    a, b = factory("A"), factory("B")
    assert a.__qualname__ == b.__qualname__  # the collision that forbids the marker
    _patch(monkeypatch, _CLEAN, provider=_provider_with(a))
    with pytest.raises(RuntimeError, match="top-level"):
        ge.carbon_fingerprint(ROOT)


def test_fingerprint_refuses_a_lambda_responder(monkeypatch):
    lam = lambda messages, **kwargs: None  # noqa: E731 — the shape under test
    lam.__module__ = "runner.tasks.cluster_h"
    _patch(monkeypatch, _CLEAN, provider=_provider_with(lam))
    with pytest.raises(RuntimeError, match="top-level"):
        ge.carbon_fingerprint(ROOT)


def test_fingerprint_refuses_a_bound_method_responder(monkeypatch):
    """Two instances of one class share the method's qualname while closing over
    different instance state — same collision as the closure case."""

    class Scripted:
        def responder(self, messages, **kwargs):
            return None

    _patch(monkeypatch, _CLEAN, provider=_provider_with(Scripted().responder))
    with pytest.raises(RuntimeError, match="top-level"):
        ge.carbon_fingerprint(ROOT)


def test_fingerprint_refuses_a_responder_from_outside_runner(monkeypatch):
    """`pinned by runner_sha` must be TRUE of the marker: runner_sha hashes only
    runner/**/*.py, so a callable from any other module has no content identity
    in the fingerprint and is refused, top-level or not."""
    _patch(monkeypatch, _CLEAN, provider=_provider_with(_scripted_responder))
    with pytest.raises(RuntimeError, match="runner"):
        ge.carbon_fingerprint(ROOT)


def test_fingerprint_refuses_a_responder_with_no_stable_name(monkeypatch):
    """A callable with no module/qualname (e.g. functools.partial) has no stable
    identity to record — an unstable marker would move the behavior key between
    runs of identical behavior. Refuse rather than guess."""
    import functools

    _patch(monkeypatch, _CLEAN, provider=_provider_with(functools.partial(print)))
    with pytest.raises(RuntimeError, match="responder"):
        ge.carbon_fingerprint(ROOT)


@pytest.mark.parametrize("pinned", [False, True])
def test_fingerprint_refuses_forbidden_url_parts_even_with_malformed_port(monkeypatch, pinned):
    """The boundary version of the ordering fix, both pinned and unpinned: a
    malformed port must not carry credentials/query/fragment past the refusals,
    and no rendering of the refusal may disclose the credential."""
    import traceback

    from runner import guard

    _patch(
        monkeypatch,
        _CLEAN,
        provider=SimpleNamespace(
            model="test-model",
            base_url="http://user:sk-secret@host.example:notaport/v1?tenant=a#frag",
            reasoning_effort=None,
            provider_order="Novita" if pinned else None,
            quantization="bf16" if pinned else None,
        ),
    )
    with pytest.raises(guard.MalformedBaseUrl) as exc:
        ge.carbon_fingerprint(ROOT)
    rendered = str(exc.value) + repr(exc.value) + "".join(traceback.format_exception(exc.value))
    assert "sk-secret" not in rendered


def test_fingerprint_refuses_a_base_url_with_a_query(monkeypatch):
    """The refusal reaches the fingerprint boundary: a query in LLM_BASE_URL breaks
    the concatenated request URL, so the state must refuse to record, not record
    a normalized identity the wire never matched."""
    from runner import guard

    _patch(
        monkeypatch,
        _CLEAN,
        provider=SimpleNamespace(
            model="test-model",
            base_url="https://host.example/v1?tenant=a",
            reasoning_effort=None,
            provider_order=None,
            quantization=None,
        ),
    )
    with pytest.raises(guard.MalformedBaseUrl):
        ge.carbon_fingerprint(ROOT)


def test_api_key_never_reaches_the_fingerprint(monkeypatch):
    """The one hard exclusion: the fingerprint is written into every record and
    results JSON, so a secret in it would be committed. Neither the key name nor
    the secret value may appear anywhere in the fingerprint."""
    import json

    _patch(
        monkeypatch,
        _CLEAN,
        provider=SimpleNamespace(
            model="test-model",
            base_url="http://localhost:1234/v1",
            api_key="sk-super-secret-value",
            reasoning_effort=None,
            provider_order=None,
            quantization=None,
        ),
    )
    fp = ge.carbon_fingerprint(ROOT)
    assert "api_key" not in fp
    assert "sk-super-secret-value" not in json.dumps(fp)


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
    fp1 = ge.carbon_fingerprint(ROOT)
    _patch(monkeypatch, {**common, ("status", "--porcelain"): "?? new_b.py\n"})
    fp2 = ge.carbon_fingerprint(ROOT)
    assert fp1["gemma_dirty"] is True
    assert fp1["dirty_sha"] is not None
    assert fp1["dirty_sha"] != fp2["dirty_sha"]
