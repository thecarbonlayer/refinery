"""The resume-guard: behavior_key is gemma_sha-independent but pins every other
behavior input. Pure/offline — guard.py imports no carbon."""

import pytest

from runner import guard

BASE = {
    "gemma_sha": "sha_A",
    "gemma_dirty": False,
    "dirty_sha": None,
    "config_version": 1,
    "model": "carbon",
    "runner_sha": "runner1",
    "provider_order": None,
    "quantization": None,
    "base_url": "http://localhost:1234/v1",
    "reasoning_effort": None,
}


def _fp(**over):
    fp = {**BASE, **over}
    fp["behavior_key"] = guard.fingerprint_behavior_key(fp)
    return fp


def test_behavior_key_ignores_committed_gemma_sha():
    """The whole point: two checkouts differing ONLY in committed gemma_sha share a
    behavior key (an additive release is not a behavior change)."""
    assert _fp(gemma_sha="sha_A")["behavior_key"] == _fp(gemma_sha="sha_B")["behavior_key"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("config_version", 2),
        ("model", "other-model"),
        ("runner_sha", "runner2"),
        ("dirty_sha", "deadbeef"),
        ("provider_order", "deepinfra"),
        ("quantization", "fp8"),
        ("base_url", "http://localhost:11434/v1"),
        ("reasoning_effort", "high"),
    ],
)
def test_behavior_key_moves_on_any_real_behavior_input(field, value):
    assert _fp()["behavior_key"] != _fp(**{field: value})["behavior_key"]


@pytest.mark.parametrize("field", guard._KEY_FIELDS)
def test_every_declared_key_field_actually_moves_the_key(field):
    """The structural half: `_KEY_FIELDS` is the load-bearing enumeration the key is
    derived from, so a field DECLARED behavior-relevant that failed to move the key
    would make the declaration a lie. Sweeps the declaration itself, not a copy."""
    assert _fp()["behavior_key"] != _fp(**{field: "moved-value"})["behavior_key"]


def test_two_local_endpoints_do_not_share_a_behavior_key():
    """The Codex finding, pinned: LM Studio and Ollama on localhost with the same
    model string are two serving bases. Both unpinned, both local — only base_url
    tells them apart, and a baseline from one must not resume on the other."""
    lmstudio = _fp(base_url="http://localhost:1234/v1")
    ollama = _fp(base_url="http://localhost:11434/v1")
    assert lmstudio["behavior_key"] != ollama["behavior_key"]
    assert guard.baseline_status(lmstudio, ollama) == "stale"


def test_behavior_key_moves_between_two_different_pins():
    """Not just pinned-vs-unpinned: the same model served by two different
    providers (or at two quantizations) is two serving bases, and a baseline
    recorded on one must not resume on the other."""
    a = _fp(provider_order="deepinfra", quantization="fp8")
    b = _fp(provider_order="together", quantization="fp8")
    c = _fp(provider_order="deepinfra", quantization="bf16")
    assert len({a["behavior_key"], b["behavior_key"], c["behavior_key"]}) == 3


def test_fingerprint_behavior_key_missing_field_is_a_bug_not_a_default():
    with pytest.raises(KeyError):
        guard.fingerprint_behavior_key({"model": "carbon"})  # no config_version/runner_sha


def test_baseline_status_current_when_keys_match():
    prior, current = _fp(gemma_sha="old"), _fp(gemma_sha="new")
    assert guard.baseline_status(prior, current) == "current"


def test_baseline_status_stale_on_real_change():
    assert guard.baseline_status(_fp(), _fp(config_version=2)) == "stale"


def test_baseline_status_stale_when_prior_has_no_behavior_key():
    """A recorded fingerprint predating the guard can't attest its behavior — stale."""
    prior = dict(BASE)  # no behavior_key
    assert guard.baseline_status(prior, _fp()) == "stale"


def test_assert_resumable_noop_on_additive_bump():
    guard.assert_resumable(_fp(gemma_sha="old"), _fp(gemma_sha="new"))  # no raise


def test_assert_resumable_raises_stale_baseline_with_both_keys():
    with pytest.raises(guard.StaleBaseline, match="behavior_key"):
        guard.assert_resumable(_fp(), _fp(model="other"))


# --- the serving pin: remote recording is refused without one -----------------


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:1234/v1",
        "http://127.0.0.1:1234/v1",
        "http://[::1]:1234/v1",
    ],
)
def test_local_serving_base_needs_no_pin(base_url):
    guard.assert_serving_pinned(base_url, None, None)  # no raise


def test_remote_unpinned_refuses_naming_both_env_vars():
    """The refusal is the feature: unpinned remote routing spreads one label's
    requests across providers with mixed quantization — a measured confound,
    never a warning. The message must say how to fix it."""
    with pytest.raises(guard.UnpinnedServing) as exc:
        guard.assert_serving_pinned("https://openrouter.ai/api/v1", None, None)
    msg = str(exc.value)
    assert "LLM_PROVIDER_ORDER" in msg
    assert "LLM_QUANTIZATION" in msg
    assert "openrouter.ai" in msg


@pytest.mark.parametrize(
    "provider_order,quantization,missing",
    [
        ("deepinfra", None, "LLM_QUANTIZATION"),
        (None, "fp8", "LLM_PROVIDER_ORDER"),
    ],
)
def test_remote_half_pinned_still_refuses(provider_order, quantization, missing):
    """A provider pin without a quantization pin (or the reverse) still leaves a
    serving degree of freedom inside one label. Both or nothing."""
    with pytest.raises(guard.UnpinnedServing, match=missing):
        guard.assert_serving_pinned("https://openrouter.ai/api/v1", provider_order, quantization)


def test_remote_fully_pinned_passes():
    guard.assert_serving_pinned("https://openrouter.ai/api/v1", "deepinfra", "fp8")  # no raise


def test_unparseable_base_url_is_treated_as_remote():
    """Fail closed: a base URL whose host cannot be read is not provably local."""
    with pytest.raises(guard.UnpinnedServing):
        guard.assert_serving_pinned("not a url", None, None)


# --- base_url normalization: cosmetic variants must not split behavior keys --


@pytest.mark.parametrize(
    "raw,expected",
    [
        # trailing slash: the HTTP client rstrips it before use, so it never
        # reaches the wire — recording it would split keys over nothing.
        ("http://localhost:1234/v1/", "http://localhost:1234/v1"),
        # scheme and host are case-insensitive per RFC; the path is not.
        ("HTTP://LocalHost:1234/v1", "http://localhost:1234/v1"),
        ("https://OpenRouter.ai/api/V1", "https://openrouter.ai/api/V1"),
        # already canonical: unchanged.
        ("http://localhost:1234/v1", "http://localhost:1234/v1"),
        # IPv6 host keeps its brackets.
        ("http://[::1]:1234/v1", "http://[::1]:1234/v1"),
    ],
)
def test_normalize_base_url_collapses_cosmetic_variants(raw, expected):
    assert guard.normalize_base_url(raw) == expected


def test_normalize_base_url_never_records_userinfo_or_query():
    """A credential embedded in the URL must not reach a fingerprint that gets
    written into every record — same scrub rule as api_key."""
    out = guard.normalize_base_url("http://user:sk-secret@host.example:8080/v1?key=tok#frag")
    assert out == "http://host.example:8080/v1"
    assert "sk-secret" not in out and "tok" not in out


def test_normalize_base_url_passes_the_unparseable_through():
    """What cannot be parsed cannot be normalized — it passes through verbatim and
    the pin gate reads it as remote (fail closed), so it can never record unpinned."""
    assert guard.normalize_base_url("not a url") == "not a url"
