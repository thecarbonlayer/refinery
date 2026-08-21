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
    "responder": None,
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
        ("responder", "runner.tasks.somewhere._scripted"),
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


def test_normalize_base_url_refuses_a_query_string():
    """The HTTP client appends /chat/completions to the raw base_url by string
    concatenation, so a query string swallows the endpoint path into the query
    value — the request is broken AND two query variants would be two different
    wire requests. Nothing legitimate to record: refuse with the remediation."""
    with pytest.raises(guard.MalformedBaseUrl, match="LLM_BASE_URL"):
        guard.normalize_base_url("https://host.example/v1?tenant=a")


def test_normalize_base_url_refuses_a_fragment():
    """Under the same concatenation, everything after # is client-side only — the
    request path silently loses /chat/completions. A config that cannot work must
    refuse loudly, not record."""
    with pytest.raises(guard.MalformedBaseUrl, match="fragment"):
        guard.normalize_base_url("https://host.example/v1#frag")


def test_normalize_base_url_refuses_userinfo_without_echoing_the_secret():
    """Embedded credentials DO reach the wire (the client turns userinfo into
    basic auth), so silently dropping them would record an identity different
    from the request actually sent — and recording them would commit a secret.
    Refuse, and the refusal itself must not echo the credential either."""
    with pytest.raises(guard.MalformedBaseUrl) as exc:
        guard.normalize_base_url("http://user:sk-secret@host.example:8080/v1")
    msg = str(exc.value)
    assert "LLM_API_KEY" in msg
    assert "sk-secret" not in msg


def test_normalize_base_url_passes_the_unparseable_through():
    """What cannot be parsed cannot be normalized — it passes through verbatim and
    the pin gate reads it as remote (fail closed), so it can never record unpinned."""
    assert guard.normalize_base_url("not a url") == "not a url"


def test_malformed_port_reads_as_remote_everywhere():
    """One shared classification for the normalizer and the classifier: a URL
    whose PORT does not parse is not provably local, whatever its hostname says.
    Before this, `http://localhost:notaport/v1` classified local (hostname parses
    fine) and recorded unpinned — contradicting the fail-closed claim."""
    url = "http://localhost:notaport/v1"
    assert guard.is_local_base_url(url) is False
    assert guard.normalize_base_url(url) == url  # verbatim; cannot be normalized
    with pytest.raises(guard.UnpinnedServing):
        guard.assert_serving_pinned(url, None, None)


@pytest.mark.parametrize(
    "url,part",
    [
        ("http://user:sk-secret@host.example:notaport/v1", "credentials"),
        ("http://host.example:notaport/v1?tenant=a", "query"),
        ("http://host.example:notaport/v1#frag", "fragment"),
        # urlsplit RAISES outright on the bad bracket — the checks must not sit
        # behind any parser at all, or this authority carries its credential
        # through the verbatim fallback exactly like the malformed port did.
        ("http://user:sk-secret@[bad/v1", "credentials"),
        # even a bare delimiter breaks the /chat/completions concatenation, and
        # the old parsed check (`parts.query` is falsy-empty) waved it through.
        ("http://host.example/v1?", "query"),
        ("http://host.example/v1#", "fragment"),
    ],
)
def test_forbidden_parts_refuse_even_with_a_malformed_port(url, part):
    """The ordering bug: the full parse bailed on the malformed PORT and passed
    the URL through verbatim, so the userinfo/query/fragment refusals never ran —
    a credential could ride a bad port past the boundary. The forbidden-part
    checks run on the raw STRING, before anything that can fail to parse."""
    with pytest.raises(guard.MalformedBaseUrl, match=part):
        guard.normalize_base_url(url)


def test_refusals_never_echo_the_raw_url():
    """The refusal must not disclose what it refuses: the raw URL may embed a
    credential, and exception text ends up in logs, records and tracebacks. Host
    only — in str, repr, AND full traceback renderings."""
    import traceback

    poison = "http://user:sk-secret@host.example:notaport/v1?tenant=a#frag"
    with pytest.raises(guard.MalformedBaseUrl) as exc:
        guard.normalize_base_url(poison)
    rendered = str(exc.value) + repr(exc.value) + "".join(traceback.format_exception(exc.value))
    assert "sk-secret" not in rendered
    assert poison not in rendered
    assert "host.example" in str(exc.value)  # the host alone identifies the endpoint


def test_refusal_on_an_unsplittable_authority_still_never_echoes():
    """urlsplit raises outright on a bad IPv6 bracket, so there is no host to
    name — the refusal says so generically and still discloses nothing."""
    import traceback

    poison = "http://user:sk-secret@[bad/v1"
    with pytest.raises(guard.MalformedBaseUrl) as exc:
        guard.normalize_base_url(poison)
    rendered = str(exc.value) + repr(exc.value) + "".join(traceback.format_exception(exc.value))
    assert "sk-secret" not in rendered
    assert poison not in rendered
    assert "unparseable" in str(exc.value)


def test_unpinned_refusal_names_the_host_never_the_raw_url():
    """Same no-echo rule for the pin gate: a malformed-port URL reaches it
    VERBATIM (the normalizer cannot canonicalize it), so echoing the raw string
    would disclose whatever the URL carries. Name the host, nothing more."""
    import traceback

    url = "http://internal-host.example:notaport/v1"
    with pytest.raises(guard.UnpinnedServing) as exc:
        guard.assert_serving_pinned(url, None, None)
    rendered = str(exc.value) + repr(exc.value) + "".join(traceback.format_exception(exc.value))
    assert url not in rendered
    assert "internal-host.example" in str(exc.value)
