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
    the pin gate reads it as remote (fail closed), so it can never record unpinned.
    The probe is whitespace-free: whitespace now refuses outright (see below), so
    the verbatim path is only for clean-but-unparseable strings."""
    assert guard.normalize_base_url("not-a-url") == "not-a-url"


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


@pytest.mark.parametrize(
    "url",
    [
        "https:/\t/user:sk-secret@localhost:1234/v1",  # the probe: a tab splits the //
        "http://localhost:1234/v1\n",  # trailing newline, urlsplit-stripped
        "http://local host:1234/v1",  # internal space
    ],
)
def test_control_or_whitespace_refuses_before_any_scan_or_parse(url):
    """urlsplit deletes tab/newline BEFORE parsing (WHATWG alignment), so the raw
    authority scan can be blinded: https:/<TAB>/user:secret@localhost shows no //
    to the raw string while urlsplit parses a credentialed netloc — the normalizer
    then records a SANITIZED identity (and classifies LOCAL, skipping the pin
    gate) for a request the client cannot even issue, since httpx refuses control
    characters on the wire. The rule: any whitespace or control character
    anywhere refuses; nothing is stripped, leading or trailing included."""
    with pytest.raises(guard.MalformedBaseUrl, match="whitespace or control"):
        guard.normalize_base_url(url)


def test_control_whitespace_probe_never_echoes_and_reads_remote():
    import traceback

    poison = "https:/\t/user:sk-secret@localhost:1234/v1"
    assert guard.is_local_base_url(poison) is False  # not provably local: fail closed
    with pytest.raises(guard.MalformedBaseUrl) as exc:
        guard.normalize_base_url(poison)
    rendered = str(exc.value) + repr(exc.value) + "".join(traceback.format_exception(exc.value))
    assert "sk-secret" not in rendered
    assert poison not in rendered


@pytest.mark.parametrize("ch", ["\x80", "\x9f"])  # the C1 bounds: U+0080 and U+009F
def test_c1_controls_refuse_like_c0(ch):
    """U+0080-U+009F are category Cc while being none of isspace(), < 0x20, or
    DEL — the previous predicate passed them into the normalizer (verbatim), the
    classifier (which read the probe as LOCAL) and the record gate. The class
    rule is Cc OR isspace(): C0, DEL and C1 refuse wholesale, at all three
    boundaries."""
    with pytest.raises(guard.MalformedBaseUrl, match="whitespace or control"):
        guard.normalize_base_url(f"https://host.example/v1{ch}")
    assert guard.is_local_base_url(f"http://localhost:1234/v1{ch}") is False
    with pytest.raises(guard.MalformedBaseUrl):
        guard.assert_recorded_base_url_clean(f"http://localhost:1234/v1{ch}", "x")
    # the accepted-URL control: plain ASCII passes the same three gates
    assert guard.normalize_base_url("https://host.example/v1") == "https://host.example/v1"
    assert guard.is_local_base_url("http://localhost:1234/v1") is True
    guard.assert_recorded_base_url_clean("https://host.example/v1", "x")


class _DecodingBase:
    """Not a str, but urllib's ``_coerce_args`` accepts ANY object carrying
    ``.decode()``: it decodes, parses the decoded text, then re-encodes, so
    ``urlsplit`` hands back a BYTES-flavoured SplitResult (hostname
    ``b'localhost'``). Iterable of clean characters, so the whitespace scan
    cannot stop it; truthy, so the normalizer's falsey early return cannot
    either. This is the shape that reached ``urlsplit`` itself."""

    def decode(self, encoding="ascii", errors="strict"):
        return "http://localhost:1234/v1"

    def __iter__(self):
        return iter("http://localhost:1234/v1")

    def __bool__(self):
        return True


class _StrBase(str):
    """A str SUBCLASS: it IS a string, so the type rule must let it through —
    isinstance, never ``type(x) is str``."""


@pytest.mark.parametrize(
    "bad",
    [
        False,  # falsey non-strings were returned UNCHANGED by the normalizer...
        0,
        b"",
        b"http://localhost:1234/v1",  # ...and truthy bytes crashed the scan (TypeError)
        _DecodingBase(),  # ...while this one reached urlsplit and decoded to bytes
    ],
)
def test_non_string_base_url_refuses_at_normalizer_and_classifier(bad):
    """The type rule is one predicate at the shared parse path, not three
    scattered checks. Pre-fix, ``normalize_base_url`` returned every FALSEY
    non-string unchanged before any validation ran — with a directly built
    pinned provider, False/0/b"" were retained into the fingerprint and folded
    into the behavior key — while truthy non-strings crashed
    (TypeError/AttributeError) instead of refusing, and a decoding-protocol
    object reached urlsplit and parsed to a bytes SplitResult, which the
    normalizer would have formatted into a b'http'-shaped identity. Now: a
    non-string refuses at the normalizer with the fail-closed contract message,
    and reads unparseable (hence remote) at the classifier. Not reachable from
    Provider.from_env, which reads os.environ and always yields str — this
    closes the class."""
    with pytest.raises(guard.MalformedBaseUrl, match="not a string"):
        guard.normalize_base_url(bad)
    assert guard.is_local_base_url(bad) is False  # unparseable reads remote: fail closed
    with pytest.raises(guard.MalformedBaseUrl):  # the record gate, already guarded
        guard.assert_recorded_base_url_clean(bad, "the baseline results file")


def test_none_passes_and_a_str_subclass_is_treated_as_a_string():
    """The two deliberate non-refusals, pinned so the type rule cannot swallow
    them: None is a real recorded state (an unset base — carbon_fingerprint
    normalizes it every run), and a str subclass is a string, so it normalizes
    and classifies exactly like its plain-str twin."""
    assert guard.normalize_base_url(None) is None
    assert guard.normalize_base_url("") == ""  # empty str: still a string, still passes
    sub = _StrBase("http://LocalHost:1234/v1/")
    assert guard.normalize_base_url(sub) == guard.normalize_base_url(str(sub))
    assert guard.normalize_base_url(sub) == "http://localhost:1234/v1"
    assert guard.is_local_base_url(sub) is True
    guard.assert_recorded_base_url_clean(sub, "x")


@pytest.mark.parametrize("bad", [123, True, 4.2, {"base_url": "x"}])
def test_recorded_base_url_gate_refuses_non_strings_gracefully(bad):
    """urlsplit(123) raises AttributeError — not the caught ValueError/TypeError —
    so DESCRIBING a scalar non-string crashed past the stated contract: a
    host-safe MalformedBaseUrl naming the results file. (Containers happened to
    raise a caught TypeError; the rule must not depend on which non-string it
    is.) Non-strings are named generically before any parse touches them; None
    stays a real recorded state (unset) and passes."""
    with pytest.raises(guard.MalformedBaseUrl, match="results file"):
        guard.assert_recorded_base_url_clean(bad, "the baseline results file")
    guard.assert_recorded_base_url_clean(None, "the baseline results file")  # None passes


def test_recorded_base_url_gate_refuses_poison_and_passes_clean():
    """The record-side twin of the live boundary: result files written by a
    PRE-fix runner can carry a verbatim poisoned URL, and delta/CLI format,
    return, and serialize recorded fingerprints — the same raw-string rules must
    gate the records on the way out, host-only, naming the offending record."""
    import traceback

    guard.assert_recorded_base_url_clean(None, "x")  # unset is a real recorded state
    guard.assert_recorded_base_url_clean("http://localhost:1234/v1", "x")  # clean passes
    poison = "http://user:sk-secret@host.example:notaport/v1?tenant=a#frag"
    with pytest.raises(guard.MalformedBaseUrl) as exc:
        guard.assert_recorded_base_url_clean(poison, "the baseline results file")
    rendered = str(exc.value) + repr(exc.value) + "".join(traceback.format_exception(exc.value))
    assert "sk-secret" not in rendered
    assert poison not in rendered
    assert "host.example" in str(exc.value)
    assert "baseline results file" in str(exc.value)
    with pytest.raises(guard.MalformedBaseUrl):  # the whitespace class reaches records too
        guard.assert_recorded_base_url_clean("https:/\t/u:sk-x@localhost:1234/v1", "x")
    with pytest.raises(guard.MalformedBaseUrl):  # non-string cannot be scanned: fail closed
        guard.assert_recorded_base_url_clean(["http://u:sk-x@h/v1"], "x")


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
