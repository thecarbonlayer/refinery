"""The resume-guard: decides whether a recorded baseline is still valid under the
current harness state, so an *additive* carbon release does not force an empty
re-baseline.

It pins on a **behavior key** — carbon's declared ``config_version`` + the model +
our ``runner_sha`` (the verifier) + ``dirty_sha`` (uncommitted carbon edits) +
``provider_order``/``quantization`` (the serving pin), i.e. the inputs that actually
determine the measured behavior — NOT the raw committed ``gemma_sha``. An additive,
default-neutral carbon release keeps ``config_version``
fixed and touches no runner code, so the key is stable and the baseline resumes; a
real config / model / verifier / working-tree change moves the key and the guard
refuses (``StaleBaseline``). ``gemma_sha`` stays recorded in every fingerprint for
provenance/audit — it just no longer gates resume.

This is our eval-integrity policy (AGENTS.md), layered on carbon's ``provenance()``
primitive: carbon exposes the state, we decide what invalidates a baseline. It
mirrors a sibling consumer's eval guard (whose behavior input is a dataset
fingerprint; ours is the verifier hash + dirty-tree identity) so the two
consumers converge on one shape instead of diverging.

Kept deliberately free of any carbon import: it compares two fingerprint dicts and
nothing more. The *live* current fingerprint is produced by
``runner.carbon_env.carbon_fingerprint`` and passed in, which keeps this module a
pure, offline-testable policy.
"""

from __future__ import annotations

import hashlib
import unicodedata
from urllib.parse import urlsplit

# The fingerprint fields that fold into the behavior key, in order — the LOAD-BEARING
# enumeration: ``fingerprint_behavior_key`` derives the key from this tuple, and a
# test sweeps it to prove every declared field actually moves the key. ``gemma_sha`` is
# pointedly absent: a committed carbon move is provenance, not behavior (carbon's own
# config_version is the behavior-version declaration we trust). ``dirty_sha`` stays,
# because an uncommitted carbon edit changes behavior that no version counter attests.
# The serving fields (``base_url``, ``reasoning_effort``, ``provider_order``,
# ``quantization``) stay too: the model string alone cannot distinguish serving bases —
# the same model name answers from LM Studio on :1234, Ollama on :11434, or different
# routed providers at different quantizations and effort levels, and every one of those
# is a behavior change no other field sees. None is real data for the optional fields
# (clean tree / unpinned local serving / no effort requested), read via ``.get`` — but
# note a record that never carried the serving fields still cannot resume, because its
# RECORDED key was computed by the pre-serving formula.
_KEY_FIELDS = (
    "config_version",
    "model",
    "runner_sha",
    "dirty_sha",
    "provider_order",
    "quantization",
    "base_url",
    "reasoning_effort",
    "responder",
)

# The key fields a fingerprint MUST carry (``KeyError`` otherwise — a fingerprint that
# cannot state its behavior identity is a bug, not a silent default). The rest read
# through ``.get`` because None there is a real recorded state, not an absence.
_REQUIRED_KEY_FIELDS = frozenset({"config_version", "model", "runner_sha"})


class StaleBaseline(Exception):
    """A recorded baseline's behavior key differs from the current harness state — it
    measured different behavior and must be re-baselined (never silently reused)."""


class UnpinnedServing(Exception):
    """A remote serving base with no full provider pin must not be recorded: unpinned
    routing spreads one label's requests across providers with mixed quantization — a
    measured confound inside the experiment, never a warning."""


class MalformedBaseUrl(Exception):
    """A base URL whose recorded identity could not match the request it produces.

    The HTTP client builds request URLs by appending ``/chat/completions`` to the raw
    ``LLM_BASE_URL`` string, so a query string, a fragment, or embedded credentials
    each change or break the wire request in a way no normalized record could honestly
    attribute. There is nothing legitimate to record — refuse with the remediation."""


def behavior_key(
    config_version,
    model: str,
    runner_sha: str,
    dirty_sha: str | None,
    provider_order: str | None,
    quantization: str | None,
    base_url: str | None,
    reasoning_effort: str | None,
    responder: str | None,
) -> str:
    """Stable identity of the behavior-determining inputs. Additive carbon releases
    (``config_version`` unchanged, no runner edit, clean tree, same serving base) keep
    this fixed; a model / verifier / config / working-tree / serving-base change moves
    it. Deliberately excludes the committed ``gemma_sha`` — that is provenance, not
    behavior. ``responder`` is the qualified-name MARKER of a scripted responder on
    the fingerprinted provider (see runner/carbon_env.py), None for a real network
    provider. Thin explicit-signature wrapper over ``fingerprint_behavior_key``: one
    derivation, one hash."""
    return fingerprint_behavior_key(
        {
            "config_version": config_version,
            "model": model,
            "runner_sha": runner_sha,
            "dirty_sha": dirty_sha,
            "provider_order": provider_order,
            "quantization": quantization,
            "base_url": base_url,
            "reasoning_effort": reasoning_effort,
            "responder": responder,
        }
    )


def fingerprint_behavior_key(fingerprint: dict) -> str:
    """The behavior key of a fingerprint dict — the currency this runner passes
    around. Derived from ``_KEY_FIELDS`` itself, so the declaration and the
    derivation cannot drift apart. Missing required fields raise ``KeyError``;
    the optional fields read through ``.get`` because None there is a real
    recorded state (clean tree, unpinned local serving, no effort), not an
    absence."""
    values = [
        fingerprint[field] if field in _REQUIRED_KEY_FIELDS else fingerprint.get(field)
        for field in _KEY_FIELDS
    ]
    raw = "|".join(str(v) for v in values)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _key_of(fingerprint: dict) -> str:
    """A fingerprint's recorded ``behavior_key`` if present, else derived on the fly.
    Freshly stamped fingerprints carry it; a legacy record predating the field has no
    recorded key and no way to attest one — it derives to ``None`` handling upstream."""
    return fingerprint.get("behavior_key") or fingerprint_behavior_key(fingerprint)


def baseline_status(prior_fingerprint: dict, current_fingerprint: dict) -> str:
    """``"current"`` if the recorded baseline still matches current behavior (safe to
    resume), else ``"stale"`` (re-baseline required). A recorded fingerprint with no
    ``behavior_key`` (predates this guard) is stale by definition — it cannot attest
    which behavior it measured."""
    prior = prior_fingerprint.get("behavior_key")
    return "current" if prior and prior == _key_of(current_fingerprint) else "stale"


def assert_resumable(prior_fingerprint: dict, current_fingerprint: dict) -> None:
    """Refuse loudly (``StaleBaseline``) if the recorded baseline's behavior key differs
    from the current harness state. A no-op when it matches (the baseline resumes)."""
    if baseline_status(prior_fingerprint, current_fingerprint) == "stale":
        cur = _key_of(current_fingerprint)
        raise StaleBaseline(
            f"recorded behavior_key={prior_fingerprint.get('behavior_key')} "
            f"(gemma_sha {prior_fingerprint.get('gemma_sha')}) != current "
            f"behavior_key={cur} (gemma_sha {current_fingerprint.get('gemma_sha')}) — "
            f"config_version / model / verifier (runner_sha) / working tree / serving "
            f"pin changed, so this baseline measured different behavior. Re-baseline "
            f"explicitly (--force to overwrite this label, or use a fresh --label)."
        )


# Hosts that name THIS machine: a serving base here is one physical server, so there
# is nothing to pin. Everything else — including a URL whose host cannot be parsed —
# is remote, and remote recording requires the full pin (fail closed).
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_non_string(base_url) -> bool:
    """The type rule, in one place: a base URL that is not a ``str`` cannot be
    scanned, parsed, or honestly recorded.

    Every site that would otherwise hand a value to ``urlsplit`` or to a
    raw-string scan asks this FIRST — the same shape the character rule ended
    up in (``_has_control_or_whitespace``), because scattered checks are how
    the gap appeared: ``_describe_base`` and the record gate were guarded while
    the shared parse path was not, so ``urlsplit`` stayed reachable. It accepts
    anything with a ``.decode()`` method (its ``_coerce_args`` decodes, parses,
    and re-encodes), which returns a BYTES-flavoured SplitResult — a hostname
    ``b'localhost'`` that matches no local host and a scheme that formats into
    a ``b'http'``-shaped identity. Falsey non-strings (``False``, ``0``,
    ``b""``) were worse: the normalizer returned them unchanged before any
    check ran, so a directly built pinned provider recorded one into the
    behavior key.

    The test is the real TYPE HIERARCHY, not ``isinstance``: ``isinstance``
    consults ``__class__``, which any object may set to ``str``, and such a
    proxy passed the predicate, reached ``urlsplit`` at all three sites, and
    classified LOCAL (skipping the pin gate) — while a falsey one was handed
    back unchanged by the normalizer and recorded into a behavior key.
    ``issubclass(type(x), str)`` reads the type slot, which no attribute can
    spoof, and still accepts a genuine str SUBCLASS — that is a string and
    keeps working. None is not filtered here — callers decide, and for the
    normalizer and the record gate an unset base is a real recorded state."""
    return not issubclass(type(base_url), str)


def _has_control_or_whitespace(base_url: str) -> bool:
    """Any whitespace or control character, anywhere in the URL — nothing is
    stripped, leading and trailing included. ``urlsplit`` deletes tab/newline/CR
    BEFORE parsing (WHATWG alignment), so such a character can hide an authority
    from a raw-string scan while the parsed result grows one — and the HTTP
    client sends the RAW string, which cannot carry control characters on the
    wire at all. A URL the wire cannot carry has no honest identity to record.

    The rule is the CLASS, not an enumeration: Unicode category Cc, or
    ``isspace()``. Cc is exactly C0, DEL and the C1 range U+0080-U+009F — and
    C1 (bar NEL, which isspace catches) slipped past the previous ordinal
    bounds (``< 0x20``, ``== 0x7F``): ``https://host.example/v1<U+0080>``
    passed verbatim into the fingerprint."""
    return any(unicodedata.category(c) == "Cc" or c.isspace() for c in base_url)


def _forbidden_parts(base_url: str) -> list[str]:
    """The raw-string scan for parts a base URL must never carry: userinfo,
    query, fragment. Deliberately parser-free — nothing here can raise — so no
    malformed authority (bad port, bad bracket) can short-circuit past it. '?'
    and '#' are reserved delimiters wherever they appear unencoded, and '@'
    marks userinfo inside the authority, which ends at the first '/', '?' or
    '#'. Callers refuse whitespace/control characters FIRST
    (``_has_control_or_whitespace``); this scan assumes the string is clean of
    them."""
    authority = base_url.partition("//")[2]
    for stop in "/?#":
        authority = authority.partition(stop)[0]
    offending = []
    if "@" in authority:
        offending.append("embedded credentials (userinfo)")
    if "?" in base_url:
        offending.append("a query string")
    if "#" in base_url:
        offending.append("a fragment")
    return offending


def _split_base_url(base_url):
    """FULL parse of a base URL, or None when any part of it fails to parse.

    The one shared classification for the normalizer and the locality check below:
    ``urlsplit().hostname`` alone is not a parse — it happily returns a hostname off
    a netloc whose PORT is garbage (``localhost:notaport``), which once let a
    malformed URL classify as local and record unpinned. Hostname AND port must both
    parse, or the URL reads as unparseable — and unparseable reads as remote, so it
    can never record unpinned. A URL carrying whitespace or control characters is
    unparseable BY RULE: urlsplit would silently strip tab/newline and parse an
    authority the raw string does not have, which once classified
    ``https:/<TAB>/user:...@localhost`` as local. A NON-STRING is unparseable by
    the same rule, checked here at the shared entry so no caller can reach
    ``urlsplit`` around it: urlsplit accepts any object with ``.decode()`` and
    answers in bytes, an identity nothing here could match or record."""
    try:
        if _is_non_string(base_url) or _has_control_or_whitespace(base_url):
            return None
        parts = urlsplit(base_url)
        host = parts.hostname
        port = parts.port  # raises ValueError on a malformed port
    except (ValueError, TypeError):
        return None
    if not host:
        return None
    return parts, host, port


def is_local_base_url(base_url: str) -> bool:
    """Whether ``base_url`` provably points at this machine. Any parse failure —
    including a malformed port — reads as remote: a base that cannot prove it is
    local is not local."""
    parsed = _split_base_url(base_url)
    return parsed is not None and parsed[1] in _LOCAL_HOSTS


def _describe_base(base_url) -> str:
    """A scrub-safe way to NAME a base URL in a message: the host when one parses,
    never the raw string — the raw string may embed a credential, and exception
    text ends up in logs, records and tracebacks. Every message in this module
    that talks about a base URL goes through here; none may format the URL
    itself. ``.hostname`` parses independently of a malformed port, so even an
    unnormalizable URL usually still gets named by its host.

    A non-string is type-checked FIRST, before urlsplit sees it: scalars raise
    AttributeError there — not the ValueError/TypeError a parse failure raises —
    and this function runs INSIDE refusal messages, so a describer that can
    crash turns a host-safe refusal into a naked traceback (a recorded
    ``"base_url": 123`` did exactly that at the record gate)."""
    if _is_non_string(base_url):
        return "an unparseable base URL"
    try:
        host = urlsplit(base_url).hostname
    except (ValueError, TypeError):
        host = None
    return f"host {host!r}" if host else "an unparseable base URL"


def normalize_base_url(base_url: str) -> str:
    """The canonical serving-base identity: ``scheme://host[:port]/path``.

    Scheme and host are lowercased (both are case-insensitive on the wire) and
    trailing slashes dropped (the HTTP client rstrips them before use), so a
    cosmetic variant of the same endpoint cannot split behavior keys.

    A query string, a fragment, or embedded credentials REFUSE
    (``MalformedBaseUrl``) instead of normalizing: the client concatenates
    ``/chat/completions`` onto the raw string, so a query swallows the endpoint
    path into the query value, a fragment truncates the request path, and
    userinfo becomes basic auth on the wire while being a credential no record
    may carry. Each of those makes the recorded identity a lie about the request
    actually sent, so there is nothing honest to record. These checks run on the
    RAW STRING, BEFORE any parse that can fail — a malformed port once bailed
    the full parse and passed the URL through verbatim, credential and all, and
    ``urlsplit`` itself raises on a bad IPv6 bracket — so NO malformed-authority
    combination can carry a forbidden part through the verbatim fallback. The
    refusal names the host, never the URL itself.

    Whitespace and control characters refuse FIRST, before even the raw scan:
    ``urlsplit`` deletes tab/newline before parsing, so
    ``https:/<TAB>/user:secret@host`` shows the raw scan no authority while
    parsing to a credentialed one — the normalizer would then record a
    SANITIZED identity for a request the client cannot even issue. The rule is
    total: any whitespace or control character anywhere refuses, nothing is
    stripped, leading and trailing included.

    A URL that does not parse, or has no host or no parseable port, passes
    through verbatim ONCE it is known to carry no forbidden part: what cannot
    be parsed cannot be quietly rewritten, and the shared classification above
    reads it as remote (fail closed), so it can never record unpinned.

    None passes through unchanged — an unset base is a real recorded state, and
    carbon_fingerprint normalizes one every run. Every OTHER non-string refuses
    (``MalformedBaseUrl``): the falsey early return used to precede all
    validation, so ``False``, ``0`` and ``b""`` were returned unchanged and,
    from a directly built pinned provider, recorded into the behavior key —
    while truthy non-strings crashed the scan instead of refusing. Nothing that
    is not a string may leave here unchanged."""
    if base_url is None:
        return base_url
    if _is_non_string(base_url):
        raise MalformedBaseUrl(
            f"refusing base URL ({_describe_base(base_url)}): it is not a string but a "
            f"{type(base_url).__name__}. What cannot be scanned cannot be declared clean, "
            f"and what cannot be parsed as text has no identity to record — urlsplit would "
            f"answer in bytes for anything carrying .decode(). LLM_BASE_URL is read from "
            f"the environment as text, so this reaches here only from a directly "
            f"constructed provider: pass a scheme://host[:port]/path string, or None for "
            f"an unset base."
        )
    if not base_url:
        return base_url
    if _has_control_or_whitespace(base_url):
        raise MalformedBaseUrl(
            f"refusing base URL for {_describe_base(base_url)}: it contains whitespace or "
            f"control characters. urlsplit strips tab/newline BEFORE parsing, so such a "
            f"URL can parse to an identity the raw string does not have, while the HTTP "
            f"client sends the raw string — which cannot carry control characters on the "
            f"wire at all. Nothing is stripped here; remove the character from "
            f"LLM_BASE_URL."
        )
    # Forbidden parts next, still on the raw string (see _forbidden_parts): no
    # parser stands in front of these checks either.
    offending = _forbidden_parts(base_url)
    if offending:
        raise MalformedBaseUrl(
            f"refusing base URL for {_describe_base(base_url)}: it carries "
            f"{' and '.join(offending)}. The HTTP client appends /chat/completions to "
            f"LLM_BASE_URL verbatim, so anything after the path changes or breaks the "
            f"request in ways the recorded identity could not attribute — and "
            f"credentials must never reach a record. Use a bare "
            f"scheme://host[:port]/path in LLM_BASE_URL; credentials belong in "
            f"LLM_API_KEY."
        )
    parsed = _split_base_url(base_url)
    if parsed is None:
        return base_url
    parts, host, port = parsed
    if ":" in host:  # IPv6 literal — urlsplit strips the brackets; restore them
        host = f"[{host}]"
    port_part = f":{port}" if port is not None else ""
    return f"{parts.scheme.lower()}://{host}{port_part}{parts.path.rstrip('/')}"


def assert_serving_pinned(
    base_url: str, provider_order: str | None, quantization: str | None
) -> None:
    """Refuse (``UnpinnedServing``) a REMOTE serving base without a full serving pin.

    A local endpoint is one physical server — unpinned is its complete, honest serving
    identity, and it keeps working exactly as before. A remote multi-provider router
    without both pins can answer one label's requests from different providers at
    different quantizations; results recorded that way carry a serving confound no
    fingerprint field could attribute afterwards. That is a measured effect (program
    working notes, 2026-08), so the refusal is a hard error with the remediation, never
    a warning."""
    if is_local_base_url(base_url):
        return
    missing = [
        name
        for name, value in (
            ("LLM_PROVIDER_ORDER", provider_order),
            ("LLM_QUANTIZATION", quantization),
        )
        if not value
    ]
    if missing:
        # The base URL is named by host only (never echoed raw): this gate can
        # receive a URL the normalizer passed through verbatim, and a raw URL in
        # an exception message would disclose whatever the URL carries.
        raise UnpinnedServing(
            f"refusing to record against remote serving base ({_describe_base(base_url)}): "
            f"no full serving pin ({' and '.join(missing)} unset). Unpinned remote routing "
            f"spreads one label's requests across providers with mixed quantization — a "
            f"measured confound. Set LLM_PROVIDER_ORDER to exactly one provider name and "
            f"LLM_QUANTIZATION to one quantization label in carbon's .env (fallbacks are "
            f"disabled automatically when a pin is set), or point LLM_BASE_URL at a "
            f"local endpoint."
        )


def assert_recorded_base_url_clean(base_url, origin: str) -> None:
    """The record-side twin of the live boundary: a RECORDED fingerprint's
    base_url must pass the same raw-string checks before anything formats,
    returns, or serializes it.

    Result files written by a PRE-fix runner can carry a verbatim poisoned URL
    (the vulnerable path recorded it when pinned) — a mismatch message that
    echoes recorded values, a returned fingerprint dict, or a CLI JSON print
    would become the disclosure path the live gate closed. So the reader
    refuses first, naming the host and WHICH record (``origin``), never the
    URL. None passes: unset is a real recorded state. A non-string refuses —
    what cannot be scanned cannot be declared clean (fail closed)."""
    if base_url is None:
        return
    if (
        _is_non_string(base_url)
        or _has_control_or_whitespace(base_url)
        or _forbidden_parts(base_url)
    ):
        raise MalformedBaseUrl(
            f"refusing to read {origin}: its recorded base_url "
            f"({_describe_base(base_url)}) carries a forbidden part (credentials, "
            f"query, fragment, or control/whitespace characters). The record was "
            f"written by a pre-fix runner; re-record it — and treat any credential "
            f"it embeds as disclosed."
        )
