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


def _split_base_url(base_url):
    """FULL parse of a base URL, or None when any part of it fails to parse.

    The one shared classification for the normalizer and the locality check below:
    ``urlsplit().hostname`` alone is not a parse — it happily returns a hostname off
    a netloc whose PORT is garbage (``localhost:notaport``), which once let a
    malformed URL classify as local and record unpinned. Hostname AND port must both
    parse, or the URL reads as unparseable — and unparseable reads as remote, so it
    can never record unpinned."""
    try:
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
    actually sent, so there is nothing honest to record. The refusal names the
    part found, never the URL itself — the URL may embed the credential.

    A URL that fails to parse at all passes through verbatim: what cannot be
    parsed cannot be quietly rewritten, and the shared classification above reads
    it as remote (fail closed), so it can never record unpinned."""
    if not base_url:
        return base_url
    parsed = _split_base_url(base_url)
    if parsed is None:
        return base_url
    parts, host, port = parsed
    offending = []
    if parts.username is not None or parts.password is not None:
        offending.append("embedded credentials (userinfo)")
    if parts.query:
        offending.append("a query string")
    if parts.fragment:
        offending.append("a fragment")
    if offending:
        raise MalformedBaseUrl(
            f"refusing base URL for host {host!r}: it carries {' and '.join(offending)}. "
            f"The HTTP client appends /chat/completions to LLM_BASE_URL verbatim, so "
            f"anything after the path changes or breaks the request in ways the recorded "
            f"identity could not attribute — and credentials must never reach a record. "
            f"Use a bare scheme://host[:port]/path in LLM_BASE_URL; credentials belong "
            f"in LLM_API_KEY."
        )
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
        raise UnpinnedServing(
            f"refusing to record against remote serving base {base_url}: no full serving "
            f"pin ({' and '.join(missing)} unset). Unpinned remote routing spreads one "
            f"label's requests across providers with mixed quantization — a measured "
            f"confound. Set LLM_PROVIDER_ORDER to exactly one provider name and "
            f"LLM_QUANTIZATION to one quantization label in carbon's .env (fallbacks are "
            f"disabled automatically when a pin is set), or point LLM_BASE_URL at a "
            f"local endpoint."
        )
