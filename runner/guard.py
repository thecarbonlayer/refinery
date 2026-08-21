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

# The fingerprint fields that fold into the behavior key, in order. ``gemma_sha`` is
# pointedly absent: a committed carbon move is provenance, not behavior (carbon's own
# config_version is the behavior-version declaration we trust). ``dirty_sha`` stays,
# because an uncommitted carbon edit changes behavior that no version counter attests.
# ``provider_order`` and ``quantization`` (the serving pin) stay too: the model string
# alone cannot distinguish serving bases — the same model name answers from different
# providers at different quantizations, and that is a behavior change no other field
# sees. None is real data for all three (clean tree / unpinned serving), read via
# ``.get`` — but note a record that never carried the serving fields still cannot
# resume, because its RECORDED key was computed by the pre-serving formula.
_KEY_FIELDS = (
    "config_version",
    "model",
    "runner_sha",
    "dirty_sha",
    "provider_order",
    "quantization",
)


class StaleBaseline(Exception):
    """A recorded baseline's behavior key differs from the current harness state — it
    measured different behavior and must be re-baselined (never silently reused)."""


class UnpinnedServing(Exception):
    """A remote serving base with no full provider pin must not be recorded: unpinned
    routing spreads one label's requests across providers with mixed quantization — a
    measured confound inside the experiment, never a warning."""


def behavior_key(
    config_version,
    model: str,
    runner_sha: str,
    dirty_sha: str | None,
    provider_order: str | None,
    quantization: str | None,
) -> str:
    """Stable identity of the behavior-determining inputs. Additive carbon releases
    (``config_version`` unchanged, no runner edit, clean tree, same serving pin) keep
    this fixed; a model / verifier / config / working-tree / serving-base change moves
    it. Deliberately excludes the committed ``gemma_sha`` — that is provenance, not
    behavior."""
    raw = f"{config_version}|{model}|{runner_sha}|{dirty_sha}|{provider_order}|{quantization}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def fingerprint_behavior_key(fingerprint: dict) -> str:
    """The behavior key of a fingerprint dict — the currency this runner passes
    around. Missing behavior-relevant fields raise ``KeyError`` (a fingerprint that
    cannot state its behavior identity is a bug, not a silent default); ``dirty_sha``
    and the serving-pin fields read through ``.get`` because None there is a real
    recorded state (clean tree, unpinned local serving), not an absence."""
    return behavior_key(
        fingerprint["config_version"],
        fingerprint["model"],
        fingerprint["runner_sha"],
        fingerprint.get("dirty_sha"),
        fingerprint.get("provider_order"),
        fingerprint.get("quantization"),
    )


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


def is_local_base_url(base_url: str) -> bool:
    """Whether ``base_url`` points at this machine. Parse failures read as remote —
    a base that cannot prove it is local is not local."""
    try:
        host = urlsplit(base_url).hostname
    except ValueError:
        return False
    return (host or "") in _LOCAL_HOSTS


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
