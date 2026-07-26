"""The resume-guard: decides whether a recorded baseline is still valid under the
current harness state, so an *additive* carbon release does not force an empty
re-baseline.

It pins on a **behavior key** — carbon's declared ``config_version`` + the model +
our ``runner_sha`` (the verifier) + ``dirty_sha`` (uncommitted carbon edits), i.e.
the inputs that actually determine the measured behavior — NOT the raw committed
``gemma_sha``. An additive, default-neutral carbon release keeps ``config_version``
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

# The fingerprint fields that fold into the behavior key, in order. ``gemma_sha`` is
# pointedly absent: a committed carbon move is provenance, not behavior (carbon's own
# config_version is the behavior-version declaration we trust). ``dirty_sha`` stays,
# because an uncommitted carbon edit changes behavior that no version counter attests.
_KEY_FIELDS = ("config_version", "model", "runner_sha", "dirty_sha")


class StaleBaseline(Exception):
    """A recorded baseline's behavior key differs from the current harness state — it
    measured different behavior and must be re-baselined (never silently reused)."""


def behavior_key(config_version, model: str, runner_sha: str, dirty_sha: str | None) -> str:
    """Stable identity of the behavior-determining inputs. Additive carbon releases
    (``config_version`` unchanged, no runner edit, clean tree) keep this fixed; a
    model / verifier / config / working-tree change moves it. Deliberately excludes
    the committed ``gemma_sha`` — that is provenance, not behavior."""
    raw = f"{config_version}|{model}|{runner_sha}|{dirty_sha}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def fingerprint_behavior_key(fingerprint: dict) -> str:
    """The behavior key of a fingerprint dict — the currency this runner passes
    around. Missing behavior-relevant fields raise ``KeyError`` (a fingerprint that
    cannot state its behavior identity is a bug, not a silent default)."""
    return behavior_key(
        fingerprint["config_version"],
        fingerprint["model"],
        fingerprint["runner_sha"],
        fingerprint.get("dirty_sha"),
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
            f"config_version / model / verifier (runner_sha) / working tree changed, so "
            f"this baseline measured different behavior. Re-baseline explicitly "
            f"(--force to overwrite this label, or use a fresh --label)."
        )
