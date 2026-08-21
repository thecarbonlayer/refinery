"""Binding to the carbon checkout under test.

The runner lives OUTSIDE carbon — its own repo, not just its own directory (the
suite must not share a home with the editable surface an external editor will
act on) — but it drives carbon's Agent in-process via the editable path
dependency. This module pins down the two environment questions that raises:
which endpoint/model to call (carbon's own .env, loaded here because
Provider.from_env reads .env from the cwd), and WHICH harness state produced a
given result (git SHA + config version stamp).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

# refinery and carbon are sibling checkouts: <root>/refinery, <root>/carbon.
# parents[2] is that shared root (runner -> repo -> root), matching the
# `../carbon` editable dependency in pyproject.toml. Keep the two in step.
CARBON_ROOT = Path(__file__).resolve().parents[2] / "carbon"


def load_carbon_env(root: Path = CARBON_ROOT) -> None:
    """Load <carbon>/.env into os.environ (setdefault — real env vars win).

    Delegates to carbon's own ``load_env`` (the embedding seam, adr/0002) instead of
    reimplementing dotenv parsing, so the suite reads .env exactly the way the
    harness's own gates do."""
    from carbon import load_env

    load_env(root)


def make_provider():
    """The Provider every task's Agent uses — carbon's .env, same as its own gates.

    ``Provider.from_env(root=)`` loads <carbon>/.env itself before reading the vars,
    so no separate dotenv step is needed."""
    from carbon import Provider

    return Provider.from_env(root=CARBON_ROOT)


def _git(root: Path, *args: str) -> str:
    """Raw stdout of a git command against ``root`` (seam for offline tests).

    A failing git command must RAISE, never return '' — an empty `status
    --porcelain` from a failed run would stamp a dirty tree as clean."""
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {root}: {proc.stderr.strip()}")
    return proc.stdout


RUNNER_ROOT = Path(__file__).resolve().parent

# Every field on carbon's Provider dataclass is either recorded in the fingerprint
# or named in the exclusion dict with the reason it is not. These lists are
# LOAD-BEARING: ``carbon_fingerprint`` builds its serving section from the first,
# and tests/test_carbon_env.py sweeps carbon's actual dataclass against both — so
# a future Provider field that alters the wire request or the serving identity
# cannot land without an explicit disposition here.
PROVIDER_FIELDS_FINGERPRINTED = (
    "model",
    "base_url",
    "reasoning_effort",
    "provider_order",
    "quantization",
    # Recorded as a MARKER, not the callable: None for a network provider, else the
    # responder's qualified name (repo code, so ``runner_sha`` pins its behavior).
    # Scripted responders are REAL in this suite — cluster H builds
    # ``Provider(..., responder=...)`` per task, and those results record under the
    # suite's environment fingerprint — but those live on per-task providers inside
    # ``runner/tasks/`` (verifier apparatus, hashed by ``runner_sha``), not on the
    # env provider this fingerprint attests. The field is here so that a responder
    # ever reaching the SUITE-LEVEL provider cannot masquerade as a network serving
    # base: it would be named in the record and would move the behavior key.
    "responder",
)
PROVIDER_FIELDS_EXCLUDED = {
    "api_key": "secret — identifies the account, never the served behavior; the "
    "fingerprint is written into every record and results JSON, so recording it "
    "would commit a credential (scrub rule). MUST NOT be fingerprinted.",
}


def runner_sha(root: Path = RUNNER_ROOT) -> str:
    """Content identity of the runner package itself — the verifier's version.

    Results are only comparable when the SAME verifier produced them: a
    verifier fix silently shifts pass fractions, so every result is stamped
    with this sha and both resume and delta gate on it. Hash covers every
    .py file's relative posix path AND bytes (a rename or an edit both count)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def carbon_fingerprint(root: Path = CARBON_ROOT) -> dict:
    """Attribute a run to a harness state: git SHA (+dirty), config version, model,
    verifier hash, and the derived ``behavior_key`` the resume-guard pins on.

    ``config_version`` and ``model`` come from carbon's own ``provenance()`` primitive
    (adr/0002) rather than reaching into ``CONFIG`` directly. We keep our OWN
    authoritative full ``gemma_sha`` and dirty-tree detection layered on top:
    ``provenance()`` is best-effort (returns a *short* sha, ``None`` outside a git
    repo, and does no dirty detection), but a measurement suite must fail loudly on an
    unattributable checkout and needs a content identity for uncommitted edits.

    A dirty flag alone has no content identity — two different uncommitted edits at the
    same SHA would be indistinguishable — so a dirty tree also carries ``dirty_sha``:
    sha256 over `status --porcelain` + `diff HEAD` (untracked files appear in the
    status text, so they perturb the hash even though the diff misses them). Clean
    tree -> ``dirty_sha`` is None.

    The serving section is every ``Provider`` field that alters the wire request or
    the serving identity — enumerated in ``PROVIDER_FIELDS_FINGERPRINTED``, read from
    the same ``Provider`` carbon's own calls use, and folded into the behavior key.
    The model string alone cannot distinguish serving bases: the same model name
    answers from LM Studio on :1234, Ollama on :11434, or different routed providers
    at different quantizations and effort levels, and each of those is different
    measured behavior. ``base_url`` is recorded NORMALIZED (see
    ``guard.normalize_base_url``) so a cosmetic variant of one endpoint cannot split
    keys. None means unset — the truthful record of a field that puts nothing on the
    wire, read via ``getattr`` because a carbon Provider predating a field sends
    nothing for it whatever the env says. ``responder`` is recorded as a marker (None,
    or the callable's qualified name — see the field list above); ``api_key`` is
    excluded by name (``PROVIDER_FIELDS_EXCLUDED``) with the reason stated there.

    A REMOTE base without the full pin is refused right here
    (``guard.assert_serving_pinned``): an unpinned remote serving state is
    unattributable, so nothing downstream — recording, resume, the mid-suite drift
    check — can even name it.

    ``behavior_key`` (see runner/guard.py) folds config_version + model + runner_sha +
    dirty_sha + the serving fields — everything that determines behavior *except* the
    committed ``gemma_sha`` — so an additive carbon release resumes instead of forcing
    a re-baseline."""
    from carbon import provenance

    from runner import guard

    sha = _git(root, "rev-parse", "HEAD").strip()
    if not sha:
        raise RuntimeError("cannot fingerprint carbon checkout: git rev-parse failed")
    status = _git(root, "status", "--porcelain")
    dirty = bool(status.strip())
    dirty_sha = (
        hashlib.sha256((status + _git(root, "diff", "HEAD")).encode()).hexdigest()
        if dirty
        else None
    )
    provider = make_provider()
    serving = {field: getattr(provider, field, None) for field in PROVIDER_FIELDS_FINGERPRINTED}
    serving["base_url"] = guard.normalize_base_url(serving["base_url"])
    # The responder marker: the callable itself is uncapturable, so what is recorded
    # is WHICH repo callable — module.qualname, whose behavior runner_sha pins. A
    # callable with no stable name (e.g. functools.partial) refuses: an unstable
    # marker would move the behavior key between runs of identical behavior.
    if serving["responder"] is not None:
        resp = serving["responder"]
        module = getattr(resp, "__module__", None)
        qualname = getattr(resp, "__qualname__", None)
        if not module or not qualname:
            raise RuntimeError(
                "cannot fingerprint the provider's responder: it has no stable "
                "qualified name to record — use a module-level callable"
            )
        serving["responder"] = f"{module}.{qualname}"
    guard.assert_serving_pinned(
        serving["base_url"] or "", serving["provider_order"], serving["quantization"]
    )
    model = serving["model"]
    prov = provenance(model=model, root=root)  # config_version + model (short sha unused)
    fp = {
        "gemma_sha": sha,
        "gemma_dirty": dirty,
        "dirty_sha": dirty_sha,
        "config_version": prov["config_version"],
        "runner_sha": runner_sha(),
        **serving,
    }
    fp["behavior_key"] = guard.fingerprint_behavior_key(fp)
    return fp
