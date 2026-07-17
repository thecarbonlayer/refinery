"""Binding to the gemma checkout under test.

The runner lives OUTSIDE gemma (the suite must not share a home with the
editable surface an external editor will act on), but it drives gemma's
Agent in-process via the editable path dependency. This module pins down the
two environment questions that raises: which endpoint/model to call (gemma's
own .env, loaded here because Provider.from_env reads .env from the cwd), and
WHICH harness state produced a given result (git SHA + config version stamp).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

GEMMA_ROOT = Path(__file__).resolve().parents[3] / "gemma"


def load_gemma_env(root: Path = GEMMA_ROOT) -> None:
    """Load <gemma>/.env into os.environ (setdefault — real env vars win).

    Delegates to gemma's own ``load_env`` (the embedding seam, adr/0002) instead of
    reimplementing dotenv parsing, so the suite reads .env exactly the way the
    harness's own gates do."""
    from gemma import load_env

    load_env(root)


def make_provider():
    """The Provider every task's Agent uses — gemma's .env, same as its own gates.

    ``Provider.from_env(root=)`` loads <gemma>/.env itself before reading the vars,
    so no separate dotenv step is needed."""
    from gemma import Provider

    return Provider.from_env(root=GEMMA_ROOT)


def _git(root: Path, *args: str) -> str:
    """Raw stdout of a git command against ``root`` (seam for offline tests).

    A failing git command must RAISE, never return '' — an empty `status
    --porcelain` from a failed run would stamp a dirty tree as clean."""
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {root}: {proc.stderr.strip()}")
    return proc.stdout


RUNNER_ROOT = Path(__file__).resolve().parent


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


def gemma_fingerprint(root: Path = GEMMA_ROOT) -> dict:
    """Attribute a run to a harness state: git SHA (+dirty), config version, model,
    verifier hash, and the derived ``behavior_key`` the resume-guard pins on.

    ``config_version`` and ``model`` come from gemma's own ``provenance()`` primitive
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

    ``behavior_key`` (see runner/guard.py) folds config_version + model + runner_sha +
    dirty_sha — everything that determines behavior *except* the committed
    ``gemma_sha`` — so an additive gemma release resumes instead of forcing a
    re-baseline."""
    from gemma import provenance

    from runner import guard

    sha = _git(root, "rev-parse", "HEAD").strip()
    if not sha:
        raise RuntimeError("cannot fingerprint gemma checkout: git rev-parse failed")
    status = _git(root, "status", "--porcelain")
    dirty = bool(status.strip())
    dirty_sha = (
        hashlib.sha256((status + _git(root, "diff", "HEAD")).encode()).hexdigest()
        if dirty
        else None
    )
    model = make_provider().model
    prov = provenance(model=model, root=root)  # config_version + model (short sha unused)
    fp = {
        "gemma_sha": sha,
        "gemma_dirty": dirty,
        "dirty_sha": dirty_sha,
        "config_version": prov["config_version"],
        "model": model,
        "runner_sha": runner_sha(),
    }
    fp["behavior_key"] = guard.fingerprint_behavior_key(fp)
    return fp
