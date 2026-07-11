"""Binding to the gemma checkout under test.

The runner lives OUTSIDE dist/gemma (the suite must not share a home with the
editable surface an external editor will act on), but it drives dist/gemma's
Agent in-process via the editable path dependency. This module pins down the
two environment questions that raises: which endpoint/model to call (gemma's
own .env, loaded here because Provider.from_env reads .env from the cwd), and
WHICH harness state produced a given result (git SHA + config version stamp).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

GEMMA_ROOT = Path(__file__).resolve().parents[2] / "dist" / "gemma"


def load_gemma_env(root: Path = GEMMA_ROOT) -> None:
    """Load <gemma>/.env into os.environ (setdefault — real env vars win)."""
    path = root / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_provider():
    """The Provider every task's Agent uses — gemma's .env, same as its own gates."""
    from model import Provider

    load_gemma_env()
    return Provider.from_env()


def gemma_fingerprint(root: Path = GEMMA_ROOT) -> dict:
    """Attribute a run to a harness state: git SHA (+dirty), config version, model."""
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    if not sha:
        raise RuntimeError("cannot fingerprint gemma checkout: git rev-parse failed")
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    from harness.harness_config import CONFIG

    return {
        "gemma_sha": sha,
        "gemma_dirty": dirty,
        "config_version": CONFIG.version,
        "model": make_provider().model,
    }
