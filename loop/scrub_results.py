"""Scrub machine-identifying paths out of recorded results, verifiably.

This repo is public and results are committed as the record. Attempt `detail` strings
quote what the model saw and said, which turns out to include per-run temp workspaces
(`/var/folders/<x>/<y>/T/workspace-…`), and — when a task's transcript brushes the
environment — an entire `$PATH` with the home directory and every installed tool in
it. The prescribed pre-commit grep covered `/Users/` and `/home/` and was blind to
`/var/folders`; two committed candidate logs carry such paths today, which is how this
module earned its place.

Three substitutions, applied to EVERY string value:

    (/private)?/var/folders/<seg>/<seg>/T   ->  <TMPDIR>   (suffix after /T kept)
    (/private)?/var/folders/<anything>      ->  <TMPDIR>   (a TRUNCATED path — one
                                                committed log holds a path cut mid-
                                                directory by the runner's own clamp)
    /Users/<user>                           ->  <HOME>
    any surviving bare username             ->  <USER>

Every string, not just `detail`: a first version scrubbed `detail` only and its own
verification caught the gap — the paths also live inside `approvals` entries, which
quote tool arguments verbatim. Substituting on measurement strings (task names, shas,
outcome labels) is a no-op, and the verifier PROVES it: every non-string field must be
identical, every string field must equal the substitution of its original, so a bug
here fails loudly instead of silently rewriting evidence.

Idempotent: scrubbing a scrubbed file is a no-op, so it can run on every commit.

The durable fix — emitting scrubbed details at generation time — belongs in `runner/`
and is deliberately NOT here: `runner_sha` moved four times yesterday and the
iteration-4 baseline was recorded minutes ago; a fifth move would invalidate it for a
cosmetic gain. It is queued for the next runner invalidation window. Until then,
`tests/test_results_are_scrubbed.py` turns the suite red if anything under `results/`
carries a machine path, which makes the scrub impossible to forget rather than
housekeeping to remember.
"""

from __future__ import annotations

import getpass
import json
import re
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

_TMPDIR = re.compile(r"(?:/private)?/var/folders/[^/\s\"']+/[^/\s\"']+/T")
# The fallback for truncated paths, applied AFTER _TMPDIR so full paths keep their
# post-/T suffix (`<TMPDIR>/workspace-x`) while a clamp-cut fragment still vanishes.
_TMPDIR_PARTIAL = re.compile(r"(?:/private)?/var/folders/[^\s\"']*")
_HOME = re.compile(r"/Users/[A-Za-z0-9._-]+")


def _username() -> str:
    try:
        return getpass.getuser()
    except Exception:  # pragma: no cover — CI without a passwd entry
        return ""


def scrub_text(text: str, username: str | None = None) -> str:
    text = _TMPDIR.sub("<TMPDIR>", text)
    text = _TMPDIR_PARTIAL.sub("<TMPDIR>", text)
    text = _HOME.sub("<HOME>", text)
    user = _username() if username is None else username
    if user:
        text = re.sub(rf"\b{re.escape(user)}\b", "<USER>", text)
    return text


def _scrub_obj(obj, username):
    """Substitute in every string value; structure and non-strings pass through."""
    if isinstance(obj, dict):
        return {k: _scrub_obj(v, username) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_scrub_obj(v, username) for v in obj]
    if isinstance(obj, str):
        return scrub_text(obj, username)
    return obj


def _verify(original, scrubbed, username, where: str) -> None:
    """Strings may change ONLY by the substitution; everything else is identical."""
    if isinstance(original, dict):
        assert isinstance(scrubbed, dict) and original.keys() == scrubbed.keys(), where
        for k in original:
            _verify(original[k], scrubbed[k], username, f"{where}.{k}")
    elif isinstance(original, list):
        assert len(original) == len(scrubbed), where
        for i, (a, b) in enumerate(zip(original, scrubbed, strict=True)):
            _verify(a, b, username, f"{where}[{i}]")
    elif isinstance(original, str):
        assert scrubbed == scrub_text(original, username), f"{where}"
    else:
        assert original == scrubbed, f"{where}: {original!r} != {scrubbed!r}"


def scrub_file(path: Path, username: str | None = None) -> bool:
    """Scrub one results file in place. Returns True if anything changed."""
    user = _username() if username is None else username
    raw = path.read_text()
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        scrubbed = [_scrub_obj(r, user) for r in rows]
        for a, b in zip(rows, scrubbed, strict=True):
            _verify(a, b, user, path.name)
        # `changed` compares PARSED values, not raw text: a first version compared
        # scrubbed raw text, which flagged files it then wrote back with the offending
        # path still inside — the flag and the write disagreed, and its own invariant
        # check is what exposed it. Parsed comparison also leaves a file alone when
        # only serialization formatting would differ.
        changed = scrubbed != rows
        if changed:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in scrubbed))
        return changed
    obj = json.loads(raw)
    scrubbed_obj = _scrub_obj(obj, user)
    _verify(obj, scrubbed_obj, user, path.name)
    changed = scrubbed_obj != obj
    if changed:
        path.write_text(json.dumps(scrubbed_obj, indent=2, ensure_ascii=False) + "\n")
    return changed


def main(argv: list[str] | None = None) -> int:
    targets = [Path(a) for a in (argv or sys.argv[1:])] or sorted(RESULTS_DIR.glob("*.json*"))
    changed = []
    for path in targets:
        if scrub_file(path):
            changed.append(path.name)
    print(
        f"scrubbed {len(changed)} of {len(targets)} files"
        + (": " + ", ".join(changed) if changed else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
