"""Scrub machine-identifying paths out of every string value.

Lives in `runner/` — not `loop/` — so rows are BORN clean: `runner/run.py`'s
`write_record` calls `scrub_text` at serialization, after every verifier has already
read the raw (unscrubbed) text. `loop/scrub_results.py` imports `scrub_text` back from
here rather than keeping a second copy, and remains the repair tool for anything
recorded before this module existed.

Three substitutions, applied to EVERY string value:

    (/private)?/var/folders/<seg>/<seg>/T   ->  <TMPDIR>   (suffix after /T kept)
    (/private)?/var/folders/<anything>      ->  <TMPDIR>   (a TRUNCATED path — one
                                                committed log holds a path cut mid-
                                                directory by the runner's own clamp)
    /Users/<user>                           ->  <HOME>
    any surviving bare username             ->  <USER>
"""

from __future__ import annotations

import getpass
import re

_TMPDIR = re.compile(r"(?:/private)?/var/folders/[^/\s\"']+/[^/\s\"']+/T")
# The fallback for truncated paths, applied AFTER _TMPDIR so full paths keep their
# post-/T suffix (`<TMPDIR>/workspace-x`) while a clamp-cut fragment still vanishes.
_TMPDIR_PARTIAL = re.compile(r"(?:/private)?/var/folders(?:/[^\s\"']*)?")
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
