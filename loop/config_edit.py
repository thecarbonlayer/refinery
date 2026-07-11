"""Apply a candidate's diff to the editable surface (harness_config.json).

Edits are SURGICAL: each changed scalar is replaced as its exact
``"field": <json-old>`` text, so the resulting git diff is precisely the
changed lines plus the version bump — no reformatting noise (the PR diff *is*
the deliverable; a rewritten file would bury the one-knob story). List-typed
fields (approval_tools, code_extensions) span multiple lines and are not
supported by this replacement; no current candidate needs them, and support
can be added when one does — loudly unsupported beats quietly reformatted.

Safety: the candidate's ``old`` values must match the file (stale-candidate
guard), every replacement must be unique in the text, and the edited file is
re-parsed AND re-validated through gemma's own ``load_config`` door before the
write is considered done.
"""

from __future__ import annotations

import json
from pathlib import Path

from loop.artifacts import Candidate

CONFIG_REL = Path("harness") / "harness_config.json"


def config_path(gemma_root: str | Path) -> Path:
    return Path(gemma_root) / CONFIG_REL


def apply_candidate(gemma_root: str | Path, candidate: Candidate) -> dict:
    """Rewrite harness_config.json with the candidate's values + version bump.

    Returns the new config as a dict. Raises (leaving the file untouched) on
    any mismatch between the candidate's ``old`` values and the file."""
    path = config_path(gemma_root)
    text = path.read_text()
    current = json.loads(text)

    expected = dict(current)
    for name, diff in candidate.fields.items():
        if name not in current:
            raise ValueError(f"candidate {candidate.id!r}: no field {name!r} in {path}")
        if current[name] != diff["old"]:
            raise ValueError(
                f"candidate {candidate.id!r} is stale: field {name!r} is "
                f"{current[name]!r} in the file, candidate expected {diff['old']!r}"
            )
        expected[name] = diff["new"]
    expected["version"] = current["version"] + 1

    for name in [*candidate.fields, "version"]:
        old_text = f'"{name}": {json.dumps(current[name])}'
        new_text = f'"{name}": {json.dumps(expected[name])}'
        n = text.count(old_text)
        if n != 1:
            raise ValueError(
                f"candidate {candidate.id!r}: cannot surgically edit {name!r} — "
                f"{old_text!r} occurs {n} times in {path} (list-typed/multiline "
                f"fields are not supported by line replacement)"
            )
        text = text.replace(old_text, new_text)

    if json.loads(text) != expected:
        raise ValueError(
            f"candidate {candidate.id!r}: edited text does not parse back to the "
            f"intended config — refusing to write"
        )
    from harness.harness_config import load_config  # gemma's own validation door

    tmp = path.with_suffix(".json.candidate-check")
    tmp.write_text(text)
    try:
        load_config(tmp)  # raises on wrong types / non-positive counts / bad regex
    finally:
        tmp.unlink(missing_ok=True)
    path.write_text(text)
    return expected
