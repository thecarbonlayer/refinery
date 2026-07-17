"""Apply a candidate's diff to the editable surface (harness_config.json).

Edits are SURGICAL: each changed scalar is replaced as its exact
``"field": <json-old>`` text, so the resulting git diff is precisely the
changed lines plus the version bump — no reformatting noise (the PR diff *is*
the deliverable; a rewritten file would bury the one-knob story). A collection
field that spans multiple lines in the file (its exact one-line ``"field": [...]``
text never appears) is loudly unsupported rather than reformatted; a single-line
collection stays editable. No current candidate needs a multiline one, and
support can be added when one does.

Which fields exist, and which are collections, is discovered from gemma's own
``config_schema()`` (adr/0002) rather than hardcoded here — so the editor tracks
the editable surface as gemma grows it, and the self-improving loop can propose a
knob the day gemma adds it instead of waiting on a matching edit here.

Safety: a candidate may only target a knob gemma's schema declares, its ``old``
values must match the file (stale-candidate guard), every replacement must be
unique in the text, and the edited file is re-parsed AND re-validated through
gemma's own ``load_config`` door before the write is considered done.
"""

from __future__ import annotations

import json
from pathlib import Path

from loop.artifacts import Candidate

CONFIG_REL = Path("harness") / "harness_config.json"


def config_path(gemma_root: str | Path) -> Path:
    return Path(gemma_root) / CONFIG_REL


def known_knobs() -> dict[str, dict]:
    """gemma's editable-surface schema, keyed by field name: what knobs exist, their
    type, and which are collections / positive-int — the generic knob catalogue the
    editor and the propose side read instead of hardcoding field names."""
    from gemma import config_schema

    return {field["name"]: field for field in config_schema()}


def apply_candidate(gemma_root: str | Path, candidate: Candidate) -> dict:
    """Rewrite harness_config.json with the candidate's values + version bump.

    Returns the new config as a dict. Raises (leaving the file untouched) on
    any mismatch between the candidate's ``old`` values and the file."""
    path = config_path(gemma_root)
    text = path.read_text()
    current = json.loads(text)
    schema = known_knobs()

    expected = dict(current)
    for name, diff in candidate.fields.items():
        if name not in schema:
            raise ValueError(
                f"candidate {candidate.id!r}: no field {name!r} in gemma's config "
                f"schema (known knobs: {', '.join(sorted(schema))})"
            )
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
            hint = (
                "collection field spans multiple lines in this file"
                if schema.get(name, {}).get("collection")
                else "value not found as a single unique line"
            )
            raise ValueError(
                f"candidate {candidate.id!r}: cannot surgically edit {name!r} — "
                f"{old_text!r} occurs {n} times in {path} ({hint}; not supported "
                f"by line replacement)"
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
