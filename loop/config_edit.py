"""Apply a candidate's diff to the editable surface (harness_config.json).

Edits are SURGICAL: each changed scalar is replaced as its exact
``"field": <json-old>`` text, so the resulting git diff is precisely the
changed lines plus the version bump — no reformatting noise (the PR diff *is*
the deliverable; a rewritten file would bury the one-knob story). A collection
field that spans multiple lines in the file (its exact one-line ``"field": [...]``
text never appears) is loudly unsupported rather than reformatted; a single-line
collection stays editable. No current candidate needs a multiline one, and
support can be added when one does.

Which fields exist, and which are collections, is discovered from carbon's own
``config_schema()`` (adr/0002) rather than hardcoded here — so the editor tracks
the editable surface as carbon grows it, and the self-improving loop can propose a
knob the day carbon adds it instead of waiting on a matching edit here.

Safety: a candidate may only target a knob carbon's schema declares, its ``old``
values must match the file (stale-candidate guard), every replacement must be
unique in the text, and the edited file is re-parsed AND re-validated through
carbon's own ``load_config`` door before the write is considered done.
"""

from __future__ import annotations

import json
from pathlib import Path

from loop.artifacts import Candidate

CONFIG_REL = Path("harness") / "harness_config.json"


def config_path(carbon_root: str | Path) -> Path:
    return Path(carbon_root) / CONFIG_REL


def known_knobs() -> dict[str, dict]:
    """carbon's editable-surface schema, keyed by field name: what knobs exist, their
    type, and which are collections / positive-int — the generic knob catalogue the
    editor and the propose side read instead of hardcoding field names."""
    from carbon import surface_manifest

    # carbon has already partitioned on `editable`; re-filtering here would be
    # dead code that hides a shape change instead of failing on it.
    return {field["name"]: field for field in surface_manifest()["editable"]}


def immutable_invariants() -> dict[str, str]:
    """Carbon's explicit do-not-propose list, keyed by invariant name."""
    from carbon import surface_manifest

    return {item["name"]: item["reason"] for item in surface_manifest()["immutable"]}


def proposal_surface() -> dict:
    """The complete proposer contract: selectable knobs and locked boundaries.

    Reads the manifest exactly once, so every section describes the same carbon
    state even if the file underneath is changing.
    """
    from carbon import surface_manifest

    manifest = surface_manifest()
    return {
        "editable": {field["name"]: field for field in manifest["editable"]},
        "locked_fields": {
            item["name"]: item.get("locked_reason")
            or item.get("deprecated")
            or "Managed by Carbon or the pipeline."
            for item in manifest["locked_fields"]
        },
        "immutable": {item["name"]: item["reason"] for item in manifest["immutable"]},
        "candidate_kinds": {
            "configuration_candidate": "Can be applied and evaluated automatically.",
            "strategy_surface_gap": "Needs a reviewed Carbon strategy implementation first.",
            "correctness_defect": "Needs a locked Carbon fix; must not become a knob.",
        },
    }


def apply_candidate(carbon_root: str | Path, candidate: Candidate) -> dict:
    """Rewrite harness_config.json with the candidate's values + version bump.

    Returns the new config as a dict. Raises (leaving the file untouched) on
    any mismatch between the candidate's ``old`` values and the file."""
    path = config_path(carbon_root)
    text = path.read_text()
    current = json.loads(text)
    schema = known_knobs()

    expected = dict(current)
    for name, diff in candidate.fields.items():
        if name not in schema:
            raise ValueError(
                f"candidate {candidate.id!r}: no field {name!r} in carbon's config "
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
    from harness.harness_config import load_config  # carbon's own validation door

    tmp = path.with_suffix(".json.candidate-check")
    tmp.write_text(text)
    try:
        load_config(tmp)  # raises on wrong types / non-positive counts / bad regex
    finally:
        tmp.unlink(missing_ok=True)
    path.write_text(text)
    return expected
