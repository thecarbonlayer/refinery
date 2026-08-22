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

**Optional knobs the file omits.** carbon lands a knob whose default changes
nothing by leaving it OUT of the shipped JSON: the file keeps its
``config_version``, every external baseline pinned to that version stays valid,
and the loaded default is today's behavior. Such a field is advertised by the
manifest and has no line to replace, so a candidate targeting it used to be
refused before it could even validate — the loop could propose a value it could
never run, which made the knob decorative. ``apply_candidate`` therefore INSERTS
a field the manifest marks ``optional`` and the file omits, at its place in
carbon's own serialization order, and validates it through the same door as
every other edit. A candidate says "absent" by declaring ``old: None``; the
stale guard still applies, in the only form it can take with no current value.

Absence is an insert opportunity ONLY when carbon calls the field optional. A
required field missing from the file is a corrupt config, and writing it back
would repair — and hide — that corruption.

Nested optional parameters (a new key inside ``compaction`` or ``tool_output``)
need no separate path: the established candidate contract addresses an object
knob WHOLE, so the object's one line is replaced with one that carries the extra
key. That is a property worth stating because it is easy to assume otherwise;
``tests/test_loop_config_edit.py`` pins it rather than trusting it.
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


def _serialization_order() -> list[str]:
    """Carbon's own field order, from ``config_schema()`` — the order the shipped
    file is written in. Read from carbon, never restated here: a hardcoded order
    would drift the day carbon adds a field, and the drift would show up as a
    knob inserted in a surprising place rather than as a failure."""
    from carbon import config_schema

    return [field["name"] for field in config_schema()]


def _is_complete_line(line: str, name: str) -> bool:
    """Is ``line`` this field's ENTIRE serialized entry?

    A multi-line collection (``code_extensions``) opens with ``"field": [`` and
    continues below, so its first line matches the name but is not a place a new
    field can follow. The test is behavioral rather than shape-guessing: strip the
    optional trailing comma, and ask whether what sits after the colon parses as
    JSON on its own."""
    stripped = line.strip().rstrip(",")
    prefix = f'"{name}":'
    if not stripped.startswith(prefix):
        return False
    try:
        json.loads(stripped[len(prefix) :].strip())
    except json.JSONDecodeError:
        return False
    return True


def _insert_field(text: str, name: str, value: object, candidate_id: str, path: Path) -> str:
    """Add ``"name": value`` to the serialized object, in carbon's field order.

    Anchored to the NEAREST PRECEDING field that is present in the file as a
    complete single line, so the new knob lands where carbon's surface says it
    belongs and the git diff stays the one-knob story this module exists to
    produce. Formatting is preserved the way the replacement path preserves it:
    the anchor's own indentation, and its comma convention (the anchor gains a
    comma only when it was the object's last entry)."""
    order = _serialization_order()
    lines = text.splitlines()
    anchor = None
    for previous in reversed(order[: order.index(name)]):
        hits = [i for i, line in enumerate(lines) if _is_complete_line(line, previous)]
        if len(hits) == 1:
            anchor = hits[0]
            break
        if len(hits) > 1:
            raise ValueError(
                f"candidate {candidate_id!r}: cannot insert {name!r} — anchor field "
                f"{previous!r} occurs {len(hits)} times in {path}"
            )
    if anchor is None:
        raise ValueError(
            f"candidate {candidate_id!r}: cannot insert {name!r} — no preceding field "
            f"from carbon's schema order appears as a complete line in {path}"
        )

    line = lines[anchor]
    indent = line[: len(line) - len(line.lstrip())]
    trailing_comma = line.rstrip().endswith(",")
    if not trailing_comma:
        lines[anchor] = line.rstrip() + ","
    lines.insert(
        anchor + 1, f'{indent}"{name}": {json.dumps(value)}' + ("," if trailing_comma else "")
    )
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def apply_candidate(carbon_root: str | Path, candidate: Candidate) -> dict:
    """Rewrite harness_config.json with the candidate's values + version bump.

    Returns the new config as a dict. Raises (leaving the file untouched) on
    any mismatch between the candidate's ``old`` values and the file."""
    path = config_path(carbon_root)
    text = path.read_text()
    current = json.loads(text)
    schema = known_knobs()

    expected = dict(current)
    inserted: list[str] = []
    for name, diff in candidate.fields.items():
        if name not in schema:
            raise ValueError(
                f"candidate {candidate.id!r}: no field {name!r} in carbon's config "
                f"schema (known knobs: {', '.join(sorted(schema))})"
            )
        if name not in current:
            # Absent from the file. Legal only for a knob carbon publishes as
            # optional — anything else missing is a corrupt config, not an insert.
            if not schema[name].get("optional"):
                raise ValueError(f"candidate {candidate.id!r}: no field {name!r} in {path}")
            if diff["old"] is not None:
                raise ValueError(
                    f"candidate {candidate.id!r} is stale: field {name!r} is absent from "
                    f"the file, candidate expected {diff['old']!r} (use null for absent)"
                )
            inserted.append(name)
            expected[name] = diff["new"]
            continue
        if current[name] != diff["old"]:
            raise ValueError(
                f"candidate {candidate.id!r} is stale: field {name!r} is "
                f"{current[name]!r} in the file, candidate expected {diff['old']!r}"
            )
        expected[name] = diff["new"]
    expected["version"] = current["version"] + 1

    for name in inserted:
        text = _insert_field(text, name, expected[name], candidate.id, path)

    for name in [*candidate.fields, "version"]:
        if name in inserted:
            continue  # written by the insert above; there is no old text to replace
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
