"""Surgical config edits: exactly the changed lines, validated through carbon's door."""

import json
import shutil
from pathlib import Path

import pytest

from loop.artifacts import Candidate
from loop.config_edit import (
    apply_candidate,
    config_path,
    immutable_invariants,
    known_knobs,
    proposal_surface,
)
from runner.carbon_env import CARBON_ROOT

REAL_CONFIG = CARBON_ROOT / "harness" / "harness_config.json"


def _bumped() -> int:
    """The version an edit should produce: whatever the real config carries now,
    plus one. Hardcoding it pinned these tests to a config that has since been
    bumped in carbon, so they broke on a change that was not theirs."""
    return json.loads(REAL_CONFIG.read_text())["version"] + 1


def make_candidate(fields, cand_id="cand-x"):
    return Candidate(
        id=cand_id,
        cluster_id="CL-1",
        proposer="Fable",
        proposer_detail="test",
        fields=fields,
        rationale="r",
        expected_effect="e",
        regression_risk="g",
    )


@pytest.fixture
def fake_carbon(tmp_path) -> Path:
    """A copy of the REAL config file in a fake carbon layout — the tests must
    exercise the file the pipeline will actually edit, not a toy fixture."""
    (tmp_path / "harness").mkdir()
    shutil.copy(REAL_CONFIG, tmp_path / "harness" / "harness_config.json")
    return tmp_path


def test_apply_changes_only_the_named_lines(fake_carbon):
    before = config_path(fake_carbon).read_text().splitlines()
    old = json.loads(config_path(fake_carbon).read_text())
    doubled = old["max_tokens"] * 2  # legal, distinct, and derived from disk
    new = apply_candidate(
        fake_carbon, make_candidate({"max_tokens": {"old": old["max_tokens"], "new": doubled}})
    )
    after = config_path(fake_carbon).read_text().splitlines()
    assert new["max_tokens"] == doubled
    assert new["version"] == old["version"] + 1
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(before) == len(after) and len(changed) == 2  # the knob + the version bump
    assert json.loads(config_path(fake_carbon).read_text()) == new


def test_stale_old_value_rejected_and_file_untouched(fake_carbon):
    before = config_path(fake_carbon).read_text()
    with pytest.raises(ValueError, match="stale"):
        apply_candidate(fake_carbon, make_candidate({"max_tokens": {"old": 9999, "new": 8192}}))
    assert config_path(fake_carbon).read_text() == before


def test_unknown_field_rejected_against_carbon_schema(fake_carbon):
    """The field catalogue comes from carbon's config_schema(), not a hardcoded list —
    a knob carbon doesn't declare is rejected with the known-knob set surfaced."""
    with pytest.raises(ValueError, match="no field 'nope' in carbon's config schema"):
        apply_candidate(fake_carbon, make_candidate({"nope": {"old": 1, "new": 2}}))


def test_known_knobs_reflects_carbon_schema():
    """known_knobs is discovered from carbon, so the real editable knobs appear with
    their collection flags — the editor tracks the surface as carbon grows it."""
    knobs = known_knobs()
    assert "max_item_chars" not in knobs
    assert "approval_tools" not in knobs
    assert "require_run" not in knobs
    assert "version" not in knobs
    # The tool_output menu is mid-transition: carbon is adding `offload_to_file`,
    # a strategy that writes the full result to a file instead of dropping the cut
    # bytes, and the E cluster is already wired to measure it (E4). Pinned to the
    # exact menu on either side of that landing, so the suite is green against a
    # carbon that has it and one that does not — and never loosened to a subset
    # check, which would stop reporting a fourth name or a rename, the one thing
    # a menu pin is for.
    assert knobs["tool_output"]["strategies"] in (
        ["head_tail", "keep_head"],
        ["head_tail", "keep_head", "offload_to_file"],
    )
    assert knobs["compaction"]["strategies"] == [
        "structured_checkpoint",
        "summarize_middle",
        "token_budget_checkpoint",
    ]


def test_bounded_strategy_object_is_editable(fake_carbon):
    old = json.loads(config_path(fake_carbon).read_text())["tool_output"]
    changed = {**old, "strategy": "keep_head"}
    new = apply_candidate(
        fake_carbon,
        make_candidate({"tool_output": {"old": old, "new": changed}}),
    )
    assert new["tool_output"] == changed
    assert json.loads(config_path(fake_carbon).read_text())["tool_output"] == changed


def test_proposal_surface_delegates_to_the_same_catalogues():
    """The single-read refactor inlined ``known_knobs()`` and
    ``immutable_invariants()``, so nothing pinned the two code paths together:
    replacing the whole ``immutable`` section with ``{}`` — deleting the proposer's
    entire do-not-propose safety list — left every test green."""
    surface = proposal_surface()
    assert surface["editable"] == known_knobs()
    assert surface["immutable"] == immutable_invariants()
    assert surface["immutable"], "the do-not-propose list must never be empty"


def test_proposal_surface_explicitly_separates_locked_invariants():
    locked = immutable_invariants()
    assert "verification_integrity" in locked
    assert "unique_atomic_edits" in locked
    surface = proposal_surface()
    assert "tool_output" in surface["editable"]
    assert "require_run" in surface["locked_fields"]
    assert "max_item_chars" in surface["locked_fields"]
    assert surface["candidate_kinds"]["correctness_defect"].startswith("Needs a locked Carbon fix")


def test_multiline_list_field_unsupported_by_surgical_edit(fake_carbon):
    """code_extensions spans multiple lines in the real file, so its exact
    `"field": <json>` text never appears — loudly unsupported, not reformatted."""
    old = json.loads(config_path(fake_carbon).read_text())["code_extensions"]
    with pytest.raises(ValueError, match="no field 'code_extensions'"):
        apply_candidate(
            fake_carbon, make_candidate({"code_extensions": {"old": old, "new": [".py"]}})
        )


def test_permission_boundary_field_is_not_editable(fake_carbon):
    """The loop may measure approval behavior but may not weaken the boundary."""
    old = ["bash", "write_file", "edit_file"]
    with pytest.raises(ValueError, match="no field 'approval_tools'"):
        apply_candidate(
            fake_carbon, make_candidate({"approval_tools": {"old": old, "new": ["bash"]}})
        )


# --- optional knobs the config file omits -------------------------------------
#
# carbon lands a knob whose default changes nothing by leaving it OUT of the
# shipped JSON: the file keeps its `config_version`, every external baseline
# pinned to that version stays valid, and the loaded default is today's
# behavior. `tool_exposure` is the live example, and `compaction.prompt_suffix`
# on a sibling carbon branch is the next one. The editor discovers such a knob
# through `known_knobs()` (carbon's manifest) but used to refuse to APPLY it,
# because the file has no line to replace — so the loop could propose a value it
# could never run. These tests pin both shapes of that insert.


def _drop_key(fake_root: Path, field: str) -> dict:
    """Remove a top-level field from the fixture's config, returning the rest.

    Surgical on the TEXT, so the rest of the file keeps its exact formatting —
    the fixture has to look like a real config that never carried the field."""
    path = config_path(fake_root)
    text = path.read_text()
    raw = json.loads(text)
    body = json.dumps(raw[field])
    middle = f'  "{field}": {body},\n'  # any field but the last
    last = f',\n  "{field}": {body}\n'  # the last field carries no trailing comma
    if text.count(middle) == 1:
        text = text.replace(middle, "")
    elif text.count(last) == 1:
        text = text.replace(last, "\n")
    else:
        raise AssertionError(f"{field} is not a single plain line in the fixture")
    path.write_text(text)
    del raw[field]
    return raw


def test_carbon_advertises_tool_exposure_as_an_optional_knob_the_file_omits():
    """The premise the insert path rests on, read from carbon rather than assumed:
    the manifest advertises the knob AND marks it optional, and the shipped config
    really does omit it. If carbon ever writes the field into the file, the insert
    path stops being exercised by the test below and this says so."""
    knob = known_knobs()["tool_exposure"]
    assert knob["optional"] is True
    assert "tool_exposure" not in json.loads(REAL_CONFIG.read_text())


def test_apply_inserts_an_optional_top_level_field_the_file_omits(fake_carbon):
    """THE fix: a candidate may SET a manifest-advertised field the JSON omits.

    `old: None` is how a candidate says "absent from the file" — the stale guard
    still applies, in the only form it can take when there is no current value."""
    before = config_path(fake_carbon).read_text()
    value = {"strategy": "query_match", "k": 8}
    new = apply_candidate(
        fake_carbon,
        make_candidate({"tool_exposure": {"old": None, "new": value}}),
    )
    assert new["tool_exposure"] == value
    assert new["version"] == _bumped()
    on_disk = json.loads(config_path(fake_carbon).read_text())
    assert on_disk == new
    # Exactly one line added, and the diff stays the one-knob story this module
    # exists to produce: the inserted line, the version bump, and (because the
    # anchor was the file's last field) the comma the anchor line had to gain.
    after = config_path(fake_carbon).read_text()
    assert len(after.splitlines()) == len(before.splitlines()) + 1
    inserted = [ln for ln in after.splitlines() if ln.lstrip().startswith('"tool_exposure":')]
    assert len(inserted) == 1
    assert inserted[0] == f'  "tool_exposure": {json.dumps(value)}'


def test_inserted_field_lands_in_carbons_own_serialization_order(fake_carbon):
    """Where the line goes is not cosmetic: the file is read by people reviewing a
    one-line PR diff, and a knob appearing in a random position reads as noise.
    The insert follows carbon's own schema order, so the file stays in the order
    the surface publishes."""
    from carbon import config_schema

    apply_candidate(
        fake_carbon,
        make_candidate({"tool_exposure": {"old": None, "new": {"strategy": "all"}}}),
    )
    order = [f["name"] for f in config_schema()]
    on_disk = [k for k in json.loads(config_path(fake_carbon).read_text()) if k in order]
    assert on_disk == [name for name in order if name in on_disk]


def test_insert_refuses_a_candidate_that_claims_the_field_already_had_a_value(fake_carbon):
    """The stale guard, in its absent-field form: a candidate whose `old` names a
    value is a candidate written against a different file than this one."""
    before = config_path(fake_carbon).read_text()
    with pytest.raises(ValueError, match="stale"):
        apply_candidate(
            fake_carbon,
            make_candidate(
                {"tool_exposure": {"old": {"strategy": "all"}, "new": {"strategy": "all", "k": 3}}}
            ),
        )
    assert config_path(fake_carbon).read_text() == before


def test_insert_is_refused_for_a_field_carbon_does_not_call_optional(fake_carbon):
    """Absence is only an insert opportunity when carbon says the field may be
    absent. A REQUIRED field missing from the file is a corrupt config, and
    quietly writing it back would repair — and hide — that corruption."""
    rest = _drop_key(fake_carbon, "max_tokens")
    assert "max_tokens" not in rest
    assert known_knobs()["max_tokens"].get("optional") is not True
    before = config_path(fake_carbon).read_text()
    with pytest.raises(ValueError, match="no field 'max_tokens'"):
        apply_candidate(fake_carbon, make_candidate({"max_tokens": {"old": None, "new": 8192}}))
    assert config_path(fake_carbon).read_text() == before


def test_apply_adds_an_optional_key_nested_inside_an_object_knob(fake_carbon):
    """The NESTED shape, proven rather than assumed — `compaction.prompt_suffix`'s
    shape on the sibling carbon branch: an optional parameter, absent from the
    serialized object, legal to the loader when present.

    The established candidate contract (every committed candidate in iter-02..06)
    addresses an object knob WHOLE: `old` and `new` are the full object, and the
    file line is replaced in one piece. The stand-in here is
    `compaction.checkpoint_fallback` — optional, defaulted, and accepted by
    carbon's real door — with the fixture edited to omit it, which is exactly the
    state `prompt_suffix` is in on that branch. Nothing here touches that branch;
    the shape is what is under test."""
    path = config_path(fake_carbon)
    raw = json.loads(path.read_text())
    without = {k: v for k, v in raw["compaction"].items() if k != "checkpoint_fallback"}
    assert "checkpoint_fallback" in raw["compaction"], "fixture premise: the key ships present"
    path.write_text(path.read_text().replace(json.dumps(raw["compaction"]), json.dumps(without)))
    current = json.loads(path.read_text())["compaction"]
    assert "checkpoint_fallback" not in current

    gained = {**current, "checkpoint_fallback": "keep_head"}
    new = apply_candidate(
        fake_carbon, make_candidate({"compaction": {"old": current, "new": gained}})
    )
    assert new["compaction"]["checkpoint_fallback"] == "keep_head"
    # Written, parsed back, and accepted by carbon's own load_config (apply_candidate
    # runs that door before writing), with the object still on one line.
    on_disk = config_path(fake_carbon).read_text()
    assert json.loads(on_disk)["compaction"] == gained
    assert len([ln for ln in on_disk.splitlines() if ln.lstrip().startswith('"compaction":')]) == 1


def test_invalid_new_value_rejected_by_carbon_door(fake_carbon):
    """A non-positive count must be caught by carbon's own load_config before
    the file is written — the pipeline can never leave a config the harness
    would refuse to load.

    ``old`` is read from disk, not spelled out: a hardcoded policy dict goes stale
    on 29 of the 30 legal `tool_output` variants, and then this test fails on a
    `stale` error instead of the `positive integer` one it exists to prove. The
    sibling `test_bounded_strategy_object_is_editable` already reads from disk.
    """
    before = config_path(fake_carbon).read_text()
    live = json.loads(before)["tool_output"]
    with pytest.raises(ValueError, match="positive integer"):
        apply_candidate(
            fake_carbon,
            make_candidate({"tool_output": {"old": live, "new": {**live, "budget": -1}}}),
        )
    assert config_path(fake_carbon).read_text() == before
    assert not list(fake_carbon.glob("harness/*.candidate-check"))  # temp check file cleaned up
