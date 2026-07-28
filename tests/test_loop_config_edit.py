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
    assert knobs["tool_output"]["strategies"] == ["head_tail", "keep_head"]
    assert knobs["compaction"]["strategies"] == [
        "structured_checkpoint",
        "summarize_middle",
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
