"""Surgical config edits: exactly the changed lines, validated through gemma's door."""

import json
import shutil
from pathlib import Path

import pytest

from loop.artifacts import Candidate
from loop.config_edit import apply_candidate, config_path
from runner.gemma_env import GEMMA_ROOT

REAL_CONFIG = GEMMA_ROOT / "harness" / "harness_config.json"


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
def fake_gemma(tmp_path) -> Path:
    """A copy of the REAL config file in a fake gemma layout — the tests must
    exercise the file the pipeline will actually edit, not a toy fixture."""
    (tmp_path / "harness").mkdir()
    shutil.copy(REAL_CONFIG, tmp_path / "harness" / "harness_config.json")
    return tmp_path


def test_apply_changes_only_the_named_lines(fake_gemma):
    before = config_path(fake_gemma).read_text().splitlines()
    old = json.loads(config_path(fake_gemma).read_text())
    new = apply_candidate(
        fake_gemma, make_candidate({"max_item_chars": {"old": 4000, "new": 12000}})
    )
    after = config_path(fake_gemma).read_text().splitlines()
    assert new["max_item_chars"] == 12000
    assert new["version"] == old["version"] + 1
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(before) == len(after) and len(changed) == 2  # the knob + the version bump
    assert json.loads(config_path(fake_gemma).read_text()) == new


def test_stale_old_value_rejected_and_file_untouched(fake_gemma):
    before = config_path(fake_gemma).read_text()
    with pytest.raises(ValueError, match="stale"):
        apply_candidate(fake_gemma, make_candidate({"max_item_chars": {"old": 9999, "new": 12000}}))
    assert config_path(fake_gemma).read_text() == before


def test_unknown_field_rejected(fake_gemma):
    with pytest.raises(ValueError, match="no field 'nope'"):
        apply_candidate(fake_gemma, make_candidate({"nope": {"old": 1, "new": 2}}))


def test_multiline_list_field_unsupported_by_surgical_edit(fake_gemma):
    """code_extensions spans multiple lines in the real file, so its exact
    `"field": <json>` text never appears — loudly unsupported, not reformatted."""
    old = json.loads(config_path(fake_gemma).read_text())["code_extensions"]
    with pytest.raises(ValueError, match="cannot surgically edit"):
        apply_candidate(
            fake_gemma, make_candidate({"code_extensions": {"old": old, "new": [".py"]}})
        )


def test_single_line_list_field_is_editable(fake_gemma):
    """approval_tools sits on one line in the real file, so the surgical
    replacement handles it — the C-cluster knob stays genuinely editable."""
    old = ["bash", "write_file", "edit_file"]
    new = apply_candidate(
        fake_gemma, make_candidate({"approval_tools": {"old": old, "new": ["bash"]}})
    )
    assert json.loads(config_path(fake_gemma).read_text())["approval_tools"] == ["bash"]
    assert new["version"] == 2


def test_invalid_new_value_rejected_by_gemma_door(fake_gemma):
    """A non-positive count must be caught by gemma's own load_config before
    the file is written — the pipeline can never leave a config the harness
    would refuse to load."""
    before = config_path(fake_gemma).read_text()
    with pytest.raises(ValueError, match="positive integer"):
        apply_candidate(fake_gemma, make_candidate({"max_item_chars": {"old": 4000, "new": -1}}))
    assert config_path(fake_gemma).read_text() == before
    assert not list(fake_gemma.glob("harness/*.candidate-check"))  # temp check file cleaned up
