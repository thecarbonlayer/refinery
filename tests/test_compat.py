"""The startup contract: a wrong carbon checkout fails loud, early, helpfully."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loop.compat import PIN_FILE, CarbonBaseError, load_pin, require_carbon_base


def _derived_pin(tmp_path: Path, required: list[list[str]]) -> Path:
    """The real pin with only required_symbols swapped — schema drift can't hide."""
    pin = load_pin()
    pin["required_symbols"] = required
    out = tmp_path / "carbon-base.json"
    out.write_text(json.dumps(pin))
    return out


def test_current_pair_is_compatible():
    assert isinstance(require_carbon_base(), list)


def test_pin_file_schema():
    # Membership/shape only — the branch and commit are allowed to change
    # (promotion will move them); pinning literals would assert "nothing has
    # ever changed".
    pin = load_pin()
    assert PIN_FILE.is_file()
    assert set(pin) >= {"carbon_branch", "carbon_commit", "required_symbols"}
    assert len(pin["carbon_commit"]) == 40
    assert pin["required_symbols"], "an empty required list checks nothing"
    assert all(len(entry) == 2 for entry in pin["required_symbols"])


def test_missing_module_fails_with_remediation(tmp_path):
    pin_file = _derived_pin(tmp_path, [["harness.no_such_module_for_this_test", "X"]])
    with pytest.raises(CarbonBaseError) as excinfo:
        require_carbon_base(pin_file)
    message = str(excinfo.value)
    assert "not the base" in message
    assert "git -C" in message
    assert load_pin()["carbon_branch"] in message


def test_missing_attr_fails_and_names_it(tmp_path):
    pin_file = _derived_pin(tmp_path, [["harness.session_env", "NO_SUCH_ATTR_FOR_TEST"]])
    with pytest.raises(CarbonBaseError) as excinfo:
        require_carbon_base(pin_file)
    assert "harness.session_env.NO_SUCH_ATTR_FOR_TEST" in str(excinfo.value)


def test_missing_pin_file_fails(tmp_path):
    with pytest.raises(CarbonBaseError) as excinfo:
        require_carbon_base(tmp_path / "nope.json")
    assert "missing pin file" in str(excinfo.value)
