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
    # Shape enforcement (keys, non-empty symbols, entry arity, commit shape)
    # lives in load_pin() itself now and is covered by the failure-mode
    # tests below; this just confirms the committed pin actually loads.
    load_pin()
    assert PIN_FILE.is_file()


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


def test_empty_required_symbols_rejected(tmp_path):
    pin_file = _derived_pin(tmp_path, [])
    with pytest.raises(CarbonBaseError) as excinfo:
        load_pin(pin_file)
    message = str(excinfo.value)
    assert str(pin_file) in message
    assert "required_symbols" in message


def test_malformed_entry_wrong_arity_rejected(tmp_path):
    pin_file = _derived_pin(tmp_path, [["a", "b", "c"]])
    with pytest.raises(CarbonBaseError) as excinfo:
        load_pin(pin_file)
    message = str(excinfo.value)
    assert str(pin_file) in message
    assert "required_symbols" in message


def test_missing_required_key_rejected(tmp_path):
    pin = load_pin()
    del pin["carbon_commit"]
    pin_file = tmp_path / "carbon-base.json"
    pin_file.write_text(json.dumps(pin))
    with pytest.raises(CarbonBaseError) as excinfo:
        load_pin(pin_file)
    message = str(excinfo.value)
    assert str(pin_file) in message
    assert "carbon_commit" in message


def test_broken_json_rejected(tmp_path):
    pin_file = tmp_path / "carbon-base.json"
    pin_file.write_text('{"carbon_branch":')
    with pytest.raises(CarbonBaseError) as excinfo:
        load_pin(pin_file)
    assert "unreadable pin file" in str(excinfo.value)


def test_empty_file_rejected(tmp_path):
    pin_file = tmp_path / "carbon-base.json"
    pin_file.write_text("")
    with pytest.raises(CarbonBaseError) as excinfo:
        load_pin(pin_file)
    assert "unreadable pin file" in str(excinfo.value)


def test_non_dict_top_level_rejected(tmp_path):
    pin_file = tmp_path / "carbon-base.json"
    pin_file.write_text(json.dumps([1, 2]))
    with pytest.raises(CarbonBaseError):
        load_pin(pin_file)


def test_non_string_carbon_commit_rejected(tmp_path):
    pin = load_pin()
    pin["carbon_commit"] = 12345
    pin_file = tmp_path / "carbon-base.json"
    pin_file.write_text(json.dumps(pin))
    with pytest.raises(CarbonBaseError):
        load_pin(pin_file)


def test_junk_module_name_folds_into_carbon_base_error(tmp_path):
    pin_file = _derived_pin(tmp_path, [["", "X"]])
    with pytest.raises(CarbonBaseError) as excinfo:
        require_carbon_base(pin_file)
    assert "import failed" in str(excinfo.value)


def test_getattr_exception_folds_into_attribute_check_failed(tmp_path, monkeypatch):
    # PEP 562: a module-level __getattr__ can raise on any attribute access.
    # hasattr() only swallows AttributeError, so anything else must be caught
    # explicitly or it escapes require_carbon_base raw.
    (tmp_path / "pep562_boom_mod.py").write_text(
        "def __getattr__(name):\n    raise ValueError('boom')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    pin_file = _derived_pin(tmp_path, [["pep562_boom_mod", "whatever"]])
    with pytest.raises(CarbonBaseError) as excinfo:
        require_carbon_base(pin_file)
    assert "attribute check failed" in str(excinfo.value)


def test_dependency_import_failure_is_not_read_as_module_missing(tmp_path, monkeypatch):
    # The module itself exists; its own import statement fails on a
    # dependency. That must not read the same as the module being absent.
    (tmp_path / "depfail_mod.py").write_text("import no_such_dependency_xyz\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    pin_file = _derived_pin(tmp_path, [["depfail_mod", "X"]])
    with pytest.raises(CarbonBaseError) as excinfo:
        require_carbon_base(pin_file)
    message = str(excinfo.value)
    assert "import failed" in message
    assert "module missing" not in message


def test_empty_carbon_branch_rejected(tmp_path):
    pin = load_pin()
    pin["carbon_branch"] = ""
    pin_file = tmp_path / "carbon-base.json"
    pin_file.write_text(json.dumps(pin))
    with pytest.raises(CarbonBaseError) as excinfo:
        load_pin(pin_file)
    assert "carbon_branch" in str(excinfo.value)


def test_malformed_carbon_commit_rejected(tmp_path):
    pin = load_pin()
    pin["carbon_commit"] = "abc"
    pin_file = tmp_path / "carbon-base.json"
    pin_file.write_text(json.dumps(pin))
    with pytest.raises(CarbonBaseError) as excinfo:
        load_pin(pin_file)
    assert "carbon_commit" in str(excinfo.value)
