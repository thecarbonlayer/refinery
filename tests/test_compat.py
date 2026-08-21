"""The startup contract: a wrong carbon checkout fails loud, early, helpfully."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
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


def _import_of_loop_compat_raises(exc: BaseException):
    """A drop-in for builtins.__import__ that fails only for ``loop.compat``.

    This is exactly the failure shape of an incompatible pair: the guard in
    ``loop/__init__.py`` raises while ``import loop.compat`` executes, so the
    statement never returns. Everything else imports normally.
    """
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "loop.compat":
            raise exc
        return real_import(name, *args, **kwargs)

    return failing_import


_STUB_MARKER = "stub remediation: carbon checkout is not the pinned base"


def _write_wrong_base_env(tmp_path: Path) -> None:
    """A temp pytest rootdir simulating an incompatible pair with the REAL hook.

    A stub ``loop`` package mirrors the real failure shape (``__init__.py``
    imports ``loop.compat`` successfully, then the guard call raises), and a
    stub conftest re-exports every ``pytest_``-named hook from refinery's real
    ``tests/conftest.py`` — by pattern, not by name, so this environment keeps
    testing the real guard even if it moves between hooks. The stub shadows
    the real ``loop`` because pytest inserts the rootdir (the temp dir) at the
    front of sys.path when loading the stub conftest.
    """
    from loop.compat import CARBON_BASE_EXIT_CODE

    stub = tmp_path / "loop"
    stub.mkdir()
    (stub / "compat.py").write_text(
        f"CARBON_BASE_EXIT_CODE = {CARBON_BASE_EXIT_CODE}\n"
        "\n"
        "\n"
        "class CarbonBaseError(RuntimeError):\n"
        "    pass\n"
        "\n"
        "\n"
        "def require_carbon_base():\n"
        f"    raise CarbonBaseError({_STUB_MARKER!r})\n"
    )
    (stub / "__init__.py").write_text(
        "from loop.compat import require_carbon_base as _require_carbon_base\n"
        "\n"
        "for _warning in _require_carbon_base():\n"
        "    pass\n"
    )
    (tmp_path / "conftest.py").write_text(
        "import tests.conftest as _real_conftest\n"
        "\n"
        "for _name in dir(_real_conftest):\n"
        '    if _name.startswith("pytest_") and callable(getattr(_real_conftest, _name)):\n'
        "        globals()[_name] = getattr(_real_conftest, _name)\n"
    )
    (tmp_path / "test_never_collected.py").write_text(
        "def test_never_runs():\n"
        "    raise AssertionError('the guard should have aborted before collection')\n"
    )


def _run_pytest_in_wrong_base_env(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(PIN_FILE.parent)}
    return subprocess.run(
        [sys.executable, "-m", "pytest", *args, "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )


def test_wrong_base_run_exits_with_designed_code_not_internalerror(tmp_path):
    """The process-level exit contract, end to end through a real pytest run.

    The run must terminate with CARBON_BASE_EXIT_CODE and the remediation
    text on the output — not pytest's INTERNALERROR (exit 3), which is what
    an UnboundLocalError inside the handler used to produce. Cheap despite
    the subprocess: the guard aborts before collection.
    """
    from loop.compat import CARBON_BASE_EXIT_CODE

    _write_wrong_base_env(tmp_path)
    proc = _run_pytest_in_wrong_base_env(tmp_path, str(tmp_path))
    output = proc.stdout + proc.stderr
    assert proc.returncode == CARBON_BASE_EXIT_CODE, output
    assert _STUB_MARKER in output
    assert "INTERNALERROR" not in output
    assert "UnboundLocalError" not in output


def test_wrong_base_help_still_prints_help_cleanly(tmp_path):
    """``pytest --help`` must keep working on an incompatible pair.

    ``--help`` (and ``--markers``) run ``config._do_configure()`` OUTSIDE
    ``wrap_session`` — the only place pytest.exit's returncode becomes the
    process status. A guard raising pytest.exit from pytest_configure
    therefore escaped these commands as a raw ``_pytest.outcomes.Exit``
    traceback, exit 1: neither the designed exit code nor working help. The
    guard belongs in pytest_sessionstart, which every session-running
    invocation passes through and informational commands never reach —
    ``--help`` touches nothing from the carbon pair, so it must simply print
    help and succeed.
    """
    _write_wrong_base_env(tmp_path)
    proc = _run_pytest_in_wrong_base_env(tmp_path, "--help")
    output = proc.stdout + proc.stderr
    assert proc.returncode == 0, output
    assert "usage:" in proc.stdout
    # Not the bare word "Traceback": pytest's own help text describes the
    # --tb option with it. The escape's signature is the traceback header.
    assert "Traceback (most recent call last)" not in output
    assert "INTERNALERROR" not in output


def test_carbon_base_error_at_import_aborts_with_designed_exit_code(monkeypatch):
    """The handler itself: a genuine CarbonBaseError raised by the import must
    become pytest.exit with CARBON_BASE_EXIT_CODE and the remediation text —
    not an UnboundLocalError because the constant's own import never ran."""
    from loop.compat import CARBON_BASE_EXIT_CODE
    from tests.conftest import pytest_sessionstart

    probe = CarbonBaseError("not the base (in-process probe)")
    monkeypatch.setattr(builtins, "__import__", _import_of_loop_compat_raises(probe))
    with pytest.raises(pytest.exit.Exception) as excinfo:
        pytest_sessionstart(session=None)
    assert excinfo.value.returncode == CARBON_BASE_EXIT_CODE
    assert "not the base (in-process probe)" in str(excinfo.value)


def test_same_named_error_from_elsewhere_is_not_swallowed(monkeypatch):
    """A RuntimeError subclass merely NAMED CarbonBaseError, defined anywhere
    but loop.compat, must re-raise with its real traceback — never be folded
    into the guard's clean exit."""
    from tests.conftest import pytest_sessionstart

    class FakeCarbonBaseError(RuntimeError):
        pass

    FakeCarbonBaseError.__name__ = "CarbonBaseError"  # same name, foreign module
    probe = FakeCarbonBaseError("imposter")
    monkeypatch.setattr(builtins, "__import__", _import_of_loop_compat_raises(probe))
    with pytest.raises(FakeCarbonBaseError, match="imposter"):
        pytest_sessionstart(session=None)


def test_unrelated_runtime_error_is_not_swallowed(monkeypatch):
    """A plain RuntimeError from the import is a genuine bug, not a wrong
    checkout; it must re-raise, not exit clean."""
    from tests.conftest import pytest_sessionstart

    probe = RuntimeError("a genuine bug elsewhere in the import")
    monkeypatch.setattr(builtins, "__import__", _import_of_loop_compat_raises(probe))
    with pytest.raises(RuntimeError, match="a genuine bug"):
        pytest_sessionstart(session=None)


def test_unrecoverable_exit_code_reraises_rather_than_guessing(monkeypatch):
    """If loop.compat is somehow NOT in sys.modules after the failure, the
    designed exit code cannot be recovered — the error must re-raise with its
    real traceback rather than exit with a guessed or wrong code."""
    from tests.conftest import pytest_sessionstart

    probe = CarbonBaseError("no module left to read the code from")
    monkeypatch.setattr(builtins, "__import__", _import_of_loop_compat_raises(probe))
    monkeypatch.delitem(sys.modules, "loop.compat")
    with pytest.raises(CarbonBaseError, match="no module left"):
        pytest_sessionstart(session=None)


def test_carbon_base_exit_code_is_distinct():
    """Exit code 7 is distinct from pytest's reserved 0-5 to avoid misread failures."""
    from loop.compat import CARBON_BASE_EXIT_CODE

    assert CARBON_BASE_EXIT_CODE == 7, "Must be 7 to avoid pytest's USAGE_ERROR (code 4)"
    assert CARBON_BASE_EXIT_CODE not in range(6), (
        "Must not collide with pytest's reserved codes (0-5)"
    )
