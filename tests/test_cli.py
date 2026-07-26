import argparse
import json

import pytest

from runner import guard
from runner.cli import check, positive_int, validate_only
from runner.tasks import TASKS


def _fp(**over):
    fp = {
        "gemma_sha": "abc123",
        "gemma_dirty": False,
        "dirty_sha": None,
        "config_version": 1,
        "model": "carbon",
        "runner_sha": "runner1",
        **over,
    }
    fp["behavior_key"] = guard.fingerprint_behavior_key(fp)
    return fp


def _write_baseline(results_dir, label, fingerprint):
    (results_dir / f"{label}.json").write_text(
        json.dumps({"fingerprint": fingerprint, "tasks": {}, "summary": {}})
    )


def _patch_check(monkeypatch, tmp_path, current_fp):
    import runner.carbon_env as ge
    import runner.suite as suite_mod

    monkeypatch.setattr(suite_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ge, "carbon_fingerprint", lambda: current_fp)


def test_check_returns_0_when_baseline_current(tmp_path, monkeypatch, capsys):
    """An additive carbon bump (only gemma_sha moved) still resumes -> exit 0."""
    _write_baseline(tmp_path, "base", _fp(gemma_sha="OLD"))
    _patch_check(monkeypatch, tmp_path, _fp(gemma_sha="NEW"))
    assert check("base") == 0
    assert "CURRENT" in capsys.readouterr().out


def test_check_returns_1_when_baseline_stale(tmp_path, monkeypatch, capsys):
    _write_baseline(tmp_path, "base", _fp())
    _patch_check(monkeypatch, tmp_path, _fp(config_version=2))
    assert check("base") == 1
    assert "STALE" in capsys.readouterr().out


def test_check_returns_1_when_no_such_baseline(tmp_path, monkeypatch, capsys):
    _patch_check(monkeypatch, tmp_path, _fp())
    assert check("missing") == 1
    assert "no such baseline" in capsys.readouterr().out


def test_validate_only_accepts_known_names():
    assert validate_only(["A1", "B2"], TASKS) == []


def test_validate_only_reports_unknown_names_sorted():
    """A typo in --only must error up front, not silently run zero attempts of
    the intended task (foot-gun (a))."""
    assert validate_only(["A1", "Z9", "b1"], TASKS) == ["Z9", "b1"]


def test_positive_int_accepts_positive_values():
    assert positive_int("1") == 1
    assert positive_int("5") == 5


def test_positive_int_rejects_zero_and_negative():
    """--attempts 0 would silently fall back to defaults (`attempts or
    spec.attempts`), and a negative would run zero attempts — both must be
    rejected at parse time."""
    with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
        positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError, match=">= 1"):
        positive_int("-3")
