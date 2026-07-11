import argparse

import pytest

from runner.cli import positive_int, validate_only
from runner.tasks import TASKS


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
