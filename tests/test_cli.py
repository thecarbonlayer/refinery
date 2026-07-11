from runner.cli import validate_only
from runner.tasks import TASKS


def test_validate_only_accepts_known_names():
    assert validate_only(["A1", "B2"], TASKS) == []


def test_validate_only_reports_unknown_names_sorted():
    """A typo in --only must error up front, not silently run zero attempts of
    the intended task (foot-gun (a))."""
    assert validate_only(["A1", "Z9", "b1"], TASKS) == ["Z9", "b1"]
