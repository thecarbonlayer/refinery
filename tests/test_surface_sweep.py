"""The sweep is the gate's newest veto, so its own mechanics are pinned here.

Every test injects ``run_suites``. The real one shells out to pytest, and a test that
reached it would spawn nested full runs — the recursion ``run_harness_gates`` already
refuses by name.
"""

from __future__ import annotations

import json

import pytest

from loop.config_edit import config_path
from loop.surface_sweep import enumerate_points, sweep

CARBON_REL = "harness/harness_config.json"


# Every field these tests name a value of, pinned to a KNOWN value. Copying the live
# config instead makes each assertion conditional on what happens to ship: the sweep
# skips the shipped value, so `tool_output.strategy=keep_head` is missing from the
# labels exactly when carbon ships keep_head, and this file reddens under five legal
# values. That is the defect this module exists to catch, reproduced one layer up —
# and it is what the first clean run of the sweep caught these tests doing.
_PINNED = {
    '"tool_output": ': '{"strategy": "head_tail", "budget": 4000, "tail_fraction": 0.6}',
    '"retry": ': '{"strategy": "backoff", "max_attempts": 5, "base_delay_ms": 2000}',
    '"compaction": ': (
        '{"strategy": "token_budget_checkpoint", "keep_head": 2, "keep_tail": 4, '
        '"trigger_fraction": 0.8, "summary_max_tokens": 1024}'
    ),
}


@pytest.fixture
def fake_carbon(tmp_path):
    """A carbon-shaped tree holding a config whose SHAPE is real and whose values are
    fixed, so the points are the real points and the assertions are not hostage to the
    shipped value. Edited as text, one field per line, because that is the contract the
    surgical editor and the sweep both hold to."""
    import re

    from runner.carbon_env import CARBON_ROOT

    text = config_path(CARBON_ROOT).read_text()
    for key, value in _PINNED.items():
        text, n = re.subn(rf"{re.escape(key)}\{{[^\n]*\}}", key + value, text)
        assert n == 1, f"{key} did not appear exactly once as a single line"
    root = tmp_path / "carbon"
    (root / "harness").mkdir(parents=True)
    (root / CARBON_REL).write_text(text)
    return root


def _all_green(_carbon, _editor):
    return True, {"carbon": {"passed": True}, "refinery": {"passed": True}}


def test_points_are_derived_from_carbons_schema_not_listed_here(fake_carbon):
    """The sweep must cover a knob the day carbon publishes it.

    Hardcoding the menu here would reproduce the defect the sweep exists to catch, one
    layer up: the surface would grow a value and the sweep would keep reporting green
    over a value it never tried.
    """
    current = json.loads((fake_carbon / CARBON_REL).read_text())
    labels = {p.label for p in enumerate_points(current)}
    for strategy in ("keep_head", "offload_to_file"):
        assert f"tool_output.strategy={strategy}" in labels
    assert "compaction.strategy=summarize_middle" in labels
    # Both retry parameters, as a cross product: `fail_fast` makes exactly one provider
    # call whatever `max_attempts` says, and that interaction is what a pinned test
    # missed. Sweeping either axis alone would not have found it.
    assert {"retry=fail_fast/1", "retry=fail_fast/5", "retry=backoff/1"} <= labels


def test_no_point_probes_a_value_the_surface_forbids(fake_carbon):
    """Every probe must survive carbon's own validation door.

    Two premise probes were once built on a `tail_fraction` of 0.0/1.0, which the
    surface never permitted; they passed until carbon grew a door to say so. A sweep
    generating illegal points would fail the whole harness on values no candidate
    could ever propose.
    """
    from harness.harness_config import load_config

    path = fake_carbon / CARBON_REL
    original = path.read_text()
    current = json.loads(original)
    for point in enumerate_points(current):
        raw = dict(current)
        raw[point.field] = point.value
        path.write_text(json.dumps(raw, indent=2) + "\n")
        load_config(path)  # raises if the point is not a legal config
    path.write_text(original)


def test_the_shipped_value_is_not_reprobed(fake_carbon):
    """The gate's plain run already covers it; a duplicate would double the cost and
    report the same verdict twice."""
    current = json.loads((fake_carbon / CARBON_REL).read_text())
    for point in enumerate_points(current):
        assert point.value != current[point.field], f"{point.label} re-probes the shipped value"


def test_a_red_point_fails_the_sweep_and_names_the_value(fake_carbon):
    """A red must identify the VALUE, not just the suite. "refinery_pytest failed"
    sends someone hunting a break that is not there; "retry=fail_fast/5 is red" says
    which test pinned which knob."""
    calls: list[str] = []

    def run_suites(carbon_root, _editor):
        applied = json.loads((carbon_root / CARBON_REL).read_text())
        calls.append(applied["retry"]["strategy"])
        if applied["retry"]["strategy"] == "fail_fast":
            return False, {
                "carbon": {"passed": False, "failed": ["FAILED tests/test_x.py::test_y"]},
                "refinery": {"passed": True},
            }
        return _all_green(carbon_root, _editor)

    report = sweep(fake_carbon, fake_carbon, run_suites=run_suites, log=lambda _: None)

    assert not report["passed"]
    reds = {label for label, point in report["points"].items() if not point["passed"]}
    assert reds == {"retry=fail_fast/1", "retry=fail_fast/5"}
    assert "fail_fast" in calls, "the point was never actually written to the config"


def test_the_config_is_restored_even_when_a_point_explodes(fake_carbon):
    """The sweep runs with a candidate already applied to the working tree, so it
    restores the exact bytes it read rather than checking out — a checkout would
    discard the candidate and leave the caller measuring the committed config."""
    path = fake_carbon / CARBON_REL
    # Stand in for an applied candidate: a value that is NOT what git holds.
    text = path.read_text().replace('"max_tool_steps": 20', '"max_tool_steps": 17')
    assert '"max_tool_steps": 17' in text
    path.write_text(text)

    def explode(_carbon, _editor):
        raise RuntimeError("suite runner died")

    with pytest.raises(RuntimeError, match="suite runner died"):
        sweep(fake_carbon, fake_carbon, run_suites=explode, log=lambda _: None)

    assert path.read_text() == text, "the sweep did not restore the candidate's config"


def test_green_everywhere_reports_every_point_it_probed(fake_carbon):
    """ "Passed" has to carry its denominator. A sweep that silently probed zero points
    would report exactly the same verdict as one that probed all of them."""
    report = sweep(fake_carbon, fake_carbon, run_suites=_all_green, log=lambda _: None)

    assert report["passed"]
    assert report["probed"] == len(report["points"]) > 0
    assert all(point["passed"] for point in report["points"].values())
