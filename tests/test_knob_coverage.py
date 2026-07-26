"""No editable knob enters the loop without tasks capable of observing it."""

from loop.config_edit import known_knobs
from runner.knob_coverage import KNOB_COVERAGE
from runner.tasks import TASKS


def test_every_editable_knob_has_miner_and_guard_coverage():
    assert set(KNOB_COVERAGE) == set(known_knobs())
    for knob, coverage in KNOB_COVERAGE.items():
        assert coverage["miners"], f"{knob} has no miner"
        assert coverage["guards"], f"{knob} has no guard"


def test_coverage_names_real_tasks_or_full_suite():
    names = {task.name for task in TASKS}
    referenced = {
        task
        for coverage in KNOB_COVERAGE.values()
        for group in ("miners", "guards")
        for task in coverage[group]
        if task != "*"
    }
    assert referenced <= names
