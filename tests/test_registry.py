from runner.spec import ATTEMPTS
from runner.tasks import TASKS


def test_registry_shape():
    names = [t.name for t in TASKS]
    assert len(names) == len(set(names)), "duplicate task names"
    for t in TASKS:
        assert t.split in ATTEMPTS
        assert t.cluster in "ABCD"
        assert t.expected_baseline in ("pass", "fail", "uncertain")


# Once all clusters land this pins the full suite; update the sets per cluster task.
def test_registry_membership():
    names = {t.name for t in TASKS}
    assert {"D1", "D2", "D3"} <= names
