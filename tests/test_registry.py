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


def test_d3_body_ground_truth():
    from runner.tasks.cluster_d import D3_COUNT, _d3_body

    body = _d3_body()
    assert body.count("TODO") == D3_COUNT
    assert sum(1 for ln in body.splitlines() if "TODO" in ln) == D3_COUNT  # one per line


def test_d3_reply_oracle():
    from runner.tasks.cluster_d import _last_int

    assert _last_int("23") == "23"
    assert _last_int("The count is 23.") == "23"
    # last-int semantics: a trailing quoted line number would (correctly) fail vs 23
    assert _last_int("I found 22; line 123 was not a TODO") == "123"
    assert _last_int("no numbers here") is None


def test_d3_body_never_contains_answer_token():
    from runner.tasks.cluster_d import D3_COUNT, _d3_body

    assert str(D3_COUNT) not in _d3_body()
