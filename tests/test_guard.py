"""The resume-guard: behavior_key is gemma_sha-independent but pins every other
behavior input. Pure/offline — guard.py imports no gemma."""

import pytest

from runner import guard

BASE = {
    "gemma_sha": "sha_A",
    "gemma_dirty": False,
    "dirty_sha": None,
    "config_version": 1,
    "model": "gemma",
    "runner_sha": "runner1",
}


def _fp(**over):
    fp = {**BASE, **over}
    fp["behavior_key"] = guard.fingerprint_behavior_key(fp)
    return fp


def test_behavior_key_ignores_committed_gemma_sha():
    """The whole point: two checkouts differing ONLY in committed gemma_sha share a
    behavior key (an additive release is not a behavior change)."""
    assert _fp(gemma_sha="sha_A")["behavior_key"] == _fp(gemma_sha="sha_B")["behavior_key"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("config_version", 2),
        ("model", "other-model"),
        ("runner_sha", "runner2"),
        ("dirty_sha", "deadbeef"),
    ],
)
def test_behavior_key_moves_on_any_real_behavior_input(field, value):
    assert _fp()["behavior_key"] != _fp(**{field: value})["behavior_key"]


def test_fingerprint_behavior_key_missing_field_is_a_bug_not_a_default():
    with pytest.raises(KeyError):
        guard.fingerprint_behavior_key({"model": "gemma"})  # no config_version/runner_sha


def test_baseline_status_current_when_keys_match():
    prior, current = _fp(gemma_sha="old"), _fp(gemma_sha="new")
    assert guard.baseline_status(prior, current) == "current"


def test_baseline_status_stale_on_real_change():
    assert guard.baseline_status(_fp(), _fp(config_version=2)) == "stale"


def test_baseline_status_stale_when_prior_has_no_behavior_key():
    """A recorded fingerprint predating the guard can't attest its behavior — stale."""
    prior = dict(BASE)  # no behavior_key
    assert guard.baseline_status(prior, _fp()) == "stale"


def test_assert_resumable_noop_on_additive_bump():
    guard.assert_resumable(_fp(gemma_sha="old"), _fp(gemma_sha="new"))  # no raise


def test_assert_resumable_raises_stale_baseline_with_both_keys():
    with pytest.raises(guard.StaleBaseline, match="behavior_key"):
        guard.assert_resumable(_fp(), _fp(model="other"))
