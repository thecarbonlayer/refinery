"""Artifact contract: the proposer (model) writes these files; code validates them."""

import json

import pytest

from loop.artifacts import load_candidates, load_clusters

GOOD_CLUSTER = {
    "id": "CL-1",
    "mechanism": "clamp suffix-drop",
    "tasks": ["A2"],
    "hypothesis": "max_item_chars truncates the tail",
    "evidence": ["reply: 'the log ends abruptly'"],
}

GOOD_CANDIDATE = {
    "id": "cand-raise-clamp-12k",
    "cluster_id": "CL-1",
    "proposer": "Fable",
    "proposer_detail": "claude-fable-5, in-session",
    "fields": {"max_item_chars": {"old": 4000, "new": 12000}},
    "rationale": "needle sits past the clamp",
    "expected_effect": "A2/A4 recover",
    "regression_risk": "window pressure on A1/A3",
}


def write(tmp_path, name, payload):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return p


def test_clusters_round_trip(tmp_path):
    (c,) = load_clusters(write(tmp_path, "c.json", [GOOD_CLUSTER]))
    assert c.id == "CL-1" and c.tasks == ("A2",)


def test_cluster_missing_key_rejected(tmp_path):
    bad = {k: v for k, v in GOOD_CLUSTER.items() if k != "hypothesis"}
    with pytest.raises(ValueError, match="missing key 'hypothesis'"):
        load_clusters(write(tmp_path, "c.json", [bad]))


def test_candidates_round_trip(tmp_path):
    (c,) = load_candidates(write(tmp_path, "k.json", [GOOD_CANDIDATE]))
    assert c.fields["max_item_chars"]["new"] == 12000
    assert c.proposer == "Fable"


def test_candidate_may_not_edit_version(tmp_path):
    bad = dict(GOOD_CANDIDATE, fields={"version": {"old": 1, "new": 7}})
    with pytest.raises(ValueError, match="pipeline owns that field"):
        load_candidates(write(tmp_path, "k.json", [bad]))


def test_candidate_noop_field_rejected(tmp_path):
    bad = dict(GOOD_CANDIDATE, fields={"max_item_chars": {"old": 4000, "new": 4000}})
    with pytest.raises(ValueError, match="no-op"):
        load_candidates(write(tmp_path, "k.json", [bad]))


def test_duplicate_candidate_ids_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_candidates(write(tmp_path, "k.json", [GOOD_CANDIDATE, GOOD_CANDIDATE]))


def test_empty_files_rejected(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        load_candidates(write(tmp_path, "k.json", []))
    with pytest.raises(ValueError, match="non-empty"):
        load_clusters(write(tmp_path, "c.json", []))
