"""Fixed artifacts of the mining and proposal steps.

Mining and proposing are done by the proposer model as reasoning, not by code
in this package — but each step's output is written to a JSON file with the
shapes below, so validation and the PR pipeline consume a frozen artifact
instead of re-deriving (or quietly reshaping) the analysis each time. A future
API-driven proposer must produce these same files; nothing downstream cares
who authored a candidate, only whether it passes the acceptance rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# The pipeline owns the version bump — a candidate that edited `version`
# directly could collide with it or smuggle a fake rollback point.
FORBIDDEN_FIELDS = {"version"}


@dataclass(frozen=True)
class Cluster:
    """One mined failure cluster: a recurring mechanism, not a surface symptom."""

    id: str
    mechanism: str  # root-cause mechanism, one line
    tasks: tuple[str, ...]  # suite tasks exhibiting it (held-in traces mined)
    hypothesis: str  # what the miner believes is causing it
    evidence: tuple[str, ...]  # example failing traces (quoted from the JSONL)


@dataclass(frozen=True)
class Candidate:
    """One proposed edit to the editable surface (harness_config.json)."""

    id: str
    cluster_id: str
    proposer: str  # short provenance name, e.g. "Fable" -> "Fable-proposed, ..."
    proposer_detail: str  # e.g. "claude-fable-5, in-session"
    fields: dict[str, dict]  # field -> {"old": <json>, "new": <json>}
    rationale: str
    expected_effect: str
    regression_risk: str


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"artifact: {msg}")


def load_clusters(path: str | Path) -> list[Cluster]:
    raw = json.loads(Path(path).read_text())
    _require(isinstance(raw, list) and raw, "clusters file must be a non-empty JSON array")
    clusters = []
    for i, c in enumerate(raw):
        for key in ("id", "mechanism", "tasks", "hypothesis", "evidence"):
            _require(key in c, f"cluster #{i} missing key {key!r}")
        _require(bool(c["tasks"]), f"cluster {c['id']!r} names no tasks")
        clusters.append(
            Cluster(
                id=c["id"],
                mechanism=c["mechanism"],
                tasks=tuple(c["tasks"]),
                hypothesis=c["hypothesis"],
                evidence=tuple(c["evidence"]),
            )
        )
    _require(len({c.id for c in clusters}) == len(clusters), "duplicate cluster ids")
    return clusters


def load_candidates(path: str | Path) -> list[Candidate]:
    raw = json.loads(Path(path).read_text())
    _require(isinstance(raw, list) and raw, "candidates file must be a non-empty JSON array")
    cands = []
    for i, c in enumerate(raw):
        for key in (
            "id",
            "cluster_id",
            "proposer",
            "proposer_detail",
            "fields",
            "rationale",
            "expected_effect",
            "regression_risk",
        ):
            _require(key in c, f"candidate #{i} missing key {key!r}")
        _require(bool(c["fields"]), f"candidate {c['id']!r} changes no fields")
        for name, diff in c["fields"].items():
            _require(
                name not in FORBIDDEN_FIELDS,
                f"candidate {c['id']!r} edits {name!r} — the pipeline owns that field",
            )
            _require(
                isinstance(diff, dict) and set(diff) == {"old", "new"},
                f"candidate {c['id']!r} field {name!r} must be {{'old': ..., 'new': ...}}",
            )
            _require(
                diff["old"] != diff["new"],
                f"candidate {c['id']!r} field {name!r} is a no-op (old == new)",
            )
        cands.append(
            Candidate(
                id=c["id"],
                cluster_id=c["cluster_id"],
                proposer=c["proposer"],
                proposer_detail=c["proposer_detail"],
                fields=c["fields"],
                rationale=c["rationale"],
                expected_effect=c["expected_effect"],
                regression_risk=c["regression_risk"],
            )
        )
    _require(len({c.id for c in cands}) == len(cands), "duplicate candidate ids")
    return cands


@dataclass(frozen=True)
class ValidationRecord:
    """The outcome of validating one candidate — accepted or not, it is kept.

    Rejected candidates never touch the dist/gemma repo, but their record (and
    the runner's results JSON it points at) is part of the iteration's honest
    history: most candidates are EXPECTED to fail acceptance."""

    candidate_id: str
    label: str  # runner results label (results/<label>.json)
    accepted: bool
    delta_in: float
    delta_ho: float
    per_task: dict[str, float] = field(default_factory=dict)
    baseline_fingerprint: dict = field(default_factory=dict)
    candidate_fingerprint: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "accepted": self.accepted,
            "delta_in": self.delta_in,
            "delta_ho": self.delta_ho,
            "per_task": self.per_task,
            "baseline_fingerprint": self.baseline_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
        }
