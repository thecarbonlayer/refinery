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
    kind: str = "configuration_candidate"


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
        kind = c.get("kind", "configuration_candidate")
        _require(
            kind == "configuration_candidate",
            f"candidate {c['id']!r} has kind {kind!r}; only configuration_candidate "
            "artifacts are executable (record strategy gaps and correctness defects as clusters)",
        )
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
                kind=kind,
            )
        )
    _require(len({c.id for c in cands}) == len(cands), "duplicate candidate ids")
    return cands


@dataclass(frozen=True)
class ValidationRecord:
    """The outcome of validating one candidate — accepted or not, it is kept.

    Rejected candidates never touch the carbon repo, but their record (and
    the runner's results JSON it points at) is part of the iteration's honest
    history: most candidates are EXPECTED to fail acceptance."""

    candidate_id: str
    label: str  # runner results label (results/<label>.json)
    accepted: bool
    delta_in: float
    delta_ho: float
    # The candidate's own {field: {old, new}} edit, as validated -- additive (None on
    # a record written, or loaded, before this existed). Lets a later confirmation
    # verify it is confirming the SAME edit the first decision was about, not merely
    # a candidate that happens to share the same id (`loop.cli._check_candidate_
    # identity`). None is a legitimate value, never coerced to `{}`: an absent field
    # means "nothing to compare", which is a different fact than "compared and empty".
    candidate_fields: dict | None = None
    per_task: dict[str, float] = field(default_factory=dict)
    aggregate_accepted: bool | None = None
    regressions: dict[str, float] = field(default_factory=dict)
    catastrophic_regressions: dict[str, float] = field(default_factory=dict)
    baseline_metrics: dict[str, float] = field(default_factory=dict)
    candidate_metrics: dict[str, float] = field(default_factory=dict)
    metric_delta: dict[str, float] = field(default_factory=dict)
    # Metrics one side did not measure, and metrics whose contributing-task count
    # differs between the two runs. Both make a mean unsafe to read as a
    # like-for-like comparison, so neither may be silently dropped.
    metric_not_compared: list[str] = field(default_factory=list)
    metric_task_counts: dict[str, dict] = field(default_factory=dict)
    metric_attempt_counts: dict[str, dict] = field(default_factory=dict)
    metric_denominator_drift: list[str] = field(default_factory=list)
    baseline_fingerprint: dict = field(default_factory=dict)
    candidate_fingerprint: dict = field(default_factory=dict)
    # Harness-suite veto, recorded whether it fired or not. A candidate is a config
    # value, and a config value can break BOTH repos' own tests without moving a
    # single task score — the suite measures behaviour on tasks, not whether the
    # harness still holds together. Three such breakages shipped unnoticed before
    # this existed, so the outcome is part of the record rather than a thing someone
    # is trusted to have checked. Empty dict on records written before the gate.
    gates: dict = field(default_factory=dict)
    # Which per-task movements the edited knob can actually reach, derived from the
    # attempt logs rather than an authored table. A delta on a task the knob cannot
    # touch is grader variance, not an effect: two tasks whose agents have no tool
    # registry at all supplied -1.33 of the -2.00 that made iteration 3's Δ_in
    # negative, against a candidate that edited `tool_output`. Recorded beside the
    # verdict, never subtracted from it — see `coverage_note` for why not.
    coverage: dict = field(default_factory=dict)
    # Acceptance with impossible attributions removed, and the raw verdict kept inside
    # it as evidence. `accepted` above is the CAUSAL one: recording the split beside a
    # verdict the noise still decided left iteration 3's failure in place.
    causal: dict = field(default_factory=dict)
    # The three-outcome rule's disposition — applied and decisive for calibrated
    # sections (tool_output today), or a stated reason it was not applied. When
    # applied, `accepted` above follows it, and can only become True through a
    # confirmation run recorded separately.
    rule: dict = field(default_factory=dict)

    @property
    def disposition(self) -> str:
        """The candidate's STATE, which `accepted: bool` cannot express.

        "Not yet accepted" and "rejected" are materially different, and collapsing
        them into one boolean made the record say the wrong thing out loud: the first
        CONFIRM this rule ever produced — a candidate whose gain was real, whose
        guards held, and whose only remaining step was a paired rerun — printed as
        "REJECTED" in the validation summary, beside a rule outcome of CONFIRM.

        `accepted` stays a bool because it is the SHIPPING gate (`pr` refuses anything
        false, and only a confirmation can make it true). This is what humans and
        reports read instead.
        """
        outcome = self.rule.get("outcome") if self.rule.get("applied") else None
        if outcome == "CONFIRM":
            return "PENDING_CONFIRMATION"
        return "ACCEPTED" if self.accepted else "REJECTED"

    def to_json(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "label": self.label,
            "accepted": self.accepted,
            "disposition": self.disposition,
            "candidate_fields": self.candidate_fields,
            "delta_in": self.delta_in,
            "delta_ho": self.delta_ho,
            "per_task": self.per_task,
            "aggregate_accepted": self.aggregate_accepted,
            "regressions": self.regressions,
            "catastrophic_regressions": self.catastrophic_regressions,
            "baseline_metrics": self.baseline_metrics,
            "candidate_metrics": self.candidate_metrics,
            "metric_delta": self.metric_delta,
            "metric_not_compared": self.metric_not_compared,
            "metric_task_counts": self.metric_task_counts,
            "metric_attempt_counts": self.metric_attempt_counts,
            "metric_denominator_drift": self.metric_denominator_drift,
            "baseline_fingerprint": self.baseline_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            # These two were computed, attached to the record, described in their own
            # docstrings as "part of the record" — and then dropped here, which is the
            # only place the record reaches disk (`loop/cli.py` writes `to_json()` and
            # nothing else). So the harness-gate outcome was never persisted for any
            # candidate, and neither was the coverage split. A field that a serializer
            # silently omits is worse than one that was never added: the code reads as
            # though the evidence exists, and every claim made about the record on the
            # strength of it was false.
            "gates": self.gates,
            "coverage": self.coverage,
            "causal": self.causal,
            "rule": self.rule,
        }


def write_validation_record(record: ValidationRecord, path: Path) -> Path:
    """The one path a validation record takes to disk.

    Extracted so it can be tested. `to_json()` silently omitted `gates` and `coverage`
    for as long as they existed, and every test aimed at those features checked the
    computation rather than the file — the claim was "the record carries it" and the
    thing under test was one call short of the claim. A test that only round-trips
    `to_json()` in memory still cannot see a caller that drops a key before writing.
    """
    path.write_text(json.dumps(record.to_json(), indent=2) + "\n")
    return path


# The only stage a ConfirmationRecord names today — a fresh PAIRED rerun of a first
# CONFIRM's selected tasks (`loop.acceptance.confirmed`). A named constant rather than
# a literal repeated at each call site, so the record and its tests read the same word.
STAGE_PAIRED_CONFIRMATION = "paired_confirmation"


@dataclass(frozen=True)
class ConfirmationRecord:
    """The outcome of one paired-confirmation rerun — the only path to ACCEPT.

    Formalizes the shape iter-06's confirmation was hand-assembled into (contract
    §5): a candidate's first CONFIRM decision, rerun fresh at higher attempt counts on
    exactly that decision's ``confirm_tasks``, judged by ``acceptance.confirmed()``.
    Kept on disk whether the confirmation lands ACCEPT or REJECT — a REJECTED
    confirmation is still the honest record of what was tried, same reasoning as
    ``ValidationRecord`` keeping rejected candidates.

    ``first_decision`` and ``confirmation`` are ``Decision.to_json()`` dicts, not
    ``Decision`` objects — this artifact is the frozen-JSON contract, the same
    boundary ``ValidationRecord.rule`` already draws around a ``Decision``.
    ``per_task`` mirrors iter-06's shape exactly: ``{task: {"base": [passes,
    attempts], "cand": [passes, attempts]}}``, for every task both arms actually
    measured.
    """

    candidate_id: str
    baseline_label: str
    candidate_label: str
    attempts_per_task_per_arm: int
    confirm_set: tuple[str, ...]
    first_decision: dict
    confirmation: dict
    per_task: dict[str, dict[str, list[int]]]
    finding: str
    stage: str = STAGE_PAIRED_CONFIRMATION

    def to_json(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "attempts_per_task_per_arm": self.attempts_per_task_per_arm,
            "confirm_set": list(self.confirm_set),
            "first_decision": self.first_decision,
            "confirmation": self.confirmation,
            "per_task": self.per_task,
            "finding": self.finding,
        }

    @classmethod
    def from_json(cls, data: dict) -> ConfirmationRecord:
        return cls(
            candidate_id=data["candidate_id"],
            stage=data.get("stage", STAGE_PAIRED_CONFIRMATION),
            baseline_label=data["baseline_label"],
            candidate_label=data["candidate_label"],
            attempts_per_task_per_arm=data["attempts_per_task_per_arm"],
            confirm_set=tuple(data["confirm_set"]),
            first_decision=data["first_decision"],
            confirmation=data["confirmation"],
            per_task=data["per_task"],
            finding=data["finding"],
        )


def write_confirmation_record(record: ConfirmationRecord, path: Path) -> Path:
    """The one path a confirmation record takes to disk — mirrors
    ``write_validation_record`` so both artifacts have exactly one writer to test."""
    path.write_text(json.dumps(record.to_json(), indent=2) + "\n")
    return path
