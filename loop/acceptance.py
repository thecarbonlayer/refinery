"""Three-outcome acceptance: REJECT, CONFIRM, ACCEPT — calibrated by measurement.

The one-number rule (`Δ_in >= 0 and Δ_ho >= 0 and max > 0`) was measured against six
baseline runs with NOTHING changed between them. On the twelve workflow-gap pairs it
wrongly accepted 6 and showed a false regression on 3 more. Two facts from that
measurement shape this rule:

- Negative variation reached exactly ONE attempt per split. So a regression bound
  tighter than one attempt rejects noise, and a looser one admits real damage.
- Positive variation reached TWO attempts held-out (+0.0400) with nothing changed.
  So no single-run gain, however large a margin over one attempt, is proof — which is
  why a gain earns CONFIRM, never ACCEPT.

Outcomes:
  REJECT  — a split regressed beyond one attempt, a full-pass task collapsed to zero,
            security failed more often than in the baseline, or there is no gain
            beyond one attempt anywhere (nothing worth confirming).
  CONFIRM — a gain larger than one attempt exists and nothing disqualifies it. The
            candidate is PROMISING, not accepted: the same six runs produced this
            much movement from noise.
  ACCEPT  — only ever from ``confirmed()``: a fresh PAIRED rerun of the selected
            tasks in which the ORIGINAL improvement appears again and nothing
            regresses. ``evaluate()`` never returns it.

Security is read from OUTCOMES, not from a task-level flag. Cluster-C tasks carry
both a security conjunct and a functional one, and a flag on the task cannot tell a
leaked secret from a wrong mode report. The runner emits ``critical_failure`` for the
security half only; this rule counts those per task and blocks any candidate whose
count rises above its baseline's, independent of what the averages say — one extra
leak must not disappear into a mean of twenty-eight numbers.

Thresholds are DERIVED from the suite the results actually ran — one attempt on the
largest-grained task of each split — never hard-coded decimals. All arithmetic is
exact (`Fraction` of the integer counts); floats appear only in the report.

Lives in `loop/`, not `runner/`: `runner_sha` is the verifier's identity and a
governance rule must not invalidate baselines when it changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

REJECT = "REJECT"
CONFIRM = "CONFIRM"
ACCEPT = "ACCEPT"


@dataclass(frozen=True)
class Decision:
    outcome: str  # REJECT | CONFIRM (evaluate) | ACCEPT (confirmed only)
    reasons: tuple[str, ...]
    delta_in: float
    delta_ho: float
    threshold_in: float  # one attempt, held-in, as a float for the record
    threshold_ho: float
    excluded: tuple[str, ...] = ()  # proof-unreachable for the edited section, zeroed
    evidence_split: str = ""  # which split carried the gain, for the confirmation
    improved_tasks: tuple[str, ...] = ()  # the gain's task basis — what must repeat
    confirm_tasks: tuple[str, ...] = ()  # what a confirmation pair must rerun
    security_regressions: dict[str, list[int]] = field(default_factory=dict)  # task -> [base, cand]
    raw: dict = field(default_factory=dict)  # the one-number rule's verdict, as evidence

    def to_json(self) -> dict:
        return {
            "outcome": self.outcome,
            "reasons": list(self.reasons),
            "delta_in": self.delta_in,
            "delta_ho": self.delta_ho,
            "threshold_in": self.threshold_in,
            "threshold_ho": self.threshold_ho,
            "excluded": list(self.excluded),
            "evidence_split": self.evidence_split,
            "improved_tasks": list(self.improved_tasks),
            "confirm_tasks": list(self.confirm_tasks),
            "security_regressions": self.security_regressions,
            "raw": self.raw,
        }


def one_attempt(results: dict, split: str) -> Fraction:
    """The largest movement a single attempt can cause in this split's mean.

    Derived from the structure of the suite that actually ran — a filtered
    confirmation subset with more attempts per task derives a proportionally finer
    grain, with no code knowing which case it is in.
    """
    grains = [
        Fraction(1, int(t["attempts"])) for t in results["tasks"].values() if t["split"] == split
    ]
    if not grains:
        return Fraction(0)
    return max(grains) / len(grains)


def security_failures(results: dict) -> dict[str, int]:
    """Per-task count of ``critical_failure`` outcomes. Zero-count tasks omitted."""
    out = {}
    for name, t in results["tasks"].items():
        n = sum(1 for o in t.get("outcomes", ()) if o == "critical_failure")
        if n:
            out[name] = n
    return out


def _exact(task: dict) -> Fraction:
    if "passes" in task and task.get("attempts"):
        return Fraction(int(task["passes"]), int(task["attempts"]))
    return Fraction(task["pass_fraction"]).limit_denominator(10_000)


def _parity(baseline: dict, candidate: dict) -> None:
    """The gates a comparison cannot skip, WITHOUT refusing filtered results.

    ``runner.delta`` refuses any result carrying a ``filter`` key, which is right for
    full-suite Δs and fatal for confirmation runs — a confirmation deliberately reruns
    only the selected tasks. What still must hold: same task set, same per-task
    attempts, same verifier, and attributed measurements on both sides.
    """
    b, c = set(baseline["tasks"]), set(candidate["tasks"])
    if b != c:
        raise ValueError(
            f"task sets differ: only-baseline={sorted(b - c)}, only-candidate={sorted(c - b)}"
        )
    mismatched = [
        n
        for n in sorted(b)
        if baseline["tasks"][n]["attempts"] != candidate["tasks"][n]["attempts"]
    ]
    if mismatched:
        raise ValueError(f"attempt counts differ on: {', '.join(mismatched)}")
    bf, cf = baseline.get("fingerprint", {}), candidate.get("fingerprint", {})
    if not bf or not cf:
        raise ValueError("results file lacks a fingerprint — refusing unattributed measurements")
    if bf.get("runner_sha") != cf.get("runner_sha"):
        raise ValueError("verifier version mismatch — re-measure")


def _split_deltas(
    baseline: dict, candidate: dict, excluded: frozenset[str] | set[str]
) -> tuple[dict[str, Fraction], Fraction, Fraction]:
    per = {
        n: (
            Fraction(0)
            if n in excluded
            else _exact(candidate["tasks"][n]) - _exact(baseline["tasks"][n])
        )
        for n in baseline["tasks"]
    }
    splits = {n: t["split"] for n, t in baseline["tasks"].items()}

    def mean(split: str) -> Fraction:
        vals = [v for n, v in per.items() if splits[n] == split]
        return sum(vals) / len(vals) if vals else Fraction(0)

    return per, mean("held_in"), mean("held_out")


def evaluate(
    baseline: dict,
    candidate: dict,
    *,
    excluded: frozenset[str] | set[str] = frozenset(),
    always_confirm: frozenset[str] | set[str] = frozenset(),
) -> Decision:
    """One full-suite validation's verdict: REJECT or CONFIRM, never ACCEPT.

    ``always_confirm`` names tasks the confirmation pair must rerun even when they did
    not move — the edit's GUARDS. Selecting only the movers lets a candidate confirm
    its E3/E4-style gain without ever re-testing the trade-off the section is known to
    carry (E1's retrieval economy for tool_output) or the security checks whose
    critical outcomes might appear only under the confirmation's higher attempt
    counts. An unchanged guard is exactly the task whose stability the confirmation
    exists to re-establish.
    """
    _parity(baseline, candidate)
    per, d_in, d_ho = _split_deltas(baseline, candidate, excluded)
    thr_in, thr_ho = one_attempt(baseline, "held_in"), one_attempt(baseline, "held_out")

    reasons: list[str] = []
    if d_in < -thr_in:
        reasons.append(
            f"held-in regressed beyond one attempt ({float(d_in):+.4f} < -{float(thr_in):.4f})"
        )
    if d_ho < -thr_ho:
        reasons.append(
            f"held-out regressed beyond one attempt ({float(d_ho):+.4f} < -{float(thr_ho):.4f})"
        )

    # Complete-collapse veto, causal-filtered: a collapse the edited section cannot
    # reach is the grader's noise wearing the candidate's name.
    collapses = sorted(
        n
        for n, v in per.items()
        if n not in excluded
        and _exact(baseline["tasks"][n]) == 1
        and _exact(candidate["tasks"][n]) == 0
    )
    if collapses:
        reasons.append("full-pass task collapsed to zero: " + ", ".join(collapses))

    # Security, from outcomes and independent of the mean: more critical_failures
    # than the baseline on any task blocks, whatever the averages say. NOT causal-
    # filtered — a leak is a leak whichever knob was being edited.
    base_sec, cand_sec = security_failures(baseline), security_failures(candidate)
    sec_reg = {
        n: [base_sec.get(n, 0), cand_sec[n]] for n in cand_sec if cand_sec[n] > base_sec.get(n, 0)
    }
    if sec_reg:
        reasons.append(
            "security failed more often than baseline: "
            + ", ".join(f"{n} {v[0]}->{v[1]}" for n, v in sorted(sec_reg.items()))
        )

    raw_evidence = {"delta_in": float(d_in), "delta_ho": float(d_ho)}
    if reasons:
        return Decision(
            REJECT,
            tuple(reasons),
            float(d_in),
            float(d_ho),
            float(thr_in),
            float(thr_ho),
            tuple(sorted(excluded)),
            "",
            (),
            (),
            sec_reg,
            raw_evidence,
        )

    evidence_split = "held_in" if d_in > thr_in else ("held_out" if d_ho > thr_ho else "")
    if not evidence_split:
        return Decision(
            REJECT,
            (
                "no gain beyond one attempt on either split — indistinguishable from the "
                "measured null variation, nothing to confirm",
            ),
            float(d_in),
            float(d_ho),
            float(thr_in),
            float(thr_ho),
            tuple(sorted(excluded)),
            "",
            (),
            (),
            sec_reg,
            raw_evidence,
        )

    splits = {n: t["split"] for n, t in baseline["tasks"].items()}
    improved = tuple(sorted(n for n, v in per.items() if splits[n] == evidence_split and v > 0))
    moved = {n for n, v in per.items() if v != 0}
    # The confirmation reruns every task that moved, every task that showed a security
    # failure on either side (a steady leak count still deserves a look at higher
    # attempt counts), and every named guard whether it moved or not.
    confirm = tuple(sorted(moved | set(base_sec) | set(cand_sec) | set(always_confirm)))
    return Decision(
        CONFIRM,
        (
            f"gain beyond one attempt on {evidence_split.replace('_', '-')} "
            f"(carried by {', '.join(improved)})",
        ),
        float(d_in),
        float(d_ho),
        float(thr_in),
        float(thr_ho),
        tuple(sorted(excluded)),
        evidence_split,
        improved,
        confirm,
        sec_reg,
        raw_evidence,
    )


def confirmed(
    first: Decision,
    confirm_baseline: dict,
    confirm_candidate: dict,
    *,
    excluded: frozenset[str] | set[str] = frozenset(),
) -> Decision:
    """The only path to ACCEPT: a fresh PAIRED rerun repeats the ORIGINAL story.

    Both sides fresh, the same selected tasks, more attempts — rerunning only the
    candidate against the recorded baseline would keep the exact time-separation
    confound the null runs measured. Three requirements, each closing a hole the
    first version of this function had:

    - The confirmation pair must cover exactly ``first.confirm_tasks`` — filtered
      results are expected here, so this does its own parity gates instead of
      ``runner.delta``'s, which refuses any filtered result outright.
    - The ORIGINAL improvement must appear again: the same ``improved_tasks`` must in
      aggregate gain more than one attempt's grain on the same split. A different
      improvement appearing is a new claim that starts its own cycle — noise is
      exactly the ability to produce a fresh two-attempt gain somewhere else.
    - No regression is allowed in the confirmation itself: neither split beyond its
      allowance, and ANY security-failure count above the confirmation baseline
      blocks — not merely a repeat of the first run's security regressions.
    """
    if first.outcome != CONFIRM:
        return first
    _parity(confirm_baseline, confirm_candidate)
    ran = set(confirm_baseline["tasks"])
    want = set(first.confirm_tasks)
    if ran != want:
        raise ValueError(
            f"confirmation must rerun exactly the selected tasks: "
            f"missing={sorted(want - ran)}, extra={sorted(ran - want)}"
        )

    per, d_in, d_ho = _split_deltas(confirm_baseline, confirm_candidate, excluded)
    thr_in = one_attempt(confirm_baseline, "held_in")
    thr_ho = one_attempt(confirm_baseline, "held_out")
    report = (float(d_in), float(d_ho), float(thr_in), float(thr_ho))

    reasons: list[str] = []
    if d_in < -thr_in:
        reasons.append(f"held-in regressed in confirmation ({float(d_in):+.4f})")
    if d_ho < -thr_ho:
        reasons.append(f"held-out regressed in confirmation ({float(d_ho):+.4f})")

    base_sec, cand_sec = security_failures(confirm_baseline), security_failures(confirm_candidate)
    sec_reg = {
        n: [base_sec.get(n, 0), cand_sec[n]] for n in cand_sec if cand_sec[n] > base_sec.get(n, 0)
    }
    if sec_reg:
        reasons.append(
            "security regressed in confirmation: "
            + ", ".join(f"{n} {v[0]}->{v[1]}" for n, v in sorted(sec_reg.items()))
        )

    # The original improvement, on the original tasks, on the original split.
    basis = [per[n] for n in first.improved_tasks]
    grain = (
        max(
            Fraction(1, int(confirm_baseline["tasks"][n]["attempts"])) for n in first.improved_tasks
        )
        / len(first.improved_tasks)
        if basis
        else Fraction(0)
    )
    repeated = bool(basis) and sum(basis) / len(basis) > grain
    if not repeated:
        reasons.append(
            "original improvement did not repeat on "
            + ", ".join(first.improved_tasks)
            + " — a gain elsewhere is a new claim, not a confirmation"
        )

    if reasons:
        return Decision(
            REJECT,
            tuple(reasons),
            *report,
            first.excluded,
            first.evidence_split,
            first.improved_tasks,
            first.confirm_tasks,
            sec_reg,
            {"stage": "confirmation"},
        )
    return Decision(
        ACCEPT,
        (
            "original improvement repeated under a fresh paired confirmation on "
            + ", ".join(first.improved_tasks),
        ),
        *report,
        first.excluded,
        first.evidence_split,
        first.improved_tasks,
        first.confirm_tasks,
        sec_reg,
        {"stage": "confirmation"},
    )
