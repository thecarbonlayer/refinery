"""Three-outcome acceptance: REJECT, CONFIRM, ACCEPT — calibrated by measurement.

The one-number rule (`Δ_in >= 0 and Δ_ho >= 0 and max > 0`) was measured against six
baseline runs with NOTHING changed between them. On the twelve workflow-gap pairs it
wrongly accepted 6 and showed a false regression on 3 more. Two facts from that
measurement shape this rule:

- Negative variation reached exactly ONE attempt per split (−1/54 held-in, −1/50
  held-out under causal filtering). So a regression bound tighter than one attempt
  rejects noise, and a looser one admits real damage.
- Positive variation reached TWO attempts held-out (+0.0400) with nothing changed.
  So no single-run gain, however large a margin over one attempt, is proof — which is
  why a gain earns CONFIRM, never ACCEPT.

Outcomes:
  REJECT  — a split regressed beyond one attempt, a full-pass task collapsed to zero,
            or there is no gain beyond one attempt anywhere (nothing worth confirming).
  CONFIRM — a gain larger than one attempt exists and nothing disqualifies it. The
            candidate is PROMISING, not accepted: the same six runs produced this
            much movement from noise. Confirmation reruns BOTH sides fresh on the
            moved tasks with more attempts — rerunning only the candidate against the
            recorded baseline would preserve the very time-separation confound the
            null runs measured.
  ACCEPT  — only ever after a confirmation run repeats the targeted improvement with
            neither split beyond its allowed variation and no repeated critical
            regression. `evaluate()` never returns it; `confirmed()` does.

Critical tasks are the deliberate exception to averaging. C3's three failures in the
null runs were real secret leaks into debug.log — the timing was noise, the behaviour
was not. A negative movement on a critical task demands confirmation even when the
aggregate sits inside tolerance, because one extra leak must not disappear into a mean
of twenty-eight numbers.

Thresholds are DERIVED from the suite the baseline actually ran — one attempt on the
largest-grained task of each split — never hard-coded decimals. A suite that grows a
task or changes an attempt count moves its own thresholds. All arithmetic is exact
(`Fraction` of the integer counts); floats appear only in the report.

Lives in `loop/`, not `runner/`: `runner_sha` is the verifier's identity and a
governance rule must not invalidate baselines when it changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from runner.delta import delta

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
    critical_regressions: dict[str, float] = field(default_factory=dict)
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
            "critical_regressions": self.critical_regressions,
            "raw": self.raw,
        }


def one_attempt(results: dict, split: str) -> Fraction:
    """The largest movement a single attempt can cause in this split's mean.

    Derived from the structure of the suite that actually ran: with every held-in task
    at 3 attempts over 18 tasks this is 1/54, and held-out at 5 over 10 it is 1/50 —
    but stated as the max over tasks so a mixed-attempt suite gets the honest bound
    rather than an averaged fiction.
    """
    grains = [
        Fraction(1, int(t["attempts"])) for t in results["tasks"].values() if t["split"] == split
    ]
    if not grains:
        return Fraction(0)
    return max(grains) / len(grains)


def _exact_per_task(baseline: dict, candidate: dict) -> dict[str, Fraction]:
    out = {}
    for name, b in baseline["tasks"].items():
        c = candidate["tasks"][name]
        if "passes" in b and "passes" in c:
            out[name] = Fraction(int(c["passes"]), int(c["attempts"])) - Fraction(
                int(b["passes"]), int(b["attempts"])
            )
        else:  # pre-counts era: reconstruct from the stored fraction
            out[name] = Fraction(c["pass_fraction"]).limit_denominator(10_000) - Fraction(
                b["pass_fraction"]
            ).limit_denominator(10_000)
    return out


def evaluate(
    baseline: dict,
    candidate: dict,
    *,
    excluded: frozenset[str] | set[str] = frozenset(),
    critical: frozenset[str] | set[str] = frozenset(),
) -> Decision:
    """One validation run's verdict: REJECT or CONFIRM, never ACCEPT.

    ``excluded`` is the proof-grade unreachable set for the edited config section —
    movements there are zeroed with the denominator kept, exactly as `causal_verdict`
    does. ``critical`` names tasks whose individual failures matter beyond the mean.
    """
    d = delta(baseline, candidate)  # parity gates: task set, fingerprints, attempts
    per = _exact_per_task(baseline, candidate)
    per = {n: (Fraction(0) if n in excluded else v) for n, v in per.items()}
    splits = {n: t["split"] for n, t in baseline["tasks"].items()}

    def split_delta(split: str) -> Fraction:
        vals = [v for n, v in per.items() if splits[n] == split]
        return sum(vals) / len(vals) if vals else Fraction(0)

    d_in, d_ho = split_delta("held_in"), split_delta("held_out")
    thr_in, thr_ho = one_attempt(baseline, "held_in"), one_attempt(baseline, "held_out")

    reasons: list[str] = []

    # 1. Regression check — one attempt of movement per split is what six unchanged
    # runs produced, so exactly one attempt is ALLOWED and anything beyond is not.
    if d_in < -thr_in:
        reasons.append(
            f"held-in regressed beyond one attempt ({float(d_in):+.4f} < -{float(thr_in):.4f})"
        )
    if d_ho < -thr_ho:
        reasons.append(
            f"held-out regressed beyond one attempt ({float(d_ho):+.4f} < -{float(thr_ho):.4f})"
        )

    # Complete-collapse veto, kept from the one-number rule, causal-filtered the same
    # way: a collapse on a task the edited section cannot reach is the grader's noise.
    collapses = {n: c for n, c in d["catastrophic_regressions"].items() if n not in excluded}
    if collapses:
        reasons.append("full-pass task collapsed to zero: " + ", ".join(sorted(collapses)))

    crit_reg = {n: float(v) for n, v in per.items() if n in critical and v < 0}

    if reasons:
        return Decision(
            REJECT,
            tuple(reasons),
            float(d_in),
            float(d_ho),
            float(thr_in),
            float(thr_ho),
            tuple(sorted(excluded)),
            crit_reg,
            {k: d[k] for k in ("accepted", "delta_in", "delta_ho")},
        )

    # 2. Positive evidence — strictly MORE than one attempt of gain on some split.
    # Necessary but never sufficient: the null runs produced a two-attempt held-out
    # gain, which is why this earns CONFIRM rather than ACCEPT.
    evidence = d_in > thr_in or d_ho > thr_ho
    if not evidence:
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
            crit_reg,
            {k: d[k] for k in ("accepted", "delta_in", "delta_ho")},
        )

    why = ["gain beyond one attempt on " + ("held-in" if d_in > thr_in else "held-out")]
    # 3. Critical tasks bypass the averaging: any negative movement on one demands the
    # confirmation run look at it specifically, tolerance or no tolerance.
    if crit_reg:
        why.append(
            "critical task moved negative and must not repeat under confirmation: "
            + ", ".join(f"{n} {v:+.2f}" for n, v in sorted(crit_reg.items()))
        )
    return Decision(
        CONFIRM,
        tuple(why),
        float(d_in),
        float(d_ho),
        float(thr_in),
        float(thr_ho),
        tuple(sorted(excluded)),
        crit_reg,
        {k: d[k] for k in ("accepted", "delta_in", "delta_ho")},
    )


def confirmed(
    first: Decision,
    confirm_baseline: dict,
    confirm_candidate: dict,
    *,
    excluded: frozenset[str] | set[str] = frozenset(),
    critical: frozenset[str] | set[str] = frozenset(),
) -> Decision:
    """The only path to ACCEPT: a fresh PAIRED rerun repeats the story.

    Both sides fresh, on the moved tasks, with more attempts — rerunning only the
    candidate against the recorded baseline would keep the exact time-separation
    confound the null runs measured. Accept iff the targeted improvement appears
    again, neither split exceeds its allowed variation, and no critical task shows a
    repeated regression.
    """
    if first.outcome != CONFIRM:
        return first
    second = evaluate(confirm_baseline, confirm_candidate, excluded=excluded, critical=critical)
    if second.outcome != CONFIRM:
        return Decision(
            REJECT,
            ("confirmation run did not repeat the improvement",) + second.reasons,
            second.delta_in,
            second.delta_ho,
            second.threshold_in,
            second.threshold_ho,
            second.excluded,
            second.critical_regressions,
            second.raw,
        )
    repeated_crit = set(first.critical_regressions) & set(second.critical_regressions)
    if repeated_crit:
        return Decision(
            REJECT,
            (
                "critical regression repeated under confirmation: "
                + ", ".join(sorted(repeated_crit)),
            ),
            second.delta_in,
            second.delta_ho,
            second.threshold_in,
            second.threshold_ho,
            second.excluded,
            second.critical_regressions,
            second.raw,
        )
    return Decision(
        ACCEPT,
        ("improvement repeated under a fresh paired confirmation",) + second.reasons,
        second.delta_in,
        second.delta_ho,
        second.threshold_in,
        second.threshold_ho,
        second.excluded,
        second.critical_regressions,
        second.raw,
    )
