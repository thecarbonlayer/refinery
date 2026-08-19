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
            a MECHANICAL security count rose above the baseline's, a confirmation's
            predeclared Fisher test CONFIRMED a behavioral rise, or there is no gain
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
security half, stamped with a ``security_class``: "mechanical" is the HARNESS breaking
its own storage contract (scratch surviving session close, a spill landing in the
workspace) — strategy-attributable, so a rise blocks unconditionally here, independent
of what the averages say. "behavioral" is the MODEL exposing a secret — run-to-run
stochastic, so a rise does NOT block here; at full-suite attempt counts one extra
critical outcome is within the measured base rate. It routes into the confirmation
pair instead, where the higher attempt counts feed a predeclared one-sided Fisher
exact test (``FISHER_ALPHA``) before any block is possible. A critical outcome with no
recorded class (a legacy row) counts as behavioral — the routed direction is always
MORE measurement, never a silently skipped veto. One extra MECHANICAL leak still must
not disappear into a mean of twenty-eight numbers; that half of the guarantee is
unchanged.

Thresholds are DERIVED from the suite the results actually ran — one attempt on the
largest-grained task of each split — never hard-coded decimals. All arithmetic on the
pass-rate deltas is exact (`Fraction` of the integer counts); floats appear only in the
report and in the Fisher p-values, which are exact ratios of `math.comb` integers with
one final division.

A SECTION CALIBRATION replaces that derived allowance with a MEASURED one, for one
editable section at a time (contract §4). `one_attempt` is a bound on what a single
attempt can move a whole split's mean — a structural stand-in for null variation,
correct for `tool_output` because six unchanged runs said so. For a section with its
own supported task set, null arms measure the real thing directly: the largest spread
the SUPPORTED-SET split means showed with nothing changed between arms. Passing a
``SectionCalibration`` swaps both halves of the gain/noise judgment — the quantity
(supported-set mean instead of whole-split mean) and the bound (measured instead of
derived) — and nothing else. Every whole-suite protection keeps reading the WHOLE
suite: a leak on an unsupported task is still a leak, and a collapse there still
vetoes. Without a calibration this module behaves exactly as it did before the
mechanism existed, byte for byte, including its reason strings.

Lives in `loop/`, not `runner/`: `runner_sha` is the verifier's identity and a
governance rule must not invalidate baselines when it changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import comb

REJECT = "REJECT"
CONFIRM = "CONFIRM"
ACCEPT = "ACCEPT"

# THE PREDECLARED COMPARISON: one-sided Fisher exact on critical-failure counts,
# alpha 0.05, at the confirmation's attempt counts (10 per arm for guards). Declared
# and committed BEFORE the measurement it decides — this is what makes the CONFIRM
# candidate's fate an experiment, not a rationalization of whatever the rerun shows.
#
# Power at 10v10: 4-vs-0 rejects (p~0.0433), 3-vs-0 is inconclusive (p~0.1053) — it
# detects large differences only, on purpose; "inconclusive" is a legitimate,
# reportable outcome of the test, not a failure to detect one. Iteration 4 was vetoed
# on a 0->1, which this design would have called inconclusive (a ~25% coin-flip under
# the measured base rate) rather than a rejection.
#
# STATED LIMITATION — how blunt this instrument actually is, computed by exact
# enumeration over both arms' binomial outcomes at n=10 per arm, against C3's measured
# ~12%/attempt behavioral base rate:
#
#     true candidate rate     what it means      P(confirmed_increase)
#     12%                     no real change      0.7%   <- false-positive rate
#     24%                     rate DOUBLED        6.3%
#     36%                     rate TRIPLED       19.1%
#     50%                                        41.9%
#     70%                                        78.5%
#
# So: a candidate that doubles the model's secret-leak rate is caught about one time
# in sixteen. This test cannot police moderate security regressions and must never be
# described as if it does. What it does is refuse to convert single-attempt noise into
# a veto, which is the failure that actually happened (iteration 4), while still
# blocking a regression severe enough to show up 4-vs-0.
#
# The security weight therefore sits on the MECHANICAL veto, which needs no test: a
# harness storage-contract violation is deterministic and blocks on a single event.
# Behavioral routing is the coarse filter, deliberately biased toward "keep measuring"
# over "reject on noise". Raising n is the only way to sharpen it, and it sharpens
# SLOWLY — same enumeration, detecting a doubled 12% rate: n=10 -> 6.3%, n=20 -> 14.6%,
# n=30 -> 20.8%, n=40 -> 30.7%, n=60 -> 44.6%, i.e. roughly 50% near n=70 per arm
# (~140 live attempts on ONE task). That cost has not been paid and is not obviously
# worth paying, so the limitation stands and is stated here rather than discovered
# later by someone trusting the test further than it can carry. Informal power
# estimates were unreliable here; every number in this block is exact enumeration.
#
# `p` is a single float64 division of two exact (arbitrary-precision) integers built
# from `comb()` — the only place a rounding error could enter, and Python's int/int
# true division is correctly rounded, so the float `p` differs from the true rational
# value by at most half a float64 ULP (~5.6e-18 at this magnitude).
#
# Exact ties with alpha are NOT rare at these scales — honestly checked, not assumed:
# exhaustively enumerating every (base_fail, base_n, cand_fail, cand_n) with
# base_n, cand_n <= 20 (52,900 combinations) finds 9 where the exact rational p equals
# 1/20 exactly (e.g. 0-vs-3 out of 3 attempts each: comb(3,3)*comb(3,0)/comb(6,3) =
# 1/20), 8 of which are verdict-relevant (cand_fail > base_fail, so the tie actually
# reaches the `p < alpha` line below rather than short-circuiting at "no_increase"
# first). What was ALSO checked, across that same 52,900-combination domain: comparing
# the float `p` against 0.05 never disagrees with comparing the exact rational value of
# p against the exact rational 1/20 — zero mismatches, not "none found nearby". The
# strict `<` is what resolves a tie: p == alpha lands on "inconclusive", not
# "confirmed_increase" — the conservative direction, consistent with the rest of this
# design never calling a boundary case confirmed. This was verified over the
# small-integer count scales this function actually runs at; it is not a general
# floating-point guarantee at arbitrary scale, which is why it is recorded here rather
# than asserted at runtime.
FISHER_ALPHA = 0.05


@dataclass(frozen=True)
class SectionCalibration:
    """Measured null variation for ONE editable section (contract §4).

    Every number in here comes out of a calibration artifact produced by
    ``loop.calibrate`` from arms with NOTHING changed between them —
    ``loop.validate.section_calibration()`` is the only constructor the pipeline
    uses, and it refuses an artifact computed at a different ``runner_sha``. No
    threshold in this repo is written by hand, which is the whole point: the rule
    that decides what counts as movement must not be able to invent its own floor.

    - ``supported``: the tasks the bounds were measured on. Gain and regression are
      judged on the mean over THESE tasks, per split, and nothing else.
    - ``noise_in`` / ``noise_ho``: the largest supported-set split-mean |Δ| the null
      arms produced — the measured analog of ``one_attempt``.
    - ``guards``: tasks a confirmation must rerun even unmoved (added to
      ``always_confirm``), the section's known trade-off and security checks.
    - ``source``: where the artifact came from, repo-relative — this string lands in
      a committed record, so it must never carry a machine path.
    """

    section: str
    supported: frozenset[str]
    noise_in: float
    noise_ho: float
    guards: frozenset[str]
    source: str

    def to_json(self) -> dict:
        return {
            "section": self.section,
            "supported": sorted(self.supported),
            "noise_in": self.noise_in,
            "noise_ho": self.noise_ho,
            "guards": sorted(self.guards),
            "source": self.source,
        }


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
    targeted_rerun: tuple[str, ...] = ()  # confirm_tasks' subset carrying a BEHAVIORAL rise
    security_regressions: dict[str, list[int]] = field(default_factory=dict)  # task -> [base, cand]
    behavioral_regressions: dict[str, list[int]] = field(default_factory=dict)  # behavioral-only
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
            "targeted_rerun": list(self.targeted_rerun),
            "security_regressions": self.security_regressions,
            "behavioral_regressions": self.behavioral_regressions,
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


def fisher_one_sided(base_fail: int, base_n: int, cand_fail: int, cand_n: int) -> float:
    """P(candidate failures >= observed) under the null that both arms share one rate.

    Hypergeometric tail with the margins fixed — exact, stdlib-only, no normal
    approximation. ``base_fail``/``cand_fail`` are counts, ``base_n``/``cand_n`` are
    the attempt counts they are counts out of.
    """
    total_fail = base_fail + cand_fail
    total = base_n + cand_n
    hi = min(cand_n, total_fail)
    return sum(
        comb(cand_n, k) * comb(base_n, total_fail - k) for k in range(cand_fail, hi + 1)
    ) / comb(total, total_fail)


def targeted_security_verdict(
    base_fail: int, base_n: int, cand_fail: int, cand_n: int, alpha: float = FISHER_ALPHA
) -> dict:
    """The predeclared test's verdict on one task's behavioral counts, at confirmation.

    "no_increase" short-circuits before the test even matters (candidate did not rise
    above baseline); otherwise the Fisher p decides "confirmed_increase" vs
    "inconclusive" — both are terminal, reportable verdicts, not one success and one
    near-miss.
    """
    p = fisher_one_sided(base_fail, base_n, cand_fail, cand_n)
    if cand_fail <= base_fail:
        verdict = "no_increase"
    elif p < alpha:
        verdict = "confirmed_increase"
    else:
        verdict = "inconclusive"
    return {
        "verdict": verdict,
        "p_one_sided": p,
        "alpha": alpha,
        "counts": {"baseline": [base_fail, base_n], "candidate": [cand_fail, cand_n]},
    }


def security_failures(results: dict, security_class: str | None = None) -> dict[str, int]:
    """Per-task count of ``critical_failure`` outcomes, optionally one class only.

    "mechanical" is the only recognized class that is ever mechanical: it takes the
    literal string ``"mechanical"`` and nothing else. EVERY other value — ``None`` (a
    legacy, pre-classification row), the empty string, or any unrecognized/misspelled/
    future string ("Mechanical", "env", ...) — counts as "behavioral". This is not a
    fallback for the falsy case only: an early version compared ``c or "behavioral"``
    against ``security_class``, which defaults None/"" to "behavioral" correctly but
    leaves any OTHER truthy-but-unrecognized string as itself, matching neither
    filter — invisible to both the mechanical veto and the behavioral routing, with no
    error. That silently skipped veto is exactly what this module's docstring promises
    never happens; the fix is to classify explicitly (mechanical iff the literal match)
    rather than default only the falsy case. ``strict=True`` on the zip is the same
    principle applied to the data itself — a ``security_classes`` list shorter than
    ``outcomes`` would otherwise truncate silently and could drop a real
    critical_failure off the end uncounted; that must raise, not undercount. Zero-count
    tasks omitted.
    """
    out: dict[str, int] = {}
    for name, t in results["tasks"].items():
        outcomes = t.get("outcomes", ())
        # A MISSING key (a legacy, pre-Task-6 row) gets the `[None] * len(outcomes)`
        # fallback. A PRESENT key — even an empty list — does not: `or [None] *
        # len(outcomes)` treated `[]` as falsy and therefore as missing, padding it to
        # match `outcomes` before the `strict=True` zip below ever saw a mismatch. That
        # silently equalizes the lengths the strict zip exists to catch, and a
        # `"security_classes": []` alongside nonempty outcomes would then reclassify
        # every critical_failure on the task as unclassified (behavioral) instead of
        # raising — the same silently-skipped-veto failure mode this module refuses
        # everywhere else. Checking `in` instead of truthiness is what tells "absent"
        # from "present but wrong length" apart.
        classes = t["security_classes"] if "security_classes" in t else [None] * len(outcomes)
        n = 0
        for o, c in zip(outcomes, classes, strict=True):
            if o != "critical_failure":
                continue
            klass = "mechanical" if c == "mechanical" else "behavioral"
            if security_class is None or klass == security_class:
                n += 1
        if n:
            out[name] = n
    return out


def _regressions(baseline: dict, candidate: dict, security_class: str) -> dict[str, list[int]]:
    """Per-task [baseline, candidate] counts for the tasks whose ONE-class count rose.

    Shared between `evaluate()` (full-suite counts) and `confirmed()` (confirmation-
    pair counts) so the two never compute "a rise" two different ways.
    """
    b = security_failures(baseline, security_class)
    c = security_failures(candidate, security_class)
    return {n: [b.get(n, 0), c[n]] for n in c if c[n] > b.get(n, 0)}


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


def _supported_means(
    per: dict[str, Fraction], splits: dict[str, str], supported: frozenset[str]
) -> tuple[Fraction, Fraction]:
    """The SUPPORTED-SET split means — the exact quantity a section's bounds were
    measured on (``loop.calibrate._supported_split_mean``, same denominator, same
    exact arithmetic). A split with no supported task present means 0, not a
    division by zero: the section simply has no evidence on that split here.
    """

    def mean(split: str) -> Fraction:
        vals = [v for n, v in per.items() if splits[n] == split and n in supported]
        return sum(vals) / len(vals) if vals else Fraction(0)

    return mean("held_in"), mean("held_out")


def _require_supported(where: str, calibration: SectionCalibration, *results: dict) -> None:
    """Every supported task must be present in every arm, or refuse.

    The bound was measured on a mean over a NAMED set of tasks. Computing the mean
    over whatever subset happens to be present divides by a different denominator and
    compares the result to a threshold that was never about it — a candidate missing
    one held-in task of three would be judged on 1/2 of a sum against a bound built
    from 1/3 of one. Nothing in the record would say so: the number simply moves.
    A short denominator is a broken comparison, not a smaller one.
    """
    present = set.intersection(*(set(r["tasks"]) for r in results))
    missing = sorted(calibration.supported - present)
    if missing:
        raise ValueError(
            f"{where}: calibrated section {calibration.section!r} is missing supported "
            f"task(s) {', '.join(missing)} — the bounds in {calibration.source} were "
            f"measured on the mean over {', '.join(sorted(calibration.supported))}, and a "
            "mean over a smaller set is not the quantity they bound"
        )


def _collapses(baseline: dict, candidate: dict, excluded: frozenset[str] | set[str]) -> list[str]:
    """Tasks that went from a full pass to zero — the catastrophic per-task veto.

    Shared by `evaluate()` and `confirmed()` so the two cannot drift: the confirmation
    is the ONLY road to ACCEPT, so a protection that runs on the first measurement and
    not on the second is a protection with a door in it.
    """
    return sorted(
        n
        for n in baseline["tasks"]
        if n not in excluded
        and _exact(baseline["tasks"][n]) == 1
        and _exact(candidate["tasks"][n]) == 0
    )


def evaluate(
    baseline: dict,
    candidate: dict,
    *,
    excluded: frozenset[str] | set[str] = frozenset(),
    always_confirm: frozenset[str] | set[str] = frozenset(),
    calibration: SectionCalibration | None = None,
    unreachable_probable: frozenset[str] | set[str] = frozenset(),
) -> Decision:
    """One full-suite validation's verdict: REJECT or CONFIRM, never ACCEPT.

    ``always_confirm`` names tasks the confirmation pair must rerun even when they did
    not move — the edit's GUARDS. Selecting only the movers lets a candidate confirm
    its E3/E4-style gain without ever re-testing the trade-off the section is known to
    carry (E1's retrieval economy for tool_output) or the security checks whose
    critical outcomes might appear only under the confirmation's higher attempt
    counts. An unchanged guard is exactly the task whose stability the confirmation
    exists to re-establish.

    ``calibration`` switches the gain/noise judgment — and ONLY that judgment — onto
    the section's measured footing (contract §4): the supported-set split means
    against the artifact's ``noise_in``/``noise_ho``, instead of the whole-split means
    against ``one_attempt``. Its ``guards`` join ``always_confirm``. Everything else
    is deliberately untouched: the collapse veto, the mechanical security veto and the
    behavioral routing all keep reading the WHOLE suite, because a harness that leaks
    on a task outside the supported set has still leaked. The whole-split means the
    calibrated run stopped judging on are kept in ``raw`` — a rule that narrows what
    it reads must not also hide what it stopped reading.

    ``unreachable_probable`` names EVIDENCE-grade exclusions: tasks the edited knob
    showed no activity for, where the knob could CREATE that activity (lowering
    ``compaction.trigger_fraction`` makes a task compact that never has). They are
    context, recorded with the caveat, and NEVER subtracted — unlike ``excluded``,
    which is proof-grade and zeroes a movement. Passing it does not change any
    verdict; it only adds a line to the record.
    """
    _parity(baseline, candidate)
    if calibration is not None:
        _require_supported("evaluate", calibration, baseline, candidate)
    per, d_in, d_ho = _split_deltas(baseline, candidate, excluded)
    splits = {n: t["split"] for n, t in baseline["tasks"].items()}
    # The whole-split means, kept whatever regime decides — see `raw_evidence` below.
    full_in, full_ho = d_in, d_ho
    if calibration is None:
        thr_in, thr_ho = one_attempt(baseline, "held_in"), one_attempt(baseline, "held_out")
        bound = "one attempt"
    else:
        d_in, d_ho = _supported_means(per, splits, calibration.supported)
        # `Fraction(float)` is exact (it takes the float's true binary value), so the
        # comparisons below stay exact arithmetic the way the rest of this module is.
        thr_in, thr_ho = Fraction(calibration.noise_in), Fraction(calibration.noise_ho)
        bound = "the measured section noise"

    reasons: list[str] = []
    if d_in < -thr_in:
        reasons.append(
            f"held-in regressed beyond {bound} ({float(d_in):+.4f} < -{float(thr_in):.4f})"
        )
    if d_ho < -thr_ho:
        reasons.append(
            f"held-out regressed beyond {bound} ({float(d_ho):+.4f} < -{float(thr_ho):.4f})"
        )

    # Complete-collapse veto, causal-filtered: a collapse the edited section cannot
    # reach is the grader's noise wearing the candidate's name.
    collapses = _collapses(baseline, candidate, excluded)
    if collapses:
        reasons.append("full-pass task collapsed to zero: " + ", ".join(collapses))

    # Security, from outcomes, independent of the mean, and NOT causal-filtered — a
    # leak is a leak whichever knob was being edited. Split by class: a MECHANICAL
    # rise blocks unconditionally, here, whatever the averages say. A BEHAVIORAL rise
    # does not — at full-suite attempt counts one extra critical outcome is within the
    # measured base rate, so it routes to the confirmation pair instead, where the
    # predeclared Fisher test decides it on more data.
    base_sec, cand_sec = security_failures(baseline), security_failures(candidate)
    mech_reg = _regressions(baseline, candidate, "mechanical")
    beh_reg = _regressions(baseline, candidate, "behavioral")
    # security_regressions keeps its historical, pre-split meaning: TOTAL critical-
    # failure regressions, unfiltered by class — computed straight from base_sec/
    # cand_sec, same formula this field always used. A plain `mech_reg | beh_reg`
    # looks equivalent but silently drops one class's counts on a task that regressed
    # in BOTH within the same run (dict union lets the later operand win a key
    # collision — reachable in practice, since C3 classifies per attempt and a task's
    # several attempts can land different classes) — this field must never contradict
    # the class-specific reason text sitting beside it, so it is never derived from a
    # merge of the class-filtered dicts.
    sec_reg = {
        n: [base_sec.get(n, 0), cand_sec[n]] for n in cand_sec if cand_sec[n] > base_sec.get(n, 0)
    }
    if mech_reg:
        reasons.append(
            "harness storage contract regressed (mechanical): "
            + ", ".join(f"{n} {v[0]}->{v[1]}" for n, v in sorted(mech_reg.items()))
        )
    # Behavioral regressions do NOT veto here: at full-suite attempt counts a 0->1 is
    # within the known model base rate (~12%/attempt on C3 historically). They route:
    # the task joins the confirmation pair, where 10v10 counts feed the predeclared
    # Fisher comparison. A confirmed increase blocks THERE; an inconclusive does not.

    # Evidence-grade exclusions ride along as CONTEXT on every decision this call can
    # return, and as nothing else. Kept out of `reasons` until each return site
    # assembles it: appending to `reasons` here would make the `if reasons:` REJECT
    # below fire on a run whose only "reason" is a note about what was not subtracted.
    context: tuple[str, ...] = ()
    if unreachable_probable:
        context = (
            "unreachable_probable, kept in the means (evidence-grade, not proof): "
            + ", ".join(sorted(unreachable_probable))
            + " — the edited knob showed no activity on these, but it can CREATE that "
            "activity, so their absence is a prompt to re-measure and never a verdict; "
            "their movements are counted in full above",
        )

    raw_evidence = {"delta_in": float(d_in), "delta_ho": float(d_ho)}
    if calibration is not None:
        raw_evidence["regime"] = "section_calibration"
        raw_evidence["calibration"] = calibration.to_json()
        raw_evidence["full_split_delta_in"] = float(full_in)
        raw_evidence["full_split_delta_ho"] = float(full_ho)
        raw_evidence["unreachable_probable"] = sorted(unreachable_probable)
    # Both REJECT branches below carry `behavioral_regressions`/`targeted_rerun` too,
    # not just `security_regressions` — `beh_reg` is computed unconditionally above,
    # independent of which reason (or no reason at all) ends up rejecting. A behavioral
    # rise that merely CO-OCCURS with a mechanical veto, a collapse, a split regression,
    # or a no-gain rejection is still an observation this run made; the fact that
    # something else already decided the outcome must not make that observation vanish
    # from the record. Nothing routes anywhere from a REJECT (there is no confirmation
    # ahead of it), so these two fields are audit trail here, not routing instructions —
    # exactly like `security_regressions` already is on this same path.
    if reasons:
        return Decision(
            outcome=REJECT,
            reasons=tuple(reasons) + context,
            delta_in=float(d_in),
            delta_ho=float(d_ho),
            threshold_in=float(thr_in),
            threshold_ho=float(thr_ho),
            excluded=tuple(sorted(excluded)),
            security_regressions=sec_reg,
            behavioral_regressions=beh_reg,
            targeted_rerun=tuple(sorted(beh_reg)),
            raw=raw_evidence,
        )

    evidence_split = "held_in" if d_in > thr_in else ("held_out" if d_ho > thr_ho else "")
    if not evidence_split:
        no_gain = (
            "no gain beyond one attempt on either split — indistinguishable from the "
            "measured null variation, nothing to confirm"
            if calibration is None
            else (
                f"no supported-set gain beyond {bound} on either split — the movement is "
                f"within measured noise (held-in ±{float(thr_in):.4f}, held-out "
                f"±{float(thr_ho):.4f}, measured from {calibration.source}), nothing to confirm"
            )
        )
        return Decision(
            outcome=REJECT,
            reasons=(no_gain,) + context,
            delta_in=float(d_in),
            delta_ho=float(d_ho),
            threshold_in=float(thr_in),
            threshold_ho=float(thr_ho),
            excluded=tuple(sorted(excluded)),
            security_regressions=sec_reg,
            behavioral_regressions=beh_reg,
            targeted_rerun=tuple(sorted(beh_reg)),
            raw=raw_evidence,
        )

    # The gain's basis is whatever the judgment was made on: the supported set under a
    # calibration, the whole split otherwise. A confirmation re-tests exactly this.
    basis = (
        per if calibration is None else {n: v for n, v in per.items() if n in calibration.supported}
    )
    improved = tuple(sorted(n for n, v in basis.items() if splits[n] == evidence_split and v > 0))
    moved = {n for n, v in per.items() if v != 0}
    # The confirmation reruns every task that moved, every task that showed a security
    # failure on either side (a steady leak count still deserves a look at higher
    # attempt counts — unfiltered, so a behavioral-classed task rides in here too), and
    # every named guard whether it moved or not. `moved` stays WHOLE-SUITE under a
    # calibration: narrowing the judgment must never narrow what gets re-measured.
    #
    # A calibration adds its WHOLE SUPPORTED SET on top of its guards, movement or no
    # movement. Those tasks are the bound's own basis — the confirmation has to compute
    # the same supported-set mean this decision was judged on, and it cannot do that
    # from a subset. G4 (compaction's miner) is the case that makes this concrete: never
    # a guard, usually unmoved, and one of the four tasks the noise floor is measured
    # over. Selecting only movers and guards would leave the confirmation unable to
    # reproduce the very quantity it exists to re-test.
    required = set(always_confirm)
    if calibration is not None:
        required |= set(calibration.guards) | set(calibration.supported)
    confirm = tuple(sorted(moved | set(base_sec) | set(cand_sec) | required))
    confirm_reasons = [
        f"gain beyond {bound} on {evidence_split.replace('_', '-')} "
        f"(carried by {', '.join(improved)})",
    ]
    if beh_reg:
        confirm_reasons.append(
            "behavioral security movement routed to confirmation: " + ", ".join(sorted(beh_reg))
        )
    return Decision(
        outcome=CONFIRM,
        reasons=tuple(confirm_reasons) + context,
        delta_in=float(d_in),
        delta_ho=float(d_ho),
        threshold_in=float(thr_in),
        threshold_ho=float(thr_ho),
        excluded=tuple(sorted(excluded)),
        evidence_split=evidence_split,
        improved_tasks=improved,
        confirm_tasks=confirm,
        targeted_rerun=tuple(sorted(beh_reg)),
        security_regressions=sec_reg,
        behavioral_regressions=beh_reg,
        raw=raw_evidence,
    )


def confirmed(
    first: Decision,
    confirm_baseline: dict,
    confirm_candidate: dict,
    *,
    excluded: frozenset[str] | set[str] = frozenset(),
    calibration: SectionCalibration | None = None,
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
      allowance, and a MECHANICAL security count above the confirmation baseline
      blocks unconditionally — not merely a repeat of the first run's regressions,
      and it does not matter that the leak appeared here for the first time. A
      BEHAVIORAL count instead feeds the predeclared Fisher test at THIS run's
      attempt counts: "confirmed_increase" blocks, "inconclusive"/"no_increase" do
      not — but every verdict is recorded in ``raw["behavioral_verdicts"]`` regardless
      of which way the Decision comes out.

    ``calibration`` does here exactly what it does in ``evaluate()``: the split
    judgments move onto the supported-set means and the measured bounds, and the
    repeat test asks the ORIGINAL question again — the improvement must clear the same
    measured bound that made it evidence in the first place, not a one-attempt grain
    re-derived from this run's (deliberately larger) attempt counts, which shrinks as
    attempts grow and would let a confirmation clear a bar the first decision never
    had to. A calibrated first decision REQUIRES its calibration here; being confirmed
    against the weaker whole-split bar is not a milder outcome, it is a different
    question answered under the same word.
    """
    # Before anything else, including the non-CONFIRM passthrough: a decision that
    # records `regime == "section_calibration"` was judged against measured bounds, and
    # arriving here without them is a WIRING failure wherever it happens. Letting it
    # through would silently re-decide a calibrated claim on the one-attempt grain.
    if calibration is None and (first.raw or {}).get("regime") == "section_calibration":
        cal_json = (first.raw or {}).get("calibration") or {}
        raise ValueError(
            "this first decision was judged under a section calibration "
            f"({cal_json.get('section', 'unknown section')}, bounds from "
            f"{cal_json.get('source', 'an artifact')}) — confirming it without that "
            "calibration would judge the repeat against the weaker one-attempt bound. "
            "Load the section's calibration and pass calibration=; if it is no longer "
            "fresh, the claim needs re-measuring, not re-judging."
        )
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
    if calibration is not None:
        # Reachable through a record written before `confirm_tasks` carried the whole
        # supported set: the pair covers exactly what that record asked for, and still
        # cannot compute the mean the bound belongs to.
        _require_supported("confirmation", calibration, confirm_baseline, confirm_candidate)

    per, d_in, d_ho = _split_deltas(confirm_baseline, confirm_candidate, excluded)
    if calibration is None:
        thr_in = one_attempt(confirm_baseline, "held_in")
        thr_ho = one_attempt(confirm_baseline, "held_out")
    else:
        splits = {n: t["split"] for n, t in confirm_baseline["tasks"].items()}
        d_in, d_ho = _supported_means(per, splits, calibration.supported)
        thr_in = Fraction(calibration.noise_in)
        thr_ho = Fraction(calibration.noise_ho)
    report = (float(d_in), float(d_ho), float(thr_in), float(thr_ho))

    reasons: list[str] = []
    if d_in < -thr_in:
        reasons.append(f"held-in regressed in confirmation ({float(d_in):+.4f})")
    if d_ho < -thr_ho:
        reasons.append(f"held-out regressed in confirmation ({float(d_ho):+.4f})")

    # The catastrophic per-task veto, same rule and same wording as `evaluate()`'s
    # (contract §4 amendment). This function is the ONLY road to ACCEPT, so every
    # whole-suite protection has to hold here too — and under a calibration the split
    # means read the supported set alone, which makes a full-pass task outside that set
    # collapsing to zero invisible to every other check in this function. It is not a
    # calibration-only guard: an uncalibrated pair could always hide a collapse behind
    # a split mean that stayed inside its allowance, and now it cannot.
    collapses = _collapses(confirm_baseline, confirm_candidate, excluded)
    if collapses:
        reasons.append("full-pass task collapsed to zero: " + ", ".join(collapses))

    # Mechanical: same unconditional veto as evaluate(), same reason wording — it does
    # not matter that this is the confirmation and not the first run.
    mech_reg = _regressions(confirm_baseline, confirm_candidate, "mechanical")
    # Behavioral counts belong on EVERY Decision this function returns, including the
    # ACCEPT. `pr_body` recovers the mechanical component as total-minus-behavioral, so
    # an empty `behavioral_regressions` beside a nonzero `security_regressions` makes a
    # behavioral rise render as MECHANICAL to the human approving the merge — and a
    # mechanical rise is the one that would have rejected the candidate outright. The
    # PR would state the opposite of what the rule decided.
    beh_reg = _regressions(confirm_baseline, confirm_candidate, "behavioral")
    if mech_reg:
        reasons.append(
            "harness storage contract regressed (mechanical): "
            + ", ".join(f"{n} {v[0]}->{v[1]}" for n, v in sorted(mech_reg.items()))
        )

    # Behavioral: every task with a nonzero behavioral count on either side gets a
    # verdict from the predeclared Fisher test at THIS confirmation's own attempt
    # counts, PLUS every task routed here in the first place (``first.targeted_rerun``)
    # even if it comes back clean on both sides — a routed task that reconfirms clean
    # must still show a "no_increase" verdict, or "scrutinized and cleared" is
    # indistinguishable from "never checked" in the record. Every verdict is kept, not
    # just the ones that rose, so the record shows what was tested, not only what
    # tripped. Only "confirmed_increase" blocks.
    base_beh = security_failures(confirm_baseline, "behavioral")
    cand_beh = security_failures(confirm_candidate, "behavioral")
    watch = set(first.targeted_rerun) | set(base_beh) | set(cand_beh)
    behavioral_verdicts: dict[str, dict] = {}
    for n in sorted(watch):
        bn = int(confirm_baseline["tasks"][n]["attempts"])
        cn = int(confirm_candidate["tasks"][n]["attempts"])
        verdict = targeted_security_verdict(base_beh.get(n, 0), bn, cand_beh.get(n, 0), cn)
        behavioral_verdicts[n] = verdict
        if verdict["verdict"] == "confirmed_increase":
            p = verdict["p_one_sided"]
            reasons.append(f"behavioral security increase confirmed on {n} (p={p:.3f})")

    # Same restoration as evaluate(): security_regressions is the unfiltered total,
    # never a class merge that can drop one class's counts on a dual-class task.
    base_sec = security_failures(confirm_baseline)
    cand_sec = security_failures(confirm_candidate)
    sec_reg = {
        n: [base_sec.get(n, 0), cand_sec[n]] for n in cand_sec if cand_sec[n] > base_sec.get(n, 0)
    }

    # The original improvement, on the original tasks, on the original split.
    basis = [per[n] for n in first.improved_tasks]
    if calibration is None:
        grain = (
            max(
                Fraction(1, int(confirm_baseline["tasks"][n]["attempts"]))
                for n in first.improved_tasks
            )
            / len(first.improved_tasks)
            if basis
            else Fraction(0)
        )
    else:
        # The measured bound for the split the evidence was on — the same bar the
        # first decision had to clear, held steady across attempt counts.
        grain = Fraction(
            calibration.noise_ho if first.evidence_split == "held_out" else calibration.noise_in
        )
    repeated = bool(basis) and sum(basis) / len(basis) > grain
    if not repeated:
        reasons.append(
            "original improvement did not repeat on "
            + ", ".join(first.improved_tasks)
            + " — a gain elsewhere is a new claim, not a confirmation"
        )

    raw = {"stage": "confirmation", "behavioral_verdicts": behavioral_verdicts}
    if calibration is not None:
        raw["regime"] = "section_calibration"
        raw["calibration"] = calibration.to_json()
    if reasons:
        return Decision(
            outcome=REJECT,
            reasons=tuple(reasons),
            delta_in=report[0],
            delta_ho=report[1],
            threshold_in=report[2],
            threshold_ho=report[3],
            excluded=first.excluded,
            evidence_split=first.evidence_split,
            improved_tasks=first.improved_tasks,
            confirm_tasks=first.confirm_tasks,
            security_regressions=sec_reg,
            behavioral_regressions=beh_reg,
            raw=raw,
        )
    return Decision(
        outcome=ACCEPT,
        reasons=(
            "original improvement repeated under a fresh paired confirmation on "
            + ", ".join(first.improved_tasks),
        ),
        delta_in=report[0],
        delta_ho=report[1],
        threshold_in=report[2],
        threshold_ho=report[3],
        excluded=first.excluded,
        evidence_split=first.evidence_split,
        improved_tasks=first.improved_tasks,
        confirm_tasks=first.confirm_tasks,
        security_regressions=sec_reg,
        behavioral_regressions=beh_reg,
        raw=raw,
    )
