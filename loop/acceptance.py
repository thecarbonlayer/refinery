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

A SECTION CALIBRATION replaces that derived allowance with a MEASURED NULL MODEL, for
one editable section at a time. `one_attempt` is a bound on
what a single attempt can move a whole split's mean — a structural stand-in for null
variation, correct for `tool_output` because six unchanged runs said so. For a section
with its own supported task set, null arms measure the real thing: one exact pooled
pass rate per supported task, with nothing changed between the arms that produced them.

The calibration carries THOSE RATES, never a threshold. Every bound this module
compares against is computed HERE, at judgment time, from the null rates and the ACTUAL
per-task attempt counts of the two runs being compared — the exact distribution of
(candidate mean − baseline mean) under "both arms drawn at the same rate", enumerated
with `Fraction` arithmetic (`loop.calibrate.null_gain_quantile` /
`null_task_quantile`), never sampled. That is the round-1 defect closed structurally: a
threshold measured at ten attempts per task is not a threshold for a judgment made at
three, and a stored number cannot know which one it is about. The gate reads its own
counts, so it cannot be wrong about them.

Passing a ``SectionCalibration`` therefore swaps both halves of the gain/noise judgment
— the quantity (supported-set mean instead of whole-split mean) and the bound (a
computed null quantile instead of one attempt) — and adds two gates the ACCEPT road did
not have: each GUARD task is adjudicated against its own null distribution (guards were
witnesses; they are gates now), and both supported-split means must be non-negative for
an ACCEPT. Every whole-suite protection keeps reading the WHOLE suite: a leak on an
unsupported task is still a leak, and a collapse there still vetoes. Without a
calibration this module behaves exactly as it did before the mechanism existed,
byte for byte, including its reason strings — except the confirmation collapse veto
(added by a later amendment): ``confirmed()`` runs the catastrophic per-task check
unconditionally rather than only under a calibration, closing a hole where an
uncalibrated confirmation pair could hide a full-pass-to-zero collapse behind a split
mean that stayed inside its allowance.

Lives in `loop/`, not `runner/`: `runner_sha` is the verifier's identity and a
governance rule must not invalidate baselines when it changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache
from math import comb
from types import MappingProxyType

from loop.calibrate import null_gain_quantile, null_task_quantile

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


def _frac(x: Fraction) -> str:
    """A ``Fraction`` as the exact ``"numerator/denominator"`` string that goes into a
    reason and a committed record. Never ``float()`` — a reason naming ``0.1067`` says
    nothing a reader can re-derive, and the comparison it reports was never made on
    that number."""
    return f"{x.numerator}/{x.denominator}"


@dataclass(frozen=True)
class SectionCalibration:
    """The measured NULL MODEL for ONE editable section.

    Every number in here comes out of a calibration artifact produced by
    ``loop.calibrate.calibrate_model`` from arms with NOTHING changed between them —
    ``loop.validate.section_calibration()`` is the only constructor the pipeline uses,
    and it refuses an artifact that is stale, unfit, or measured in a different world.
    No threshold in this repo is written by hand, which is the whole point: the rule
    that decides what counts as movement must not be able to invent its own floor.

    What it deliberately does NOT carry is a threshold. Round 1 stored bounds, and a
    stored bound cannot know the attempt counts of the judgment it is about to gate —
    a bound measured at ten attempts per task gated a three-attempt run at a tenth of
    the resolution that run could even produce. The rates are the model; the bound is
    computed per judgment, from these rates and that judgment's own counts.

    - ``supported``: the tasks the model covers. Gain and regression are judged on the
      mean over THESE tasks, per split, and nothing else.
    - ``null_rates``: each supported task's pooled null pass rate, as an EXACT
      ``Fraction`` of the pooled integer counts, in a read-only mapping (the memos key
      on these values, so a post-construction edit would move the bound underneath
      quantiles already cached at the old rate). Every rate must be strictly interior,
      ``0 < rate < 1``: a rate of exactly 0 or 1 gives that
      task a null quantile of 0 at every attempt count, which is a gate nothing can
      fail. Exact, and checked to be exact: a
      float rate is refused here rather than at the arithmetic, because
      ``loop.calibrate``'s memoized binomial pmf keys on ``(n, p)`` and Python hashes
      ``0.5`` and ``Fraction(1, 2)`` as the same key — one float would silently hand
      every later exact call a cached tuple computed for a value that had already
      rounded.
    - ``coverage_level``: the one-sided coverage every quantile is computed at
      (97.5%, i.e. 39/40 exactly), exact for the same reason.
    - ``guards``: tasks a confirmation must rerun even unmoved (added to
      ``always_confirm``) AND now adjudicate against their own null distribution
      the guard gate. A guard the model has no rate for could not be adjudicated at
      all, so ``guards`` must be a subset of ``supported`` — a guard silently skipped
      is exactly the not-a-gate this phase exists to remove.
    - ``source``: where the artifact came from, repo-relative — this string lands in
      a committed record, so it must never carry a machine path.
    - ``computed_at_runner_sha``: the verifier version the arms behind this model were
      measured at. Carried so a PR body can state the artifact's IDENTITY — path plus
      the version it was computed at — instead of naming a file and leaving the reader
      to guess which build of it. Empty only for a calibration built by hand in a test.
    """

    section: str
    supported: frozenset[str]
    null_rates: dict[str, Fraction]
    coverage_level: Fraction
    guards: frozenset[str]
    source: str
    computed_at_runner_sha: str = ""

    def __post_init__(self) -> None:
        if not self.supported:
            raise ValueError("SectionCalibration: the supported set cannot be empty")
        missing = sorted(set(self.supported) - set(self.null_rates))
        extra = sorted(set(self.null_rates) - set(self.supported))
        if missing or extra:
            raise ValueError(
                f"SectionCalibration({self.section!r}): null_rates must cover exactly the "
                f"supported set — missing a rate for {missing}, carrying a rate for the "
                f"unsupported {extra}. A supported task with no null rate has no "
                "distribution to be judged against, and a rate for a task outside the "
                "set is a number nothing will ever read."
            )
        bad = sorted(t for t, r in self.null_rates.items() if not isinstance(r, Fraction))
        if bad:
            raise ValueError(
                f"SectionCalibration({self.section!r}): null_rates must be exact Fraction "
                f"values, got a non-Fraction rate for {bad} — a float rate compares equal "
                "to an exact Fraction and silently poisons the memoized binomial cache "
                "every later exact call at the same (attempts, rate) reads from"
            )
        out_of_range = sorted(t for t, r in self.null_rates.items() if not 0 <= r <= 1)
        if out_of_range:
            raise ValueError(
                f"SectionCalibration({self.section!r}): a null rate outside [0, 1] is not "
                f"a pass rate: {out_of_range}"
            )
        # Contract §4.1 amendment: DEGENERATE rates do not install. A pooled rate of
        # exactly 0 or 1 makes that task's null difference distribution a point mass at
        # zero, so its quantile is 0 at EVERY attempt count and any movement at all
        # clears it — the per-task twin of the grain check's "a threshold below the
        # finest representable movement gates nothing". Reviewer's reproduction: an
        # artifact with G4 pooled 0/49 passes every fitness check, installs, and lets a
        # single-attempt repeat ACCEPT. A task that never passes (or never fails) across
        # all arms is not calibrated evidence; the honest move is to refuse and let a
        # human look at the task.
        degenerate = sorted(t for t, r in self.null_rates.items() if r in (0, 1))
        if degenerate:
            named = ", ".join(f"{t}={_frac(self.null_rates[t])}" for t in degenerate)
            raise ValueError(
                f"SectionCalibration({self.section!r}): degenerate null rate(s) {named} — "
                "every supported task needs 0 < rate < 1. A rate of exactly 0 or 1 gives "
                "that task a null quantile of 0 at every attempt count, which is a gate "
                "nothing can fail. Re-measure the task or take it to a human; do not "
                "install a bound that is really an absence."
            )
        if not isinstance(self.coverage_level, Fraction):
            raise ValueError(
                f"SectionCalibration({self.section!r}): coverage_level must be an exact "
                f"Fraction, got {type(self.coverage_level).__name__} "
                f"({self.coverage_level!r})"
            )
        if not 0 < self.coverage_level <= 1:
            raise ValueError(
                f"SectionCalibration({self.section!r}): coverage_level must be in (0, 1], "
                f"got {self.coverage_level}"
            )
        ungated = sorted(set(self.guards) - set(self.supported))
        if ungated:
            raise ValueError(
                f"SectionCalibration({self.section!r}): guard(s) {ungated} are outside the "
                "supported set, so the null model has no distribution to adjudicate their "
                "drop against. Contract §2 made guards GATES; a guard that cannot be "
                "gated must not be installed as one."
            )
        # Read-only from here on. `_cached_gain_quantile`/`_cached_task_quantile` memoize
        # on the rate values, so a caller mutating this mapping after construction would
        # move the bound for every judgment that followed while every already-computed
        # quantile stayed at the old rate — a threshold silently disagreeing with the
        # model it claims to come from, and no error anywhere. A frozen dataclass blocks
        # rebinding the attribute; it does nothing about mutating the dict behind it.
        object.__setattr__(self, "null_rates", MappingProxyType(dict(self.null_rates)))

    def to_json(self) -> dict:
        return {
            "section": self.section,
            "supported": sorted(self.supported),
            "null_rates": {t: _frac(self.null_rates[t]) for t in sorted(self.null_rates)},
            "coverage_level": _frac(self.coverage_level),
            "guards": sorted(self.guards),
            "source": self.source,
            "computed_at_runner_sha": self.computed_at_runner_sha,
        }


def _sha256_json(payload: dict) -> str:
    """A stable sha256 over a JSON payload — sorted keys, no incidental whitespace.

    Both digests below are only as good as their canonicalization: a digest that
    changes when a dict is re-serialized in a different order refuses honest
    records and teaches whoever meets it to route around the check.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def calibration_digest(calibration: SectionCalibration) -> str:
    """A digest of the three things a calibrated judgment actually depends on: the
    null RATES, the COVERAGE LEVEL, and the SOURCE the two came from.

    Both stages of this pipeline judge against a null model, and until this existed
    nothing tied them to the SAME one. A first pass under a strict model and a
    confirmation under a loose one both look clean in their own records: each check
    (fit, freshness, pinned supported set) passes independently on each artifact, and
    the pair of decisions reads as one coherent story it never was. The first
    decision records this digest; ``confirmed()`` recomputes it from the calibration
    in its hand and refuses a mismatch.

    Deliberately NOT over the whole artifact: `fitness`, `provenance` and the power
    rows are audit material, and re-running `loop.calibrate --model` on the same arms
    to add a published number would otherwise invalidate every in-flight claim. What
    must not move under a claim is the numbers the claim was judged with.
    """
    return _sha256_json(
        {
            "null_rates": {t: _frac(r) for t, r in sorted(calibration.null_rates.items())},
            "coverage_level": _frac(calibration.coverage_level),
            "source": calibration.source,
        }
    )


def _decision_digest_payload(improved_tasks, confirm_tasks, regime: str | None) -> dict:
    return {
        "improved_tasks": list(improved_tasks),
        "confirm_tasks": list(confirm_tasks),
        "regime": regime,
    }


def decision_digest(decision: Decision) -> str:
    """A digest of the first decision's own CLAIM: which tasks carried the gain,
    which tasks the confirmation must rerun, and under which regime.

    The confirmation re-tests every task in ``improved_tasks``, each against its own
    null quantile, and ALL of them must repeat. So a record edited from three
    carriers down to one is not a smaller claim, it is an easier exam — and the
    record is a JSON file on disk that a later step reloads and trusts. The digest is
    written by the decision that made the claim and checked by whatever reloads it.
    """
    return _sha256_json(
        _decision_digest_payload(
            decision.improved_tasks, decision.confirm_tasks, (decision.raw or {}).get("regime")
        )
    )


@dataclass(frozen=True)
class Decision:
    outcome: str  # REJECT | CONFIRM (evaluate) | ACCEPT (confirmed only)
    reasons: tuple[str, ...]
    delta_in: float
    delta_ho: float
    # The bound the split was actually judged against, as a float for the record: one
    # attempt uncalibrated, the computed null-model quantile under a calibration. The
    # EXACT fraction and the counts it was computed at live in `raw["null_quantiles"]`.
    threshold_in: float
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


@cache
def _cached_gain_quantile(
    rates: tuple[tuple[str, Fraction], ...],
    attempts_a: tuple[tuple[str, int], ...],
    attempts_b: tuple[tuple[str, int], ...],
    level: Fraction,
) -> Fraction:
    """``null_gain_quantile`` behind a memo, keyed on hashable views of its inputs.

    Pure function, exact inputs, exact output — the memo cannot change an answer, only
    skip recomputing it. It matters because the enumeration is the expensive part of a
    judgment (a three-task split at fifty attempts a side convolves distributions over
    big-integer rationals), and one decision computes the same quantile several times:
    both splits in ``evaluate()``, both again in ``confirmed()``, and a loop judging
    several candidates against ONE calibration repeats every one of them.
    """
    return null_gain_quantile(dict(rates), dict(attempts_a), dict(attempts_b), level)


@cache
def _cached_task_quantile(rate: Fraction, attempts_a: int, attempts_b: int, level: Fraction):
    """``null_task_quantile`` behind the same memo, for the per-carrier and per-guard
    gates, which ask for one quantile per task and often the same one twice."""
    return null_task_quantile(rate, attempts_a, attempts_b, level)


def _attempts(results: dict, task: str) -> int:
    return int(results["tasks"][task]["attempts"])


def _split_tasks(calibration: SectionCalibration, splits: dict[str, str], split: str) -> list[str]:
    return sorted(t for t in calibration.supported if splits.get(t) == split)


def _gain_quantile(
    calibration: SectionCalibration,
    tasks: list[str],
    baseline: dict,
    candidate: dict,
) -> Fraction:
    """The gain gate's bound for one split, computed NOW, at judgment time.

    The null distribution of (candidate supported-split mean − baseline supported-split
    mean) with both arms binomial at the pooled rates, at each side's REAL attempt
    counts, read off the results being judged rather than off the artifact. A split
    with no supported task present has no such distribution and no evidence either —
    its mean is a structural zero (``_supported_means``), and a bound of zero is the
    only bound consistent with that: nothing to clear, nothing to fall below.
    """
    if not tasks:
        return Fraction(0)
    return _cached_gain_quantile(
        tuple((t, calibration.null_rates[t]) for t in tasks),
        tuple((t, _attempts(baseline, t)) for t in tasks),
        tuple((t, _attempts(candidate, t)) for t in tasks),
        calibration.coverage_level,
    )


def _task_quantile(
    calibration: SectionCalibration, task: str, baseline: dict, candidate: dict
) -> Fraction:
    """The same construction for ONE task — the repeat gate's bound per carrier and the
    guard gate's bound per guard. Refuses a task the model has no rate for instead of
    substituting the split's mean bound, which is round 1's third named defect ("the
    confirmation gate applies a 3-task-mean bound to single-task deltas")."""
    rate = calibration.null_rates.get(task)
    if rate is None:
        raise ValueError(
            f"{calibration.section!r} calibration has no null rate for {task!r} — its "
            f"model covers {', '.join(sorted(calibration.supported))}, and a task's own "
            "repeat or drop cannot be judged against another task's distribution or "
            "against a mean over several. Re-measure, or re-decide: do not substitute."
        )
    return _cached_task_quantile(
        rate, _attempts(baseline, task), _attempts(candidate, task), calibration.coverage_level
    )


def _counts_note(tasks: list[str], baseline: dict, candidate: dict) -> str:
    """``"A1 50v50, G4 50v50"`` — the attempt counts a quantile was computed for, in
    every reason that names one. A threshold with no counts beside it is exactly the
    round-1 artifact: a number nobody could tell was about the wrong run."""
    return ", ".join(f"{t} {_attempts(baseline, t)}v{_attempts(candidate, t)}" for t in tasks)


def _quantile_record(
    calibration: SectionCalibration,
    tasks: list[str],
    quantile: Fraction,
    baseline: dict,
    candidate: dict,
) -> dict:
    """One computed bound, as JSON for the decision's ``raw`` — the tasks it covered,
    the counts it was computed at, the exact value, and the coverage it holds. The
    decision must be re-derivable from its own record."""
    return {
        "tasks": list(tasks),
        "attempts": {t: [_attempts(baseline, t), _attempts(candidate, t)] for t in tasks},
        "quantile": _frac(quantile),
        "coverage_level": _frac(calibration.coverage_level),
    }


def _require_supported(where: str, calibration: SectionCalibration, *results: dict) -> None:
    """Every supported task must be present in every arm, or refuse.

    The quantile is computed over a mean across a NAMED set of tasks. Computing the
    observed mean over whatever subset happens to be present divides by a different
    denominator and compares it against a bound built for another — a candidate missing
    one held-in task of three would be judged on 1/2 of a sum against a quantile
    enumerated over 1/3 of one. Nothing in the record would say so: the number simply
    moves. A short denominator is a broken comparison, not a smaller one.
    """
    present = set.intersection(*(set(r["tasks"]) for r in results))
    missing = sorted(calibration.supported - present)
    if missing:
        raise ValueError(
            f"{where}: calibrated section {calibration.section!r} is missing supported "
            f"task(s) {', '.join(missing)} — the null model in {calibration.source} covers "
            f"{', '.join(sorted(calibration.supported))}, and the quantile it produces is "
            "for a mean over all of them, not over whichever ones showed up"
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
    the section's measured null model: the supported-set split means
    against a quantile computed HERE, from the pooled null rates and THESE two runs'
    own per-task attempt counts, instead of the whole-split means against
    ``one_attempt``. CONFIRM requires the observed supported-set mean to exceed that
    quantile strictly; the other split is judged by the same construction in the
    losing direction (both arms share a rate and, past ``_parity``, a count, so the
    null distribution is symmetric about zero and ``-quantile`` is its lower edge).
    Its ``guards`` join ``always_confirm``. Everything else is deliberately untouched:
    the collapse veto, the mechanical security veto and the behavioral routing all
    keep reading the WHOLE suite, because a harness that leaks on a task outside the
    supported set has still leaked. The whole-split means the calibrated run stopped
    judging on are kept in ``raw``, next to the quantiles it computed and the counts
    it computed them at — a rule that narrows what it reads must not also hide what it
    stopped reading, or what it replaced it with.

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
    reasons: list[str] = []
    if calibration is None:
        thr_in, thr_ho = one_attempt(baseline, "held_in"), one_attempt(baseline, "held_out")
        bound = "one attempt"
        if d_in < -thr_in:
            reasons.append(
                f"held-in regressed beyond {bound} ({float(d_in):+.4f} < -{float(thr_in):.4f})"
            )
        if d_ho < -thr_ho:
            reasons.append(
                f"held-out regressed beyond {bound} ({float(d_ho):+.4f} < -{float(thr_ho):.4f})"
            )
    else:
        d_in, d_ho = _supported_means(per, splits, calibration.supported)
        # The bound, computed now, at these runs' own counts — nothing is read off the
        # artifact but the rates. See `_gain_quantile`.
        tasks_in = _split_tasks(calibration, splits, "held_in")
        tasks_ho = _split_tasks(calibration, splits, "held_out")
        thr_in = _gain_quantile(calibration, tasks_in, baseline, candidate)
        thr_ho = _gain_quantile(calibration, tasks_ho, baseline, candidate)
        for label, delta, thr, tasks in (
            ("held-in", d_in, thr_in, tasks_in),
            ("held-out", d_ho, thr_ho, tasks_ho),
        ):
            if delta < -thr:
                reasons.append(
                    f"{label} supported-set mean regressed beyond the null model "
                    f"({float(delta):+.4f} < -{_frac(thr)}, the "
                    f"{_frac(calibration.coverage_level)} null-model quantile computed at "
                    f"{_counts_note(tasks, baseline, candidate)})"
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
        # The bounds this decision was actually made against, exact, with the counts
        # they were computed at. Floats reach `threshold_in`/`threshold_ho` for the
        # report; the record keeps the fractions the comparison really used.
        raw_evidence["null_quantiles"] = {
            "held_in": _quantile_record(calibration, tasks_in, thr_in, baseline, candidate),
            "held_out": _quantile_record(calibration, tasks_ho, thr_ho, baseline, candidate),
        }
        # STAGE BINDING, half one: the calibration this decision was judged against,
        # digested. `confirmed()` recomputes it from the calibration IT was handed and
        # refuses a mismatch, so the two stages cannot be answered under two different
        # null models. Written on every calibrated decision, not only the CONFIRMs — a
        # REJECT's record is evidence too, and a reader should be able to tell which
        # model produced it.
        raw_evidence["calibration_digest"] = calibration_digest(calibration)
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
                "no supported-set gain beyond the null-model quantile on either split — "
                f"held-in {float(d_in):+.4f} vs {_frac(thr_in)} computed at "
                f"{_counts_note(tasks_in, baseline, candidate) or 'no supported task'}; "
                f"held-out {float(d_ho):+.4f} vs {_frac(thr_ho)} computed at "
                f"{_counts_note(tasks_ho, baseline, candidate) or 'no supported task'}; "
                f"both at {_frac(calibration.coverage_level)} coverage from the null rates "
                f"in {calibration.source}, nothing to confirm"
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
    if calibration is None:
        confirm_reasons = [
            f"gain beyond {bound} on {evidence_split.replace('_', '-')} "
            f"(carried by {', '.join(improved)})",
        ]
    else:
        won = d_in if evidence_split == "held_in" else d_ho
        won_thr = thr_in if evidence_split == "held_in" else thr_ho
        won_tasks = tasks_in if evidence_split == "held_in" else tasks_ho
        confirm_reasons = [
            f"gain beyond the null-model quantile on {evidence_split.replace('_', '-')} "
            f"(carried by {', '.join(improved)}): supported-set mean {float(won):+.4f} > "
            f"{_frac(won_thr)}, the {_frac(calibration.coverage_level)} null-model quantile "
            f"computed at {_counts_note(won_tasks, baseline, candidate)}",
        ]
    if beh_reg:
        confirm_reasons.append(
            "behavioral security movement routed to confirmation: " + ", ".join(sorted(beh_reg))
        )
    if calibration is not None:
        # STAGE BINDING, half two: this decision's own claim, digested — the carrier
        # set the confirmation must reproduce and the task set it must rerun. Only a
        # CONFIRM is ever reloaded and acted on, so only a CONFIRM carries it.
        raw_evidence["decision_digest"] = _sha256_json(
            _decision_digest_payload(improved, confirm, "section_calibration")
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

    ``calibration`` does here what it does in ``evaluate()``, and then three things
    more:

    - The split judgments move onto the supported-set means against quantiles computed
      at THIS pair's own attempt counts.
    - The repeat test is PER CARRIER, not per mean: each task in
      ``first.improved_tasks`` must gain beyond ITS OWN null quantile at these counts,
      and ALL of them must. Round 1 judged single-task deltas against a three-task-mean
      bound, which is how a null pair reproduced an end-to-end false ACCEPT: A1 moved
      two attempts out of ten between two arms with nothing changed between them, and
      that cleared a bound built for a mean.
    - Each GUARD is adjudicated the same way in the losing direction, and the
      supported-set means must be non-negative on BOTH splits. Guards were witnesses —
      rerun so a human could look at them. They are gates now: a guard dropping beyond
      its own null distribution REJECTs, naming the guard and the quantile.

    A calibrated first decision REQUIRES its calibration here; being confirmed against
    the weaker whole-split bar is not a milder outcome, it is a different question
    answered under the same word.
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
    if calibration is not None and (first.raw or {}).get("regime") == "section_calibration":
        # The other half of the same wiring check: a calibration IS in hand, but is it
        # the one this claim was judged against? Every other check in the loader passes
        # independently on any fit, fresh, correctly-pinned artifact, so two different
        # null models both install cleanly and the swap leaves no trace anywhere else.
        # An ABSENT digest refuses too: a record that never bound itself to a model is
        # exactly the record this check exists to stop being trusted.
        recorded = (first.raw or {}).get("calibration_digest")
        current = calibration_digest(calibration)
        if recorded != current:
            raise ValueError(
                "this first decision was judged against a DIFFERENT calibration than the "
                f"one handed to the confirmation: the record's calibration digest is "
                f"{recorded!r} and the calibration in hand digests to {current!r} (rates, "
                f"coverage level and source from {calibration.source}). Confirming a claim "
                "under a null model it was never judged against answers a different "
                "question under the same word. Re-validate the candidate against the "
                "calibration you intend to confirm it with."
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
    reasons: list[str] = []
    tasks_in: list[str] = []
    tasks_ho: list[str] = []
    if calibration is None:
        thr_in = one_attempt(confirm_baseline, "held_in")
        thr_ho = one_attempt(confirm_baseline, "held_out")
        if d_in < -thr_in:
            reasons.append(f"held-in regressed in confirmation ({float(d_in):+.4f})")
        if d_ho < -thr_ho:
            reasons.append(f"held-out regressed in confirmation ({float(d_ho):+.4f})")
    else:
        splits = {n: t["split"] for n, t in confirm_baseline["tasks"].items()}
        d_in, d_ho = _supported_means(per, splits, calibration.supported)
        tasks_in = _split_tasks(calibration, splits, "held_in")
        tasks_ho = _split_tasks(calibration, splits, "held_out")
        # Computed at THIS pair's counts, not the first decision's: the question is the
        # same one, asked of the measurement in hand. See `_gain_quantile`.
        thr_in = _gain_quantile(calibration, tasks_in, confirm_baseline, confirm_candidate)
        thr_ho = _gain_quantile(calibration, tasks_ho, confirm_baseline, confirm_candidate)
        for label, delta, thr, tasks in (
            ("held-in", d_in, thr_in, tasks_in),
            ("held-out", d_ho, thr_ho, tasks_ho),
        ):
            if delta < -thr:
                reasons.append(
                    f"{label} supported-set mean regressed in confirmation "
                    f"({float(delta):+.4f} < -{_frac(thr)}, the "
                    f"{_frac(calibration.coverage_level)} null-model quantile computed at "
                    f"{_counts_note(tasks, confirm_baseline, confirm_candidate)})"
                )
    report = (float(d_in), float(d_ho), float(thr_in), float(thr_ho))

    # The catastrophic per-task veto, same rule and same wording as `evaluate()`'s
    # (added by amendment). This function is the ONLY road to ACCEPT, so every
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
    carrier_quantiles: dict[str, dict] = {}
    guard_quantiles: dict[str, dict] = {}
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
        repeated = bool(basis) and sum(basis) / len(basis) > grain
        if not repeated:
            reasons.append(
                "original improvement did not repeat on "
                + ", ".join(first.improved_tasks)
                + " — a gain elsewhere is a new claim, not a confirmation"
            )
    else:
        # PER CARRIER, never the mean of them: a carrier that stayed flat
        # cannot be carried over the line by a louder one beside it, and a task's own
        # null distribution is the only distribution its own movement is drawn from.
        if not basis:
            reasons.append(
                "original improvement did not repeat: the first decision named no carrier "
                "task, so there is nothing for this pair to reproduce"
            )
        for n in first.improved_tasks:
            q = _task_quantile(calibration, n, confirm_baseline, confirm_candidate)
            carrier_quantiles[n] = _quantile_record(
                calibration, [n], q, confirm_baseline, confirm_candidate
            )
            if per[n] <= q:
                reasons.append(
                    f"original improvement did not repeat on {n}: {float(per[n]):+.4f} does "
                    f"not beat {_frac(q)}, the {_frac(calibration.coverage_level)} null-model "
                    f"quantile computed at "
                    f"{_counts_note([n], confirm_baseline, confirm_candidate)} — a gain "
                    "elsewhere is a new claim, not a confirmation"
                )
        # GUARDS ARE GATES. Each guard's DROP is judged against its own
        # null distribution, which is the only check in this function that can see a
        # trade the supported-set mean cancels out to zero.
        for g in sorted(calibration.guards):
            q = _task_quantile(calibration, g, confirm_baseline, confirm_candidate)
            guard_quantiles[g] = _quantile_record(
                calibration, [g], q, confirm_baseline, confirm_candidate
            )
            if per[g] < -q:
                reasons.append(
                    f"guard {g} dropped beyond its own null quantile: {float(per[g]):+.4f} is "
                    f"worse than -{_frac(q)}, the {_frac(calibration.coverage_level)} "
                    f"null-model quantile computed at "
                    f"{_counts_note([g], confirm_baseline, confirm_candidate)}"
                )
        # POSITIVITY: an ACCEPT ships the edit. A supported-set mean below
        # zero on either split is a net loss the section's own evidence recorded, small
        # enough to stay inside the null band and real enough not to ship.
        for label, delta in (("held-in", d_in), ("held-out", d_ho)):
            if delta < 0:
                reasons.append(
                    f"{label} supported-set mean is negative ({float(delta):+.4f}) — ACCEPT "
                    "requires a non-negative supported-set mean on BOTH splits, whichever "
                    "one carried the evidence"
                )

    raw = {"stage": "confirmation", "behavioral_verdicts": behavioral_verdicts}
    if calibration is not None:
        raw["regime"] = "section_calibration"
        raw["calibration"] = calibration.to_json()
        raw["null_quantiles"] = {
            "held_in": _quantile_record(
                calibration, tasks_in, thr_in, confirm_baseline, confirm_candidate
            ),
            "held_out": _quantile_record(
                calibration, tasks_ho, thr_ho, confirm_baseline, confirm_candidate
            ),
        }
        raw["carrier_quantiles"] = carrier_quantiles
        raw["guard_quantiles"] = guard_quantiles
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
    accepted_reason = "original improvement repeated under a fresh paired confirmation on " + (
        ", ".join(first.improved_tasks)
    )
    if calibration is not None:
        accepted_reason += (
            " — each beyond its OWN "
            f"{_frac(calibration.coverage_level)} null-model quantile ("
            + "; ".join(
                f"{n} {carrier_quantiles[n]['quantile']} at "
                f"{_attempts(confirm_baseline, n)}v{_attempts(confirm_candidate, n)}"
                for n in first.improved_tasks
            )
            + ")"
        )
    return Decision(
        outcome=ACCEPT,
        reasons=(accepted_reason,),
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
