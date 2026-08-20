"""Measured noise from NULL arms: nothing changed between them, so every |Δ| this
module reports is sampling noise, not signal. Where `runner.delta` computes a Δ
between a baseline and a candidate and refuses anything partial, this tool exists
to consume the null-run PROTOCOL itself: one full-suite arm plus several
`--only <supported set>` subset arms, all at the same harness state, and turn
their spread into thresholds a later rule extension can use instead of borrowing
`one_attempt` wholesale.

THE PROTOCOL THIS MODULE ASSUMES, stated here because the code depends on it and
a citation to a document living somewhere else would be a fact nobody can check
from this repo:

- Every arm is a run of this suite against the same harness state, with NOTHING
  edited between arms. An arm therefore measures noise; a Δ across arms measures
  nothing else.
- Every arm shares `runner_sha`, `config_version`, `model`, the carbon revision
  under test, and tree state. A Δ across any of those is a measurement of the
  thing that changed, not of noise, so a disagreement refuses.
- Round-2 arm shapes: three full-suite arms at the suite's standard attempt
  counts (3 held-in, 5 held-out) and five `--only A1 G2 G4 G5 --attempts 10`
  subset arms. Pooled per-task attempts therefore run to 59 (65 for G2).
- The section this calibrates is `compaction`. Two pinned sets, not one, from
  Phase 2c on: `SUPPORTED` ({A1, G2, G4, G5}) is what the rule's GAIN judgment
  averages over and what every fitness check below is computed on, while
  `MODEL_TASKS` (supported ∪ guards) is what the artifact must carry a pooled
  RATE for -- a guard is adjudicated one task at a time and cannot be judged
  without its own null distribution. They coincided through round 2, which is
  why `calibrate_model`'s `coverage=` defaults to `supported`.

Pure computation: no subprocess, no network, no writes except the one analysis
artifact `main()` produces. Every rate comparison stays exact (`Fraction` of the
integer pass/attempt counts, mirroring `runner.delta`'s style) until the very
last step, so a real spread of zero can never surface as a tiny nonzero float.

`pairwise_outcomes` records what the CURRENT, uncalibrated `evaluate()` rule
would have decided on every arm pair -- the honest baseline this whole exercise
calibrates against. Two arms drawn from the same protocol (both subset, same
task set, same attempts) evaluate cleanly. A full-suite arm paired with a subset
arm never can: their task sets differ, so `evaluate()` is first restricted to the
INTERSECTION, and even then their per-task attempt counts still differ (3 or 5 vs
10) -- `evaluate()`'s own parity gate refuses that, by design, the same way it
refuses a candidate re-measured at a different sample size. That refusal is not a
bug to route around: it is recorded as an honest `"outcome": "ERROR"` entry
rather than allowed to crash the whole analysis.

ROUND 2: the round-1 threshold shape above was measured, then withdrawn -- see
iterations/calibration-compaction/README.md for why (a bound below a real
validation's own grain, an end-to-end false ACCEPT reproduced from two of its own
null arms, a 3-task-mean bound applied to single-task deltas, no stated
coverage). `calibrate()` and its output shape stay exactly as they were, kept for
history; `calibrate_model()` is the round-2 replacement, alongside it, not
instead of it. It stops storing THRESHOLDS and starts storing the null MODEL
itself -- per-task pooled pass rates, exact fractions, with provenance -- so a
rule extension can compute a quantile at whatever attempt counts the judgment it
is deciding actually used (the round-1 defect: a bound measured at n=10 applied
at n=3). `null_gain_quantile` and `null_task_quantile` are the exact-enumeration
primitives that computation needs; they are exported so `loop.acceptance` can
import them directly rather than re-deriving exact binomial quantiles a second
time.

The artifact is also the place this program publishes what its rule CANNOT do:
end-to-end power across both stages, the false-CONFIRM rate conditional on the
one baseline arm every real judgment is made against, and the leave-one-out
margins behind the stability verdict. `recompute_model()` is the other half of
that honesty -- it re-derives every one of those checks from the artifact's own
per-arm counts so a reader (and the loader) never has to take the recorded
verdict's word for itself.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from functools import cache
from itertools import combinations, product
from math import comb
from pathlib import Path

from runner.suite import RESULTS_DIR

EDITOR_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = EDITOR_ROOT / "iterations" / "calibration-compaction" / "analysis.json"
MODEL_PATH = EDITOR_ROOT / "iterations" / "calibration-compaction" / "model-r2.json"

# The `compaction` section's pinned supported set: A1, G2, G4, G5 (aliases
# CMP-1..CMP-4). Only `main()`'s CLI default depends on this; `calibrate()`/
# `calibrate_model()` themselves always take `supported` explicitly, so tests
# never need to touch this. Round 1 and round 2 pin the SAME four tasks, so one
# constant serves both entry points.
SUPPORTED = frozenset({"A1", "G2", "G4", "G5"})

# The tasks a confirmation pair must rerun for `compaction` EVEN IF UNMOVED, and
# which are adjudicated against their own null distribution when it does: the
# section's known trade-off guards. G4 is the MINER and is deliberately absent --
# the task a candidate is mined from cannot also be the task that vouches for it.
# G2 is the held-out member, so the guard set spans both splits.
#
# It lives HERE, beside the null model, rather than only in `loop.validate`, for
# one concrete reason: the end-to-end power rows below enumerate the SECOND stage
# of the pipeline, and the second stage applies the guard gate. A power number
# computed against a different guard set than the one the confirmation actually
# applies would describe a pipeline nobody runs. `loop.validate` imports this
# constant instead of restating it, so the two cannot drift.
#
# The Phase 2c scenario guards (phase2c-guards-contract.md §1-§3, §6): CMP-5
# (supersession), CMP-6 (judged meaning-preservation), CMP-7 (buried facts).
# They exist to catch a compaction fix mined from G4 that generalizes only to
# G4's shape, so they are guards and never miners.
SCENARIO_GUARDS = frozenset({"CMP-5", "CMP-6", "CMP-7"})
CONFIRMATION_GUARDS = frozenset({"A1", "G2", "G5"}) | SCENARIO_GUARDS

# The null model's task COVERAGE: supported ∪ guards (contract §6). Wider than
# `SUPPORTED`, which stays the set the rule's GAIN judgment averages over -- the
# guards are judged one task at a time, against their own null distribution, so
# each needs a pooled rate of its own while none of them enters a split mean.
# DERIVED from the sets it unions rather than typed out again: a hand-listed copy
# is how a guard gets added in one place and missed by the campaign that has to
# produce its rate.
#
# Consequence, stated where it will be met: no artifact on disk covers these seven
# tasks yet, so `compaction` is LOUDLY uncalibrated until the campaign at the new
# runner hash re-records (contract amendment 2). That is not a transition hazard —
# this branch already moved the runner hash, so every artifact recorded before it
# was stale for these measurements anyway.
MODEL_TASKS = SUPPORTED | CONFIRMATION_GUARDS

# Round-2's one-sided coverage, as the exact Fraction every quantile in this
# module is computed at. 975/1000 reduces to 39/40 -- kept spelled out at the
# protocol's own stated precision ("97.5%") rather than hand-simplified, so a
# reader can see where it came from. The loader pins the installed artifact to
# exactly this value.
COVERAGE_LEVEL = Fraction(975, 1000)

# The power-floor construction: a true candidate rate of POOLED + 0.2 on the
# carrier, measured at the subset arms' own attempt count (`--attempts 10`) --
# the confirmation-shaped count a repeat/guard gate actually judges at, not the
# standard 3/5 a first pass uses.
POWER_OFFSET = Fraction(1, 5)
POWER_ATTEMPTS = 10

# The end-to-end alternatives the artifact publishes a joint detection rate for:
# +0.2 on ONE carrier (every supported task gets its own row) and +0.3/+0.5
# applied uniformly across the whole supported set. Three sizes, deliberately --
# a single number invites the reader to treat it as "the" power, and the shape of
# the curve is the honest answer.
END_TO_END_UNIFORM_OFFSETS = (Fraction(3, 10), Fraction(1, 2))

# The DESIGNATED baseline arm: the one recorded run every Phase-2b comparison is
# actually made against. A human chose it (the first full-suite arm of the
# protocol) and the choice is recorded in the artifact rather than left implicit,
# because the false-CONFIRM probability CONDITIONAL on it is a different -- and
# more relevant -- number than the marginal one averaged over baselines that will
# never be used again.
DESIGNATED_BASELINE = "r2-null-full-a"

# The four fields the null-run protocol requires every arm to share (see this
# module's docstring for the protocol itself): a Δ across runner versions, config
# versions, models, or uncommitted carbon
# states is not a measurement of noise, it is a measurement of the thing that
# changed. `dirty_sha` is None for a clean tree -- None==None across arms is a
# consistent (both clean) digest, not a mismatch; only a genuine difference (two
# arms dirty in different, unrelated ways, or one dirty and one clean) refuses.
_FINGERPRINT_FIELDS = ("runner_sha", "config_version", "model", "dirty_sha")

# Round-2's provenance record: the same four fields PLUS `carbon_sha` --
# carbon is the section under test here, not the verifier, so its identity
# belongs in the record too. There is no literal `carbon_sha` key in a results
# JSON's fingerprint; the runner already stamps the model/carbon revision it
# drove under the name `gemma_sha` (checked against a committed result file,
# e.g. results/null-cmp-a.json), and `_fingerprint_field` is where that mapping
# lives -- one place, so a future rename of either name only needs one edit.
_PROVENANCE_FIELDS = ("runner_sha", "config_version", "model", "carbon_sha", "dirty_sha")


def _exact_pass_fraction(task: dict) -> Fraction:
    """A task's pass rate as an exact `Fraction` of its integer counts.

    Same fallback and the same reason as `runner.delta`'s and `loop.acceptance`'s
    private `_exact`: `pass_fraction` is stored rounded to 4 places for display,
    and comparing rounded values can manufacture a nonzero spread between two
    results with IDENTICAL integer counts. Falls back to the stored fraction only
    for a legacy row with no counts recorded.
    """
    if "passes" in task and task.get("attempts"):
        return Fraction(int(task["passes"]), int(task["attempts"]))
    return Fraction(task["pass_fraction"]).limit_denominator(10_000)


def _max_pairwise_abs_delta(values: list[Fraction]) -> Fraction:
    if len(values) < 2:
        return Fraction(0)
    return max(abs(a - b) for a, b in combinations(values, 2))


def _load_arm(results_dir: Path, label: str) -> dict:
    path = results_dir / f"{label}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no results for arm {label!r} at {path}")
    return json.loads(path.read_text())


def _check_labels(labels: list[str], who: str) -> None:
    """Shared empty/duplicate-label refusal for both entry points below -- the
    same physical result standing in for two independent null arms is wrong
    whichever artifact shape is being computed, and an empty arm list has
    nothing to pool."""
    if not labels:
        raise ValueError(f"{who}: at least one arm label is required")
    dupes = sorted({label for label in labels if labels.count(label) > 1})
    if dupes:
        raise ValueError(
            f"{who}: duplicate arm label(s): {', '.join(dupes)} -- the same "
            "physical result cannot stand in for two independent null arms"
        )


def _fingerprint_field(fp: dict, field: str):
    """One provenance field's value from a raw results `fingerprint` dict.

    `carbon_sha` is not a literal key the runner writes -- the round-2 provenance
    record sources it from `gemma_sha`, the field the runner already stamps with the model/
    carbon revision under test.
    """
    if field == "carbon_sha":
        return fp.get("gemma_sha")
    return fp.get(field)


def _check_fingerprint_fields(
    arm_results: dict[str, dict], labels: list[str], fields: tuple[str, ...]
) -> dict:
    """Core of both fingerprint/provenance checks below: every named field must
    be identical across arms, or refuse loudly, naming the first field that
    differs and every arm's disagreeing value, rather than silently comparing
    unattributed or mismatched measurements."""
    first_fp = arm_results[labels[0]]["fingerprint"]
    shared: dict = {}
    for field in fields:
        values = {
            label: _fingerprint_field(arm_results[label]["fingerprint"], field) for label in labels
        }
        if len(set(values.values())) > 1:
            detail = ", ".join(f"{label}={v!r}" for label, v in values.items())
            raise ValueError(f"fingerprint mismatch on {field!r} across arms: {detail}")
        shared[field] = _fingerprint_field(first_fp, field)
    return shared


def _check_fingerprints(arm_results: dict[str, dict], labels: list[str]) -> dict:
    """The round-1 four-field fingerprint check, unchanged: reuse/extend point for
    `_check_fingerprint_fields` below, which `_check_provenance` (round-2)
    shares the same core with."""
    return _check_fingerprint_fields(arm_results, labels, _FINGERPRINT_FIELDS)


def _check_provenance(arm_results: dict[str, dict], labels: list[str]) -> dict:
    """The round-2 five-field provenance check: round-1's 4 fields plus
    `carbon_sha`, extending `_check_fingerprints` rather than duplicating it.

    Plus one thing round-1's check could not do, because round-1 read every field
    through `dict.get`: an arm whose fingerprint LACKS the `dirty_sha` key is
    refused outright, before any comparison. `dirty_sha` is the one field where
    None is real data -- it is exactly what a CLEAN tree records -- so `.get`
    reading an ABSENT key as None makes an arm that never stated its tree state
    compare equal to an arm that stated it was clean. Absent is not clean, it is
    unknown, and the entire claim this artifact rests on is that nothing differed
    between the arms. An unknown cannot support that claim, so it refuses here
    rather than passing silently into the pool.
    """
    for label in labels:
        if "dirty_sha" not in arm_results[label].get("fingerprint", {}):
            raise ValueError(
                f"arm {label!r} records no dirty_sha key in its fingerprint -- an absent "
                "key is not a clean tree. A clean checkout records dirty_sha=None, so an "
                "unrecorded tree state would compare equal to a recorded-clean one and "
                "pool as if the two arms provably matched. Re-record the arm with a "
                "runner that stamps tree state, or leave it out of the pool."
            )
    return _check_fingerprint_fields(arm_results, labels, _PROVENANCE_FIELDS)


def _build_arms(arm_results: dict[str, dict], labels: list[str]) -> list[dict]:
    return [
        {
            "label": label,
            **{
                field: arm_results[label]["fingerprint"].get(field) for field in _FINGERPRINT_FIELDS
            },
        }
        for label in labels
    ]


def _build_provenance(arm_results: dict[str, dict], labels: list[str]) -> list[dict]:
    """Per-arm provenance records, the round-2 analog of
    `_build_arms` -- audit trail alongside the consistency check
    `_check_provenance` already ran: every arm's own values are recorded even
    though they are required to already agree."""
    return [
        {
            "label": label,
            **{
                field: _fingerprint_field(arm_results[label]["fingerprint"], field)
                for field in _PROVENANCE_FIELDS
            },
        }
        for label in labels
    ]


def _per_task(arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]) -> dict:
    """Per supported task: pass counts per arm, and the max pairwise |Δ pass_fraction|.

    An arm missing the task entirely (should not happen under the protocol, but
    defended rather than assumed) simply does not contribute a count or a fraction
    for that task -- it is never imputed as a measured zero.
    """
    per_task: dict[str, dict] = {}
    for task in sorted(supported):
        present = [
            (label, arm_results[label]["tasks"][task])
            for label in labels
            if task in arm_results[label]["tasks"]
        ]
        per_task[task] = {
            "passes_by_arm": {
                label: {"passes": int(t["passes"]), "attempts": int(t["attempts"])}
                for label, t in present
            },
            "max_abs_delta": float(
                _max_pairwise_abs_delta([_exact_pass_fraction(t) for _, t in present])
            ),
        }
    return per_task


def _supported_split_mean(results: dict, split: str, supported: frozenset[str]) -> Fraction:
    fracs = [
        _exact_pass_fraction(t)
        for name, t in results["tasks"].items()
        if name in supported and t["split"] == split
    ]
    return sum(fracs) / len(fracs) if fracs else Fraction(0)


def _section_noise(
    arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]
) -> tuple[dict[str, float], dict[str, str], dict[str, list[str]]]:
    """Per split, the max pairwise |Δ supported-set mean| -- the measured analog of
    `one_attempt`, from arms that actually cover the WHOLE supported set for that
    split at a SHARED attempt count.

    A full-suite arm (standard 3/5 attempts) and the subset arms (10 attempts,
    10 per task) never share an attempt count, so a bound mixing them would
    average fractions of unequal precision -- the same reason `evaluate()`'s own
    parity gate refuses mismatched attempts. Group arms by their (uniform, within
    the arm) attempt count for this split's supported tasks, and use the LARGEST
    such group; `section_noise_arms` names exactly which arms fed the bound, so an
    excluded arm is never silently dropped.

    Returns the bound three ways: `noise` (float, for display/backward-compat),
    `noise_exact` (the SAME bound as an exact `"numerator/denominator"` string), and
    `noise_arms`. The float is a rounded VIEW of the bound, not the bound itself --
    `evaluate()`/`confirmed()` compare a measurement against this threshold with exact
    `Fraction` arithmetic everywhere else, and a threshold rebuilt by re-fractioning
    the float would compare against the float's binary value, not the true rational
    spread the arms actually measured (most denominators, including as plain a one as
    10, are not exact in binary). `noise_exact` is what lets a caller skip that lossy
    round trip.
    """
    task_split: dict[str, str] = {}
    for label in labels:
        for name, t in arm_results[label]["tasks"].items():
            if name in supported:
                task_split.setdefault(name, t["split"])

    noise: dict[str, float] = {}
    noise_exact: dict[str, str] = {}
    noise_arms: dict[str, list[str]] = {}
    for split in sorted(set(task_split.values())):
        split_tasks = {name for name, sp in task_split.items() if sp == split}
        groups: dict[int, list[str]] = {}
        for label in labels:
            tasks = arm_results[label]["tasks"]
            if not split_tasks <= set(tasks):
                continue  # this arm does not cover the entire supported set here
            attempts_values = {int(tasks[name]["attempts"]) for name in split_tasks}
            if len(attempts_values) != 1:
                continue  # not uniform within the arm -- no single grain to group on
            (attempts,) = attempts_values
            groups.setdefault(attempts, []).append(label)
        if not groups:
            noise[split] = 0.0
            noise_exact[split] = "0/1"
            noise_arms[split] = []
            continue
        best = max(groups, key=lambda a: (len(groups[a]), a))
        chosen = groups[best]
        means = [_supported_split_mean(arm_results[label], split, supported) for label in chosen]
        bound = _max_pairwise_abs_delta(means)
        noise[split] = float(bound)
        noise_exact[split] = f"{bound.numerator}/{bound.denominator}"
        noise_arms[split] = chosen
    return noise, noise_exact, noise_arms


def _restrict(results: dict, names: set[str]) -> dict:
    return {**results, "tasks": {n: results["tasks"][n] for n in names}}


def _pairwise_outcomes(arm_results: dict[str, dict], labels: list[str]) -> dict:
    """What `evaluate()` (the CURRENT, uncalibrated rule -- no `calibration=`, no
    named guards: `always_confirm=frozenset()`) returns for every arm pair.

    A full-suite arm and a subset arm never share a task set; restrict both sides
    to the INTERSECTION first and record it. `evaluate()` may still refuse (their
    attempt counts differ even on the shared tasks) or otherwise raise -- that is
    caught and recorded as an honest ERROR entry rather than allowed to crash the
    whole analysis, exactly as the design calls for.

    `evaluate` is imported HERE rather than at module scope, and the reason is a
    real one: since Task 3, `loop.acceptance` imports `null_gain_quantile` and
    `null_task_quantile` from this module to compute its judgment-time thresholds.
    A module-level import in both directions is a cycle Python cannot resolve in
    either order (whichever module is imported first hits a half-initialized
    partner and the `from ... import` name does not exist yet). Only this one
    round-1 legacy function needs `evaluate`, and it runs once per artifact build,
    so the deferred import is paid nowhere that matters -- and it keeps the rule
    module's own import list at module scope, where an import error surfaces at
    import time instead of inside a judgment.
    """
    from loop.acceptance import evaluate

    outcomes: dict[str, dict] = {}
    for a, b in combinations(labels, 2):
        key = f"{a}::{b}"
        ra, rb = arm_results[a], arm_results[b]
        tasks_a, tasks_b = set(ra["tasks"]), set(rb["tasks"])
        entry: dict = {}
        if tasks_a != tasks_b:
            inter = sorted(tasks_a & tasks_b)
            entry["task_intersection"] = inter
            ra, rb = _restrict(ra, set(inter)), _restrict(rb, set(inter))
        try:
            decision = evaluate(ra, rb, always_confirm=frozenset())
            entry["outcome"] = decision.outcome
        except Exception as exc:  # honest record of whatever evaluate() raised, never a crash
            entry["outcome"] = "ERROR"
            entry["error"] = str(exc)
        outcomes[key] = entry
    return outcomes


def calibrate(labels: list[str], results_dir: Path, supported: frozenset[str]) -> dict:
    """The round-1 analysis artifact, from the named arms' results JSONs
    under `results_dir`. Refuses (`ValueError` naming the field) on a fingerprint
    mismatch across arms; accepts filtered/subset arms outright -- that is the
    entire point of this tool, where `runner.delta` refuses them.

    Round-1 shape, kept intact for history -- see `calibrate_model()` for the
    round-2 replacement.
    """
    _check_labels(labels, "calibrate")
    arm_results = {label: _load_arm(results_dir, label) for label in labels}
    shared_fp = _check_fingerprints(arm_results, labels)
    section_noise, section_noise_exact, section_noise_arms = _section_noise(
        arm_results, labels, supported
    )
    return {
        "arms": _build_arms(arm_results, labels),
        "per_task": _per_task(arm_results, labels, supported),
        "section_noise": section_noise,
        "section_noise_exact": section_noise_exact,
        "section_noise_arms": section_noise_arms,
        "pairwise_outcomes": _pairwise_outcomes(arm_results, labels),
        "computed_at_runner_sha": shared_fp["runner_sha"],
    }


# ---------------------------------------------------------------------------
# Round 2: exact binomial
# enumeration -- the primitives, then the null-model artifact built on them.
# ---------------------------------------------------------------------------


def _binom_pmf(n: int, p: Fraction) -> tuple[Fraction, ...]:
    """P(X = k) for X ~ Binomial(n, p), k = 0..n -- exact `Fraction`, `math.comb`
    only, memoized via `_binom_pmf_cached` below. The same (n, p) pair recurs
    constantly across a single artifact's fitness checks: grain and stability
    both rebuild the SAME per-task pmf many times over (stability alone does
    it once per leave-one-arm-out pool), and goodness/power reuse the pooled
    rate across every arm -- memoizing is what keeps the whole artifact fast
    the protocol's own ceiling of <= 4 supported tasks at <= 20
    attempts keeps it cheap.

    This thin, UNCACHED wrapper is where `p` is checked to be exactly a
    `Fraction`, not merely numerically equal to one (`isinstance`, not `==`).
    That check has to live OUTSIDE the `@cache`d function, not inside it:
    Python hashes and compares `0.5` and `Fraction(1, 2)` as equal, so
    `@cache`'s own key lookup treats `_binom_pmf_cached(n, 0.5)` and
    `_binom_pmf_cached(n, Fraction(1, 2))` as the SAME cache entry -- a guard
    written inside the cached function only runs on a cache MISS, and is
    silently skipped on a cache HIT. Concretely: call the exact form first
    (populating the cache), then call the float form with the same numeric
    value, and a guard inside the cached function would never even execute,
    handing back the cached exact tuple as if the float call had been
    accepted. Checking here, before the cached function is ever reached in
    either direction, means a float `p` is refused every time, regardless of
    what has or hasn't already been cached.
    """
    if not isinstance(p, Fraction):
        raise TypeError(
            f"_binom_pmf: p must be an exact Fraction, got {type(p).__name__} ({p!r}) -- "
            "a float here would silently collide, by numeric equality, with an exact "
            "Fraction at the same (n, rate) in this function's memoized cache"
        )
    return _binom_pmf_cached(n, p)


@cache
def _binom_pmf_cached(n: int, p: Fraction) -> tuple[Fraction, ...]:
    """The memoized computation itself -- only ever reached through
    `_binom_pmf`'s type check above, never called directly."""
    q = 1 - p
    return tuple(Fraction(comb(n, k)) * p**k * q ** (n - k) for k in range(n + 1))


def _task_diff_pmf(n_a: int, p_a: Fraction, n_b: int, p_b: Fraction) -> dict[Fraction, Fraction]:
    """Exact pmf of ONE task's own (b/n_b - a/n_a), a ~ Binomial(n_a, p_a),
    b ~ Binomial(n_b, p_b), independent -- the per-task building block both
    `null_gain_quantile` (the NULL construction, called with `p_a == p_b`:
    nothing changed) and the power computations below (the ALTERNATIVE
    construction, called with `p_b = p_a + offset` on the carrier task: a
    true improvement is assumed) convolve across a task set. One function,
    not two, so the null and alternative constructions can never accidentally
    diverge in how they build a task's own diff distribution. Each side keeps
    its OWN attempt count as the fraction's denominator here, before anything
    is averaged across tasks -- this signature SUPPORTS `n_a != n_b` (a task
    judged at different attempt counts on each side) even though no real
    caller in this repo produces that shape today (`evaluate()`/`confirmed()`
    always run `_parity`-matched, symmetric attempts first); the capability
    exists for a future asymmetric shape, not because one occurs now.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError(f"_task_diff_pmf: attempts must be positive, got n_a={n_a}, n_b={n_b}")
    pmf_a = _binom_pmf(n_a, p_a)
    pmf_b = _binom_pmf(n_b, p_b)
    out: dict[Fraction, Fraction] = {}
    for a_k, a_p in enumerate(pmf_a):
        if a_p == 0:
            continue
        a_frac = Fraction(a_k, n_a)
        for b_k, b_p in enumerate(pmf_b):
            if b_p == 0:
                continue
            d = Fraction(b_k, n_b) - a_frac
            out[d] = out.get(d, Fraction(0)) + a_p * b_p
    return out


def null_gain_quantile(
    rates: dict[str, Fraction],
    attempts_a: dict[str, int],
    attempts_b: dict[str, int],
    level: Fraction,
) -> Fraction:
    """The exact null-model quantile that gates a supported-split mean gain
    at judgment time.

    Under H0 (nothing changed), EACH task's pass count on both sides is
    Binomial(attempts, rate) at the SAME pooled `rates[task]` -- there is only
    one `rates` argument, deliberately, because the null hypothesis is that
    the two runs differ only in attempt count, never in the rate they were
    drawn from. `attempts_a`/`attempts_b` are each side's REAL attempt counts,
    which may differ from each other and across tasks: a bound measured at
    n=10 must never be smeared onto a real n=3 judgment (the round-1 defect
    this whole module exists to close), and computing the quantile fresh at
    the judgment's own attempt counts is what prevents that structurally.

    The quantity gated is D = mean_b - mean_a, the mean (not sum) over the
    given tasks of EACH task's own (b_t/attempts_b[t] - a_t/attempts_a[t]) --
    never a pooled-count difference divided by task count, which would be the
    wrong quantity whenever attempts differ across tasks or sides (as they do
    the moment one side is a full-suite run and the other a confirmation
    rerun). `level` (e.g. `COVERAGE_LEVEL`, 97.5%) is a one-sided coverage
    level: the returned quantile `q` is the SMALLEST value D can take such
    that P(D <= q) >= level -- so a caller comparing an observed gain against
    `q` with strict `>` gets exactly "beats a 97.5%-covered null band", never
    an off-by-one at the boundary.

    Exact enumeration only -- `Fraction` arithmetic and `math.comb`, never
    floats or sampling. `rates` may carry a denominator far larger than either
    side's own attempt count (the round-2 pool runs to 59 attempts per task,
    65 for G2): the exact arithmetic is what keeps raising that denominator safe
    (`Fraction(21, 49) ** 10` is exact, not a float that has already lost the
    difference between 21/49 and its neighbors).

    Performance note: every real caller (`evaluate()`/`confirmed()`, gated by
    their own `_parity` check) has `attempts_a[t] == attempts_b[t]` for every
    task `t` -- a baseline and candidate always ran the SAME suite config.
    That symmetric case stays fast even at the protocol's ceiling (<=4
    tasks, <=20 attempts: ~60ms measured). Fully asymmetric attempts across
    several multi-attempt tasks at once is supported by this signature but not
    a real call shape, and its per-task diff pmf can carry far more distinct
    Fraction keys, so it is not the case this function is optimized for.

    Rejects (`ValueError`) a non-`Fraction` `rate` or `level`: this is the
    public boundary every caller (including `loop.acceptance`, once Task 3
    imports it) crosses before a rate ever reaches the memoized `_binom_pmf`
    cache, so a float mistake is refused HERE, loudly, rather than silently
    poisoning that cache for every later exact call (see `_binom_pmf`'s own
    docstring for the mechanism).
    """
    tasks = sorted(rates)
    if not tasks:
        raise ValueError("null_gain_quantile: at least one task is required")
    if set(attempts_a) != set(tasks) or set(attempts_b) != set(tasks):
        raise ValueError(
            "null_gain_quantile: attempts_a/attempts_b must cover exactly the tasks "
            f"in rates ({sorted(tasks)}); got attempts_a={sorted(attempts_a)}, "
            f"attempts_b={sorted(attempts_b)}"
        )
    non_fraction = sorted(t for t in tasks if not isinstance(rates[t], Fraction))
    if non_fraction:
        raise ValueError(
            f"null_gain_quantile: rates must be exact Fraction values, got a non-Fraction "
            f"rate for {non_fraction} -- a float rate silently poisons the memoized "
            "binomial pmf cache for every later exact call at the same (attempts, rate)"
        )
    if not isinstance(level, Fraction):
        raise ValueError(
            f"null_gain_quantile: level must be an exact Fraction, got "
            f"{type(level).__name__} ({level!r})"
        )
    if not (0 < level <= 1):
        raise ValueError(f"null_gain_quantile: level must be in (0, 1], got {level}")
    non_positive = sorted(t for t in tasks if int(attempts_a[t]) <= 0 or int(attempts_b[t]) <= 0)
    if non_positive:
        raise ValueError(
            f"null_gain_quantile: attempts_a/attempts_b must be positive for every task, "
            f"got a non-positive attempt count for {non_positive}"
        )
    pmf = _mean_diff_pmf(rates, attempts_a, rates, attempts_b)
    cdf = Fraction(0)
    for value in sorted(pmf):
        cdf += pmf[value]
        if cdf >= level:
            return value
    raise AssertionError(  # pragma: no cover -- a normalized pmf always reaches level<=1
        "null_gain_quantile: CDF never reached `level` -- the enumerated pmf did not "
        "sum to 1, which should be impossible"
    )


def _mean_diff_pmf(
    rates_a: dict[str, Fraction],
    attempts_a: dict[str, int],
    rates_b: dict[str, Fraction],
    attempts_b: dict[str, int],
) -> dict[Fraction, Fraction]:
    """Exact pmf of D = mean over `rates_a`'s tasks of each task's own
    (b_t/attempts_b[t] - a_t/attempts_a[t]).

    The one convolution every enumeration in this module runs, written once.
    Under the NULL both sides are drawn at the same rates (`rates_a is rates_b`);
    under an ALTERNATIVE the b-side carries the improved rates. Keeping one
    function means the null band and the power computed against it can never
    disagree about how a task's own difference is built.
    """
    tasks = sorted(rates_a)
    n = len(tasks)
    total: dict[Fraction, Fraction] = {Fraction(0): Fraction(1)}
    for t in tasks:
        task_pmf = _task_diff_pmf(attempts_a[t], rates_a[t], attempts_b[t], rates_b[t])
        nxt: dict[Fraction, Fraction] = {}
        for s, sp in total.items():
            if sp == 0:
                continue
            for d, dp in task_pmf.items():
                if dp == 0:
                    continue
                nxt[s + d] = nxt.get(s + d, Fraction(0)) + sp * dp
        total = nxt
    return {s / n: p for s, p in total.items()}


def null_gain_cdf(
    rates: dict[str, Fraction],
    attempts_a: dict[str, int],
    attempts_b: dict[str, int],
    value: Fraction,
) -> Fraction:
    """P(D <= `value`) under the same null construction `null_gain_quantile` takes
    the quantile of -- the COVERAGE a given threshold actually holds.

    The quantile answers "what bound holds 97.5% coverage?"; this answers "how much
    coverage does THIS bound hold?", which is the question a stability margin asks:
    leave an arm out, re-pool, and see how much of the 39/40 the full pool's own
    threshold still carries. A verdict that passed with a hair of slack and one that
    passed with room to spare are different facts, and the artifact publishes both
    rather than only the boolean they collapse into.
    """
    pmf = _mean_diff_pmf(rates, attempts_a, rates, attempts_b)
    return sum((p for d, p in pmf.items() if d <= value), Fraction(0))


def null_task_quantile(
    rate: Fraction, attempts_a: int, attempts_b: int, level: Fraction
) -> Fraction:
    """Single-task specialization of `null_gain_quantile` -- the
    carrier/guard construction: a repeat gate (per carrier task) and a guard
    adjudication both judge ONE task's own null distribution, never a
    supported-set mean. Implemented as `null_gain_quantile` over a single
    synthetic task key so the two constructions can never silently diverge.

    Validates `rate`/`level` are exact `Fraction`s itself, with a message
    naming THIS function, before delegating -- `null_gain_quantile` would
    catch the same mistake, but under its own name, which is confusing from
    a caller that never mentioned it.
    """
    if not isinstance(rate, Fraction):
        raise ValueError(
            f"null_task_quantile: rate must be an exact Fraction, got "
            f"{type(rate).__name__} ({rate!r})"
        )
    if not isinstance(level, Fraction):
        raise ValueError(
            f"null_task_quantile: level must be an exact Fraction, got "
            f"{type(level).__name__} ({level!r})"
        )
    return null_gain_quantile({"_task": rate}, {"_task": attempts_a}, {"_task": attempts_b}, level)


def _fraction_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def _pooled_counts(arm_results: dict[str, dict], labels: list[str], task: str) -> tuple[int, int]:
    passes = sum(int(arm_results[label]["tasks"][task]["passes"]) for label in labels)
    attempts = sum(int(arm_results[label]["tasks"][task]["attempts"]) for label in labels)
    return passes, attempts


def _check_supported_set(
    arm_results: dict[str, dict], labels: list[str], coverage: frozenset[str]
) -> None:
    """The pinned COVERAGE set, held exactly -- {A1, G2, G4, G5} through round 2,
    and supported ∪ guards from Phase 2c on (`MODEL_TASKS`).

    A subset (`--only`, `filter` key present) arm carrying a task OUTSIDE
    `coverage` refuses -- a filtered run is a promise that only the
    covered set was measured, and a task the pool was never meant to cover
    appearing there means the arm does not match the protocol this artifact
    assumes. Independently, ANY arm (filtered or full-suite) missing a
    covered task refuses -- the pool needs every arm's count on every
    covered task, never a partial denominator standing in for the whole; a
    real full-suite arm always carries the whole suite, so this branch only
    ever fires on a truncated or mislabeled result file.

    The guards are the reason this is coverage rather than support: a guard is
    adjudicated against its OWN pooled rate, so an arm that skipped it leaves the
    guard gated on nothing while the artifact still reads as complete.
    """
    for label in labels:
        result = arm_results[label]
        tasks = set(result["tasks"])
        if "filter" in result:
            extra = tasks - coverage
            if extra:
                raise ValueError(
                    f"arm {label!r} is a subset run carrying task(s) outside the "
                    f"pinned supported set {sorted(coverage)}: {sorted(extra)}"
                )
        missing = coverage - tasks
        if missing:
            raise ValueError(
                f"arm {label!r} is missing supported task(s) {sorted(missing)} -- the "
                f"null model needs every arm to cover the pinned supported set "
                f"{sorted(coverage)}"
            )


def _task_splits(
    arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]
) -> dict[str, str]:
    """Each supported task's split (`held_in`/`held_out`), read from every arm
    that carries it -- not just the first. A task's split is a property of
    the SUITE, not of any one arm's run, so two arms disagreeing on it is not
    "take the first and move on", it is a sign the arms are not measuring the
    same suite (a stale/mislabeled result file); refuse naming the task and
    both disagreeing values rather than silently keeping whichever arm
    happened to come first in `labels`.
    """
    splits: dict[str, str] = {}
    for label in labels:
        for task in supported:
            if task not in arm_results[label]["tasks"]:
                continue
            split = arm_results[label]["tasks"][task]["split"]
            if task in splits and splits[task] != split:
                raise ValueError(
                    f"calibrate_model: arm {label!r} disagrees with an earlier arm on "
                    f"task {task!r}'s split ({split!r} vs {splits[task]!r}) -- a task's "
                    "split is a property of the suite, not of one arm's run"
                )
            splits[task] = split
    return splits


def _standard_attempts(
    arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]
) -> dict[str, int]:
    """The full-suite attempt count per task -- 3 held-in / 5 held-out in the
    real protocol, never the subset arms' 10. Read from EVERY
    arm that is NOT a `--only` subset run (no `filter` key present): the same
    literal signal `_pairwise_outcomes` already uses to tell a full-suite arm
    apart from a filtered one. Refuses if no such arm is present among
    `labels` -- the fitness checks (grain, stability, power) cannot define
    "standard attempt counts" without at least one arm that ran the real
    suite, and the round-2 protocol always includes three (`r2-null-full-*`).

    Two full-suite arms disagreeing on a task's own attempt count is refused,
    not silently resolved by keeping whichever arm came first -- the suite's
    attempt count for a task is fixed, so a disagreement means the arms are
    not measuring the same suite config. A recorded attempt count of zero (or
    less) is refused too, naming the arm and task: `_split_grain` divides by
    this value, and a bare `ZeroDivisionError` two calls later would not say
    which arm's data caused it.
    """
    attempts: dict[str, int] = {}
    for label in labels:
        result = arm_results[label]
        if "filter" in result:
            continue
        for task in supported:
            if task not in result["tasks"]:
                continue
            value = int(result["tasks"][task]["attempts"])
            if value <= 0:
                raise ValueError(
                    f"calibrate_model: arm {label!r} records {value} attempts for "
                    f"standard-attempt task {task!r} -- must be positive"
                )
            if task in attempts and attempts[task] != value:
                raise ValueError(
                    f"calibrate_model: arm {label!r} disagrees with an earlier full-suite "
                    f"arm on task {task!r}'s standard attempt count ({value} vs "
                    f"{attempts[task]}) -- a task's standard attempt count is fixed by the "
                    "suite, not by one arm's run"
                )
            attempts[task] = value
    missing = sorted(supported - set(attempts))
    if missing:
        raise ValueError(
            "calibrate_model: no un-filtered (full-suite) arm covers standard-attempt "
            f"task(s) {', '.join(missing)} -- fitness needs at least one arm that ran "
            "the real suite, not only `--only` subset runs, to know the standard "
            "attempt counts a real validation judges at"
        )
    return attempts


def _build_null_model(
    arm_results: dict[str, dict],
    labels: list[str],
    supported: frozenset[str],
    null_counts: dict[str, tuple[int, int]],
) -> dict:
    """The per-task null model: a pooled `null_rate` as an exact,
    UNREDUCED "passes/attempts" string -- the denominator IS the pooled
    attempt count (59 held-in, 65 for G2 across the eight round-2 arms), so reducing
    it the way `section_noise_exact` reduces a bound would hide exactly the
    count the arm protocol exists to guarantee -- plus the per-arm counts
    behind the pool, for audit and for the goodness check.
    """
    return {
        task: {
            "null_rate": f"{passes}/{attempts}",
            "per_arm": {
                label: [
                    int(arm_results[label]["tasks"][task]["passes"]),
                    int(arm_results[label]["tasks"][task]["attempts"]),
                ]
                for label in labels
            },
        }
        for task, (passes, attempts) in null_counts.items()
    }


def _split_grain(
    task_split: dict[str, str],
    standard_attempts: dict[str, int],
    supported: frozenset[str],
    split: str,
) -> tuple[list[str], Fraction]:
    """The smallest representable movement of a split's supported-set mean at
    standard attempt counts -- the same construction as `loop.acceptance
    .one_attempt`: the largest single-attempt grain among the split's tasks,
    divided by how many tasks the mean averages over.
    """
    tasks = sorted(t for t in supported if task_split.get(t) == split)
    if not tasks:
        return tasks, Fraction(0)
    grain = max(Fraction(1, standard_attempts[t]) for t in tasks) / len(tasks)
    return tasks, grain


def _present_splits(task_split: dict[str, str], supported: frozenset[str]) -> list[str]:
    return sorted({task_split[t] for t in supported if t in task_split})


def _check_grain(
    null_counts: dict[str, tuple[int, int]],
    task_split: dict[str, str],
    standard_attempts: dict[str, int],
    supported: frozenset[str],
    level: Fraction,
) -> dict:
    """Fitness check 1, GRAIN: at standard attempt counts, the
    computed coverage-level gain quantile must EXCEED the split's grain -- a
    threshold sitting at or below the finest movement a real run can even
    produce gates nothing, the round-1 defect that sank the withdrawn artifact
    (its held-in bound, 0.1, sat below the 3-attempt grain of 1/9).
    """
    result: dict = {}
    overall = True
    for split in _present_splits(task_split, supported):
        tasks, grain = _split_grain(task_split, standard_attempts, supported, split)
        if not tasks:
            continue
        rates = {t: Fraction(*null_counts[t]) for t in tasks}
        attempts = {t: standard_attempts[t] for t in tasks}
        quantile = null_gain_quantile(rates, attempts, attempts, level)
        ok = quantile > grain
        overall = overall and ok
        result[split] = {
            "tasks": tasks,
            "attempts": dict(attempts),
            "quantile": _fraction_str(quantile),
            "grain": _fraction_str(grain),
            "pass": ok,
        }
    result["pass"] = overall
    return result


def _two_sided_binomial_tail(k: int, n: int, p: Fraction) -> Fraction:
    """Exact two-sided binomial tail p-value: the sum of every pmf value at
    most as large as the observed count's own -- the standard exact-test
    definition (equivalent to `scipy`/R's `binom.test`, computed here without
    either), never a normal approximation, and every term an exact `Fraction`
    comparison, so a probability that lands exactly on the observed value's
    own is never mis-included or mis-excluded by float rounding.
    """
    pmf = _binom_pmf(n, p)
    observed = pmf[k]
    return sum((prob for prob in pmf if prob <= observed), Fraction(0))


_GOODNESS_ALPHA = Fraction(1, 100)


def _check_goodness(
    arm_results: dict[str, dict],
    labels: list[str],
    supported: frozenset[str],
    null_counts: dict[str, tuple[int, int]],
) -> dict:
    """Fitness check 2, GOODNESS: for each task, an exact binomial
    two-sided tail test of each arm's own count against the SAME pooled rate
    every arm contributed to -- any arm whose tail probability falls below
    0.01 disagrees with a single-rate model badly enough that pooling it in
    is not a defensible null measurement (an outlier arm masquerading as
    noise), recorded per task+arm so the artifact shows exactly which
    arm/task combination failed, not only the aggregate verdict.
    """
    per_task: dict[str, dict] = {}
    overall = True
    for task in sorted(supported):
        p = Fraction(*null_counts[task])
        per_arm: dict[str, dict] = {}
        for label in labels:
            t = arm_results[label]["tasks"][task]
            k, n = int(t["passes"]), int(t["attempts"])
            tail = _two_sided_binomial_tail(k, n, p)
            ok = tail >= _GOODNESS_ALPHA
            overall = overall and ok
            per_arm[label] = {
                "passes": k,
                "attempts": n,
                "tail_p": _fraction_str(tail),
                "pass": ok,
            }
        per_task[task] = {"pooled_rate": _fraction_str(p), "per_arm": per_arm}
    return {"per_task": per_task, "pass": overall}


def _check_stability(
    arm_results: dict[str, dict],
    labels: list[str],
    supported: frozenset[str],
    task_split: dict[str, str],
    standard_attempts: dict[str, int],
    level: Fraction,
) -> dict:
    """Fitness check 3, STABILITY: leave-one-arm-out pooled rates must
    not shift a standard-count threshold across a REPRESENTABLE-MOVEMENT
    boundary. Representable movements are multiples of the split's own grain
    (`_split_grain`, the same quantity fitness check 1 gates on): a threshold
    that moves but stays inside the same grain bucket cannot change which
    real, quantized observation clears it, so only a bucket CROSSING counts
    as instability. `floor(quantile / grain)` is that bucket index, computed
    with exact `Fraction` floor division.

    Guards its own inputs first (Task 2 review): a leave-one-arm-out pool whose
    REMAINING arms all recorded zero attempts for a task divides by zero, and
    `Fraction(passes, 0)` raises a bare `ZeroDivisionError` naming nothing --
    no arm, no task, no hint that a result file is truncated. `_standard_attempts`
    only ever inspects the un-filtered arms, so a subset arm carrying a
    zero-attempt row reaches here unchecked. Every arm, filtered or not, must
    record positive attempts for every supported task; the refusal names the arm
    and the task, the way every other refusal in this module does.
    """
    for label in labels:
        for task in sorted(supported):
            row = arm_results[label]["tasks"].get(task)
            if row is None:
                continue  # `_check_supported_set` already refuses a missing task
            attempts = int(row["attempts"])
            if attempts <= 0:
                raise ValueError(
                    f"calibrate_model: arm {label!r} records {attempts} attempts for "
                    f"supported task {task!r} -- every arm must contribute a positive "
                    "attempt count to the pool, or a leave-one-arm-out pooled rate "
                    "divides by zero"
                )
    result: dict = {}
    overall = True
    for split in _present_splits(task_split, supported):
        tasks, grain = _split_grain(task_split, standard_attempts, supported, split)
        if not tasks or grain == 0:
            continue
        attempts = {t: standard_attempts[t] for t in tasks}
        full_rates = {t: Fraction(*_pooled_counts(arm_results, labels, t)) for t in tasks}
        full_q = null_gain_quantile(full_rates, attempts, attempts, level)
        full_bucket = full_q // grain
        moved: dict[str, str] = {}
        margins: dict[str, dict] = {}
        if len(labels) > 1:
            for held_out in labels:
                remaining = [label for label in labels if label != held_out]
                loo_rates = {t: Fraction(*_pooled_counts(arm_results, remaining, t)) for t in tasks}
                loo_q = null_gain_quantile(loo_rates, attempts, attempts, level)
                if loo_q // grain != full_bucket:
                    moved[held_out] = _fraction_str(loo_q)
                # The MARGIN, published whether the verdict moved or not: how much
                # coverage the FULL pool's own threshold still holds once this arm
                # is removed. `pass` says only whether the bucket changed, which is
                # a boolean over a continuous quantity -- a pool that passed with
                # 39/40 to spare and one that passed by a hair read identically. A
                # negative `slack` is a threshold that has stopped covering its
                # declared level under the reduced pool even though it stayed in the
                # same grain bucket, which is exactly the state a reader deciding
                # whether to append more arms needs to see.
                coverage = null_gain_cdf(loo_rates, attempts, attempts, full_q)
                margins[held_out] = {
                    "quantile": _fraction_str(loo_q),
                    "bucket": int(loo_q // grain),
                    "coverage_at_full_quantile": _fraction_str(coverage),
                    "slack": _fraction_str(coverage - level),
                    "slack_float": float(coverage - level),
                }
        ok = not moved
        overall = overall and ok
        result[split] = {
            "tasks": tasks,
            "full_quantile": _fraction_str(full_q),
            "grain": _fraction_str(grain),
            "moved_excluding": moved,
            "leave_one_out": margins,
            "pass": ok,
        }
    result["pass"] = overall
    return result


def _weakest_carrier(per_carrier: dict[str, Fraction]) -> str:
    """The carrier whose power is the FLOOR of a split's gain gate (Task 2
    review). Contract §4.4 publishes power so nobody mistakes a weak test for a
    strong one, and a split's gain gate has as many powers as it has tasks that
    could carry the improvement -- publishing one of them (round 2 published the
    alphabetically first) states the power of the case that happened to be named,
    not the power of the gate. The gate is only as strong as its weakest carrier,
    so that is the number the split-level row reports; the per-carrier rows stay
    beside it, so the floor never hides the spread it came from.

    Ties break on the task name, so the choice is deterministic for a reader
    comparing two artifacts.
    """
    return min(sorted(per_carrier), key=lambda t: per_carrier[t])


def _gain_power(
    rates: dict[str, Fraction],
    attempts: dict[str, int],
    carrier: str,
    offset: Fraction,
    threshold: Fraction,
) -> Fraction:
    """P[supported-split mean gain > threshold | `carrier`'s true rate is
    `rates[carrier] + offset` (capped at 1), every OTHER task in `rates`
    still at its pooled null rate] -- the gain gate's own power (fitness
    check 4 covers "each gate", and this is the row `_check_power` was missing: a first-pass
    `evaluate()` judges the split MEAN, not one task alone, so its power must
    be measured on that same mean, not borrowed from a single-task number).

    Reuses `_task_diff_pmf`, the exact building block `null_gain_quantile`
    itself convolves across tasks -- same per-task pmf construction, just
    with the carrier's `b`-side rate substituted for the alternative instead
    of held equal to `a`-side's (the null hypothesis `null_gain_quantile`
    always assumes). `threshold` is the caller's own already-computed
    coverage-level quantile at the SAME rates/attempts (`_check_grain`'s
    quantity) -- recomputing it here would risk the two silently drifting.
    """
    alt = {t: (min(Fraction(1), rates[t] + offset) if t == carrier else rates[t]) for t in rates}
    pmf = _mean_diff_pmf(rates, attempts, alt, attempts)
    return sum((p for d, p in pmf.items() if d > threshold), Fraction(0))


# The two splits this suite has. Spelled out rather than derived, because the
# end-to-end rows below publish a per-split breakdown that must carry a row for a
# split even when the alternative never touches it -- a missing key would read as
# "not applicable" where the truth is "zero".
SPLITS = ("held_in", "held_out")


def _split_means(diffs: dict[str, Fraction], task_split: dict[str, str]) -> dict[str, Fraction]:
    """Each split's supported-set mean movement -- the exact quantity both stages
    of the rule judge on (`loop.acceptance._supported_means`, same denominator,
    same exact arithmetic). A split with no supported task present means 0."""
    out: dict[str, Fraction] = {}
    for split in SPLITS:
        vals = [d for t, d in diffs.items() if task_split.get(t) == split]
        out[split] = sum(vals) / len(vals) if vals else Fraction(0)
    return out


def _stage1_verdict(
    diffs: dict[str, Fraction],
    task_split: dict[str, str],
    quantiles: dict[str, Fraction],
) -> tuple[bool, str, tuple[str, ...]]:
    """STAGE 1 as a pure predicate: does `loop.acceptance.evaluate()` CONFIRM, on
    which split, carried by which tasks?

    A restatement of `evaluate()`'s calibrated branch over the SUPPORTED tasks
    alone, so the end-to-end enumeration can be run over count vectors instead of
    over constructed results dicts (which would make it thousands of times slower
    and no more true). Every clause below is the same clause `evaluate()` applies:

    - either split's supported-set mean below `-quantile` REJECTs;
    - a full-pass task collapsing to zero REJECTs -- and that event is exactly
      `diff == -1`, since (b - a)/n = -1 only when a = n and b = 0;
    - the evidence split is held-in if its mean strictly exceeds its quantile,
      else held-out on the same test, else there is no gain and the verdict is
      REJECT;
    - the carriers are the evidence split's tasks that moved up.

    What it deliberately does NOT model, because the enumeration covers only the
    supported set: the whole-suite protections `evaluate()` keeps reading (a
    collapse or a mechanical security rise on a task OUTSIDE the supported set).
    Those can only ever turn a CONFIRM into a REJECT, so every published
    detection rate is an UPPER bound on the real one. `tests/test_p2b_closing.py`
    checks this predicate against the real `evaluate()` on sampled vectors, which
    is the only reason it is safe to enumerate against instead of the function.
    """
    means = _split_means(diffs, task_split)
    for split in SPLITS:
        if means[split] < -quantiles.get(split, Fraction(0)):
            return False, "", ()
    if any(d == -1 for d in diffs.values()):
        return False, "", ()
    evidence = ""
    for split in SPLITS:
        if means[split] > quantiles.get(split, Fraction(0)):
            evidence = split
            break
    if not evidence:
        return False, "", ()
    carriers = tuple(sorted(t for t, d in diffs.items() if task_split.get(t) == evidence and d > 0))
    return True, evidence, carriers


def _stage2_accepts(
    diffs: dict[str, Fraction],
    task_split: dict[str, str],
    quantiles: dict[str, Fraction],
    task_quantiles: dict[str, Fraction],
    carriers: tuple[str, ...] | frozenset[str],
    guards: frozenset[str],
) -> bool:
    """STAGE 2 as a pure predicate: does `loop.acceptance.confirmed()` ACCEPT?

    The same restatement, for the second gate: no split regressed past its own
    quantile, no collapse, EVERY named carrier beat its OWN per-task quantile
    (strictly -- a carrier that merely ties is not a repeat), no guard dropped
    past its own quantile, and both supported-set means are non-negative. A first
    decision naming no carrier has nothing to reproduce and cannot ACCEPT.

    Same modeling limitation as `_stage1_verdict`, same direction: the whole-suite
    vetoes and the behavioral Fisher comparison are not enumerated, so the number
    published is an upper bound.
    """
    means = _split_means(diffs, task_split)
    for split in SPLITS:
        if means[split] < -quantiles.get(split, Fraction(0)):
            return False
    if any(d == -1 for d in diffs.values()):
        return False
    if not carriers:
        return False
    for t in carriers:
        if diffs[t] <= task_quantiles[t]:
            return False
    for g in sorted(guards):
        if g in diffs and diffs[g] < -task_quantiles[g]:
            return False
    return all(means[split] >= 0 for split in SPLITS)


def _diff_grid(
    rates_a: dict[str, Fraction],
    attempts_a: dict[str, int],
    rates_b: dict[str, Fraction],
    attempts_b: dict[str, int],
    tasks: tuple[str, ...],
):
    """Every joint per-task difference vector and its exact probability.

    Enumerated over each task's own DIFFERENCE distribution (2n+1 values) rather
    than over both sides' count pairs ((n+1)^2 of them): the predicates above read
    only the differences, and 21 values per task at ten attempts beats 121 pairs
    by a factor that decides whether this artifact takes seconds or minutes to
    build.
    """
    per_task = []
    for t in tasks:
        pmf = _task_diff_pmf(attempts_a[t], rates_a[t], attempts_b[t], rates_b[t])
        per_task.append(tuple((d, p) for d, p in sorted(pmf.items()) if p))
    for combo in product(*per_task):
        prob = Fraction(1)
        for _, p in combo:
            prob *= p
        yield {t: d for t, (d, _) in zip(tasks, combo, strict=True)}, prob


def _stage1_mass(
    rates: dict[str, Fraction],
    attempts: dict[str, int],
    alt_rates: dict[str, Fraction],
    task_split: dict[str, str],
    quantiles: dict[str, Fraction],
) -> dict[tuple[str, tuple[str, ...]], Fraction]:
    """{(evidence split, carriers): probability} over every stage-1 CONFIRM shape.

    Keyed by the two things a first decision hands the second stage, because those
    are exactly what stage 2's answer depends on."""
    mass: dict[tuple[str, tuple[str, ...]], Fraction] = {}
    tasks = tuple(sorted(rates))
    for diffs, prob in _diff_grid(rates, attempts, alt_rates, attempts, tasks):
        ok, split, carriers = _stage1_verdict(diffs, task_split, quantiles)
        if ok:
            key = (split, carriers)
            mass[key] = mass.get(key, Fraction(0)) + prob
    return mass


def _conditional_stage1_mass(
    rates: dict[str, Fraction],
    attempts: dict[str, int],
    alt_rates: dict[str, Fraction],
    baseline_counts: dict[str, int],
    task_split: dict[str, str],
    quantiles: dict[str, Fraction],
) -> dict[tuple[str, tuple[str, ...]], Fraction]:
    """The same {(evidence split, carriers): probability} map as `_stage1_mass`, but
    with the BASELINE side FIXED at one recorded arm's own pass counts.

    Every judgment this pipeline makes compares a candidate against ONE designated
    baseline run, not against a fresh draw from the null. Marginalising over baselines
    answers "how would this rule behave across all the baselines we might have had",
    which is a fact about the design; conditioning on the arm actually in use answers
    "how will it behave on the comparisons we are going to make", which is a fact about
    this program. Both are published, and they differ a lot here: the designated arm
    drew low on two tasks, which makes movement easier to demonstrate in BOTH
    directions -- more true detections and more false ones.
    """
    tasks = tuple(sorted(rates))
    pmfs = {t: _binom_pmf(attempts[t], alt_rates[t]) for t in tasks}
    mass: dict[tuple[str, tuple[str, ...]], Fraction] = {}
    for combo in product(*(range(attempts[t] + 1) for t in tasks)):
        prob = Fraction(1)
        diffs = {}
        for t, k in zip(tasks, combo, strict=True):
            prob *= pmfs[t][k]
            diffs[t] = Fraction(k - baseline_counts[t], attempts[t])
        if prob == 0:
            continue
        ok, split, carriers = _stage1_verdict(diffs, task_split, quantiles)
        if ok:
            key = (split, carriers)
            mass[key] = mass.get(key, Fraction(0)) + prob
    return mass


def _stage2_split_mass(
    rates: dict[str, Fraction],
    attempts: dict[str, int],
    alt_rates: dict[str, Fraction],
    tasks: tuple[str, ...],
    quantile: Fraction,
    task_quantiles: dict[str, Fraction],
    guards: frozenset[str],
) -> dict[frozenset[str], Fraction]:
    """{carriers on this split: P(this split's half of an ACCEPT holds)}.

    Stage 2's predicate FACTORS across splits -- every clause in it is either
    per-task or per-split-mean, and the two splits' draws are independent -- so
    the joint acceptance probability for a carrier set is the product of the two
    splits' masses. That factoring is what keeps the enumeration at 21^3 + 21
    vectors instead of 21^4, and it is exact, not an approximation.
    """
    subsets = [
        frozenset(combo) for size in range(len(tasks) + 1) for combo in combinations(tasks, size)
    ]
    out: dict[frozenset[str], Fraction] = dict.fromkeys(subsets, Fraction(0))
    n = len(tasks)
    for diffs, prob in _diff_grid(rates, attempts, alt_rates, attempts, tasks):
        if any(d == -1 for d in diffs.values()):
            continue
        mean = sum(diffs.values()) / n
        if mean < -quantile or mean < 0:
            continue
        if any(t in guards and diffs[t] < -task_quantiles[t] for t in tasks):
            continue
        passing = frozenset(t for t in tasks if diffs[t] > task_quantiles[t])
        for subset in subsets:
            if subset <= passing:
                out[subset] += prob
    return out


def _end_to_end_row(
    kind: str,
    offset: Fraction,
    carrier: str | None,
    rates: dict[str, Fraction],
    standard_attempts: dict[str, int],
    task_split: dict[str, str],
    stage1_quantiles: dict[str, Fraction],
    stage2_quantiles: dict[str, Fraction],
    task_quantiles: dict[str, Fraction],
    split_tasks: dict[str, tuple[str, ...]],
    guards: frozenset[str],
    attempts: int,
    baseline_counts: dict[str, int] | None,
) -> dict:
    """One published joint detection rate: P(stage 1 CONFIRMs AND stage 2 ACCEPTs).

    `carrier` names the single task the improvement lands on, or None for a
    uniform improvement across the whole supported set. Stage 1 is judged at the
    suite's STANDARD attempt counts (that is what a first validation runs) and
    stage 2 at the confirmation's own, higher count.

    Reported twice, and the second is the one that describes this program. The
    MARGINAL figure averages over every baseline the null could have produced. The
    CONDITIONAL figure fixes the designated baseline arm at the counts it actually
    recorded, which is the comparison every real judgment makes. The false-CONFIRM
    block publishes exactly the same pair against the same arm, so a reader can put
    a detection rate beside a false-alarm rate without silently comparing a
    conditional number against a marginal one -- which is what the first version of
    this artifact invited.
    """
    alt = {
        t: min(Fraction(1), rates[t] + (offset if carrier in (None, t) else Fraction(0)))
        for t in rates
    }
    stage1 = _stage1_mass(rates, standard_attempts, alt, task_split, stage1_quantiles)
    per_split_mass = {
        split: _stage2_split_mass(
            {t: rates[t] for t in tasks},
            {t: attempts for t in tasks},
            {t: alt[t] for t in tasks},
            tasks,
            stage2_quantiles.get(split, Fraction(0)),
            task_quantiles,
            guards,
        )
        for split, tasks in split_tasks.items()
        if tasks
    }

    def accept_probability(carriers: tuple[str, ...]) -> Fraction:
        p = Fraction(1)
        for split, mass in per_split_mass.items():
            on_split = frozenset(t for t in carriers if task_split.get(t) == split)
            p *= mass.get(on_split, Fraction(0))
        return p

    conditional = (
        _conditional_stage1_mass(
            rates, standard_attempts, alt, baseline_counts, task_split, stage1_quantiles
        )
        if baseline_counts is not None
        else {}
    )

    def fold(mass: dict[tuple[str, tuple[str, ...]], Fraction], prefix: str) -> dict:
        by_split = {
            split: {f"{prefix}stage1_confirm": Fraction(0), f"{prefix}joint": Fraction(0)}
            for split in SPLITS
        }
        for (split, carriers), prob in mass.items():
            row = by_split.setdefault(
                split, {f"{prefix}stage1_confirm": Fraction(0), f"{prefix}joint": Fraction(0)}
            )
            row[f"{prefix}stage1_confirm"] += prob
            row[f"{prefix}joint"] += prob * accept_probability(carriers)
        return by_split

    marginal_split = fold(stage1, "")
    conditional_split = fold(conditional, "conditional_")

    def totals(by_split: dict, prefix: str) -> tuple[Fraction, Fraction]:
        return (
            sum((v[f"{prefix}stage1_confirm"] for v in by_split.values()), Fraction(0)),
            sum((v[f"{prefix}joint"] for v in by_split.values()), Fraction(0)),
        )

    stage1_total, joint_total = totals(marginal_split, "")
    cond_stage1_total, cond_joint_total = totals(conditional_split, "conditional_")
    return {
        "kind": kind,
        "offset": _fraction_str(offset),
        "carrier": carrier,
        "improved_tasks": sorted(rates) if carrier is None else [carrier],
        "stage1_attempts": {t: standard_attempts[t] for t in sorted(rates)},
        "stage2_attempts": attempts,
        "stage1_confirm": _fraction_str(stage1_total),
        "stage1_confirm_float": float(stage1_total),
        "joint": _fraction_str(joint_total),
        "joint_float": float(joint_total),
        "conditional_stage1_confirm": _fraction_str(cond_stage1_total),
        "conditional_stage1_confirm_float": float(cond_stage1_total),
        "conditional_joint": _fraction_str(cond_joint_total),
        "conditional_joint_float": float(cond_joint_total),
        "by_evidence_split": {
            split: {
                "stage1_confirm": _fraction_str(marginal_split[split]["stage1_confirm"]),
                "stage1_confirm_float": float(marginal_split[split]["stage1_confirm"]),
                "joint": _fraction_str(marginal_split[split]["joint"]),
                "joint_float": float(marginal_split[split]["joint"]),
                "conditional_stage1_confirm": _fraction_str(
                    conditional_split[split]["conditional_stage1_confirm"]
                ),
                "conditional_joint": _fraction_str(conditional_split[split]["conditional_joint"]),
                "conditional_joint_float": float(conditional_split[split]["conditional_joint"]),
            }
            for split in sorted(marginal_split)
        },
    }


def _check_end_to_end(
    null_counts: dict[str, tuple[int, int]],
    supported: frozenset[str],
    task_split: dict[str, str],
    standard_attempts: dict[str, int],
    level: Fraction,
    guards: frozenset[str],
    baseline: tuple[str, dict[str, int]] | None,
    *,
    offset: Fraction = POWER_OFFSET,
    attempts: int = POWER_ATTEMPTS,
    uniform_offsets: tuple[Fraction, ...] = END_TO_END_UNIFORM_OFFSETS,
) -> dict:
    """The rows nobody could compute from the stage-1 numbers alone.

    A published stage-1 power says what fraction of a real improvement earns a
    CONFIRM. It does NOT say what fraction gets SHIPPED, and the difference is
    large: the confirmation re-tests every carrier against its own distribution,
    adjudicates every guard, and requires both supported-set means to be
    non-negative -- three more ways for a true improvement to die. Publishing only
    the first factor is how a weak pipeline reads as a strong one.

    Three alternatives, so the reader gets a curve rather than a number: +0.2 on
    one carrier at a time (the same offset the stage-1 rows use, one row per
    supported task) and +0.3/+0.5 applied uniformly across the supported set.

    `baseline` is `(arm label, per-task pass counts)` for the DESIGNATED baseline arm,
    or None when this pooling has no such arm. Every row carries a conditional rate
    against it beside the marginal one, so the detection rates and the false-CONFIRM
    rate published in the same artifact are conditional on the same recorded run and
    can honestly be read against each other.
    """
    rates = {t: Fraction(*null_counts[t]) for t in sorted(supported)}
    split_tasks = {
        split: tuple(sorted(t for t in supported if task_split.get(t) == split)) for split in SPLITS
    }
    stage1_quantiles = {
        split: (
            null_gain_quantile(
                {t: rates[t] for t in tasks},
                {t: standard_attempts[t] for t in tasks},
                {t: standard_attempts[t] for t in tasks},
                level,
            )
            if tasks
            else Fraction(0)
        )
        for split, tasks in split_tasks.items()
    }
    stage2_quantiles = {
        split: (
            null_gain_quantile(
                {t: rates[t] for t in tasks},
                {t: attempts for t in tasks},
                {t: attempts for t in tasks},
                level,
            )
            if tasks
            else Fraction(0)
        )
        for split, tasks in split_tasks.items()
    }
    task_quantiles = {
        t: null_task_quantile(rates[t], attempts, attempts, level) for t in sorted(supported)
    }
    rows = [
        _end_to_end_row(
            "single_carrier",
            offset,
            carrier,
            rates,
            standard_attempts,
            task_split,
            stage1_quantiles,
            stage2_quantiles,
            task_quantiles,
            split_tasks,
            guards,
            attempts,
            baseline[1] if baseline else None,
        )
        for carrier in sorted(supported)
    ]
    rows += [
        _end_to_end_row(
            "uniform",
            uniform,
            None,
            rates,
            standard_attempts,
            task_split,
            stage1_quantiles,
            stage2_quantiles,
            task_quantiles,
            split_tasks,
            guards,
            attempts,
            baseline[1] if baseline else None,
        )
        for uniform in uniform_offsets
    ]
    return {
        "confirmation_attempts": attempts,
        "guards": sorted(guards),
        "baseline_arm": baseline[0] if baseline else None,
        "baseline_counts": (
            {t: [c, standard_attempts[t]] for t, c in sorted(baseline[1].items())}
            if baseline
            else None
        ),
        "stage1_quantiles": {s: _fraction_str(q) for s, q in sorted(stage1_quantiles.items())},
        "stage2_quantiles": {s: _fraction_str(q) for s, q in sorted(stage2_quantiles.items())},
        "task_quantiles": {t: _fraction_str(q) for t, q in sorted(task_quantiles.items())},
        "note": (
            "P(stage-1 CONFIRM and stage-2 ACCEPT), enumerated exactly over both "
            "stages' binomial outcomes at the supported set alone. The whole-suite "
            "vetoes (a collapse or a security rise on an unsupported task) and the "
            "behavioral Fisher comparison are NOT modeled, and can only lower a real "
            "detection rate, so every number here is an upper bound."
        ),
        "rows": rows,
    }


def designated_baseline_counts(
    arm_results: dict[str, dict],
    supported: frozenset[str],
    standard_attempts: dict[str, int],
    baseline_arm: str = DESIGNATED_BASELINE,
) -> tuple[dict[str, int] | None, str]:
    """The designated baseline arm's own per-task pass counts, or None and the reason.

    Resolved ONCE and handed to both blocks that condition on it -- the false-CONFIRM
    rate and the end-to-end detection rates. Two resolvers would be two chances to
    condition on two different arms and publish the pair as if they were comparable,
    which is the exact defect the conditional rows were added to fix.

    Refuses (None, reason) when the arm is not in this pooling, or when it did not run
    at the suite's standard attempt counts: a designated baseline has to be a full-suite
    run for "conditional on it" to describe a judgment anyone will actually make.
    """
    baseline = arm_results.get(baseline_arm)
    if baseline is None:
        return None, (
            f"no arm labeled {baseline_arm!r} is in this pooling, so there is no "
            "designated baseline to be conditional on"
        )
    counts: dict[str, int] = {}
    for task in sorted(supported):
        row = baseline["tasks"][task]
        if int(row["attempts"]) != standard_attempts[task]:
            return None, (
                f"arm {baseline_arm!r} ran {task} at {row['attempts']} attempts, not the "
                f"suite's standard {standard_attempts[task]} -- a designated baseline must "
                "be a full-suite arm for a conditional rate to describe a real judgment"
            )
        counts[task] = int(row["passes"])
    return counts, ""


def _check_false_confirm(
    arm_results: dict[str, dict],
    null_counts: dict[str, tuple[int, int]],
    supported: frozenset[str],
    task_split: dict[str, str],
    standard_attempts: dict[str, int],
    level: Fraction,
    *,
    baseline_arm: str = DESIGNATED_BASELINE,
) -> dict:
    """The probability a first pass CONFIRMs when NOTHING changed -- twice.

    `marginal` draws both arms from the null: the average false-CONFIRM rate over
    every pair of runs that could have happened. `conditional` fixes the DESIGNATED
    baseline arm at the counts it actually recorded and draws only the candidate
    side. The second is the number that describes this pipeline, because every
    Phase-2b judgment is made against that one recorded baseline and no other --
    and a baseline that happened to land low makes a false CONFIRM likelier for
    every candidate that will ever be compared against it. Publishing only the
    marginal averages that away.
    """
    rates = {t: Fraction(*null_counts[t]) for t in sorted(supported)}
    split_tasks = {
        split: tuple(sorted(t for t in supported if task_split.get(t) == split)) for split in SPLITS
    }
    quantiles = {
        split: (
            null_gain_quantile(
                {t: rates[t] for t in tasks},
                {t: standard_attempts[t] for t in tasks},
                {t: standard_attempts[t] for t in tasks},
                level,
            )
            if tasks
            else Fraction(0)
        )
        for split, tasks in split_tasks.items()
    }

    def summarize(mass: dict[tuple[str, tuple[str, ...]], Fraction]) -> tuple[Fraction, dict]:
        by_split = {split: Fraction(0) for split in SPLITS}
        for (split, _carriers), prob in mass.items():
            by_split[split] = by_split.get(split, Fraction(0)) + prob
        return sum(by_split.values(), Fraction(0)), by_split

    marginal_mass = _stage1_mass(rates, standard_attempts, rates, task_split, quantiles)
    marginal, marginal_split = summarize(marginal_mass)

    out: dict = {
        "baseline_arm": None,
        "quantiles": {s: _fraction_str(q) for s, q in sorted(quantiles.items())},
        "marginal": _fraction_str(marginal),
        "marginal_float": float(marginal),
        "note": (
            "P(evaluate() returns CONFIRM) with nothing changed. `marginal` draws "
            "both arms from the null model; `conditional` fixes the designated "
            "baseline arm's own recorded counts and draws only the candidate arm, "
            "which is the comparison this pipeline actually makes."
        ),
    }
    counts, why_not = designated_baseline_counts(
        arm_results, supported, standard_attempts, baseline_arm
    )
    if counts is None:
        out["conditional"] = None
        out["conditional_float"] = None
        out["by_evidence_split"] = {
            s: {"marginal": _fraction_str(p), "conditional": None}
            for s, p in sorted(marginal_split.items())
        }
        out["baseline_note"] = why_not
        return out

    tasks = tuple(sorted(supported))
    conditional_mass = _conditional_stage1_mass(
        rates, standard_attempts, rates, counts, task_split, quantiles
    )
    conditional, conditional_split = summarize(conditional_mass)
    out["baseline_arm"] = baseline_arm
    out["baseline_counts"] = {t: [counts[t], standard_attempts[t]] for t in tasks}
    out["conditional"] = _fraction_str(conditional)
    out["conditional_float"] = float(conditional)
    out["by_evidence_split"] = {
        split: {
            "marginal": _fraction_str(marginal_split[split]),
            "marginal_float": float(marginal_split[split]),
            "conditional": _fraction_str(conditional_split[split]),
            "conditional_float": float(conditional_split[split]),
        }
        for split in SPLITS
    }
    return out


def _check_power(
    null_counts: dict[str, tuple[int, int]],
    supported: frozenset[str],
    task_split: dict[str, str],
    standard_attempts: dict[str, int],
    level: Fraction,
    *,
    guards: frozenset[str] = CONFIRMATION_GUARDS,
    baseline: tuple[str, dict[str, int]] | None = None,
    offset: Fraction = POWER_OFFSET,
    attempts: int = POWER_ATTEMPTS,
) -> dict:
    """Fitness check 4, POWER: recorded, never gating -- "each gate",
    not one, and then the whole PIPELINE.

    Three blocks, and the third is a different question from the first two.
    `stage1_only` answers "what does each individual gate detect?"; `end_to_end`
    answers "what actually ships?", which is the product of both stages and the
    only number a reader should compare against a claim that this loop can find
    improvements. The stage-1 rows are labeled as such rather than left unlabeled,
    because an unlabeled power number beside a two-stage pipeline reads as the
    pipeline's power and is several times too high.

    The two stage-1 shapes:

    - `per_task`: for each supported task (any of which can be a
      confirmation's carrier or guard), the probability that a TRUE candidate
      rate of pooled + 0.2 clears that task's OWN null quantile at
      `attempts` v `attempts` (the subset arms' own attempt count, the
      confirmation-shaped one) -- the repeat/guard gate's power.
    - `gain_gate`: for each split with supported tasks, the probability that
      the SAME true improvement clears the split's supported-set-MEAN quantile
      at STANDARD attempt counts -- the first-pass evaluate() gain gate's own
      power, the weakest of the two shapes (a mean over several tasks dilutes
      one task's movement, and standard attempts are fewer than the
      confirmation's 10) and the one this artifact must not omit just because a
      single-task number already exists nearby. Every task on the split is
      enumerated as the carrier (`per_carrier`) and the row's headline `power`
      is the MINIMUM across them -- a floor, not a sample: a gate advertised at
      its luckiest carrier's power overstates what it can actually detect.

    Both enumerated exactly over both sides' binomial outcomes, published so
    nobody mistakes a weak test for a strong one (this module's whole reason
    for existing).
    """
    per_task: dict[str, dict] = {}
    for task in sorted(supported):
        p = Fraction(*null_counts[task])
        # Capped at 1: a pooled rate already close to 1 (a near-ceiling task)
        # plus the flat +0.2 offset can overshoot a valid probability, and an
        # alternative rate above 1 is not a rate.
        p_alt = min(Fraction(1), p + offset)
        threshold = null_task_quantile(p, attempts, attempts, level)
        cleared = _gain_power({task: p}, {task: attempts}, task, offset, threshold)
        per_task[task] = {
            "pooled_rate": _fraction_str(p),
            "alt_rate": _fraction_str(p_alt),
            "attempts": attempts,
            "threshold": _fraction_str(threshold),
            "power": _fraction_str(cleared),
            "power_float": float(cleared),
        }

    gain_gate: dict[str, dict] = {}
    for split in _present_splits(task_split, supported):
        tasks, grain = _split_grain(task_split, standard_attempts, supported, split)
        if not tasks:
            continue
        rates = {t: Fraction(*null_counts[t]) for t in tasks}
        split_attempts = {t: standard_attempts[t] for t in tasks}
        threshold = null_gain_quantile(rates, split_attempts, split_attempts, level)
        per_carrier = {t: _gain_power(rates, split_attempts, t, offset, threshold) for t in tasks}
        carrier = _weakest_carrier(per_carrier)
        power = per_carrier[carrier]
        gain_gate[split] = {
            "tasks": tasks,
            "carrier": carrier,
            "attempts": dict(split_attempts),
            "threshold": _fraction_str(threshold),
            "power": _fraction_str(power),
            "power_float": float(power),
            "per_carrier": {
                t: {"power": _fraction_str(p), "power_float": float(p)}
                for t, p in per_carrier.items()
            },
        }

    return {
        "offset": _fraction_str(offset),
        "attempts": attempts,
        "stage1_only": {"per_task": per_task, "gain_gate": gain_gate},
        "end_to_end": _check_end_to_end(
            null_counts,
            supported,
            task_split,
            standard_attempts,
            level,
            guards,
            baseline,
            offset=offset,
            attempts=attempts,
        ),
    }


def calibrate_model(
    labels: list[str],
    results_dir: Path,
    supported: frozenset[str],
    *,
    coverage: frozenset[str] | None = None,
) -> dict:
    """The round-2 null-MODEL artifact, from the named arms' results JSONs
    under `results_dir`.

    `supported` is the set the rule's GAIN judgment averages over, and it is what
    every fitness check below is computed on. `coverage` is the set the model
    carries a pooled RATE for -- supported ∪ guards from Phase 2c on
    (`MODEL_TASKS`), because a guard is adjudicated against its own null
    distribution and cannot be judged without one. It defaults to `supported`,
    which reproduces the round-2 artifact byte for byte.

    The asymmetry is deliberate and is the contract's (§6): widening the gain set
    would change the denominator of every split mean the rule judges, which is a
    change to the rule, not to its coverage.

    Refuses (`ValueError`, naming what's wrong) on: no labels; a duplicate
    label; a provenance mismatch across arms (the round-1 4 fields plus
    `carbon_sha`); a subset (`--only`) arm carrying a task outside the pinned
    {A1, G2, G4, G5} supported set; any arm missing a supported task; two arms
    disagreeing on a task's split or standard attempt count (a property of
    the SUITE, not of one arm's run); a non-positive recorded attempt count;
    or no un-filtered arm to read standard attempt counts from at all. It
    does NOT refuse on a failed fitness check -- `fitness.fit` is computed
    and recorded either way (an unfit pooling is written out with
    `fit: false` rather than raising), and it is the LOADER's job, not this function's, to
    refuse installing an unfit artifact.

    Round-1's `calibrate()` and its output shape are untouched; this is a
    separate entry point, not a replacement in place.
    """
    coverage = supported if coverage is None else coverage
    if not supported <= coverage:
        raise ValueError(
            f"calibrate_model: the supported set {sorted(supported - coverage)} is not "
            f"covered by the model's task coverage {sorted(coverage)} -- the gain judgment "
            "cannot average over a task the model has no rate for"
        )
    _check_labels(labels, "calibrate_model")
    arm_results = {label: _load_arm(results_dir, label) for label in labels}
    _check_provenance(arm_results, labels)
    _check_supported_set(arm_results, labels, coverage)

    provenance = _build_provenance(arm_results, labels)
    # Rates, splits and attempt counts are collected over the whole COVERAGE set --
    # the guards need theirs -- while every fitness check below stays on `supported`.
    null_counts = {task: _pooled_counts(arm_results, labels, task) for task in sorted(coverage)}
    null_model = _build_null_model(arm_results, labels, coverage, null_counts)
    task_split = _task_splits(arm_results, labels, coverage)
    standard_attempts = _standard_attempts(arm_results, labels, coverage)

    grain = _check_grain(null_counts, task_split, standard_attempts, supported, COVERAGE_LEVEL)
    goodness = _check_goodness(arm_results, labels, supported, null_counts)
    stability = _check_stability(
        arm_results, labels, supported, task_split, standard_attempts, COVERAGE_LEVEL
    )
    # Resolved once, so the conditional detection rates and the conditional
    # false-CONFIRM rate are conditional on the SAME recorded arm -- the whole point
    # of publishing them together.
    baseline_counts, _ = designated_baseline_counts(arm_results, supported, standard_attempts)
    baseline = (DESIGNATED_BASELINE, baseline_counts) if baseline_counts is not None else None
    power = _check_power(
        null_counts,
        supported,
        task_split,
        standard_attempts,
        COVERAGE_LEVEL,
        baseline=baseline,
    )
    false_confirm = _check_false_confirm(
        arm_results, null_counts, supported, task_split, standard_attempts, COVERAGE_LEVEL
    )
    fit = bool(grain["pass"] and goodness["pass"] and stability["pass"])

    return {
        "null_model": null_model,
        "provenance": provenance,
        "coverage_level": "0.975",
        "fitness": {
            "grain": grain,
            "goodness": goodness,
            "stability": stability,
            "power": power,
            "false_confirm": false_confirm,
            "fit": fit,
        },
        "computed_at_runner_sha": provenance[0]["runner_sha"],
    }


def recompute_model(
    model: dict, level: Fraction = COVERAGE_LEVEL
) -> tuple[dict[str, Fraction], list[str]]:
    """Re-derive an artifact's rates and re-run its fitness checks from its OWN
    per-arm counts, and report every disagreement with what it recorded.

    The hole this closes: `fitness.fit` was computed by the tool that wrote the
    file and then READ OFF that same file by the loader. Editing the rates and
    leaving the verdict alone therefore installed a null model nothing had ever
    checked -- the reviewer's reproduction was to set every `null_rate` to
    "1/1000", which passes every recorded check because no check re-ran. A
    self-certifying artifact certifies nothing; the reader has to do the checking.

    What is re-derived, and from what:

    - Each task's pooled `null_rate`, summed from `per_arm` -- the raw counts the
      artifact carries for audit, which a rate-only tamper leaves untouched. Every
      task the model COVERS, guards included.
    - Each task's split and the suite's standard attempt counts, read from the
      recorded `fitness.grain` rows (they carry both), so no extra field has to be
      trusted or added.
    - GRAIN, GOODNESS and STABILITY, re-run by the very functions that wrote them,
      against the re-derived rates, and compared to the recorded blocks whole.

    The fitness checks are re-run over the GAIN set, and that set is read from the
    grain rows' own `tasks` lists rather than from the null model's keys. From Phase
    2c the two differ: the model rates supported ∪ guards, while the gain judgment
    still averages over the supported tasks alone. Deriving the gain set from the
    rates would re-run grain over the wider set, disagree with the rows the artifact
    actually recorded, and refuse a perfectly good model. The rows say what they
    were computed over; that is the authority.

    Returns `(rates, problems)`. A non-empty `problems` means REFUSE: the rates
    are still returned so a caller can report them, never so it can install them.
    """
    problems: list[str] = []
    null_model = model.get("null_model") or {}
    fitness = model.get("fitness") or {}
    covered = frozenset(null_model)
    if not covered:
        return {}, ["the artifact carries no null_model, so there is nothing to recompute"]

    task_split: dict[str, str] = {}
    standard_attempts: dict[str, int] = {}
    for split, row in (fitness.get("grain") or {}).items():
        if split == "pass" or not isinstance(row, dict):
            continue
        for task in row.get("tasks") or []:
            task_split[task] = split
        for task, value in (row.get("attempts") or {}).items():
            try:
                standard_attempts[task] = int(value)
            except (TypeError, ValueError):
                problems.append(
                    f"recomputation cannot proceed: the recorded grain row for {split} "
                    f"gives task {task} a non-integer attempt count ({value!r})"
                )
    # The GAIN set: exactly the tasks the recorded grain rows were computed over.
    # `covered` (the rated tasks) may be wider from Phase 2c on.
    gain = frozenset(task_split) | frozenset(standard_attempts)
    if not gain:
        problems.append(
            "recomputation cannot proceed: the recorded grain rows name no task, so there "
            "is no gain set to re-run the fitness checks over"
        )
        return {}, problems
    ungated = sorted(gain - covered)
    missing = sorted(gain - set(task_split)) + sorted(gain - set(standard_attempts)) + ungated
    if missing or problems:
        problems.append(
            "recomputation cannot proceed: the recorded grain rows do not state a split "
            "and a standard attempt count for every supported task (missing "
            f"{sorted(set(missing))})"
        )
        return {}, problems

    rates: dict[str, Fraction] = {}
    labels: list[str] | None = None
    counts: dict[str, dict[str, tuple[int, int]]] = {}
    for task in sorted(covered):
        per_arm = (null_model[task] or {}).get("per_arm") or {}
        if not per_arm:
            problems.append(
                f"recomputation cannot proceed: {task} carries no per_arm counts, so its "
                "recorded null_rate cannot be re-derived from anything"
            )
            return {}, problems
        arm_labels = sorted(per_arm)
        if labels is None:
            labels = arm_labels
        elif arm_labels != labels:
            problems.append(
                f"recomputation cannot proceed: {task}'s per_arm names {arm_labels}, which "
                f"is not the arm set {labels} the other tasks pooled over -- one pool, or "
                "the rates are not comparable"
            )
            return {}, problems
        try:
            task_counts = {
                label: (int(per_arm[label][0]), int(per_arm[label][1])) for label in arm_labels
            }
        except (TypeError, ValueError, IndexError, KeyError):
            problems.append(
                f"recomputation cannot proceed: {task}'s per_arm is not a "
                "{arm: [passes, attempts]} map"
            )
            return {}, problems
        counts[task] = task_counts
        passes = sum(p for p, _ in task_counts.values())
        attempts = sum(n for _, n in task_counts.values())
        if attempts <= 0:
            problems.append(
                f"recomputation cannot proceed: {task} pooled {attempts} attempts across "
                "its arms, so it has no rate at all"
            )
            return {}, problems
        derived = f"{passes}/{attempts}"
        recorded = (null_model[task] or {}).get("null_rate")
        if recorded != derived:
            problems.append(
                f"recomputing {task}'s null_rate from the artifact's own per_arm counts "
                f"gives {derived!r}, but the artifact records {recorded!r} -- the rate every "
                "quantile is computed from does not match the counts it claims to come from"
            )
        rates[task] = Fraction(passes, attempts)
    if problems:
        return rates, problems

    # Rebuilt over the GAIN set alone: these three checks are the ones the artifact
    # recorded, and it recorded them over the tasks its grain rows name.
    arm_results = {
        label: {
            "tasks": {
                task: {
                    "split": task_split[task],
                    "attempts": counts[task][label][1],
                    "passes": counts[task][label][0],
                }
                for task in sorted(gain)
            }
        }
        for label in (labels or [])
    }
    null_counts = {
        task: (
            sum(p for p, _ in counts[task].values()),
            sum(n for _, n in counts[task].values()),
        )
        for task in sorted(gain)
    }
    try:
        recomputed = {
            "grain": _check_grain(null_counts, task_split, standard_attempts, gain, level),
            "goodness": _check_goodness(arm_results, labels or [], gain, null_counts),
            "stability": _check_stability(
                arm_results, labels or [], gain, task_split, standard_attempts, level
            ),
        }
    except (ValueError, ZeroDivisionError, KeyError) as exc:
        return rates, [
            f"recomputing the fitness checks from the artifact's own counts raised: {exc}"
        ]
    for name, value in recomputed.items():
        where = _first_disagreement(fitness.get(name), value, name)
        if where:
            problems.append(
                f"recomputing the {name} check from the artifact's own counts disagrees with "
                f"the recorded {name} block at {where}"
            )
    return rates, problems


def _first_disagreement(recorded, recomputed, path: str) -> str:
    """The first place two recorded/recomputed blocks differ, as a dotted path and
    both values -- never the whole block.

    A refusal that dumps two nested dictionaries side by side is a refusal nobody
    reads. The useful fact is WHICH number stopped being true, and the fitness
    blocks are small trees, so walking them costs nothing and the message stays one
    line a person can act on.
    """
    if isinstance(recorded, dict) and isinstance(recomputed, dict):
        for key in sorted(set(recorded) | set(recomputed)):
            if key not in recorded:
                return f"{path}.{key} (recomputed only: {recomputed[key]!r})"
            if key not in recomputed:
                return f"{path}.{key} (recorded only: {recorded[key]!r})"
            deeper = _first_disagreement(recorded[key], recomputed[key], f"{path}.{key}")
            if deeper:
                return deeper
        return ""
    if recorded != recomputed:
        return f"{path}: recorded {recorded!r}, recomputed {recomputed!r}"
    return ""


def main(argv: list[str] | None = None) -> None:
    """`python -m loop.calibrate <label> [<label> ...]` -- writes the round-1
    shape to `iterations/calibration-compaction/analysis.json`.

    `python -m loop.calibrate --model <label> [<label> ...]` -- writes the
    round-2 null-model artifact (`calibrate_model()`) to
    `iterations/calibration-compaction/model-r2.json` instead.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--model":
        labels = args[1:]
        if not labels:
            raise SystemExit("usage: python -m loop.calibrate --model <label> [<label> ...]")
        # `MODEL_TASKS`, not `SUPPORTED`: the artifact must rate every guard, or the
        # guards it names are gated on nothing (contract §6). The gain judgment stays
        # on `SUPPORTED`, which is what `calibrate_model` computes fitness over.
        model = calibrate_model(labels, RESULTS_DIR, SUPPORTED, coverage=MODEL_TASKS)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(model, indent=2) + "\n")
        print(f"wrote {MODEL_PATH}")
        return

    labels = args
    if not labels:
        raise SystemExit("usage: python -m loop.calibrate <label> [<label> ...]")
    analysis = calibrate(labels, RESULTS_DIR, SUPPORTED)
    ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2) + "\n")
    print(f"wrote {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
