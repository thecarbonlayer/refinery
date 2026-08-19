"""Measured noise from NULL arms: nothing changed between them, so every |Δ| this
module reports is sampling noise, not signal. Where `runner.delta` computes a Δ
between a baseline and a candidate and refuses anything partial, this tool exists
to consume the null-run PROTOCOL itself (contracts/phase2-calibration-contract.md
§2): one full-suite arm plus several `--only <supported set>` subset arms, all at
the same harness state, and turn their spread into thresholds a later rule
extension can use instead of borrowing `one_attempt` wholesale.

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

ROUND 2 (contracts/phase2b-calibration-contract.md): the round-1 threshold shape
above was measured, then withdrawn -- see iterations/calibration-compaction/
README.md for why (a bound below a real validation's own grain, an end-to-end
false ACCEPT reproduced from two of its own null arms, a 3-task-mean bound
applied to single-task deltas, no stated coverage). `calibrate()` and its output
shape stay exactly as they were, kept for history; `calibrate_model()` is the
round-2 replacement, alongside it, not instead of it. It stops storing
THRESHOLDS and starts storing the null MODEL itself -- per-task pooled pass
rates, exact fractions, with provenance -- so a rule extension can compute a
quantile at whatever attempt counts the judgment it is deciding actually used
(the round-1 defect: a bound measured at n=10 applied at n=3). `null_gain_quantile`
and `null_task_quantile` are the exact-enumeration primitives that computation
needs; they are exported so `loop.acceptance` can import them directly rather
than re-deriving exact binomial quantiles a second time.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from functools import cache
from itertools import combinations
from math import comb
from pathlib import Path

from loop.acceptance import evaluate
from runner.suite import RESULTS_DIR

EDITOR_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = EDITOR_ROOT / "iterations" / "calibration-compaction" / "analysis.json"
MODEL_PATH = EDITOR_ROOT / "iterations" / "calibration-compaction" / "model-r2.json"

# Contract §1: A1, G2, G4, G5 (aliases CMP-1..CMP-4) -- the section this phase
# calibrates. Only `main()`'s CLI default depends on this; `calibrate()`/
# `calibrate_model()` themselves always take `supported` explicitly, so tests
# never need to touch this. The round-2 contract (phase2b) pins the SAME four
# tasks as round-1 (phase2), so one constant serves both entry points.
SUPPORTED = frozenset({"A1", "G2", "G4", "G5"})

# Round-2's declared one-sided coverage (contract §1: "0.975"), as the exact
# Fraction every quantile in this module is computed at. 975/1000 reduces to
# 39/40 -- kept spelled out at the contract's own precision rather than
# hand-simplified, so a reader can see the "97.5%" the contract states.
COVERAGE_LEVEL = Fraction(975, 1000)

# Round-2's power-floor construction (contract §4.4): a true candidate rate of
# POOLED + 0.2 on the carrier, measured at the subset arms' own attempt count
# (contract §3's `--attempts 10`) -- the confirmation-shaped count a repeat/
# guard gate actually judges at, not the standard 3/5 a first pass uses.
POWER_OFFSET = Fraction(1, 5)
POWER_ATTEMPTS = 10

# The four fields the null-run protocol requires every arm to share (contract
# §2): a Δ across runner versions, config versions, models, or uncommitted carbon
# states is not a measurement of noise, it is a measurement of the thing that
# changed. `dirty_sha` is None for a clean tree -- None==None across arms is a
# consistent (both clean) digest, not a mismatch; only a genuine difference (two
# arms dirty in different, unrelated ways, or one dirty and one clean) refuses.
_FINGERPRINT_FIELDS = ("runner_sha", "config_version", "model", "dirty_sha")

# Round-2's provenance (contract §1): the same four fields PLUS `carbon_sha` --
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

    `carbon_sha` is not a literal key the runner writes -- contract §1 sources
    it from `gemma_sha`, the field the runner already stamps with the model/
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
    """contract §2's 4-field check (round-1, unchanged): reuse/extend point for
    `_check_fingerprint_fields` below, which `_check_provenance` (round-2)
    shares the same core with."""
    return _check_fingerprint_fields(arm_results, labels, _FINGERPRINT_FIELDS)


def _check_provenance(arm_results: dict[str, dict], labels: list[str]) -> dict:
    """contract §1's 5-field provenance check: round-1's 4 fields plus
    `carbon_sha`, extending `_check_fingerprints` rather than duplicating it."""
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
    """Per-arm provenance records (contract §1), the round-2 analog of
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
    contract §2) never share an attempt count, so a bound mixing them would
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
    """
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
    """The analysis artifact of contract §3, from the named arms' results JSONs
    under `results_dir`. Refuses (`ValueError` naming the field) on a fingerprint
    mismatch across arms; accepts filtered/subset arms outright -- that is the
    entire point of this tool, where `runner.delta` refuses them.

    Round-1 shape, kept intact for history -- see `calibrate_model()` for the
    round-2 (contracts/phase2b-calibration-contract.md) replacement.
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
# Round 2 (contracts/phase2b-calibration-contract.md): exact binomial
# enumeration -- the primitives, then the null-model artifact built on them.
# ---------------------------------------------------------------------------


@cache
def _binom_pmf(n: int, p: Fraction) -> tuple[Fraction, ...]:
    """P(X = k) for X ~ Binomial(n, p), k = 0..n -- exact `Fraction`, `math.comb`
    only, memoized. The same (n, p) pair recurs constantly across a single
    artifact's fitness checks: grain and stability both rebuild the SAME
    per-task pmf many times over (stability alone does it once per leave-one-
    arm-out pool), and goodness/power reuse the pooled rate across every arm --
    memoizing here is what keeps the whole artifact fast (contract's own
    "supported sets are <= 4 tasks at <= 20 attempts -- cheap").
    """
    q = 1 - p
    return tuple(Fraction(comb(n, k)) * p**k * q ** (n - k) for k in range(n + 1))


def _task_diff_pmf(n_a: int, n_b: int, p: Fraction) -> dict[Fraction, Fraction]:
    """Exact pmf of ONE task's own (b/n_b - a/n_a), a ~ Binomial(n_a, p),
    b ~ Binomial(n_b, p), independent -- the per-task building block
    `null_gain_quantile` convolves across the given task set. Each side keeps
    its OWN attempt count as the fraction's denominator here, before anything
    is averaged across tasks: a task judged at n_a=3 on one side and n_b=5 on
    the other (a real asymmetric evaluate()/confirmed() pair) must never be
    forced through a shared denominator that belongs to neither side.
    """
    pmf_a = _binom_pmf(n_a, p)
    pmf_b = _binom_pmf(n_b, p)
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
    (contract §2).

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
    side's own attempt count (a pooled rate over 49+ null attempts, contract
    §3): the exact arithmetic is what keeps raising that denominator safe
    (`Fraction(21, 49) ** 10` is exact, not a float that has already lost the
    difference between 21/49 and its neighbors).

    Performance note: every real caller (`evaluate()`/`confirmed()`, gated by
    their own `_parity` check) has `attempts_a[t] == attempts_b[t]` for every
    task `t` -- a baseline and candidate always ran the SAME suite config.
    That symmetric case stays fast even at the contract's stated ceiling (<=4
    tasks, <=20 attempts: ~60ms measured). Fully asymmetric attempts across
    several multi-attempt tasks at once is supported by this signature but not
    a real call shape, and its per-task diff pmf can carry far more distinct
    Fraction keys, so it is not the case this function is optimized for.
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
    if not (0 < level <= 1):
        raise ValueError(f"null_gain_quantile: level must be in (0, 1], got {level}")
    n = len(tasks)
    total: dict[Fraction, Fraction] = {Fraction(0): Fraction(1)}
    for t in tasks:
        task_pmf = _task_diff_pmf(attempts_a[t], attempts_b[t], rates[t])
        nxt: dict[Fraction, Fraction] = {}
        for s, sp in total.items():
            if sp == 0:
                continue
            for d, dp in task_pmf.items():
                if dp == 0:
                    continue
                nxt[s + d] = nxt.get(s + d, Fraction(0)) + sp * dp
        total = nxt
    cdf = Fraction(0)
    for s in sorted(total):
        cdf += total[s]
        if cdf >= level:
            return s / n
    raise AssertionError(  # pragma: no cover -- a normalized pmf always reaches level<=1
        "null_gain_quantile: CDF never reached `level` -- the enumerated pmf did not "
        "sum to 1, which should be impossible"
    )


def null_task_quantile(
    rate: Fraction, attempts_a: int, attempts_b: int, level: Fraction
) -> Fraction:
    """Single-task specialization of `null_gain_quantile` -- contract §2's
    carrier/guard construction: a repeat gate (per carrier task) and a guard
    adjudication both judge ONE task's own null distribution, never a
    supported-set mean. Implemented as `null_gain_quantile` over a single
    synthetic task key so the two constructions can never silently diverge.
    """
    return null_gain_quantile({"_task": rate}, {"_task": attempts_a}, {"_task": attempts_b}, level)


def _fraction_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def _pooled_counts(arm_results: dict[str, dict], labels: list[str], task: str) -> tuple[int, int]:
    passes = sum(int(arm_results[label]["tasks"][task]["passes"]) for label in labels)
    attempts = sum(int(arm_results[label]["tasks"][task]["attempts"]) for label in labels)
    return passes, attempts


def _check_supported_set(
    arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]
) -> None:
    """contract §1's pin: {A1, G2, G4, G5}, held exactly.

    A subset (`--only`, `filter` key present) arm carrying a task OUTSIDE
    `supported` refuses -- a filtered run is a promise that only the
    supported set was measured, and a task the pool was never meant to cover
    appearing there means the arm does not match the protocol this artifact
    assumes. Independently, ANY arm (filtered or full-suite) missing a
    supported task refuses -- the pool needs every arm's count on every
    supported task, never a partial denominator standing in for the whole; a
    real full-suite arm always carries the whole suite, so this branch only
    ever fires on a truncated or mislabeled result file.
    """
    for label in labels:
        result = arm_results[label]
        tasks = set(result["tasks"])
        if "filter" in result:
            extra = tasks - supported
            if extra:
                raise ValueError(
                    f"arm {label!r} is a subset run carrying task(s) outside the "
                    f"pinned supported set {sorted(supported)}: {sorted(extra)}"
                )
        missing = supported - tasks
        if missing:
            raise ValueError(
                f"arm {label!r} is missing supported task(s) {sorted(missing)} -- the "
                f"null model needs every arm to cover the pinned supported set "
                f"{sorted(supported)}"
            )


def _task_splits(
    arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]
) -> dict[str, str]:
    splits: dict[str, str] = {}
    for label in labels:
        for task in supported:
            if task in arm_results[label]["tasks"] and task not in splits:
                splits[task] = arm_results[label]["tasks"][task]["split"]
    return splits


def _standard_attempts(
    arm_results: dict[str, dict], labels: list[str], supported: frozenset[str]
) -> dict[str, int]:
    """The full-suite attempt count per task -- 3 held-in / 5 held-out in the
    real protocol (contract §3), never the subset arms' 10. Read from any arm
    that is NOT a `--only` subset run (no `filter` key present): the same
    literal signal `_pairwise_outcomes` already uses to tell a full-suite arm
    apart from a filtered one. Refuses if no such arm is present among
    `labels` -- the fitness checks (grain, stability, power) cannot define
    "standard attempt counts" without at least one arm that ran the real
    suite, and the round-2 protocol always includes three (`r2-null-full-*`).
    """
    attempts: dict[str, int] = {}
    for label in labels:
        result = arm_results[label]
        if "filter" in result:
            continue
        for task in supported:
            if task in result["tasks"] and task not in attempts:
                attempts[task] = int(result["tasks"][task]["attempts"])
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
    """contract §1's per-task null model: a pooled `null_rate` as an exact,
    UNREDUCED "passes/attempts" string -- the denominator IS the pooled
    attempt count (contract §3's ">= 49 held-in, >= 55 for G2"), so reducing
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
    """Fitness check 1 (contract §4.1): at standard attempt counts, the
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
    """Fitness check 2 (contract §4.2): for each task, an exact binomial
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
    """Fitness check 3 (contract §4.3): leave-one-arm-out pooled rates must
    not shift a standard-count threshold across a REPRESENTABLE-MOVEMENT
    boundary. Representable movements are multiples of the split's own grain
    (`_split_grain`, the same quantity fitness check 1 gates on): a threshold
    that moves but stays inside the same grain bucket cannot change which
    real, quantized observation clears it, so only a bucket CROSSING counts
    as instability. `floor(quantile / grain)` is that bucket index, computed
    with exact `Fraction` floor division.
    """
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
        if len(labels) > 1:
            for held_out in labels:
                remaining = [label for label in labels if label != held_out]
                loo_rates = {t: Fraction(*_pooled_counts(arm_results, remaining, t)) for t in tasks}
                loo_q = null_gain_quantile(loo_rates, attempts, attempts, level)
                if loo_q // grain != full_bucket:
                    moved[held_out] = _fraction_str(loo_q)
        ok = not moved
        overall = overall and ok
        result[split] = {
            "tasks": tasks,
            "full_quantile": _fraction_str(full_q),
            "grain": _fraction_str(grain),
            "moved_excluding": moved,
            "pass": ok,
        }
    result["pass"] = overall
    return result


def _check_power(
    null_counts: dict[str, tuple[int, int]],
    supported: frozenset[str],
    level: Fraction,
    *,
    offset: Fraction = POWER_OFFSET,
    attempts: int = POWER_ATTEMPTS,
) -> dict:
    """Fitness check 4 (contract §4.4): recorded, never gating. For each
    supported task (any of which can be a confirmation's carrier), the
    probability that a TRUE candidate rate of pooled + 0.2 clears that task's
    own null quantile at `attempts` v `attempts` (contract §3's subset attempt
    count, the confirmation-shaped one) -- enumerated exactly over both sides'
    binomial outcomes, published so nobody mistakes a weak test for a strong
    one (this module's whole reason for existing).
    """
    per_task: dict[str, dict] = {}
    for task in sorted(supported):
        p = Fraction(*null_counts[task])
        # Capped at 1: a pooled rate already close to 1 (a near-ceiling task)
        # plus the flat +0.2 offset can overshoot a valid probability, and an
        # alternative rate above 1 is not a rate.
        p_alt = min(Fraction(1), p + offset)
        threshold = null_task_quantile(p, attempts, attempts, level)
        pmf_a = _binom_pmf(attempts, p)
        pmf_b = _binom_pmf(attempts, p_alt)
        cleared = Fraction(0)
        for a_k, a_p in enumerate(pmf_a):
            if a_p == 0:
                continue
            for b_k, b_p in enumerate(pmf_b):
                if b_p == 0:
                    continue
                if Fraction(b_k - a_k, attempts) > threshold:
                    cleared += a_p * b_p
        per_task[task] = {
            "pooled_rate": _fraction_str(p),
            "alt_rate": _fraction_str(p_alt),
            "attempts": attempts,
            "threshold": _fraction_str(threshold),
            "power": _fraction_str(cleared),
            "power_float": float(cleared),
        }
    return {"offset": _fraction_str(offset), "attempts": attempts, "per_task": per_task}


def calibrate_model(labels: list[str], results_dir: Path, supported: frozenset[str]) -> dict:
    """The round-2 null-MODEL artifact (contracts/phase2b-calibration-contract.md
    §1+§4), from the named arms' results JSONs under `results_dir`.

    Refuses (`ValueError`, naming what's wrong) on: no labels; a duplicate
    label; a provenance mismatch across arms (the round-1 4 fields plus
    `carbon_sha`); a subset (`--only`) arm carrying a task outside the pinned
    {A1, G2, G4, G5} supported set; any arm missing a supported task; or no
    un-filtered arm to read standard attempt counts from. It does NOT refuse
    on a failed fitness check -- `fitness.fit` is computed and recorded either
    way (contract §4: "the artifact is written with `fit: false`"), and it is
    the LOADER's job, not this function's, to refuse installing an unfit
    artifact.

    Round-1's `calibrate()` and its output shape are untouched; this is a
    separate entry point, not a replacement in place.
    """
    _check_labels(labels, "calibrate_model")
    arm_results = {label: _load_arm(results_dir, label) for label in labels}
    _check_provenance(arm_results, labels)
    _check_supported_set(arm_results, labels, supported)

    provenance = _build_provenance(arm_results, labels)
    null_counts = {task: _pooled_counts(arm_results, labels, task) for task in sorted(supported)}
    null_model = _build_null_model(arm_results, labels, supported, null_counts)
    task_split = _task_splits(arm_results, labels, supported)
    standard_attempts = _standard_attempts(arm_results, labels, supported)

    grain = _check_grain(null_counts, task_split, standard_attempts, supported, COVERAGE_LEVEL)
    goodness = _check_goodness(arm_results, labels, supported, null_counts)
    stability = _check_stability(
        arm_results, labels, supported, task_split, standard_attempts, COVERAGE_LEVEL
    )
    power = _check_power(null_counts, supported, COVERAGE_LEVEL)
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
            "fit": fit,
        },
        "computed_at_runner_sha": provenance[0]["runner_sha"],
    }


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
        model = calibrate_model(labels, RESULTS_DIR, SUPPORTED)
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
