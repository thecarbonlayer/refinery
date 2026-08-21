"""Δ between two harness states' suite results, and the acceptance rule.

The rule (Self-Harness): an edit is accepted iff Δ_in ≥ 0, Δ_ho ≥ 0, and
max(Δ_in, Δ_ho) > 0 — never a regression on either split, a real gain on one.
Split rates are the MEAN OF PER-TASK PASS FRACTIONS (each task weighs equally
regardless of attempt count), and fractions come from averaged repeats, never
majority vote.
"""

from __future__ import annotations

from fractions import Fraction

# The recorded Provider fields the serving-parity gate compares. ``model`` is absent
# only because it has its own dedicated gate (with a message naming both models);
# together they cover exactly ``runner.carbon_env.PROVIDER_FIELDS_FINGERPRINTED`` —
# a cross-module test pins that, so a field recorded in the fingerprint cannot go
# ungated here.
_SERVING_FIELDS = ("base_url", "reasoning_effort", "provider_order", "quantization")


def _exact(task: dict) -> Fraction:
    """A task's pass rate as an EXACT fraction of its integer counts.

    Never `pass_fraction`. That field is written rounded to 4 places for display
    (`suite.py`), and summing rounded values loses information the counts still have:
    2/3 stores as 0.6667 while two thirds sum to 0.3333 + 0.3333 = 0.6666. Two runs
    with IDENTICAL integer counts then reported Δ_in = -5.56e-06 — a false regression
    signal manufactured by the storage format, which the acceptance rule read as a real
    one because it tests `>= 0`. Measured on six unchanged runs: it fired on 2 of 15
    pairs, and correcting it flips both from REJECT to ACCEPT.

    Falls back to the stored fraction only for results written before counts were
    recorded, so an old baseline still compares rather than crashing.
    """
    if "passes" in task and task.get("attempts"):
        return Fraction(int(task["passes"]), int(task["attempts"]))
    return Fraction(task["pass_fraction"]).limit_denominator(10_000)


def split_rate(results: dict, split: str) -> float:
    """Mean of per-task pass rates, each task weighing equally.

    Computed exactly and converted to float ONCE, at the boundary. Every intermediate
    stays a `Fraction`, so a set of movements that cancels lands on exactly 0.0 rather
    than on a tiny negative the rule would reject.
    """
    return float(exact_split_rate(results, split))


def exact_split_rate(results: dict, split: str) -> Fraction:
    fracs = [_exact(t) for t in results["tasks"].values() if t["split"] == split]
    return sum(fracs) / len(fracs) if fracs else Fraction(0)


def acceptance(delta_in: float, delta_ho: float) -> dict:
    return {
        "accepted": delta_in >= 0 and delta_ho >= 0 and max(delta_in, delta_ho) > 0,
        "delta_in": delta_in,
        "delta_ho": delta_ho,
    }


def delta(baseline: dict, candidate: dict) -> dict:
    """Compare two results JSONs (same task set required — a candidate that
    dropped a task would silently skew the split mean). Filtered (--only)
    results are refused outright: a partial suite skews both split means."""
    for label, res in (("baseline", baseline), ("candidate", candidate)):
        if "filter" in res:
            raise ValueError(
                f"refusing to compute Δ from filtered (partial) results: "
                f"{label} was produced with --only {res['filter']}"
            )
    base_names = set(baseline["tasks"])
    cand_names = set(candidate["tasks"])
    if base_names != cand_names:
        raise ValueError(
            f"task sets differ: only-baseline={sorted(base_names - cand_names)}, "
            f"only-candidate={sorted(cand_names - base_names)}"
        )
    # per-task attempts parity: fractions from unequal sample sizes have
    # unequal precision, so a Δ over them is not like-for-like.
    mismatched = [
        f"{name} (baseline {baseline['tasks'][name]['attempts']} vs "
        f"candidate {candidate['tasks'][name]['attempts']})"
        for name in sorted(base_names)
        if baseline["tasks"][name]["attempts"] != candidate["tasks"][name]["attempts"]
    ]
    if mismatched:
        raise ValueError(
            "sample-size mismatch — rerun candidate with matching attempts: "
            + ", ".join(mismatched)
        )
    # one read of each fingerprint, used for every parity gate and the echo
    base_fp = baseline.get("fingerprint", {})
    cand_fp = candidate.get("fingerprint", {})
    # presence gate: two fingerprint-less results would pass every parity
    # check below on None==None — unattributed measurements prove nothing.
    if not base_fp or not cand_fp:
        raise ValueError(
            "results file lacks a fingerprint — refusing to compare unattributed measurements"
        )
    # model parity: a Δ across models measures the model swap, not the edit.
    base_model = base_fp.get("model")
    cand_model = cand_fp.get("model")
    if base_model != cand_model:
        raise ValueError(
            f"model mismatch: baseline ran {base_model!r}, candidate ran {cand_model!r} "
            f"— Δ across models is meaningless"
        )
    # verifier parity: the runner IS part of the measurement apparatus; a Δ
    # across runner versions measures the verifier change, not the edit.
    if base_fp.get("runner_sha") != cand_fp.get("runner_sha"):
        raise ValueError(
            "verifier version mismatch — results were produced by different "
            "runner versions; re-measure"
        )
    # serving parity: the same model string served from a different endpoint, by a
    # different provider, at a different quantization, or at a different reasoning
    # effort is a different serving base in everything but name — a Δ across serving
    # bases measures the serving swap, not the edit. None == None is a genuine match
    # (the field was unset on both sides). A record predating these fields cannot
    # slip past against one carrying them: the runner that stamps them hashes to a
    # different runner_sha, so the verifier gate above refuses that pair first.
    for field in _SERVING_FIELDS:
        if base_fp.get(field) != cand_fp.get(field):
            raise ValueError(
                f"serving mismatch on {field}: baseline {base_fp.get(field)!r} vs "
                f"candidate {cand_fp.get(field)!r} — Δ across serving bases measures "
                f"the serving swap, not the edit"
            )
    # Exact throughout, floated once at the end. Subtracting two rounded means is what
    # produced a -5.56e-06 "regression" between runs with identical integer counts.
    d_in = float(exact_split_rate(candidate, "held_in") - exact_split_rate(baseline, "held_in"))
    d_ho = float(exact_split_rate(candidate, "held_out") - exact_split_rate(baseline, "held_out"))
    per_task = {
        name: float(_exact(candidate["tasks"][name]) - _exact(baseline["tasks"][name]))
        for name in sorted(base_names)
    }
    regressions = {name: change for name, change in per_task.items() if change < 0}
    # The aggregate Self-Harness rule can average a total collapse on one task
    # against a gain on another task in the same split. Refinery observed this
    # in iteration 1: A1 moved 1.0 -> 0.0 while A2 moved 0.0 -> 1.0, leaving
    # Δ_in unchanged. Treat a full-pass -> zero-pass movement as a promotion
    # veto. Smaller negative movements stay visible as regression warnings;
    # with only 3/5 stochastic attempts, making every single flip a hard veto
    # would mostly measure sampling noise.
    catastrophic_regressions = {
        name: change
        for name, change in per_task.items()
        if _exact(baseline["tasks"][name]) == 1 and _exact(candidate["tasks"][name]) == 0
    }
    aggregate = acceptance(d_in, d_ho)
    base_summary = baseline.get("summary", {})
    cand_summary = candidate.get("summary", {})
    base_metrics = base_summary.get("metrics", {})
    candidate_metrics = cand_summary.get("metrics", {})
    # A metric absent on one side was NOT measured as zero. Subtracting against a
    # default of 0 turns "not measured" into the most favourable number available
    # — a large negative cost delta — which the PR body then renders under prose
    # saying negative means less work for the same score. Compare only what both
    # sides actually measured, and name the rest instead of imputing it.
    comparable = sorted(set(base_metrics) & set(candidate_metrics))
    metric_delta = {
        name: float(candidate_metrics[name]) - float(base_metrics[name]) for name in comparable
    }
    metric_not_compared = sorted(set(base_metrics) ^ set(candidate_metrics))

    # Denominator drift: an attempt that raises records metrics={}, so a task the
    # candidate BREAKS drops out of a metric's population entirely. Both sides can
    # then report the same mean over different task sets. Diagnostic only — it
    # never moves acceptance — but it must be visible, not averaged across.
    def _pair(key: str) -> dict[str, dict]:
        base, cand = base_summary.get(key, {}), cand_summary.get(key, {})
        return {
            name: {"baseline": base.get(name), "candidate": cand.get(name)}
            for name in sorted(set(base) | set(cand))
        }

    metric_task_counts = _pair("metric_task_counts")
    # Attempt counts as well as task counts. Task counts alone are blind to the
    # commonest case: a candidate whose attempts start ERRORING keeps every task in
    # the population (same task count) while the mean silently covers a third as
    # many attempts, which reads as a large cost win.
    metric_attempt_counts = _pair("metric_attempt_counts")
    metric_denominator_drift = sorted(
        {
            name
            for counts in (metric_task_counts, metric_attempt_counts)
            for name, pair in counts.items()
            if pair["baseline"] != pair["candidate"]
        }
    )
    return {
        "baseline_fingerprint": base_fp,
        "candidate_fingerprint": cand_fp,
        "delta_in": d_in,
        "delta_ho": d_ho,
        "per_task": per_task,
        "regressions": regressions,
        "catastrophic_regressions": catastrophic_regressions,
        "baseline_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_delta": metric_delta,
        "metric_not_compared": metric_not_compared,
        "metric_task_counts": metric_task_counts,
        "metric_attempt_counts": metric_attempt_counts,
        "metric_denominator_drift": metric_denominator_drift,
        "aggregate_accepted": aggregate["accepted"],
        "accepted": aggregate["accepted"] and not catastrophic_regressions,
    }
