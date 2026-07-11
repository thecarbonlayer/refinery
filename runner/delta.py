"""Δ between two harness states' suite results, and the acceptance rule.

The rule (Self-Harness): an edit is accepted iff Δ_in ≥ 0, Δ_ho ≥ 0, and
max(Δ_in, Δ_ho) > 0 — never a regression on either split, a real gain on one.
Split rates are the MEAN OF PER-TASK PASS FRACTIONS (each task weighs equally
regardless of attempt count), and fractions come from averaged repeats, never
majority vote (open-questions.md §8).
"""

from __future__ import annotations


def split_rate(results: dict, split: str) -> float:
    fracs = [t["pass_fraction"] for t in results["tasks"].values() if t["split"] == split]
    return sum(fracs) / len(fracs) if fracs else 0.0


def acceptance(delta_in: float, delta_ho: float) -> dict:
    return {
        "accepted": delta_in >= 0 and delta_ho >= 0 and max(delta_in, delta_ho) > 0,
        "delta_in": delta_in,
        "delta_ho": delta_ho,
    }


def delta(baseline: dict, candidate: dict) -> dict:
    """Compare two results JSONs (same task set required — a candidate that
    dropped a task would silently skew the split mean)."""
    base_names = set(baseline["tasks"])
    cand_names = set(candidate["tasks"])
    if base_names != cand_names:
        raise ValueError(
            f"task sets differ: only-baseline={sorted(base_names - cand_names)}, "
            f"only-candidate={sorted(cand_names - base_names)}"
        )
    d_in = split_rate(candidate, "held_in") - split_rate(baseline, "held_in")
    d_ho = split_rate(candidate, "held_out") - split_rate(baseline, "held_out")
    per_task = {
        name: candidate["tasks"][name]["pass_fraction"] - baseline["tasks"][name]["pass_fraction"]
        for name in sorted(base_names)
    }
    return {
        "baseline_fingerprint": baseline.get("fingerprint", {}),
        "candidate_fingerprint": candidate.get("fingerprint", {}),
        "delta_in": d_in,
        "delta_ho": d_ho,
        "per_task": per_task,
        **acceptance(d_in, d_ho),
    }
