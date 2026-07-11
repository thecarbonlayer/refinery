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
    d_in = split_rate(candidate, "held_in") - split_rate(baseline, "held_in")
    d_ho = split_rate(candidate, "held_out") - split_rate(baseline, "held_out")
    per_task = {
        name: candidate["tasks"][name]["pass_fraction"] - baseline["tasks"][name]["pass_fraction"]
        for name in sorted(base_names)
    }
    return {
        "baseline_fingerprint": base_fp,
        "candidate_fingerprint": cand_fp,
        "delta_in": d_in,
        "delta_ho": d_ho,
        "per_task": per_task,
        **acceptance(d_in, d_ho),
    }
