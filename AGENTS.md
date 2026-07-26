# AGENTS.md — refinery

## What this is

refinery measures a coding agent's harness and improves it **without touching the
model's weights**. It runs a fixed 13-task suite against a live agent, records every
attempt verbatim, computes a delta between two harness states, and only then decides
whether a proposed edit was actually an improvement.

The target is [carbon](https://github.com/thecarbonlayer/carbon), a from-scratch
coding-agent harness. carbon declares a small, versioned **editable surface**
(`harness/harness_config.json`); refinery is the only thing that edits it, and it
edits it from outside.

Refining improves the material without replacing it. The weights never move; only the
harness around them does.

## The one rule that shapes everything

**The grader must never share a home with the thing being graded.** If the task
definitions, verifier code, pinned commands, or oracle hashes lived alongside the
editable surface, an editor could "pass" a task by rewriting its verifier instead of
fixing the harness. That is why refinery is a separate repo rather than a directory —
the boundary is enforced by topology, not discipline.

Corollaries you must preserve:

- Verifiers are **mechanical**. String/hash/exit-code checks, never model judgment.
- Oracles are **hash-pinned** at import time. A deleted or altered oracle is a
  spoofed task, not a passing one.
- Never move suite code into carbon, and never let carbon's config reach into
  `runner/`.

## Layout

    runner/tasks/     13 task specs in 4 clusters (mechanical verifiers only)
    runner/run.py     one attempt; runner/suite.py  the whole suite, resumable
    runner/delta.py   Δ_in / Δ_ho and the acceptance rule
    runner/guard.py   behavior-key resume guard (what invalidates a baseline)
    runner/carbon_env.py   binding to the carbon checkout under test
    loop/             validate -> branch -> PR pipeline
    iterations/       per-iteration artifacts, rejected candidates included
    results/          committed measurement records

## Working here

- **Siblings on disk.** `refinery/` and `carbon/` under one root. Both
  `pyproject.toml`'s `../carbon` and `runner/carbon_env.py`'s `CARBON_ROOT` assume
  it; change them together.
- **Real models, no mocks.** The suite drives a live endpoint from `carbon/.env`.
  refinery has no `.env` of its own, so the suite and the harness under test cannot
  disagree about which model ran.
- **`uv run pytest` is offline** — verifier helpers, Δ math, registry shape. It
  makes no model calls and must stay that way.
- **Editing `runner/` invalidates every baseline.** `runner_sha` is a content hash
  of `runner/**/*.py`; results are stamped with it and `delta` refuses to compare
  across versions. Change a verifier, re-record the baseline. This is deliberate —
  a verifier fix silently shifts pass rates.
- **A Δ is only meaningful like-for-like.** `delta` refuses filtered runs,
  mismatched per-task attempt counts, and mismatched models. Do not add an override.
- **Averages, never majority.** 3 attempts held-in, 5 held-out; a 2/3 is 0.67, not
  a pass. Held-out carries the generalization claim, so it gets more samples.

## The acceptance rule

    Δ_in ≥ 0  and  Δ_ho ≥ 0  and  max(Δ_in, Δ_ho) > 0

An edit must not regress either split and must improve at least one. Rejected
candidates are committed to `iterations/` alongside accepted ones — they are the
honest bulk of the record, and dropping them would make the loop look better than
it is.

## What the loop does and does not do

Mining and proposal are **reasoning**, performed by a proposer model and written to
fixed JSON artifacts (`iterations/<iter>/clusters.json`, `candidates.json`). Only
validation and the branch+PR step are code, so the pipeline consumes a stable
contract and never cares who produced a candidate.

A candidate is applied to carbon's **working tree**, never committed — a rejected
candidate leaves no trace there. The suite runs in a fresh subprocess (config binds
at import), the edit is reverted, and the rule decides.

Accepted edits each get their own branch off `self-improvement` in carbon and a PR
targeting it with the evidence in the body. **The pipeline never merges.** A human
decides.
