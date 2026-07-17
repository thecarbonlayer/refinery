# harness-editor — task-suite runner + Δ measurement + the self-improvement loop

The measurement half of the self-evolving-harness project: runs the 13-task
suite (docs/research/self-evolving-harness/task-suite-v2.md) against a live
Gemma agent driven by the gemma harness, N times per task (3 held-in /
5 held-out), aggregates pass fractions (averaged, never majority-voted), and
computes Δ_in/Δ_ho between two harness states for the acceptance rule
`Δ_in ≥ 0, Δ_ho ≥ 0, max(Δ_in, Δ_ho) > 0`.

**Why it lives here and not in gemma:** the task definitions, verifier
code, pinned commands, and oracle hashes must never share a home with the
editable surface an external editor (Sol) acts on — otherwise the editor can
"pass" a task by rewriting its verifier. This directory is outside the fork
Sol edits, by design. (Review flag (d)-1; open-questions.md §7.)

## Running

LM Studio must be serving the model in gemma/.env (real models, no mocks).

    uv sync
    uv run python -m runner.cli run --label baseline-main          # full suite, resumable
    uv run python -m runner.cli run --label x --only D1 --attempts 1   # spot-check
    uv run python -m runner.cli delta results/baseline-main.json results/candidate.json

Results stream to results/<label>.jsonl per attempt (a killed run resumes,
skipping finished attempts; records are pinned to the gemma SHA + config
version they measured, and a resume refuses records from a different harness
state); aggregates land in results/<label>.json. Partial runs (--only) are
stamped with a "filter" field and refuse to overwrite a full run's JSON.
`delta` refuses filtered inputs, mismatched per-task attempt counts, and
mismatched models by design — a Δ is only meaningful like-for-like.
Results are also stamped with `runner_sha` (the verifier's own version);
deltas across different runner versions are refused, so re-measure the
baseline after changing runner code.

To measure a different harness state: check out the branch in gemma
(the editable dependency points at that working tree), run with a new label,
then `delta` the two JSONs.

## The loop (mine -> propose -> validate -> PR)

`loop/` is the pipeline half: the mining and proposal steps are done by the
proposer model directly as reasoning (Fable, in-session, for iteration 1 —
see docs/research/self-evolving-harness/todo-begin-self-improvement-loop.md)
and land as fixed JSON artifacts in `iterations/<iter>/` (`clusters.json`,
`candidates.json`); only validation and the branch+PR step are code. A
candidate is applied to the gemma WORKING TREE (never committed — a
rejected candidate leaves no trace there), the suite runs in a fresh
subprocess (config values bind at import), the edit is reverted, and the
acceptance rule `Δ_in ≥ 0, Δ_ho ≥ 0, max > 0` decides. Accepted edits each get
their own branch off `self-improvement` in gemma and a PR targeting it
(explicit base — never `main`), with the evidence (cluster, knobs, per-task Δ,
provenance) in the body. The pipeline never merges.

    uv run python -m loop.cli dry-run  --iteration iter-01 --candidate clamp-12k --tasks A2 D1
    uv run python -m loop.cli validate --iteration iter-01 [--candidate clamp-12k]
    uv run python -m loop.cli pr       --iteration iter-01 --candidate clamp-12k

## Layout

- runner/tasks/ — the 13 task specs (mechanical verifiers only; sentinels,
  pinned commands, and seed-file sha256s authored at import time)
- runner/{run,suite}.py — attempt/suite drivers; runner/delta.py — Δ + rule
- runner/helpers.py — approve-and-log approver (flag (d)-2), environ guard,
  transcript/hash utilities
- results/ — committed measurement artifacts
- loop/ — the validate→branch→PR pipeline (imports runner.suite / runner.delta)
- iterations/ — per-iteration artifacts: mining notes, clusters, candidates,
  validation records (rejected candidates included — they are the honest bulk)

## Offline tests

    uv run pytest    # verifier helpers, Δ math, registry shape — no model calls
