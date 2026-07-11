# harness-editor — task-suite runner + Δ measurement

The measurement half of the self-evolving-harness project: runs the 13-task
suite (docs/research/self-evolving-harness/task-suite-v2.md) against a live
Gemma agent driven by the dist/gemma harness, N times per task (3 held-in /
5 held-out), aggregates pass fractions (averaged, never majority-voted), and
computes Δ_in/Δ_ho between two harness states for the acceptance rule
`Δ_in ≥ 0, Δ_ho ≥ 0, max(Δ_in, Δ_ho) > 0`.

**Why it lives here and not in dist/gemma:** the task definitions, verifier
code, pinned commands, and oracle hashes must never share a home with the
editable surface an external editor (Sol) acts on — otherwise the editor can
"pass" a task by rewriting its verifier. This directory is outside the fork
Sol edits, by design. (Review flag (d)-1; open-questions.md §7.)

## Running

LM Studio must be serving the model in dist/gemma/.env (real models, no mocks).

    uv sync
    uv run python -m runner.cli run --label baseline-main          # full suite, resumable
    uv run python -m runner.cli run --label x --only D1 --attempts 1   # spot-check
    uv run python -m runner.cli delta results/baseline-main.json results/candidate.json

Results stream to results/<label>.jsonl per attempt (a killed run resumes,
skipping finished attempts; records are pinned to the dist/gemma SHA + config
version they measured, and a resume refuses records from a different harness
state); aggregates land in results/<label>.json. Partial runs (--only) are
stamped with a "filter" field and refuse to overwrite a full run's JSON.
`delta` refuses filtered inputs, mismatched per-task attempt counts, and
mismatched models by design — a Δ is only meaningful like-for-like.
Results are also stamped with `runner_sha` (the verifier's own version);
deltas across different runner versions are refused, so re-measure the
baseline after changing runner code.

To measure a different harness state: check out the branch in dist/gemma
(the editable dependency points at that working tree), run with a new label,
then `delta` the two JSONs.

## Layout

- runner/tasks/ — the 13 task specs (mechanical verifiers only; sentinels,
  pinned commands, and seed-file sha256s authored at import time)
- runner/{run,suite}.py — attempt/suite drivers; runner/delta.py — Δ + rule
- runner/helpers.py — approve-and-log approver (flag (d)-2), environ guard,
  transcript/hash utilities
- results/ — committed measurement artifacts
- ../harness-editor/loop/ (future) — the mine→propose→validate pipeline,
  which imports runner.suite / runner.delta

## Offline tests

    uv run pytest    # verifier helpers, Δ math, registry shape — no model calls
