# refinery — task-suite runner + Δ measurement + the self-improvement loop

> Refining improves the material without replacing it. The weights never move;
> only the harness around them does. The pipeline proposes, validates, and
> opens a PR — it never merges.

The measurement half of the self-evolving-harness project: runs a fixed task
suite (`runner/tasks/`) against a live Carbon agent driven by the
[carbon](https://github.com/thecarbonlayer/carbon) harness, repeatedly per task
(held-out gets more samples than held-in), aggregates pass fractions
(averaged, never majority-voted), and computes Δ_in/Δ_ho between two harness
states for the aggregate acceptance rule
`Δ_in ≥ 0, Δ_ho ≥ 0, max(Δ_in, Δ_ho) > 0`.

There is one additional promotion veto: a candidate that moves any task from a
1.0 baseline pass fraction to 0.0 is rejected even if another task's gain hides
the collapse inside the split average. Smaller per-task regressions remain
visible as warnings rather than hard failures because three- and five-attempt
fractions are noisy.

**Why this is its own repo:** the task definitions, verifier code, pinned
commands, and oracle hashes must never share a home with the editable surface
the external editor acts on — otherwise the editor can "pass" a task by
rewriting its verifier. The boundary is a repo boundary, not a directory
convention.

**Layout it expects:** `refinery/` and `carbon/` as sibling checkouts under one
root. `pyproject.toml`'s `../carbon` and `runner/carbon_env.py`'s `CARBON_ROOT`
both assume that — change them together if you nest things differently.

## Running

LM Studio must be serving the model named in `carbon/.env` (real models, no
mocks). refinery has no `.env` of its own — it reads carbon's, so the suite
and the harness under test always agree on the endpoint.

    uv sync
    uv run python -m runner.cli run --label baseline-main          # full suite, resumable
    uv run python -m runner.cli run --label x --only D1 --attempts 1   # spot-check
    uv run python -m runner.cli delta results/baseline-main.json results/candidate.json

Results stream to results/<label>.jsonl per attempt (a killed run resumes,
skipping finished attempts; records are pinned to the carbon SHA + config
version they measured, and a resume refuses records from a different harness
state); aggregates land in results/<label>.json. Partial runs (--only) are
stamped with a "filter" field and refuse to overwrite a full run's JSON.
`delta` refuses filtered inputs, mismatched per-task attempt counts, and
mismatched models by design — a Δ is only meaningful like-for-like.
Results are also stamped with `runner_sha` (the verifier's own version);
deltas across different runner versions are refused, so re-measure the
baseline after changing runner code. The harness-quality tasks added after the
first recorded iteration deliberately change this hash; old results remain
historical evidence and are not comparable to a new run.

Every attempt can also record tokens, cost, model/tool calls, compactions,
tool errors, and incomplete responses. Candidate deltas report those values
next to success rates. They explain tradeoffs; they never turn a failing
candidate into a passing one.

To measure a different harness state: check out the branch in carbon
(the editable dependency points at that working tree), run with a new label,
then `delta` the two JSONs.

## The loop (mine -> propose -> validate -> PR)

`loop/` is the pipeline half: the mining and proposal steps are done by the
proposer model directly as reasoning (Fable, in-session, for iteration 1)
and land as fixed JSON artifacts in `iterations/<iter>/` (`clusters.json`,
`candidates.json`); only validation and the branch+PR step are code. A
candidate is applied to the carbon WORKING TREE (never committed — a
rejected candidate leaves no trace there), the suite runs in a fresh
subprocess (config values bind at import), the edit is reverted, and the
aggregate rule plus the catastrophic per-task regression veto decides.
Accepted edits each get their own branch off `self-improvement` in carbon and a PR targeting it
(explicit base — never `main`), with the evidence (cluster, knobs, per-task Δ,
provenance) in the body. The pipeline never merges.

Before proposing, inspect the live contract:

    uv run python -m loop.cli surface

It lists only fields Refinery may edit, including each bounded strategy menu,
and separately lists Carbon's immutable invariants. `require_run`,
`approval_tools`, code-extension gate triggers, workspace/secret boundaries,
tool validation, unique edits, worker workspace identity, and strategy
registration are deliberately unavailable to candidates. An unexpressible
policy improvement is a `strategy_surface_gap`; a broken invariant is a
`correctness_defect`. Neither should be disguised as a configuration edit.

    uv run python -m loop.cli dry-run  --iteration iter-02 --candidate output-policy --tasks E2 D1
    uv run python -m loop.cli validate --iteration iter-02 [--candidate output-policy]
    uv run python -m loop.cli pr       --iteration iter-02 --candidate output-policy

## Layout

- runner/tasks/ — the task specs (mechanical verifiers only; sentinels,
  pinned commands, and seed-file sha256s authored at import time)
- runner/{run,suite}.py — attempt/suite drivers; runner/delta.py — Δ + rule
- runner/helpers.py — approve-and-log approver, environ guard,
  transcript/hash utilities
- results/ — committed measurement artifacts
- loop/ — the validate→branch→PR pipeline (imports runner.suite / runner.delta)
- loop/knob_coverage.py — which tasks can observe each editable knob, and in what
  role. Governance metadata, deliberately outside runner/ so correcting a row does
  not change runner_sha and invalidate every recorded baseline
- iterations/ — per-iteration artifacts: mining notes, clusters, candidates,
  validation records (rejected candidates included — they are the honest bulk)

The task clusters currently cover context loss, verification integrity,
containment, tool use, large-item access, tool semantics and execution depth,
response completeness, repeated compaction, and subagent workspace binding.
See `docs/carbon-quality-review.md` for the failure each newer diagnostic is
designed to isolate and the recommended Carbon implementation order.

## Offline tests

    uv run pytest    # verifier helpers, Δ math, registry shape — no model calls
