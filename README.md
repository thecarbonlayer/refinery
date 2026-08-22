# refinery — measuring whether a change to an agent actually helped

> Refining improves the material without replacing it. The weights never move;
> only the harness around them does. The pipeline proposes, validates, and
> opens a PR. It never merges.

This is the measurement half of a two-repo project. The other half is
[carbon](https://github.com/thecarbonlayer/carbon), a coding agent built one
harness primitive at a time. Refinery's job is to answer a question that sounds
easy and is not: **did that change make the agent better, or did we get lucky?**

## The problem this exists to solve

Run an agent on a task twice and you may get two different answers. So when you
change something in the harness and the pass rate goes from 6/10 to 8/10, you
have learned almost nothing. That gap is well inside what the same code produces
on a quiet day.

The naive fix is to run more attempts, and it is not enough on its own. You also
need to know what *no change at all* looks like, measured on the same tasks
against the same model. Only then can you say whether a result sits outside the
noise.

So most of this repo is not about running tasks. It is machinery for not fooling
yourself:

- A fixed suite of tasks, each with a mechanical verifier. No model grades
  another model's work except at one carefully bounded seam.
- Repeated attempts per task, scored as pass **fractions** and averaged. A
  majority vote would throw away exactly the information that matters.
- A **null model**: recorded runs where nothing changed, used to learn how much
  a rate moves on its own. Thresholds come from that, not from a round number
  someone liked.
- A **three-outcome decision**: reject, ask for confirmation, or accept. A single
  run can never accept a change. Only a fresh, independent confirmation can.
- **Fail-closed everywhere.** If the calibration is missing, stale, or does not
  pass its own fitness checks, the rule refuses to judge rather than falling back
  to a weaker one.

That last point has teeth. The compaction suite is currently uncalibrated on
purpose, because its tasks now pass every attempt and a rate pinned at 1.0 has
no variation to model. The machinery refuses to install a gate that would gate
nothing.

## Who this is for

Engineers evaluating changes to an LLM system who have hit the same wall: the
numbers move, and you cannot tell whether you improved anything. The specific
tasks here are about coding-agent harnesses, but the calibration and
decision-rule ideas transfer to any evaluation where the thing you measure is
noisy and the change you are testing is small.

## Vocabulary

Three terms appear throughout and are worth having up front:

- **held-in / held-out** — the task split. Held-in tasks are the ones a change is
  allowed to target; held-out tasks watch for collateral damage. Held-out gets
  more attempts, because it is guarding rather than measuring.
- **Δ_in / Δ_ho** — the change in average pass fraction on each split between two
  harness states.
- **baseline / arm** — a recorded run. An *arm* where nothing was changed is a
  null arm, and null arms are what the calibration is built from.

`runner delta` reports Δ_in/Δ_ho against the original two-sided rule
(`Δ_in ≥ 0, Δ_ho ≥ 0, max > 0`). Treat that as a report, never a decision. This
loop measured that rule against six runs with nothing changed between them: of
the twelve pairs, it wrongly accepted 6 and reported a false regression on 3
more. It failed in both directions, which is why the real gate in
[Deciding](#deciding) looks the way it does.
## Carbon base

This repo is built against the carbon base named in
[`carbon-base.json`](carbon-base.json) — today the `self-improvement`
branch at a pinned commit. Importing the `loop` package (`python -m loop.cli`,
and any pytest run) checks the sibling checkout and fails with remediation if it
is missing required symbols. The runner CLI (`python -m runner.cli`) is not
guarded — `runner/` is frozen by the baseline content hash — so a wrong
checkout there still fails at import, just without remediation.

The promotion of `self-improvement` into carbon `main` landed on 2026-08-22
(carbon PR #16), so `main` now carries the symbols the suite imports and
`main` + `main` is an operable pair. The pin stays on `self-improvement`
deliberately: new carbon work lands there first, and following the promotion to
`main` would pin the base a step behind whatever is being developed. A carbon
HEAD that differs from the pinned COMMIT is a warning printed to stderr, not an
error — iteration work legitimately moves the checkout, and baseline reuse is
decided by the recorded behavior key, not the SHA.

One veto sits on top of the Δ arithmetic wherever a decision is made: a
candidate that moves any task from a 1.0 baseline pass fraction to 0.0 is
rejected even if another task's gain hides the collapse inside the split
average. Smaller per-task regressions remain visible as warnings rather than
hard failures because three- and five-attempt fractions are noisy. It is not
the only veto any more — see [Deciding](#deciding).

**Why this is its own repo:** the task definitions, verifier code, pinned
commands, and oracle hashes must never share a home with the editable surface
the external editor acts on — otherwise the editor can "pass" a task by
rewriting its verifier. The boundary is a repo boundary, not a directory
convention.

**Layout it expects:** `refinery/` and `carbon/` as sibling checkouts under one
root. `pyproject.toml`'s `../carbon` and `runner/carbon_env.py`'s `CARBON_ROOT`
both assume that — change them together if you nest things differently.

## Running

A real model must be serving; there are no mocks. refinery has no `.env` of its
own — it reads `carbon/.env` (via carbon's own loader, so real environment
variables still win), and the suite and the harness under test therefore cannot
disagree about what ran.

**Which serving base.** Carbon's checked-in `.env` points at a local LM Studio
endpoint. That is carbon's own dev and accept-gate default, and it still works
for spot-checks. It is not the base this program MEASURES on: the arms recorded
from 2026-08-22 onward (`results/p3-null-cmp-a.json` is the first) run against
OpenRouter with the provider and the quantization pinned, because unpinned
remote routing spreads one model label across providers
at mixed quantization and puts a serving confound inside the experiment.
`runner/guard.py` refuses outright to record against a remote base without both
pins. To point refinery at that base, set these in `carbon/.env` or export them:

    LLM_BASE_URL=https://openrouter.ai/api/v1
    LLM_MODEL=<routed model id>
    LLM_API_KEY=<OpenRouter key>
    LLM_PROVIDER_ORDER=<exactly one provider name>
    LLM_QUANTIZATION=<one quantization label>

A local base needs no pins — one physical server is its own complete serving
identity. The serving fields are folded into every result's `behavior_key`, so
switching bases forces a re-baseline rather than silently pooling two
populations. Do not copy values out of this paragraph: the most recent recorded
arm is `results/p3-null-cmp-a.json` and its `fingerprint` block is the authority
on what was actually served.

    uv sync
    uv run python -m runner.cli run --label baseline-main          # full suite, resumable
    uv run python -m runner.cli run --label x --only D1 --attempts 1   # spot-check
    uv run python -m runner.cli check baseline-main                # does that baseline still resume? (no model run)
    uv run python -m runner.cli delta results/baseline-main.json results/candidate.json

Results stream to results/<label>.jsonl per attempt (a killed run resumes,
skipping finished attempts; records are pinned to the carbon SHA + config
version they measured, and a resume refuses records from a different harness
state); aggregates land in results/<label>.json. Partial runs (--only) are
stamped with a "filter" field and refuse to overwrite a full run's JSON.
`delta` refuses filtered inputs, mismatched per-task attempt counts, mismatched
models, and mismatched serving fields (base URL, provider order, quantization,
reasoning effort, responder) by design — a Δ is only meaningful like-for-like.
`--force` on `run` discards a label's prior records and bypasses the
resume-guard; `runner check <label>` answers the same question read-only.
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
subprocess (config values bind at import), the edit is reverted, and a
decision is recorded (see [Deciding](#deciding)).
Accepted edits each get their own branch off `self-improvement` in carbon and a PR targeting it
(explicit base — never `main`), with the evidence (cluster, knobs, per-task Δ,
provenance) in the body. The pipeline never merges.

### Deciding

`loop/acceptance.py` is authoritative; this is the shape of it. Which path a
candidate takes depends on which editable section its edit maps to
(`loop/validate.py`):

- **A section the three-outcome rule covers** (`tool_output` always;
  `compaction` only while a fresh, fit calibration covers the measurements
  being judged) is decided by `loop.acceptance.evaluate()`, which returns
  REJECT or CONFIRM and *never* ACCEPT. CONFIRM means promising, not accepted.
  The only road to ACCEPT is a fresh paired rerun — `loop.cli confirm` →
  `loop.acceptance.confirmed()` — in which the original gain reappears and
  nothing regresses.
- **A calibration-required section with no fit calibration** (`compaction`
  today) is REFUSED with the reason recorded. It does not fall back to the
  Δ rule: two of this program's own no-change arms satisfy that rule outright,
  so falling back is how a candidate that changed nothing reaches ACCEPTED.
- **Everywhere else** the causal verdict decides: the Self-Harness Δ rule with
  movements on tasks the edited knob cannot reach zeroed out first.

Besides the collapse veto, a REJECT also follows from a rise in MECHANICAL
security failures (the harness breaking its own storage contract — blocked
unconditionally, never averaged away) and from a behavioral security rise that
the confirmation's predeclared one-sided Fisher test confirms.

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
    uv run python -m loop.cli validate --iteration iter-02 [--candidate output-policy] [--baseline results/<label>.json]
    uv run python -m loop.cli confirm  --iteration iter-02 --candidate output-policy \
        --baseline-label confirm-base --candidate-label confirm-cand --attempts 10
    uv run python -m loop.cli pr       --iteration iter-02 --candidate output-policy

## Layout

- runner/tasks/ — the task specs (deterministic verifiers by default; sentinels,
  pinned commands, and seed-file sha256s authored at import time; the one judged
  seam is the hash-pinned meaning judge in runner/judge.py, activation-gated on a
  committed validation artifact)
- runner/{run,suite}.py — attempt/suite drivers; runner/delta.py — Δ + rule
- runner/helpers.py — approve-and-log approver, environ guard,
  transcript/hash utilities
- runner/guard.py — the behavior key, the resume gate, and the serving-pin
  refusal that blocks recording against an unpinned remote base
- results/ — committed measurement artifacts
- loop/ — the validate→branch→PR pipeline (imports runner.suite / runner.delta)
- loop/acceptance.py — the three-outcome rule and the security vetoes;
  loop/calibrate.py — the measured null models the rule reads
- loop/compat.py — the carbon-base check (`carbon-base.json`) run on `loop` import
- loop/knob_coverage.py — which tasks can observe each editable knob, and in what
  role. Governance metadata, deliberately outside runner/ so correcting a row does
  not change runner_sha and invalidate every recorded baseline
- iterations/ — per-iteration artifacts: mining notes, clusters, candidates,
  validation records (rejected candidates included — they are the honest bulk)

The lettered clusters A-H cover context loss, verification integrity,
containment, tool use, large-item access, tool semantics and execution depth,
response completeness, repeated compaction, and subagent workspace binding.
`docs/carbon-quality-review.md` describes the failure each of those diagnostics
is designed to isolate; it was written for that generation of the suite and does
not cover the candidate suites below.

Four CANDIDATE suites were authored on 2026-08-21/22 ahead of their human gate.
They are registered — they appear in `TASKS`, they run, their offline premise
proofs are part of `pytest` — and they sit deliberately OUTSIDE every calibrated
gate, confirmation-guard set, null-model coverage set and knob-coverage row until
each runs its own null campaign. Their isolation is pinned by tests, not by
convention (`tests/test_cluster_v.py`, `tests/test_sel_tasks.py`,
`tests/test_registry.py`, `tests/test_calibrate.py`):

- **CTX-3..CTX-7** (`runner/tasks/cluster_ctx.py`) — context delivery through the
  `@path` injection door. Cluster id stays **A**, the mechanism family that owns
  that door via A4/A5.
- **ONB-1..ONB-5**, cluster **I** (`cluster_i.py`) — session onboarding and
  loaded instructions.
- **VER-4, LOOP-2..LOOP-6**, cluster **V** (`cluster_v.py`) — verification
  integrity and loop discipline.
- **SEL-2..SEL-5**, cluster **S** (`cluster_s.py`) — tool exposure. "S" for
  Select, out of letter sequence on purpose so parallel authoring streams did not
  all claim "I".

## Offline tests

    uv run pytest    # verifier helpers, Δ math, registry shape — no model calls

Green means all pass and exactly one skips. The skip is `tests/test_round2_attack.py`'s
calibrated sweep suspending ITSELF because `iterations/calibration-compaction/model-r2.json`
records `fitness.fit=false`: the compaction null model does not pass its own goodness
and stability checks, so it installs nothing and gates nothing, and a sweep built on it
would be asserting against a model that refused. That is the fail-closed design working.
It is restored by a fit artifact at this runner hash or a successor — not by unskipping
the test. `iterations/calibration-compaction/README.md` has the measurement history.
