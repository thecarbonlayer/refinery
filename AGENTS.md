# AGENTS.md — refinery

## What this is

refinery runs a fixed task suite against a live coding agent, records every attempt
verbatim, computes a delta between two harness states, and decides whether a proposed
edit was an improvement. It edits the harness, never the model.

The target is [carbon](https://github.com/thecarbonlayer/carbon), which declares a
small, versioned *editable surface*. refinery is the only thing that edits it, and it
edits it from outside.

## The one rule that shapes everything

**The grader must never share a home with the thing being graded.** If the task
definitions, verifier code, pinned commands, or oracle hashes lived alongside the
editable surface, an editor could pass a task by rewriting its verifier instead of
fixing the harness. That is why refinery is a separate repo rather than a directory.

Never move suite code into carbon. Never let carbon's config reach into `runner/`.

## Before you commit

- **No absolute filesystem paths.** Logs and results are written by a real machine and
  will record `/Users/<someone>/...`. This repo is public. Grep `/Users/`, `/home/` and
  `/var/folders/` before committing under `results/`, `iterations/` **or `docs/`** — a
  plan document sat in public history with a home directory in three shell commands,
  because the gate covered the record and not the prose beside it. Replace what you
  find with `<HOME>` / `<TMPDIR>`, keeping the line's meaning.
  `tests/test_results_are_scrubbed.py` enforces all of it (`results/*.json*`,
  `results/*.log`, `docs/**/*.md`); the grep is for catching it before the suite does.
- **No private names.** Sibling consumers and internal projects get described by role.
  If you cannot say it on a stranger's screen, it does not go in.
- **Tests green** — all of them, not "all but the known ones". `uv run pytest`.
- **Lint clean** — `uv run ruff check .` and `uv run ruff format --check .`.
- **Every claim self-contained.** Do not cite a document that does not live in this
  repo; a dangling citation is worse than none.

## Gotchas

- **Changing `runner/` invalidates every baseline.** The verifier's version is a content
  hash of the runner package, results are stamped with it, and `delta` refuses to
  compare across versions. Touch a verifier, re-record the baseline — a verifier fix
  shifts pass rates silently otherwise.
- **A Δ is only meaningful like-for-like.** `delta` refuses filtered runs, mismatched
  per-task attempt counts, and mismatched models. Do not add an override; the refusal
  is the feature.
- **Averages, never majority.** A task that passes some attempts and fails others is a
  fraction, not a pass. Held-out carries the generalization claim, so it gets more
  samples than held-in.
- **Verifiers are mechanical** — string, hash, and exit-code checks, never model
  judgment. Oracles are hash-pinned at import; a deleted or altered oracle is a spoofed
  task, not a passing one.
- **Tests read carbon's live config.** Fixtures copy the real file, so a change in
  carbon can turn this suite red on its own. Derive expectations from disk, never
  hardcode a value.
- **No `.env` here, by design.** refinery reads carbon's, so the suite and the harness
  under test cannot disagree about which model ran.
- **Siblings on disk.** refinery and carbon sit under one root; the dependency path and
  the checkout constant both assume it. Change them together.
- **Offline tests stay offline.** `uv run pytest` makes no model calls. Keep it that way.
- **The sibling carbon must be the pinned base.** `carbon-base.json` names it;
  `loop/compat.py` enforces it loudly when the `loop` package is imported
  (pytest exits early with remediation). `runner/` is not guarded — a wrong
  checkout there still fails at import, just without remediation. A fresh
  clone of both `main`s is not an operable pair until the promotion lands.

## The acceptance rule

    Δ_in ≥ 0  and  Δ_ho ≥ 0  and  max(Δ_in, Δ_ho) > 0

No regression on either split, improvement on at least one. Rejected candidates stay
committed alongside accepted ones — dropping them would make the loop look better than
it is.

## What the loop does and does not do

Mining and proposal are reasoning, done by a proposer model and written to fixed JSON
artifacts under `iterations/`. Only validation and the branch+PR step are code, so the
pipeline consumes a stable contract and never cares who produced a candidate.

A candidate is applied to carbon's **working tree**, never committed — a rejected
candidate leaves no trace there. The suite runs in a fresh subprocess (config binds at
import), the edit is reverted, and the rule decides.

Accepted edits each get a branch off `self-improvement` in carbon and a PR targeting
it, evidence in the body. **The pipeline never merges.** A human decides.

## Where to look

`runner/` measures; `loop/` decides and ships; `iterations/` and `results/` are the
committed record. Counts and specifics live in the code, deliberately not here.
