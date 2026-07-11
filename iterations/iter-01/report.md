# Iteration 1 — report

Run 2026-07-11. Proposer: Fable (claude-fable-5), in-session. Target: dist/gemma
`edac4be` (config v1), `google/gemma-4-26b-a4b` via LM Studio.
**Deliverable: PR #1 — https://github.com/thecarbonlayer/gemma/pull/1**
(`evolve/iter-01-clamp-12k` -> `self-improvement`, not merged by the pipeline).

## What happened, in order

1. **Mining** (held-in traces only): one recurring cluster,
   `CL-1-clamp-suffix-drop` — A2 fails 3/3 with Gemma correctly *detecting* the
   truncation and declining to guess. Harness failure, not model failure. B2's
   flake triaged as an isolated mistake, not proposed against. See
   `mining-notes.md`, `clusters.json`.
2. **Proposal**: K=3 blunt knob turns (`candidates.json`) — `clamp-8k`,
   `clamp-12k`, `clamp-12k-window-8k`. All numeric: no instruction edit can
   recover data that never enters the window.
3. **The dry-run caught a verifier bug.** Under `clamp-12k`, A2 errored at 0.0s:
   the A2/A4 fixture asserts compared fixture size against the LIVE
   `max_item_chars` — verifier behavior depending on the editable surface,
   which hard constraint 2 of task-suite-v2.md forbids. Every clamp-raising
   candidate (the suite's own predicted "dumb accepted fix") was structurally
   unpassable. Fixed by pinning `AUTHORED_CLAMP = 4000` (the authoring-time
   config-v1 value) in `runner/tasks/cluster_a.py`; verifier predicates
   unchanged. This changed `runner_sha`, so per the runner's own parity gate
   the baseline was **re-recorded** before any Δ was computed.
4. **Baseline r2** (`results/baseline-main-r2.json`, runner `1e472769…`):
   held-in 0.7917, held-out 0.8. A2 0/3 and A4 0/5 as before; B2 came in 1/3
   this time (vs 2/3 in the first baseline) — that task is noisy at 3 attempts.
5. **Validation** (full 49-attempt suite per candidate, applied as a
   working-tree edit, reverted after; rejected candidates never touch git):

| Candidate | Δ_in | Δ_ho | Rule | Per-task movement |
|---|---|---|---|---|
| clamp-8k | +0.0000 | +0.2000 | accepted | A4 0→1; A2 still 0 (raise too small) |
| **clamp-12k** | **+0.1667** | **+0.2000** | **accepted — selected** | A2 0→1, A4 0→1, B2 +0.33 (noise) |
| clamp-12k-window-8k | +0.0000 | +0.2000 | accepted | A2 0→1, A4 0→1, **A1 1→0 regression** |

## Selection

All three candidates alternate values of the same knob against the same
cluster — mutually exclusive, so one PR (open-questions.md §5: multiple PRs in
one iteration only for *distinct* clusters). `clamp-12k` strictly dominates:
both mechanism surfaces fixed, no per-task regression, highest Δ_in.

Honesty notes on the numbers:

- `clamp-12k`'s Δ_in decomposes as +0.125 from A2 (the real, deterministic fix)
  plus +0.0417 of B2 flake noise (1/3 → 2/3). It is accepted even with B2 held
  flat.
- `clamp-12k-window-8k` is the acceptance rule's known blind spot on display:
  A1 regressed 1.0 → 0.0 (doubling `default_context_limit` changes compaction
  timing, and A1's compaction-era recall collapsed), but A2's +1.0 cancels it
  inside the aggregate Δ_in, so the rule technically accepts. Recorded as
  rule-accepted / not-selected. Worth a beat on camera: aggregate split deltas
  can hide an intra-split swap; the per-task table is what catches it.
- The predicted "capability-prior mismatch" (proposer edits too clever for the
  target) did not bite in this iteration — but only because the winning edit is
  a number, not prose. The two non-selected candidates still show the proposer
  guessing wrong about magnitude (8k) and second-order effects (window-8k).

## Suite after the accepted edit

held-in 0.9583 (A2 fixed; B2 flake remains the only non-1.0), held-out 1.0.
The next iteration's mining pool is nearly empty: B2's flake is the only
held-in signal left, and it is noise-grade at 3 attempts. If iteration 2 is
wanted, either raise B2's attempt count to firm the signal or accept that this
suite is close to saturated (task-suite-v2.md anticipated a v3 with more
held-out tasks per mechanism).

## Process notes

- One iteration only — the sequential-acceptance/held-out-erosion caveat
  (open-questions.md §9 (d)-3) does not yet apply, but note `clamp-12k` was
  selected partly on Δ_ho; A4 is now effectively "spent" as a held-out check
  for the clamp mechanism.
- Provenance is a candidate field, rendered "Fable-proposed,
  task-suite-validated" in the commit and PR body — not a hardcoded name.
- One-time setup done by the pipeline: `self-improvement` branch created at
  main tip `edac4be` and pushed. Still the owner's to do on GitHub: branch
  protection (require ≥1 approving review) on `self-improvement`.
- All raw measurements are committed: `results/baseline-main-r2.json(l)`,
  `results/cand-*.json(l)`, `results/dryrun-clamp-12k.json(l)` (the run that
  exposed the verifier bug), and per-candidate validation records here.
