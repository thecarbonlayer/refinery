# Compaction calibration: where this stands

Short version: compaction is **not calibrated**. The committed artifact,
`model-r2.json`, records `fit: false` because the GOODNESS and STABILITY checks
refused, so the loader will not install it and the section falls back to the causal
verdict. That is the fail-closed design working, not a defect. Nothing here gates a
candidate today.

The rest of this file is the history, because each round was withdrawn for a different
reason and the reasons are the useful part.

## Round 1: measured, then withdrawn

`analysis-r1-unfit.json` holds the first measured calibration. Four 10-attempt null
arms, a held-in bound of 0.1, a held-out bound of 0.3. It is NOT installed, and since
round 2 there is no longer any code that could read its shape: the loader reads a null
MODEL and computes bounds at judgment time, never a stored threshold. An independent
audit showed round 1's thresholds are unfit for the judgments they would gate:

- The held-in bound (0.1) is below the 3-attempt grain of a real validation run
  (1/9 = 0.111). A single attempt flipping on a single task clears it.
- An end-to-end false ACCEPT was reproduced using two of these very null arms as the
  confirmation pair.
- The confirmation gate applied a 3-task-mean bound to single-task deltas
  (~54-58% null pass rate for A1/G5 carriers).
- Max-pairwise over four arms carries no coverage guarantee. The committed bound sat at
  the ~31st percentile of its own sampling distribution.

The null-arm DATA (`results/null-cmp-a..d`) remains valid measurement.
`tests/test_acceptance.py` replays the false ACCEPT against those exact committed
files, now as a REJECT.

## Round 2: eleven arms, four tasks, superseded

Round 2 answered each finding structurally rather than by picking better numbers. The
artifact stores pooled per-task null RATES, and every bound is computed at judgment
time from those rates and the two runs' own attempt counts, at a stated 97.5% coverage.
Per-carrier and per-guard bounds replaced the split-mean bound at the ACCEPT gate,
which also requires non-negative supported-set means. The artifact records its own
fitness checks, and the loader re-derives them from the artifact's own per-arm counts
before honoring the verdict, so editing a rate and leaving the verdict alone does not
install anything.

That model pooled **eleven arms over four tasks** (A1, G2, G4, G5). It is superseded.
Phase 2c added three scenario guards (CMP-5, CMP-6, CMP-7) and the loader now pins the
covered set to seven tasks, so a four-task model does not install regardless of its
fitness. The eleven round-2 arms were also recorded before the guards existed, so no
seven-task model could judge them either. Re-recording was the only route forward, and
that is what the Phase 2c campaign did.

The round-2 arms stay committed. `tests/test_round2_attack.py` still holds the sweep
built on them, suspended, with the reason it is suspended asserted rather than
described.

## Phase 2c: ten arms, seven tasks, `fit: false`

`model-r2.json` is now the Phase 2c pooling:

- **Ten arms**: `p2c-null-full-a/b/c` (full suite, standard attempt counts) and
  `p2c-null-cmp-a` through `p2c-null-cmp-g` (`--only A1 G2 G4 G5 CMP-5 CMP-6 CMP-7
  --attempts 10`), all at one runner hash, recorded in `computed_at_runner_sha`.
  That hash is no longer the branch's: this close-out added an `attempted` metric to
  CMP-5 and CMP-6, and any edit under `runner/` advances the content hash by design.
  Nothing in flight is affected because the campaign is complete, but a round-3
  measurement re-records at the new hash regardless.
- **Seven rated tasks**: the four-task gain set plus the three scenario guards. The
  gain judgment still averages over the four supported tasks; the guards need rates
  because the confirmation adjudicates them per task.
- **`fitness.fit: false`.** GRAIN passes. GOODNESS does not. STABILITY does not.

### Fitness now certifies what the model rates

Through most of Phase 2c the fitness checks ran over the four supported tasks alone,
so the three scenario guards were RATED and never CHECKED — each one gated on a pooled
rate nothing had questioned. At the phase's close the per-task halves of all three
checks were extended: goodness and leave-one-out stability over all seven rated tasks,
and a per-task grain row for every guard, comparing its own bound at the
confirmation's ten attempts against the single-attempt grain of 1/10.

The guards do not survive it, and that is the finding rather than a setback:

- **GOODNESS.** `p2c-null-full-a` passed CMP-5 on all 3 of its attempts, against a
  pooled rate of 17/79. The exact two-sided binomial tail is **0.00996**, just inside
  the 0.01 alpha. One arm disagrees with a single-rate model for that guard.
- **STABILITY, per task.** CMP-5, CMP-6 and G4 each have a per-task bound that crosses
  a grain bucket when one arm is dropped — CMP-5 under four different arms, CMP-6
  under three, G4 under `p2c-null-cmp-d`.
- **GRAIN passes**, including every guard's own row: each of the six confirmation
  guards has a bound of 3/10 or 2/5 at ten attempts, comfortably above 1/10.

The held-in gain refusal is unchanged and still worth stating in numbers. The held-in
split's pooled quantile over all ten arms is **4/9**. Drop one arm, `p2c-null-cmp-d`,
and it becomes **1/3**: one arm moves the bound a whole grain bucket. A threshold that
depends on which arms happen to be in the pool is not a threshold. Held-out (G2 alone)
is stable at 3/5 under every leave-one-out.

Because `fit` is false the loader refuses, loudly, naming both checks. Two consequences
worth knowing:

- The section is uncalibrated, so `evaluate()` uses the causal verdict. Nothing is
  silently gated on an unstable bound.
- The artifact has **no designated baseline arm** in its pooling (`r2-null-full-a`
  belongs to the superseded round-2 protocol), so every number that would be
  conditional on that arm is published as `null` with a `baseline_note`, in both the
  `false_confirm` block and the `power.end_to_end` rows. It is absent, not zero. An
  earlier writer folded the empty mass into `0/1` and `0.0`, which reads as "this
  pipeline ships a real improvement with probability exactly zero" sitting beside a
  real marginal number.

The marginal numbers are all still published and are all independently re-derived in
`tests/test_p2b_closing.py` and `tests/test_p2b_final.py`.

## What a round-3 measurement needs

Three things, and the second is the one that is easy to get wrong.

**1. Bounds at the attempt counts judgments actually use.** The stability check pools
arms recorded at 3 and 5 attempts with arms recorded at 10, then publishes one
quantile. A bound is only a bound for the comparison it was computed at.

Round 3 also has to answer the guards' own two refusals, which are new. CMP-5's
goodness failure sits on a single 3-attempt arm and could be either a real outlier or
the 3-attempt grain again; more held-in attempts is the same remedy. The per-task
stability crossings on CMP-5, CMP-6 and G4 are the same shape as the held-in split's:
low pooled rates on a coarse grid, where dropping one arm moves the bound a whole
bucket.

**2. More held-in ATTEMPTS, not more arms.** This is the auditor's finding and it is
the whole reason round 3 is not just "run more arms". Held-in tasks run at **n=3**, so
the finest movement observable on a single task is **1/9** and every quantile lands on
a multiple of it. Adding arms shrinks the sampling noise around a quantile that can
still only take a handful of values, so the leave-one-out margins tighten and STABILITY
starts passing. That buys a PASS. It does not buy resolution: the grain is unchanged,
so the calibration still cannot tell a real 1/9 movement from the discretization.
Raising the **held-in attempts** is what buys resolution. Nothing else does.

**3. Re-measurement on a pinned provider and quantization.** The ten arms ran against
local LM Studio, and six attempts died on `HTTP 400` from that endpoint (see
`CORRECTION.md`). The planned round-3 measurement moves to OpenRouter with the provider
AND the quantization pinned. Routing across providers with mixed quantization puts a
serving confound inside the experiment: two arms that differ only in which backend
served them are not a null pair, and nothing in the record would say so.

Until a FIT artifact exists at the arms' own runner hash or a successor, compaction
stays uncalibrated and the suspended sweeps stay suspended.
