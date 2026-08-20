# Compaction calibration — round 1: measured, then withdrawn

`analysis-r1-unfit.json` holds the first measured calibration (four
10-attempt null arms; held-in bound 0.1, held-out 0.3). It is NOT installed,
and since round 2 there is no longer any code that could read its shape:
the loader reads `model-r2.json` and a null MODEL, never a stored threshold.
An independent audit showed round 1's thresholds are unfit for the judgments
they would gate:

- The held-in bound (0.1) is below the 3-attempt grain of a real validation
  run (1/9 = 0.111): a single attempt flipping on a single task clears it.
- An end-to-end false ACCEPT was reproduced using two of these very null
  arms as the confirmation pair.
- The confirmation gate applies a 3-task-mean bound to single-task deltas
  (~54-58% null pass rate for A1/G5 carriers).
- Max-pairwise over four arms carries no coverage guarantee; the committed
  bound sits at the ~31st percentile of its own sampling distribution.

The null-arm DATA (results/null-cmp-a..d) remains valid measurement, and
`tests/test_acceptance.py` replays the false ACCEPT against those exact
committed files, now as a REJECT.

Round 2 (contracts/phase2b-calibration-contract.md) answers each finding
structurally rather than by picking better numbers: `model-r2.json` stores
pooled per-task null RATES, and every bound is computed at judgment time from
those rates and the two runs' own attempt counts, at a stated 97.5% coverage.
Per-carrier and per-guard bounds replace the split-mean bound at the ACCEPT
gate, which also now requires non-negative supported-set means. The artifact
records its own fitness checks and the loader refuses it unless they all
passed.

`model-r2.json` has not been written yet — the round-2 arms are still being
recorded. Until it exists at the load path, `section_calibration` returns None
and compaction stays on the causal verdict, which is the safe state.
