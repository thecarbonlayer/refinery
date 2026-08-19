# Compaction calibration — round 1: measured, then withdrawn

`analysis-r1-unfit.json` holds the first measured calibration (four
10-attempt null arms; held-in bound 0.1, held-out 0.3). It is NOT installed
at the load path (`analysis.json`) because an independent audit showed the
thresholds are unfit for the judgments they would gate:

- The held-in bound (0.1) is below the 3-attempt grain of a real validation
  run (1/9 = 0.111): a single attempt flipping on a single task clears it.
- An end-to-end false ACCEPT was reproduced using two of these very null
  arms as the confirmation pair.
- The confirmation gate applies a 3-task-mean bound to single-task deltas
  (~54-58% null pass rate for A1/G5 carriers).
- Max-pairwise over four arms carries no coverage guarantee; the committed
  bound sits at the ~31st percentile of its own sampling distribution.

The null-arm DATA (results/null-cmp-a..d) remains valid measurement. What
must change before a round-2 artifact is installed: bounds measured at the
attempt counts the judgment uses (3/5 for validation, confirm-shaped for
confirmation); per-carrier bounds, not only split means; a positivity
requirement and guard adjudication at the ACCEPT gate; artifact value
integrity (re-derive section_noise from per_task); an estimator with stated
coverage. Redesign is queued as a human gate in the program ledger.

With no artifact at the load path, `section_calibration` returns None and
compaction stays on the causal verdict — the safe state.
