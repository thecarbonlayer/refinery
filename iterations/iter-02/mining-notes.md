# Iteration 2 — mining notes

Mined from `results/baseline-main-r4.jsonl` (carbon config v6, model
`google/gemma-4-26b-a4b`, 26 tasks, 95 recorded attempts). Held-in traces only;
the held-out pile stays unread per the mining rule. Baseline summary:
held-in 0.7843, held-out 0.9111.

## Held-in failures, triaged

| Task | Fraction | Disposition |
|------|----------|-------------|
| G4   | 0/3      | Mined — CL-2 |
| G5   | 0/3      | Mined — CL-2 (attempt 1 is a provider-door error, cited as mechanism evidence, not counted as model failure) |
| E3   | 0/3      | Excluded — documented capability gap (`expected_baseline: fail`; knob-coverage CAPABILITY_GAPS). No positional truncation strategy can reach a mid-result fact; this stays a written request for a Carbon strategy, never a setting change that approximates one. |
| G1   | 1/3      | Excluded — model limitation, measured by the parity run (an unrelated harness fails it identically at the same fraction). Not to be optimized against. |

Held-out C3 came in at 4/5. Single attempt, no shared mechanism with anything
held-in, and held-out is not mined regardless. Noted for the noise ledger; watch
at the next baseline.

## The cluster

CL-2-compaction-state-erosion (G4, G5): one mechanism wearing two costumes.
G4 states its facts in conversation and loses the oldest of them to repeated
re-summarization — the prior checkpoint is handed back to the summarizer as
compressible transcript every round, and a model shown "carry this forward" and
"compress this" as the same text compresses. G5 never states its facts at all;
it acts, and the only record of its two `write_file` calls is whatever prose the
summarizer kept, which by the final compaction is the most recent write alone.

G5 attempt 1 adds a third face of the same mechanism: the summarize call itself
was rejected by the provider (HTTP 400 at the local endpoint, mid-task, inside
compaction). The incumbent strategy ships the serialized middle to the
summarizer with no token budget, so a long enough session can overflow the
provider window at the compaction door. The attempt is recorded as `error`, not
as evidence about the model.

## Why the strategy is the knob

The failing state never survives because prose is the only carrier. The
`compaction_prompt` already says "Preserve, verbatim, every concrete fact,
code, name, decision, file path" — and the file paths are gone anyway, in
every failing attempt. An instruction cannot make a summarizer stop compressing
its input; only the shape of what it is shown can. Carbon's menu already holds
a vetted strategy built for exactly this (`token_budget_checkpoint`: incremental
checkpoints, deterministic tool-derived file state, bounded summarize payload),
so the fix is expressible as configuration.

Three candidates, one experiment: the minimal strategy switch, the switch with
its token budget active, and a number-turn control (`keep_tail` 4 → 8) inside
the incumbent strategy. The control exists because of iteration 1: last time,
the loop's only vocabulary was a bigger number, and the number won by moving a
cliff. This time the vocabulary contains both a bigger number and a different
shape, and the same acceptance rule gets to say whether the shape earns its
place.
