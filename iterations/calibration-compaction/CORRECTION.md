# Correction: the Phase 2c campaign's six error attempts

**The recorded rates stand. The diagnosis written beside them was wrong.**

## What was claimed

Commit `8de800c` ("p2c-null-cmp-c: third 10-try run") recorded that run's two CMP-7
errors as **setup errors**, with the note "premise mostly but not always fires". That
describes a task that ran, tried to build its premise, and could not: for CMP-7 the
premise is that compaction fires often enough to bury the fact, and a premise that only
sometimes fires is a fact about the task's design.

## What the record shows

Six attempts across the ten committed arms carry the `error` outcome:

| arm | task | attempt |
| --- | --- | --- |
| `p2c-null-cmp-c` | CMP-7 | 3 |
| `p2c-null-cmp-c` | CMP-7 | 7 |
| `p2c-null-cmp-e` | CMP-7 | 1 |
| `p2c-null-cmp-e` | CMP-7 | 9 |
| `p2c-null-cmp-g` | CMP-7 | 8 |
| `p2c-null-cmp-g` | G5 | 2 |

Every one of the six is the same thing:

```
httpx.HTTPStatusError: Client error '400 Bad Request' for url
'http://localhost:1234/v1/chat/completions'
```

A serving fault from the local LM Studio endpoint, surfacing as an uncaught exception
that `runner/run.py` records as an `error` attempt with the traceback in the detail.
None of the six is a premise failure. A premise failure exits through the task's own
guard (`Attempt(False, "error", "repeated-compaction setup did not fire twice ...")`)
with a written reason and no traceback, and no attempt in this campaign did that.

Three of the six died inside compaction's own summarizer call
(`harness/compaction.py` `_summarize`), which reaches the model directly rather than
through the retry wrapper the agent's ordinary calls use. A single 400 there kills the
attempt with no retry. The other three came through `model_call_with_recovery` and
exhausted it.

## The denominator effect

The runner's visible-error policy is that a recorded error is a recorded non-pass:
`TaskResult.pass_fraction` (`runner/run.py`) divides passes by EVERY recorded attempt.
So the faults are in the published pooled rates, and both numbers matter:

| task | published (all attempts) | excluding the serving faults |
| --- | --- | --- |
| CMP-7 | 67/79 = 84.8% | **67/74 = 90.5%** |
| G5 | 49/79 = 62.0% | **49/78 = 62.8%** |

The published figures are the ones `model-r2.json` carries and the ones every bound in
it was computed from. They are not being restated here. The point is that roughly six
points of CMP-7's published rate is endpoint availability rather than anything about
compaction, and a reader comparing CMP-7 across future arms served by a different
backend would be reading that difference as behavior.

## What this does and does not change

- It does not change any verdict. `fitness.fit` is false because STABILITY refused on
  the held-in quantile, which is unrelated to these six attempts.
- It does not change the artifact. The six rows keep their `error` outcome and their
  tracebacks, and the pooled rates stand unedited, exactly as iteration 4's correction
  left that record.
- It does change what the errors are evidence FOR. "The premise does not always fire"
  would be a finding about CMP-7's design and would argue for redesigning the task.
  "The endpoint returned 400" is a finding about the serving setup and argues for
  pinning it, which is what the round-3 plan in `README.md` does: OpenRouter with the
  provider and the quantization both pinned.

## Bottom line

The campaign is complete and its arms are sound. Six of 835 recorded attempts were lost
to a local serving fault, three of them inside an unretried compaction call, and the
commit message that first described two of them named the wrong cause. The record stands;
this file corrects the sentence beside it.
