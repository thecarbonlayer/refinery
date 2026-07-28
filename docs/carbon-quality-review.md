# Carbon quality and evolution contract

Updated July 27, 2026 for Carbon config v3 and Refinery's 23-task quality suite.

## What Carbon now fixes directly

These are correctness properties. Refinery measures them but cannot turn them
off:

- `edit_file` rejects zero or multiple matches, writes atomically, and returns
  a unified diff.
- Tool arguments are structurally validated before dispatch.
- Tool calls from `finish_reason=length` responses never execute.
- A delegated worker can be bound to the parent's workspace tools and
  instruction root.
- Workspace confinement, secret-file refusal, verification freshness, and
  fail-closed test receipts remain invariant.

Carbon publishes this list through `carbon.surface_manifest()`. Refinery's
`loop surface` command prints it for proposers.

## What Refinery can evolve

The editable vocabulary is deliberately broader than scalar limits but still
contains no executable hooks:

| Field | Shape |
|---|---|
| `file_injection`, `tool_output` | bounded truncation policy: strategy, budget, tail fraction |
| `compaction` | bounded strategy plus keep-head, keep-tail, trigger fraction, summary budget |
| `compaction_prompt`, `system_prompt` | free text |
| `retry` | bounded strategy plus max attempts, base delay |
| `max_tokens`, `max_tool_steps`, `default_context_limit`, `verify_attempts` | positive int |
| `temperature` | float |

Current values are not repeated here — read them from
`carbon/harness/harness_config.json`, or run `loop surface`. Nor is the
task-to-knob mapping: it used to live in this table as a "tasks that can see it"
column, drifted from the code on five of nine rows within a single change, and
the code is the enforced version.

`loop/knob_coverage.py` is authoritative. It declares three roles per knob —
`observers` (tasks whose verdict some legal value of the knob can actually move),
`miners` ⊆ observers (observers with failure headroom), and `guards` ⊆ observers
(observers that can still regress). Offline tests fail if Carbon exposes a field
with no coverage, if a miner or guard is not an observer, if any miner already
passes, or if a guard is pinned to a task that already fails and so could not
detect a regression. It lives in `loop/` rather than `runner/` on purpose:
`runner_sha` stamps every recorded result, so governance metadata kept there
would invalidate every baseline each time a row was corrected.

Two exemptions are recorded explicitly rather than papered over, each with a
rationale the tests require. `max_tokens`, `max_tool_steps`, `retry`,
`tool_output`, and `default_context_limit` are **guard-only**: no observer fails
*because of* them today, so the loop may defend them but has nothing to mine.
`file_injection` is **unguarded**: `deliver()` applies it only on an `@path` send,
A4 is the suite's only `@path` task, and A4 is the miner — guarding it needs a
new task, not a table edit.

A third table, `CAPABILITY_GAPS`, records tasks that fail where *no* value of any
setting can help, because the strategy they need is not on the menu. E1 is the
current entry: both truncation strategies are positional, so neither reaches a
fact in the middle of a large result, and the budget that would reach it floods
the window. That is a Carbon feature request, not a candidate.

The `observers` claims are assertions about Carbon's internals and the tests
cannot check them. An audit found six false at one point — a guard whose task is
invariant to the knob looks like coverage and is worse than none, because the PR
body tells a reviewer it was watching. Re-measure before editing a row.

`max_item_chars` remains in Carbon for chapter/API compatibility but is no
longer editable; the two strategy-specific budgets replace it. Likewise,
`memory_search_limit` stays locked until memory-specific external tasks exist.

## Quality changes relative to the reference-harness review

The first pass closes the sharpest gaps:

- 20 tool rounds instead of six.
- 4,096 completion tokens instead of 1,024.
- Ranged file reads with line counts and continuation hints.
- Tail-preserving tool and sandbox output.
- Unique atomic diff-producing edits.
- Structured, tool-aware, cumulative compaction with an earlier trigger.
- Explicit incomplete-response handling.
- Worker workspace binding.
- Forced context-overflow compaction and bounded transient-provider retries.

Layered nested instructions, richer file search tools, and side-channel
full-output offload remain future Carbon work. They should land one seam at a
time with a Refinery cluster, not as an unmeasured bundle.

## What candidate comparison records

Promotion still requires:

```text
Δ_in >= 0 and Δ_ho >= 0 and max(Δ_in, Δ_ho) > 0
```

A full-pass to zero-pass task collapse is an additional veto. Alongside
per-task pass fractions, Refinery now reports mean:

- tokens and estimated cost
- model and tool calls
- compactions
- tool errors
- incomplete responses
- duration and transcript turns

Those metrics expose tradeoffs but never override correctness.

## Run the new loop

The runner and Carbon behavior both changed, so record a new baseline with LM
Studio serving the model in `carbon/.env`:

```bash
uv sync
uv run python -m loop.cli surface
uv run python -m runner.cli run --label baseline-strategy-v3
```

Cheap mechanism probes:

```bash
uv run python -m runner.cli run --label probe-output --only A2 E2 --attempts 1
uv run python -m runner.cli run --label probe-tools --only E1 F1 F2 G3 --attempts 1
uv run python -m runner.cli run --label probe-context --only A1 A3 G1 G2 --attempts 1
```

For each iteration, mine failures, write `clusters.json` and `candidates.json`,
then:

```bash
uv run python -m loop.cli dry-run \
  --iteration <iteration> --candidate <candidate> --tasks <miner> <guard>
uv run python -m loop.cli validate \
  --iteration <iteration> --candidate <candidate> \
  --baseline results/baseline-strategy-v3.json
uv run python -m loop.cli pr \
  --iteration <iteration> --candidate <candidate> \
  --baseline results/baseline-strategy-v3.json
```

Candidates may replace a whole strategy object, for example:

```json
{
  "fields": {
    "tool_output": {
      "old": {"strategy": "head_tail", "budget": 4000, "tail_fraction": 0.6},
      "new": {"strategy": "keep_head", "budget": 8000, "tail_fraction": 0.6}
    }
  }
}
```

Carbon validates the menu and parameters before Refinery writes the candidate.
The pipeline still reverts rejected edits and never merges accepted PRs.
