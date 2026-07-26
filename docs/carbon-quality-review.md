# Carbon quality and evolution contract

Updated July 26, 2026 for Carbon config v4 and Refinery's 23-task quality suite.

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

| Field | Current choice | Tasks that can see it |
|---|---|---|
| `file_injection` | `head_tail`, budget 4,000 | A4 |
| `tool_output` | `head_tail`, budget 4,000 | A2, E2 |
| `compaction` | `structured_checkpoint`, trigger 0.8 | A1, A3, G2 |
| `max_tokens` | 4,096 | G1 |
| `max_tool_steps` | 20 | F2 |
| `default_context_limit` | 4,000 | A1, A3, G2 |
| `verify_attempts` | 3 | B1, B2, B3 |
| `retry` | `backoff`, at most 3 attempts | H1, H2, H3 |
| `system_prompt`, `temperature` | current defaults | full suite plus B/D guards |

The full mapping is executable policy in `runner/knob_coverage.py`. Offline
tests fail if Carbon exposes another field without a miner and guard.

`max_item_chars` remains in Carbon for chapter/API compatibility but is no
longer editable; the two strategy-specific budgets replace it. Likewise,
`memory_search_limit` stays locked until memory-specific external tasks exist.

## Quality changes relative to the original Pi comparison

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

Layered nested instructions, richer file search tools, and Pi-style full-output
offload remain future Carbon work. They should land one seam at a time with a
Refinery cluster, not as an unmeasured bundle.

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
uv run python -m runner.cli run --label baseline-strategy-v4
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
  --baseline results/baseline-strategy-v4.json
uv run python -m loop.cli pr \
  --iteration <iteration> --candidate <candidate> \
  --baseline results/baseline-strategy-v4.json
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
