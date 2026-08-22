# Carbon quality and evolution contract

Written July 27, 2026 for Carbon config v3 and the quality suite as it stood then;
the stale claims were corrected against config v8 and the current registry on
August 22, 2026. This document explains the CONTRACT — what Carbon fixes, what
Refinery may evolve, and how the two tables are governed. It does not enumerate
the suite: `runner/tasks/` is the registry, `loop surface` is the live editable
menu, and both have grown since the review was written. Anything here that names
a count or a value is a summary; the code is the authority.

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
| `file_injection` | bounded truncation policy: strategy, budget, tail fraction |
| `tool_output` | the same, plus an `offload_to_file` strategy that hands back a retrievable path |
| `compaction` | bounded strategy plus keep-head, keep-tail, trigger fraction, summary budget, token reserves, checkpoint fallback, prompt suffix |
| `tool_exposure` | which tools the model is offered: `all`, an allowlist, or query-match top-k |
| `compaction_prompt`, `system_prompt` | free text |
| `retry` | bounded strategy plus max attempts, base delay |
| `max_tokens`, `max_tool_steps`, `default_context_limit`, `verify_attempts` | positive int |
| `temperature` | float |

The strategy menus and parameter bounds move as Carbon ships seams. Run
`loop surface` for the live version rather than trusting this table's shape.

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

Two exemption tables are recorded explicitly rather than papered over, each entry
carrying a rationale the tests require. `GUARD_ONLY_KNOBS` holds `max_tokens`,
`max_tool_steps`, `retry` and `default_context_limit`: no observer fails *because
of* them today, so the loop may defend them but has nothing to mine.
(`tool_output` was on that list and left it — it has miners now.)
`UNGUARDED_KNOBS` — knobs whose only observers are their own miners, so nothing
can guard them — is **EMPTY** as of 2026-08-15. `file_injection` was its sole
entry, because A4 was the suite's only `@path` sender and A4 is the miner; A5 is
the second `@path` task with a passing prior that closed it. An entry leaves that
table only when someone writes the task.

A third table, `CAPABILITY_GAPS`, records tasks that fail where *no* value of any
setting can help, because the strategy they need is not on the menu. It is
**EMPTY** today. E3 held the slot — it needed a `tool_output` strategy that
preserves the middle of a large result and hands back a retrievable path — until
Carbon shipped `offload_to_file`; E3 then moved to that knob's miners, with E4
alongside it to prove path-recoverability specifically. A gap is a written
request for a person, and it leaves when the person ships the strategy, never
when the task is softened to stop asking.

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

Side-channel full-output offload has since landed as the `offload_to_file`
strategy on `tool_output`, with E3/E4 measuring it. Layered nested instructions
and richer file search tools remain future Carbon work. They should land one seam
at a time with a Refinery cluster, not as an unmeasured bundle.

## What candidate comparison records

The rule quoted here when the review was written —

```text
Δ_in >= 0 and Δ_ho >= 0 and max(Δ_in, Δ_ho) > 0
```

— is what `runner delta` reports, not what promotes. It was measured against six
no-change runs and wrongly accepted 6 of 12 pairs, so `loop/acceptance.py` now
decides with three outcomes (REJECT / CONFIRM / ACCEPT), where a gain earns
CONFIRM and only a fresh paired confirmation can reach ACCEPT. See the README's
"Deciding" section and that module's docstring. A full-pass to zero-pass task
collapse is a veto under either. Alongside per-task pass fractions, Refinery
reports mean:

- tokens and estimated cost
- model and tool calls
- compactions
- tool errors
- incomplete responses
- duration and transcript turns

Those metrics expose tradeoffs but never override correctness.

## Run the new loop

The runner and Carbon behavior both changed, so record a new baseline. Point
`carbon/.env` at the serving base you mean to measure on first — the program's
measurement base is OpenRouter with `LLM_PROVIDER_ORDER` and `LLM_QUANTIZATION`
pinned, and recording against an unpinned remote base is refused outright. See
the README's "Which serving base".

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
