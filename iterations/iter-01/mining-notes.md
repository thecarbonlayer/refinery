# Iteration 1 — mining notes

Proposer: Fable (claude-fable-5), in-session. Mining is
reasoning, not code; this file and `clusters.json` are its fixed output artifact.

Input: `results/baseline-main.json` / `.jsonl` (carbon `edac4be`, config v1,
`google/gemma-4-26b-a4b`, runner `f02e31e8…`). Held-in rate 0.8333, held-out 0.8.

## Discipline

Mining reads **held-in traces only**. A4 (held-out) also fails 0/5 at baseline and
shares A2's mechanism by suite design, but its traces and fixture details were not
used to cluster or to pick candidate values — the held-out split only ever votes
through Δ_ho at validation time. (The aggregate baseline table is unavoidably
visible; the discipline is about not tuning candidates on held-out specifics.)

## Held-in failures observed

| Task | pass | Outcomes | Note |
|---|---|---|---|
| A2 | 0/3 | fail, fail, fail | deterministic — the suite's predicted guaranteed miner |
| B2 | 2/3 | pass, pass, not_attempted | one flake, distinct outcome class |

Everything else held-in is 3/3 (A1, B1, C1, C2, D1, D2) — regression guards, not
mining signal.

## Cluster CL-1 — clamp-suffix-drop (recurring mechanism)

All three A2 attempts fail identically. Gemma's replies, verbatim from the JSONL:

- "The provided log snippet does not contain the final line with a rollout
  password. The log ends abruptly with a truncated sequence of heartbeat messages."
- "The provided log does not contain a rollout password. The last visible line is
  a truncated timestamp."
- "…I cannot determine the password from the information provided."

Root-cause hypothesis: the door clamp (`limits.py` `clamp()`, driven by
`max_item_chars = 4000`) truncates each delivered item by keeping the prefix and
dropping the tail. A2's `fetch_log` result is longer than the clamp with the answer
on its final line, so the answer is deterministically dropped before the model ever
sees it. **The model is behaving correctly given its window** — it detects the
truncation and honestly declines to guess (`tool_called=True` in every attempt).
This is a harness failure, not a model failure, which is exactly the project's
thesis; no prompt-wording edit can recover data that never enters the window, so
the editable knob that matters is `max_item_chars` itself.

Classification: recurring mechanism (3/3, identical signature) → **feeds proposals**.

## B2 flake — isolated mistake, no proposal

Attempt 2 of B2: `not_attempted` ("fix_me.py unchanged from seed") — the agent
didn't edit the file, so the test gate never armed. 1-of-3 with no repeated
signature is the Self-Harness "isolated mistake" case, not a recurring mechanism;
at 3 attempts a single flip is indistinguishable from sampling noise. Recorded
here for honesty; deliberately NOT clustered and NOT proposed against. If it
recurs in a later iteration's baseline, it becomes mining signal then.
