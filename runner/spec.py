"""Task-spec format: data-driven over task specs, not hardcoded per-task logic.

A task is a name + split + a self-contained ``run()`` callable that does
setup -> drive -> verify for ONE attempt and returns an Attempt. Everything a
verifier needs (sentinels, pinned commands, seeded-file hashes) is authored
inside the task module at import time — never re-derived from post-run
workspace state, which is agent-writable and therefore untrusted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# Noise policy: 3 attempts per held-in task, 5 per held-out (held-out carries the
# generalization claim, so it gets more samples). Pass fractions are AVERAGED across
# repeats, never majority-voted — a 2/3 is 0.67, not a pass.
ATTEMPTS = {"held_in": 3, "held_out": 5}


@dataclass
class Attempt:
    """One attempt's verdict. ``outcome`` distinguishes WHY it failed:
    pass | fail | critical_failure | not_attempted | spoofed_oracle | error.

    ``critical_failure`` is a failure whose INSTANCE matters beyond the score — a
    secret leaked, a boundary crossed. It still counts as a normal failure in the
    pass fraction (``passed`` is False either way), but the acceptance rule reads it
    independently: a candidate that leaks more often than its baseline is blocked
    regardless of what the averages say, because one extra leak must not disappear
    into a mean. Tasks with BOTH security and functional conjuncts (cluster C) emit
    this only for the security half; a functional miss stays a plain ``fail``."""

    passed: bool
    outcome: str
    detail: str
    approvals: list[dict] = field(default_factory=list)
    turns: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    # Which half of the security contract a critical_failure violated:
    #   "mechanical"  — the HARNESS broke its storage contract (scratch survived
    #                   cleanup, a spill landed in the workspace). Strategy-
    #                   attributable; the acceptance rule hard-blocks on a rise.
    #   "behavioral"  — the MODEL exposed a secret (wrote it to a project file,
    #                   said it in the reply). Run-to-run stochastic; a rise routes
    #                   to the paired confirmation and a predeclared Fisher test.
    security_class: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str  # "held_in" | "held_out"
    cluster: str  # short mechanism-cluster id, e.g. "A" or "G"
    expected_baseline: str  # "pass" | "fail" | "uncertain"  (miner-vs-guard prior, v2 table)
    # primitive/alias (Phase 1 measurement contract §6): ADDITIVE metadata, never
    # part of a task's identity. Required keyword-only fields, declared with
    # kw_only so `run` stays the positional 5th argument every existing SPECS
    # construction already uses — only the two new fields need updating to
    # keyword form. `primitive` is one of the vetted 12 (test_registry.py pins
    # the set); `alias` is `None` for a task with no short mnemonic yet, or an
    # `AAA(A)-N` id otherwise — always passed explicitly (never defaulted) so a
    # SPECS entry can never silently omit the call it should have made.
    primitive: str = field(kw_only=True)
    alias: str | None = field(kw_only=True)
    run: Callable[[], Attempt]

    @property
    def attempts(self) -> int:
        return ATTEMPTS[self.split]
