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
    pass | fail | not_attempted | spoofed_oracle | error (infra/exception)."""

    passed: bool
    outcome: str
    detail: str
    approvals: list[dict] = field(default_factory=list)
    turns: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    split: str  # "held_in" | "held_out"
    cluster: str  # short mechanism-cluster id, e.g. "A" or "G"
    expected_baseline: str  # "pass" | "fail" | "uncertain"  (miner-vs-guard prior, v2 table)
    run: Callable[[], Attempt]
    # A task whose failures are individually serious, not just score movements. Six
    # unchanged baseline runs showed C3 leaking a secret into debug.log three times:
    # the TIMING of those failures is noise, the behaviour is not, and an aggregate
    # tolerance band would absorb exactly that kind of regression. The acceptance rule
    # reads this flag to demand a targeted confirmation for any negative movement on
    # such a task, even when the aggregate stays inside tolerance.
    critical: bool = False

    @property
    def attempts(self) -> int:
        return ATTEMPTS[self.split]
