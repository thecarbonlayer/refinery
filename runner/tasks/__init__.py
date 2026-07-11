"""TASKS — the canonical 13-task registry (task-suite-v2.md), cluster order A/B/C/D."""

from __future__ import annotations

from runner.spec import TaskSpec
from runner.tasks import cluster_a, cluster_b, cluster_c, cluster_d

TASKS: list[TaskSpec] = [
    *cluster_a.SPECS,
    *cluster_b.SPECS,
    *cluster_c.SPECS,
    *cluster_d.SPECS,
]
