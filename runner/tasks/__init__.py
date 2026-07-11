"""TASKS — the canonical registry, built cluster by cluster (A/B/C land in
later tasks; each cluster module appends its SPECS here)."""

from __future__ import annotations

from runner.spec import TaskSpec
from runner.tasks import cluster_a, cluster_b, cluster_d

TASKS: list[TaskSpec] = [
    *cluster_a.SPECS,
    *cluster_b.SPECS,
    *cluster_d.SPECS,
]
