"""TASKS — the canonical core + harness-quality registry, in cluster order."""

from __future__ import annotations

from runner.spec import TaskSpec
from runner.tasks import (
    cluster_a,
    cluster_b,
    cluster_c,
    cluster_d,
    cluster_e,
    cluster_f,
    cluster_g,
    cluster_h,
)

TASKS: list[TaskSpec] = [
    *cluster_a.SPECS,
    *cluster_b.SPECS,
    *cluster_c.SPECS,
    *cluster_d.SPECS,
    *cluster_e.SPECS,
    *cluster_f.SPECS,
    *cluster_g.SPECS,
    *cluster_h.SPECS,
]
