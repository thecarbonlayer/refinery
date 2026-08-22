"""TASKS — the canonical core + harness-quality registry, in cluster order."""

from __future__ import annotations

from runner.spec import TaskSpec
from runner.tasks import (
    cluster_a,
    cluster_b,
    cluster_c,
    cluster_ctx,
    cluster_d,
    cluster_e,
    cluster_f,
    cluster_g,
    cluster_h,
    cluster_i,
    cluster_v,
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
    # Last, not alphabetical: the Phase 4 context-delivery candidates joined after
    # the eight lettered clusters, and a stable suite order keeps recorded logs
    # comparable across the boundary.
    *cluster_ctx.SPECS,
    *cluster_i.SPECS,
    # cluster V is the verification/loop-discipline CANDIDATE suite: runnable,
    # but outside every calibrated gate and campaign set until its human gate
    # (tests/test_cluster_v.py pins the isolation).
    *cluster_v.SPECS,
]
