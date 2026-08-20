"""No editable knob enters the loop without tasks capable of observing it.

Non-emptiness is not coverage, and neither is a correct prior. A guard pinned to
a task that already fails cannot detect a regression; a miner pinned to a task
that already passes has nothing to mine; and either one pinned to a task whose
mechanism never touches the knob watches nothing at all while the table looks
full. These tests check all three properties, and check that the two exemption
tables stay honest instead of becoming a place to hide.
"""

import ast
import inspect

from loop.config_edit import known_knobs
from loop.knob_coverage import (
    GUARD_ONLY_KNOBS,
    KNOB_COVERAGE,
    LIVE_ALL,
    LIVE_GUARDS,
    LIVE_MINERS,
    SUITE_WIDE_KNOBS,
    UNGUARDED_KNOBS,
)
from runner.tasks import TASKS, cluster_h

PRIORS = {task.name: task.expected_baseline for task in TASKS}
ROLES = ("observers", "miners", "guards")


def _has_headroom(name: str) -> bool:
    """Can still fail at baseline, so turning it green is a measurable gain."""
    return PRIORS[name] != "pass"


def _can_regress(name: str) -> bool:
    """Not already expected to fail — a task at 0.0 has nothing left to lose."""
    return PRIORS[name] != "fail"


# Derived from the registry, never hand-listed. Splitting the live set by prior is
# what makes the headroom property hold by CONSTRUCTION: a single wildcard over
# all 20 live tasks satisfied an existential headroom check on one lucky failing
# task while filing seven already-passing tasks as miners.
LIVE_TASKS = frozenset(task.name for task in TASKS if task.cluster != "H")
SENTINELS = {
    LIVE_ALL: LIVE_TASKS,
    LIVE_MINERS: frozenset(name for name in LIVE_TASKS if _has_headroom(name)),
    LIVE_GUARDS: frozenset(name for name in LIVE_TASKS if not _has_headroom(name)),
}


def _expand(names: tuple[str, ...]) -> set[str]:
    """Resolve sentinels to real task names. A sentinel EXPANDS, never waives."""
    resolved: set[str] = set()
    for name in names:
        resolved |= SENTINELS.get(name, frozenset({name}))
    return resolved


def test_live_sentinels_are_both_non_empty():
    """The partition and headroom properties hold BY CONSTRUCTION — the two sets are
    complementary comprehensions over the same predicate, so asserting them restates
    the definition. What is not guaranteed is that either side is non-empty: if every
    live task came to share one prior, one sentinel would expand to nothing and the
    knobs using it would silently lose all coverage while every check still passed.
    """
    assert SENTINELS[LIVE_MINERS], "no live task has failure headroom — miners vanish"
    assert SENTINELS[LIVE_GUARDS], "no live task is expected to pass — guards vanish"


def test_live_expansion_excludes_scripted_provider_clusters():
    """The live sentinels claim "every task that sends a system prompt". Cluster H
    builds its agents with no `system=` argument, so it sends none. If that ever
    changes, the exclusion is stale and this catches it at the source."""
    # AST, not a substring search: this file's own prose mentions `system=`, and a
    # text match on source cannot tell a docstring from a call site.
    passes_system = any(
        isinstance(node, ast.Call) and any(kw.arg == "system" for kw in node.keywords)
        for node in ast.walk(ast.parse(inspect.getsource(cluster_h)))
    )
    assert not passes_system, (
        "cluster_h now sets a system prompt — those tasks must be included in LIVE"
    )
    # Not `LIVE_TASKS == {the same comprehension}` — that restates this file's own
    # definition and verifies nothing. What matters is the property: H is out, and
    # the set is non-empty.
    assert LIVE_TASKS and not (LIVE_TASKS & {task.name for task in TASKS if task.cluster == "H"})


def test_every_editable_knob_is_covered():
    assert set(KNOB_COVERAGE) == set(known_knobs())
    for knob, coverage in KNOB_COVERAGE.items():
        assert set(coverage) == set(ROLES), f"{knob} must declare exactly {ROLES}"
        assert coverage["observers"], f"{knob} names no observer"
        # A guard-only knob declares `miners: ()`. Demanding a non-empty tuple made
        # those three rows name a fossil miner that the exemption then asserted had
        # no headroom — the table stating something it also denied.
        if knob not in GUARD_ONLY_KNOBS:
            assert coverage["miners"], f"{knob} names no miner and is not guard-only"


def test_coverage_names_real_tasks():
    names = {task.name for task in TASKS}
    for knob, coverage in KNOB_COVERAGE.items():
        for role in ROLES:
            unknown = sorted(_expand(coverage[role]) - names)
            assert not unknown, f"{knob}.{role} names tasks that do not exist: {unknown}"


def test_miners_and_guards_can_actually_observe_the_knob():
    """The load-bearing check. A guard that cannot see the knob is worse than no
    guard, because the evidence in the PR body claims it is watching."""
    for knob, coverage in KNOB_COVERAGE.items():
        observers = _expand(coverage["observers"])
        for role in ("miners", "guards"):
            blind = sorted(_expand(coverage[role]) - observers)
            assert not blind, f"{knob}.{role} cannot observe the knob: {blind}"


def test_every_guard_can_actually_regress():
    for knob, coverage in KNOB_COVERAGE.items():
        dead = sorted(g for g in _expand(coverage["guards"]) if not _can_regress(g))
        assert not dead, (
            f"{knob}: guards already expected to fail cannot detect a regression: {dead}"
        )


def test_every_miner_has_failure_headroom():
    """UNIVERSAL, not existential. An ``any()`` check was satisfied by a single
    failing task while the same tuple filed seven already-passing tasks as miners —
    so the wildcard bypass survived the check that was meant to close it."""
    for knob, coverage in KNOB_COVERAGE.items():
        if knob in GUARD_ONLY_KNOBS:
            continue
        idle = sorted(m for m in _expand(coverage["miners"]) if not _has_headroom(m))
        assert not idle, (
            f"{knob}: miners {idle} already pass, so there is nothing to mine — "
            f"move them to guards, or declare the knob guard-only"
        )


def test_every_knob_has_a_guard_unless_declared_unguardable():
    for knob, coverage in KNOB_COVERAGE.items():
        if knob in UNGUARDED_KNOBS:
            continue
        assert coverage["guards"], f"{knob} names no guard and is not in UNGUARDED_KNOBS"


def test_only_suite_wide_knobs_may_use_the_live_sentinels():
    """The sentinels make every enforced property true BY CONSTRUCTION, so a knob
    claiming them passes the whole contract without naming real coverage.

    Note what this does and does not do. It stops a knob grabbing the sentinels
    silently. It does NOT stop someone adding that knob to `SUITE_WIDE_KNOBS` in the
    same edit — a reviewer demonstrated exactly that with carbon's locked
    memory-recall knob. Pinning the membership by name below at least makes growing
    the set a deliberate act — an assertion spelling out both members has to be
    edited too, not just a data line.

    And it does not close the exploit CLASS. A reviewer reached the identical
    universal pass with no test edit at all, simply by writing the sentinel's
    expansion out by hand: `observers` = all 20 live task names, `miners` = the 13
    with headroom, `guards` = the 7 without. Nothing here can tell a hand-written
    expansion from a measured claim. See this module's own note on what no test can
    enforce.
    """
    for knob, coverage in KNOB_COVERAGE.items():
        used = {name for role in ROLES for name in coverage[role]} & set(SENTINELS)
        if knob in SUITE_WIDE_KNOBS:
            continue
        assert not used, (
            f"{knob} uses suite-wide sentinel(s) {sorted(used)} but is not suite-wide — "
            f"name the tasks that actually observe it"
        )
    assert set(SUITE_WIDE_KNOBS) == {"system_prompt", "temperature"}, (
        "SUITE_WIDE_KNOBS changed. Only a knob every live model call carries belongs "
        "here; adding one waives every coverage property for it."
    )
    for knob in SUITE_WIDE_KNOBS:
        assert knob in KNOB_COVERAGE, f"SUITE_WIDE_KNOBS names unknown knob {knob}"


def test_exemption_tables_are_disjoint():
    """`GUARD_ONLY_KNOBS` waives the miner-headroom rule and `UNGUARDED_KNOBS`
    waives the guard rule. Together they reduce a knob to one arbitrary observer
    with nothing checked, so no knob may hold both."""
    both = sorted(set(GUARD_ONLY_KNOBS) & set(UNGUARDED_KNOBS))
    assert not both, f"{both} claim both exemptions, which waives every property at once"


def test_exemption_tables_are_current_minimal_and_argued():
    """Both exemptions must rot loudly when the CODE moves under them.

    Scope, honestly: the checks below are structural. The rationale check requires a
    task name from the knob's own coverage to appear in the string — it cannot tell
    whether the sentence is true, and a reviewer passed it with prose that named a
    task while asserting something false. A word count was worse, not different in
    kind. Only a human reading the rationale against the measurement can validate it;
    this keeps a rationale from being empty or citing nothing, and no more.
    """
    for table, label in ((GUARD_ONLY_KNOBS, "GUARD_ONLY"), (UNGUARDED_KNOBS, "UNGUARDED")):
        assert set(table) <= set(KNOB_COVERAGE), f"{label} names an unknown knob"
        for knob, reason in table.items():
            # A word count is theatre — "because we say so right here okay" passed
            # it. Require the rationale to name a task from the knob's own coverage,
            # so it points at evidence someone can go and check.
            cited = {
                name
                for role in ROLES
                for name in _expand(KNOB_COVERAGE[knob][role])
                if name in reason
            }
            assert cited, (
                f"{label}[{knob}] must cite a task from its own coverage as evidence, "
                f"got {reason!r}"
            )

    for knob in GUARD_ONLY_KNOBS:
        # Empty, not merely headroom-free: a named miner that the exemption then
        # asserts cannot fail is the table contradicting itself.
        assert not KNOB_COVERAGE[knob]["miners"], (
            f"{knob} is guard-only, so it must declare `miners: ()`; if it now has a "
            f"minable observer, remove it from GUARD_ONLY_KNOBS instead"
        )
    for knob in UNGUARDED_KNOBS:
        coverage = KNOB_COVERAGE[knob]
        # Exactly ONE observer is what makes a knob genuinely unguardable: with two
        # or more, one can guard while another mines. Checking `observers - miners`
        # instead was vacuous whenever every observer was also listed as a miner,
        # which let a knob with three usable guards be parked here.
        observers = _expand(coverage["observers"])
        assert len(observers) == 1, (
            f"{knob} has {len(observers)} observers ({sorted(observers)}) — with more "
            f"than one, some observer can guard it; name a guard instead of an exemption"
        )
        assert observers == _expand(coverage["miners"]), (
            f"{knob} is declared unguardable, so its one observer must be the miner"
        )
        assert not coverage["guards"], f"{knob} is declared unguardable but names guards"


def test_the_scenario_guards_watch_compaction_without_becoming_miners():
    """Phase 2c contract §6. CMP-5/6/7 join `compaction`'s observers and guards; the
    miner stays G4 alone.

    The mining rule is the reason the second half matters: a candidate is mined
    against ONE task and then has to survive the guards. Filing the new guards as
    miners as well would let a compaction fix be tuned against the very tasks that
    exist to catch it overfitting — which is the whole failure this phase is built
    to make mechanical.
    """
    compaction = KNOB_COVERAGE["compaction"]
    scenario = {"CMP-5", "CMP-6", "CMP-7"}
    assert scenario <= set(compaction["observers"])
    assert scenario <= set(compaction["guards"])
    assert set(compaction["miners"]) == {"G4"}
    # G2 stays a guard and G4 stays out of the guard set (a task cannot vouch for a
    # candidate mined from it).
    assert "G2" in compaction["guards"] and "G4" not in compaction["guards"]
