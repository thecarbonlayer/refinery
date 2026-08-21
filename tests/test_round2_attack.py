"""The round-2 attack, pinned: nothing changed, so nothing may be CONFIRMED or ACCEPTED.

Round 1's artifact was withdrawn because an end-to-end FALSE ACCEPT was reproduced
from two of its own null arms — two runs of the same harness, the same config, the
same carbon revision, with no edit between them, judged as an improvement worth
shipping. A calibration that cannot survive its own null data is not a calibration.

So the check is committed rather than performed once and described. Every same-shape
ordered pair of the eight recorded round-2 arms is judged by the REAL
`loop.acceptance.evaluate()` under the REAL installed `model-r2.json`, and then, with
a fabricated first CONFIRM naming every carrier set a first pass could plausibly have
produced, by the REAL `confirmed()`. Zero CONFIRMs and zero ACCEPTs, or this test is
red and the artifact does not install.

"Same shape" is what `evaluate()`'s own parity gate demands: identical task sets and
identical per-task attempt counts. The three full-suite arms are same-shape with each
other (3/5 attempts over the whole suite); the five `--only A1 G2 G4 G5 --attempts 10`
subset arms are same-shape with each other. A full arm against a subset arm is refused
by the parity gate before any rule runs, which is behavior pinned elsewhere
(`loop/replay_check.py`) and not a case this sweep can judge.

THE ARM LIST IS SPELLED OUT, never globbed. Appending an arm to the protocol must
force a deliberate edit here, because a new arm is new evidence this claim has to be
re-established against.

WHERE THIS STANDS AT THE CLOSE OF PHASE 2. The calibrated half of the sweep is
SUSPENDED, and the reason has moved. It was once "the artifact rates four tasks and the
loader pins seven". The Phase 2c campaign ran to completion — ten `p2c-null-*` arms,
all seven tasks, at one runner hash — and the artifact it produced records
`fitness.fit = false`: STABILITY refused, because one arm moves the held-in quantile
from 4/9 to 1/3, and GOODNESS joined it at the phase's close, when fitness started
certifying the guards the model rates instead of the gain set alone. So the artifact
still does not install, now by its own verdict rather than by shape, and there is still
nothing calibrated to sweep WITH. That is the fail-closed design working, not a defect.

Restoration has one condition: a FIT artifact at the arms' own runner hash or a
successor (the close-out's `attempted` metric has since advanced the branch's). The
`calibration` fixture below resolves the installed artifact on every run, so the moment
one exists the sweep stops skipping and runs — and it will go red until its arm list
and its task set are re-keyed to that artifact's own pooling, which is the deliberate
edit the paragraph above demands. Restoring the artifact without restoring the sweep is
not a state this file lets anyone stay in quietly.
"""

from __future__ import annotations

import dataclasses
import json
from itertools import combinations, permutations
from pathlib import Path

import pytest

from loop.acceptance import (
    ACCEPT,
    CONFIRM,
    Decision,
    calibration_digest,
    confirmed,
    decision_digest,
)
from loop.validate import calibration_status

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
MODEL = REPO_ROOT / "iterations" / "calibration-compaction" / "model-r2.json"

FULL_ARMS = ("r2-null-full-a", "r2-null-full-b", "r2-null-full-c")
SUBSET_ARMS = (
    "r2-null-cmp-a",
    "r2-null-cmp-b",
    "r2-null-cmp-c",
    "r2-null-cmp-d",
    "r2-null-cmp-e",
    "r2-null-cmp-f",
    "r2-null-cmp-g",
    "r2-null-cmp-h",
)
ALL_ARMS = FULL_ARMS + SUBSET_ARMS
SUPPORTED = ("A1", "G2", "G4", "G5")
SPLIT_OF = {"A1": "held_in", "G4": "held_in", "G5": "held_in", "G2": "held_out"}

# The Phase 2c campaign's own arms, in the order `calibrate_model` pooled them. Spelled
# out for the same reason the round-2 list is: the committed artifact IS this pooling,
# and a claim about "the artifact" that globbed its inputs would not notice one moving.
P2C_ARMS = (
    "p2c-null-full-a",
    "p2c-null-full-b",
    "p2c-null-full-c",
    "p2c-null-cmp-a",
    "p2c-null-cmp-b",
    "p2c-null-cmp-c",
    "p2c-null-cmp-d",
    "p2c-null-cmp-e",
    "p2c-null-cmp-f",
    "p2c-null-cmp-g",
)
# What the artifact rates: the gain set plus the three scenario guards.
COVERED = frozenset(SUPPORTED) | {"CMP-5", "CMP-6", "CMP-7"}


def _arm(label: str) -> dict:
    return json.loads((RESULTS / f"{label}.json").read_text())


def _restrict(results: dict, names) -> dict:
    kept = {n: results["tasks"][n] for n in sorted(names) if n in results["tasks"]}
    return {**results, "filter": sorted(kept), "tasks": kept}


@pytest.fixture(scope="module")
def refusal() -> str:
    """Why the installed artifact does not load — the state this sweep is suspended in.

    Asked at the artifact's OWN provenance (the campaign's first arm), so nothing about
    a stale fingerprint can be doing the refusing: this is the artifact judged against
    the very measurements it was pooled from, and it still comes back None. That makes
    this fixture the tripwire. A fit artifact at this hash installs here, `cal` stops
    being None, and every test taking `refusal` fails until someone restores the sweeps
    below rather than only the artifact.
    """
    cal, why = calibration_status("compaction", _arm(P2C_ARMS[0])["fingerprint"], model_path=MODEL)
    assert cal is None, (
        "the committed artifact must not install: it records fitness.fit=false. If it "
        "now installs, the calibrated sweeps in this file are no longer suspended and "
        "must be re-keyed to the pooling that installs — see the module docstring."
    )
    return why


@pytest.fixture(scope="module")
def calibration():
    """The installed artifact, when one installs — otherwise the sweep skips, saying why.

    Resolved per run rather than pinned to a decision made once: the suspension is a
    fact about what is on disk today, so the code asks disk. A fit artifact turns the
    skip off by itself, which is the point — the alternative is a static `skip` marker
    that stays in force long after its reason has gone and quietly retires the attack.
    """
    cal, why = calibration_status("compaction", _arm(P2C_ARMS[0])["fingerprint"], model_path=MODEL)
    if cal is None:
        pytest.skip(
            f"calibrated sweep suspended — {why} Restored by a FIT artifact at this "
            "runner hash or a successor; when one lands, re-key ALL_ARMS and the "
            "restricted task set below to that artifact's own pooling."
        )
    return cal


def _same_shape_pairs():
    for group in (FULL_ARMS, SUBSET_ARMS):
        yield from permutations(group, 2)


def test_the_sweep_covers_every_same_shape_ordered_pair_of_the_eight_arms():
    """The denominator, stated: 3x2 full-suite pairs plus 8x7 subset pairs."""
    pairs = list(_same_shape_pairs())
    assert len(pairs) == 62
    assert len(set(pairs)) == 62
    assert {label for pair in pairs for label in pair} == set(ALL_ARMS)


def test_the_calibrated_null_sweep_is_suspended_by_the_artifacts_own_refusal(refusal):
    """The sweep's own precondition, asserted rather than assumed — and NAMED.

    "Every same-shape ordered pair judged by the REAL rule under the REAL artifact" is
    a claim about a pairing that does not exist while nothing installs. What is doing
    the refusing has moved, and this test pins the current reason rather than the one it
    used to have: the campaign is complete (ten arms, all seven tasks, one runner hash)
    and the artifact it produced sets `fitness.fit = false` because GOODNESS and
    STABILITY refused. "The arms are missing" and "the arms are in and the model is
    neither well-fitted nor stable" are different remedies, so the refusal has to say
    which one this is.

    The reason set GREW at the phase's close, and the growth is the finding. Fitness now
    certifies the tasks the model rates rather than only the four the gain judgment
    averages over, and the guards it had been rating without checking do not survive it:
    one arm's CMP-5 count (3 of 3, against a pooled 17/79) disagrees with a single-rate
    model at p = 0.00996, just inside the 0.01 alpha, and CMP-5, CMP-6 and G4 each have
    a per-task bound that crosses a grain bucket when one arm is dropped. Stability
    already refused; goodness joined it once anything looked at the guards.

    Restoration condition, stated once and asserted here: a FIT artifact at this runner
    hash or a successor. Nothing weaker — not a hand-edited `fit`, which the loader
    re-derives and rejects, and not a fresh pooling that still fails either check.
    """
    assert "not calibrated" in refusal
    assert "fitness.fit=False" in refusal, "the refusal must name the artifact's own verdict"
    assert "failed: goodness, stability" in refusal, "and which checks produced it"
    assert "re-run the arms" in refusal

    # The same facts read off the artifact, so the refusal cannot be the only thing
    # saying them. `fit` is false; goodness and stability are the checks that made it so.
    fitness = json.loads(MODEL.read_text())["fitness"]
    assert fitness["fit"] is False
    assert fitness["stability"]["pass"] is False
    assert fitness["goodness"]["pass"] is False
    assert fitness["grain"]["pass"] is True, (
        "goodness and stability refuse here — if grain has started failing too, the "
        "suspension reason above is no longer the whole reason"
    )
    # And WHERE each one refuses: both land on tasks the guards' own certification
    # brought into scope, which is why the set grew.
    failing_arms = {
        task: sorted(a for a, row in block["per_arm"].items() if not row["pass"])
        for task, block in fitness["goodness"]["per_task"].items()
    }
    assert {t: a for t, a in failing_arms.items() if a} == {"CMP-5": ["p2c-null-full-a"]}
    assert {t for t, row in fitness["stability"]["per_task"].items() if not row["pass"]} == {
        "CMP-5",
        "CMP-6",
        "G4",
    }


def _carrier_sets():
    """Every carrier set a first CONFIRM could have named: a nonempty subset of ONE
    split's supported tasks (``evaluate()`` only ever names the evidence split's
    movers)."""
    for split in ("held_in", "held_out"):
        tasks = sorted(t for t in SUPPORTED if SPLIT_OF[t] == split)
        for size in range(1, len(tasks) + 1):
            for combo in combinations(tasks, size):
                yield split, combo


def test_no_fabricated_first_confirm_can_be_accepted_on_a_null_pair(calibration):
    """Stage 2, attacked directly. `evaluate()` refuses to CONFIRM any of these
    pairs, so the only way to reach `confirmed()` with them is to hand it a first
    decision it never produced. Do exactly that — for every carrier set a real first
    pass could have named — and the confirmation must still refuse every one.

    This is the round-1 false ACCEPT's own shape: a fabricated CONFIRM whose carrier
    moved between two unchanged arms, cleared against a bound built for something
    else. Under the round-2 rule each carrier is judged against its OWN null
    distribution at these counts, and the pair has to survive the guard gate and the
    positivity gate as well.

    Suspended while nothing installs (the `calibration` fixture skips), and it comes
    back on its own the moment something does. `judged` is counted and asserted against
    the sweep's own denominator so that return cannot be a quiet one: a restored
    artifact pooled over other arms than `ALL_ARMS`, or over a wider task set than the
    `_restrict` below hands it, makes this sweep cover nothing or raise — never pass.
    """
    accepted = []
    judged = 0
    for a, b in _same_shape_pairs():
        base, cand = _arm(a), _arm(b)
        fb, fc = _restrict(base, SUPPORTED), _restrict(cand, SUPPORTED)
        for split, carriers in _carrier_sets():
            first = Decision(
                outcome=CONFIRM,
                reasons=(),
                delta_in=0.0,
                delta_ho=0.0,
                threshold_in=0.0,
                threshold_ho=0.0,
                evidence_split=split,
                improved_tasks=carriers,
                confirm_tasks=tuple(sorted(SUPPORTED)),
                raw={
                    "regime": "section_calibration",
                    "calibration_digest": calibration_digest(calibration),
                },
            )
            # Bound to its own claim as well: `confirmed()` requires both digests
            # whenever a calibration is in hand, so a fabricated record has to be as
            # well-formed as a real one before it can be an attack on the RULE rather
            # than an attack on the bookkeeping.
            first = dataclasses.replace(
                first, raw={**first.raw, "decision_digest": decision_digest(first)}
            )
            decision = confirmed(first, fb, fc, calibration=calibration)
            judged += 1
            if decision.outcome == ACCEPT:
                accepted.append((f"{a}::{b}", carriers, decision.reasons))
    assert judged == 62 * 8, (
        "the sweep must actually have judged every same-shape pair against every "
        "carrier set — a restored artifact that covers other arms proves nothing here"
    )
    assert not accepted, accepted


def test_the_committed_artifact_is_the_ten_arm_seven_task_pooling(refusal):
    """The artifact this file talks about, identified rather than assumed.

    It is the Phase 2c pooling: ten arms, seven rated tasks, computed at those arms' own
    runner hash — and it is still the file the loader names when it refuses. A refusal
    about some other model would prove nothing, and neither would a sweep against one.

    This is the second tripwire. It reads the artifact whole, so swapping in a different
    pooling — a fit one included — fails here as well as at the `refusal` fixture, and
    whoever swaps it has to come through this file rather than around it.
    """
    model = json.loads(MODEL.read_text())
    assert set(model["null_model"]) == COVERED
    assert tuple(p["label"] for p in model["provenance"]) == P2C_ARMS
    assert model["computed_at_runner_sha"] == _arm(P2C_ARMS[0])["fingerprint"]["runner_sha"]
    assert "model-r2.json" in refusal
    # The old four-task round-2 pooling is genuinely gone, not merely widened: none of
    # its arms is in this artifact's provenance.
    assert not {p["label"] for p in model["provenance"]} & set(ALL_ARMS)


# ---------------------------------------------------------------------------
# replay_check: a calibrated-path mode, and skipped files counted rather than
# silently dropped (contract amendments 7 and 9).
# ---------------------------------------------------------------------------


def _seed_results(tmp_path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    for label in ALL_ARMS:
        (d / f"{label}.json").write_text(json.dumps(_arm(label)))
    return d


def test_load_results_reports_what_it_skipped_instead_of_dropping_it(tmp_path):
    """A file that cannot be parsed is not a file that does not matter: it is a
    recorded measurement the replay never covered, and a gate that quietly narrows
    its own input can pass by covering nothing."""
    from loop.replay_check import load_results

    d = _seed_results(tmp_path)
    (d / "truncated.json").write_text('{"tasks": {')
    (d / "not-a-suite.json").write_text('{"hello": "world"}')
    results, skipped = load_results(d)
    assert set(results) == set(ALL_ARMS)
    assert sorted(skipped) == ["not-a-suite", "truncated"]


def test_replay_main_exits_nonzero_when_a_result_file_was_skipped(tmp_path, capsys):
    from loop.replay_check import main

    d = _seed_results(tmp_path)
    (d / "truncated.json").write_text("{oops")
    code = main(["--results-dir", str(d), "--calibrated"])
    out = capsys.readouterr().out
    assert "truncated" in out
    assert code != 0


def test_the_calibrated_replay_mode_reports_what_it_could_not_judge(tmp_path, capsys):
    """`--calibrated` is not a byte-identity replay — the calibrated regime is new
    behavior with nothing to be identical to. It is a COVERAGE gate: every same-shape
    ordered pair whose baseline the installed artifact is fresh for is judged through
    `evaluate(calibration=...)`, every CONFIRM is carried into `confirmed()`, and any
    pair of NULL ARMS reaching CONFIRM or ACCEPT fails the run.

    With no installable artifact the gate has nothing to judge WITH, and the property
    that matters becomes the other one: it must COUNT what it could not cover rather
    than reporting a clean run over an empty set."""
    from loop.replay_check import main, replay_calibrated

    d = _seed_results(tmp_path)
    report = replay_calibrated(d, model_path=MODEL)
    # Eleven arms make 110 ordered pairs. With the artifact refusing to install at the
    # seven-task pin, EVERY pair lands in `uncalibrated_pairs` — the mode reports what
    # it could not judge instead of judging it against nothing, which is the behavior
    # worth pinning while the section waits for its campaign. The counted-pairs shape
    # (110 = 62 same-shape + 48 cross-shape) is unchanged and is re-established as a
    # calibrated sweep once the new arms exist.
    assert report["results"] == len(ALL_ARMS)
    assert report["uncalibrated_pairs"] == 110
    assert report["pairs"] == 0
    assert report["decided"] == 0
    assert report["confirms"] == 0
    assert report["accepts"] == 0
    assert report["false_outcomes"] == []
    assert report["skipped"] == []

    assert main(["--results-dir", str(d), "--calibrated"]) == 0
    out = capsys.readouterr().out
    assert "calibrated" in out.lower()
