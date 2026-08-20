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
"""

from __future__ import annotations

import json
from itertools import combinations, permutations
from pathlib import Path

import pytest

from loop.acceptance import ACCEPT, CONFIRM, Decision, calibration_digest, confirmed, evaluate
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
)
ALL_ARMS = FULL_ARMS + SUBSET_ARMS
SUPPORTED = ("A1", "G2", "G4", "G5")
SPLIT_OF = {"A1": "held_in", "G4": "held_in", "G5": "held_in", "G2": "held_out"}


def _arm(label: str) -> dict:
    return json.loads((RESULTS / f"{label}.json").read_text())


def _restrict(results: dict, names) -> dict:
    kept = {n: results["tasks"][n] for n in sorted(names) if n in results["tasks"]}
    return {**results, "filter": sorted(kept), "tasks": kept}


@pytest.fixture(scope="module")
def calibration():
    cal = calibration_status("compaction", _arm("r2-null-full-a")["fingerprint"], model_path=MODEL)[
        0
    ]
    assert cal is not None, "the installed artifact must load, or this sweep proves nothing"
    return cal


def _same_shape_pairs():
    for group in (FULL_ARMS, SUBSET_ARMS):
        yield from permutations(group, 2)


def test_the_sweep_covers_every_same_shape_ordered_pair_of_the_eight_arms():
    """The denominator, stated: 3x2 full-suite pairs plus 5x4 subset pairs."""
    pairs = list(_same_shape_pairs())
    assert len(pairs) == 26
    assert len(set(pairs)) == 26
    assert {label for pair in pairs for label in pair} == set(ALL_ARMS)


def test_no_ordered_pair_of_null_arms_reaches_confirm(calibration):
    """Stage 1 over the real arms with the real artifact: every verdict REJECT."""
    outcomes = {}
    for a, b in _same_shape_pairs():
        decision = evaluate(_arm(a), _arm(b), calibration=calibration)
        outcomes[f"{a}::{b}"] = decision.outcome
    confirms = {k: v for k, v in outcomes.items() if v == CONFIRM}
    assert not confirms, confirms
    assert set(outcomes.values()) == {"REJECT"}


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
    """
    accepted = []
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
            decision = confirmed(first, fb, fc, calibration=calibration)
            if decision.outcome == ACCEPT:
                accepted.append((f"{a}::{b}", carriers, decision.reasons))
    assert not accepted, accepted


def test_the_installed_artifact_is_the_one_this_sweep_judged_with(calibration):
    """A sweep against some other model would prove nothing about what ships."""
    assert calibration.source == "iterations/calibration-compaction/model-r2.json"
    assert calibration.computed_at_runner_sha == _arm("r2-null-full-a")["fingerprint"]["runner_sha"]


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


def test_the_calibrated_replay_mode_exercises_the_calibrated_branch(tmp_path, capsys):
    """`--calibrated` is not a byte-identity replay — the calibrated regime is new
    behavior with nothing to be identical to. It is a COVERAGE gate: every same-shape
    ordered pair whose baseline the installed artifact is fresh for is judged through
    `evaluate(calibration=...)`, every CONFIRM is carried into `confirmed()`, and any
    pair of NULL ARMS reaching CONFIRM or ACCEPT fails the run."""
    from loop.replay_check import main, replay_calibrated

    d = _seed_results(tmp_path)
    report = replay_calibrated(d, model_path=MODEL)
    # Eight arms make 56 ordered pairs. The 30 cross-shape ones are refused by the
    # parity gates before any rule runs, which is behavior, not a skipped case.
    assert report["pairs"] == 56
    assert report["refused"] == 30
    assert report["decided"] == 26
    assert report["null_arm_pairs"] == 26
    assert report["confirms"] == 0
    assert report["accepts"] == 0
    assert report["false_outcomes"] == []
    assert report["skipped"] == []

    assert main(["--results-dir", str(d), "--calibrated"]) == 0
    out = capsys.readouterr().out
    assert "calibrated" in out.lower()
    assert "26" in out
