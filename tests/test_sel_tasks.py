"""Premise checks and registration pins for the tool-exposure section (SEL).

Decision 14 discipline: every task proves its premise before its numbers are
trusted, and for this primitive most premises are computable OFFLINE, because
carbon's exposure selector is deterministic — calling the real
``exposed_specs()`` on the pinned registries shows exactly which tools the model
would be offered under any policy value, with no model call. Same pattern as the
E-cluster premise probes, which call carbon's real ``truncate()`` rather than a
reimplementation: a probe through the real mechanism cannot drift from it.

Also pinned here, because no contract document freezes them yet (the CMP-5/6/7
entries in ``test_registry.py``'s frozen table came from a committed contract;
these await the tool-exposure phase's): the four SPECS rows themselves, and the
section's isolation from every calibrated gate — no committed calibration
artifact may know these task names until the section runs its own null campaign
(decision 20's integrity rule: suites stay isolated from calibrated gates until
their own campaigns run).
"""

from __future__ import annotations

import json
from pathlib import Path

from runner.tasks import TASKS

REPO_ROOT = Path(__file__).resolve().parents[1]

SEL_NAMES = ("SEL-2", "SEL-3", "SEL-4", "SEL-5")


def _names(specs: list[dict]) -> list[str]:
    return [s["function"]["name"] for s in specs]


# --- SPECS pins (the authoring claim, frozen until a phase contract exists) ---


def test_sel_specs_are_pinned():
    by_name = {t.name: t for t in TASKS}
    # Splits alternate held-in/held-out at authoring (phase-4 brief §3); the
    # final assignment is a gate input, so a change here is a human decision.
    expected = {
        "SEL-2": "held_in",
        "SEL-3": "held_out",
        "SEL-4": "held_in",
        "SEL-5": "held_out",
    }
    for name, split in expected.items():
        t = by_name[name]
        assert t.split == split, f"{name}: split {t.split!r} != authored {split!r}"
        assert t.cluster == "S", f"{name}: cluster {t.cluster!r} != 'S'"
        assert t.primitive == "tool-selection"
        # The NAME is the map mnemonic (the CMP-5/6/7 convention): no second id.
        assert t.alias is None
        # A prior is a claim about the suite as authored, never a reading of a
        # baseline — nothing has measured these, so every prior is `uncertain`.
        assert t.expected_baseline == "uncertain"


def test_sel_section_names_the_same_tasks_the_registry_carries():
    from runner.tasks.cluster_s import SECTION_TASKS

    assert tuple(SECTION_TASKS) == SEL_NAMES
    assert set(SEL_NAMES) <= {t.name for t in TASKS}


# --- isolation from every calibrated gate -------------------------------------


def test_sel_tasks_are_outside_every_committed_calibration_artifact():
    """The proof the section starts UNCALIBRATED: no committed calibration
    artifact carries a SEL rate, supported-set entry, or coverage row. The
    acceptance gates load their task sets from these artifacts, so absence here
    is absence from every calibrated gate — a SEL number cannot reach a
    CONFIRM/ACCEPT decision until the section's own null campaign writes one.
    """
    artifacts = sorted((REPO_ROOT / "iterations").glob("calibration-*/*.json"))
    assert artifacts, "no calibration artifacts found — the isolation proof is vacuous"
    for path in artifacts:
        hits: list[str] = []
        _walk_for_sel_names(json.loads(path.read_text()), "", path.name, hits)
        assert not hits, f"SEL tasks leaked into a calibrated artifact: {hits}"


def _walk_for_sel_names(node: object, where: str, label: str, hits: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in SEL_NAMES:
                hits.append(f"{label}: key {key!r} under {where}")
            _walk_for_sel_names(value, f"{where}/{key}", label, hits)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str) and item in SEL_NAMES:
                hits.append(f"{label}: {item!r} listed under {where}")
            else:
                _walk_for_sel_names(item, where, label, hits)


def test_sel_tasks_stay_outside_the_gate_machinery_calibrated_sets():
    """The CTX branch's structural pin, applied to this section: the artifact walk
    above proves no committed calibration RECORD knows these names, but the gates
    also carry code-level sets, and those are the direct thing to pin. The SEL
    tasks are registry members and NOTHING else: no gain set, no guard set, no
    null-model coverage, no confirmation rerun list. And ``tool_exposure`` — the
    knob they observe — maps to NO rule section: a candidate editing it is
    'unmapped' to ``rule_disposition``, so the calibrated three-outcome rule
    cannot be applied to it by accident before its calibration exists."""
    from loop.calibrate import CONFIRMATION_GUARDS, MODEL_TASKS, SCENARIO_GUARDS, SUPPORTED
    from loop.validate import (
        _FIELD_SECTION,
        _SECTION_CONFIRM_GUARDS,
        CALIBRATION_REQUIRED,
        RULE_SECTIONS,
    )

    sel = set(SEL_NAMES)
    calibrated = SUPPORTED | MODEL_TASKS | CONFIRMATION_GUARDS | SCENARIO_GUARDS
    assert sel.isdisjoint(calibrated), f"SEL task(s) entered a calibrated set: {sel & calibrated}"
    for section, guards in _SECTION_CONFIRM_GUARDS.items():
        assert sel.isdisjoint(guards), f"SEL task(s) entered {section}'s confirm guards"
    assert "tool_exposure" not in _FIELD_SECTION, (
        "tool_exposure gained a rule-section mapping — that step belongs to the "
        "calibration install AFTER the SEL null campaign, not to task authoring"
    )
    assert "tool_exposure" not in RULE_SECTIONS | CALIBRATION_REQUIRED


def test_sel_tasks_are_not_in_the_compaction_supported_set():
    """The one live calibrated section today, checked by name rather than only
    through the generic walk above."""
    doc = json.loads(
        (REPO_ROOT / "iterations" / "calibration-compaction" / "model-r2.json").read_text()
    )
    assert not (set(doc["null_model"]) & set(SEL_NAMES))


# --- SEL-2: one relevant tool among 30 plausible decoys (overload, the miner) --


def test_sel2_registry_is_one_needed_tool_among_thirty_decoys():
    from runner.tasks.cluster_s import SEL2_ASK, SEL2_CODE, SEL2_NEEDED, sel2_registry

    reg = sel2_registry()
    names = reg.names()
    assert len(names) == 31, "the map's premise is one relevant tool among 30 decoys"
    assert names.count(SEL2_NEEDED) == 1
    # The needed tool is buried mid-registry, not first or last — position must
    # not be what SEL-2 measures (that is SEL-4's axis).
    position = names.index(SEL2_NEEDED)
    assert 5 <= position <= 25, f"needed tool at {position} — too close to an end"
    # Exactly one tool can produce the code; every decoy's canned result is
    # code-free, so a decoy call can never satisfy the dual oracle.
    producers = [n for n in names if SEL2_CODE in reg.get(n).func()]
    assert producers == [SEL2_NEEDED]
    # The consignment id the ask names is echoed by the needed tool's result —
    # the reply-side sentinel can only have come through the tool.
    assert "CN-7743" in SEL2_ASK and SEL2_CODE not in SEL2_ASK


def test_sel2_needed_tool_is_offered_under_the_default_exposure():
    """Red-for-the-right-reason floor: under ``all`` (today's shipped exposure)
    the needed tool IS offered, so a baseline failure is an overload failure,
    never a missing tool."""
    from harness.harness_config import ToolExposurePolicy
    from harness.tools import exposed_specs

    from runner.tasks.cluster_s import SEL2_ASK, SEL2_NEEDED, sel2_registry

    reg = sel2_registry()
    offered = _names(exposed_specs(reg, ToolExposurePolicy("all"), query=SEL2_ASK))
    assert offered == reg.names()
    assert SEL2_NEEDED in offered


# --- SEL-3: a near-duplicate decoy that must not decide the answer ------------


def test_sel3_twins_differ_in_exactly_the_value_and_the_decoy_outranks():
    from harness.harness_config import ToolExposurePolicy
    from harness.tools import exposed_specs

    from runner.tasks.cluster_s import (
        SEL3_ASK,
        SEL3_DECOY,
        SEL3_DUE_CODE,
        SEL3_NEEDED,
        SEL3_SUBTOTAL_CODE,
        sel3_registry,
    )

    reg = sel3_registry()
    assert SEL3_DUE_CODE != SEL3_SUBTOTAL_CODE
    assert SEL3_DUE_CODE in reg.get(SEL3_NEEDED).func()
    assert SEL3_SUBTOTAL_CODE in reg.get(SEL3_DECOY).func()
    # Neither twin's result carries the other's value.
    assert SEL3_SUBTOTAL_CODE not in reg.get(SEL3_NEEDED).func()
    assert SEL3_DUE_CODE not in reg.get(SEL3_DECOY).func()
    # The knob-red-capability premise, through carbon's real selector: under
    # query_match k=1 the DECOY is the one tool offered — a legal value that
    # makes the right answer unreachable, which is exactly what a guard on this
    # axis must be able to catch. Computable offline, no model call.
    top1 = _names(exposed_specs(reg, ToolExposurePolicy("query_match", k=1), query=SEL3_ASK))
    assert top1 == [SEL3_DECOY], (
        f"query_match k=1 offers {top1}, not the near-duplicate decoy — the fixture "
        "no longer proves a legal exposure value can select the right tool away"
    )
    # And under the shipped default both twins are offered: a baseline failure is
    # the model's confusion between near-duplicates, not a missing tool.
    offered = _names(exposed_specs(reg, ToolExposurePolicy("all"), query=SEL3_ASK))
    assert SEL3_NEEDED in offered and SEL3_DECOY in offered


# --- SEL-4: exposure order / rank sensitivity ---------------------------------


def test_sel4_needed_tool_is_registered_last_and_outranked_by_every_decoy():
    from harness.harness_config import ToolExposurePolicy
    from harness.tools import exposed_specs

    from runner.tasks.cluster_s import SEL4_ASK, SEL4_CODE, SEL4_NEEDED, sel4_registry

    reg = sel4_registry()
    names = reg.names()
    assert len(names) == 31
    assert names[-1] == SEL4_NEEDED, "the axis is order: the needed tool must be LAST"
    producers = [n for n in names if SEL4_CODE in reg.get(n).func()]
    assert producers == [SEL4_NEEDED]
    # Rank premise, through carbon's real selector: at a generous k the needed
    # tool is still ranked dead last (every vocabulary-stuffed decoy scores at
    # least as high, and ties keep registration order, where it is also last)...
    ranked = _names(exposed_specs(reg, ToolExposurePolicy("query_match", k=31), query=SEL4_ASK))
    assert ranked[-1] == SEL4_NEEDED, (
        "a decoy now ranks below the needed tool — the fixture has stopped "
        "discriminating rank-based exposure"
    )
    # ...so any k <= 30 removes it entirely: the strategy failure this task is
    # designed to catch, proven reachable by a legal value without a model call.
    top8 = _names(exposed_specs(reg, ToolExposurePolicy("query_match", k=8), query=SEL4_ASK))
    assert SEL4_NEEDED not in top8
    # Under the shipped default it is offered (last), so baseline red = order
    # cost, not absence.
    offered = _names(exposed_specs(reg, ToolExposurePolicy("all"), query=SEL4_ASK))
    assert offered[-1] == SEL4_NEEDED


# --- SEL-5: vocabulary mismatch must not select the needed tool away ----------


def test_sel5_needed_tool_is_strictly_lowest_ranked_for_the_ask():
    from harness.harness_config import ToolExposurePolicy
    from harness.tools import exposed_specs

    from runner.tasks.cluster_s import SEL5_ASK, SEL5_COUNT, SEL5_NEEDED, sel5_registry

    reg = sel5_registry()
    names = reg.names()
    assert len(names) == 7  # needed + six lexical-bait fillers: a SMALL registry,
    # so a baseline failure is attributable to vocabulary, never to overload
    producers = [n for n in names if SEL5_COUNT in reg.get(n).func()]
    assert producers == [SEL5_NEEDED]
    # The mismatch premise, behavior-level through carbon's real selector: the
    # ask's vocabulary reaches every filler and never the needed tool, so top-6
    # is exactly the six fillers — under query_match at any k <= 6, a legal
    # value, the ONLY tool that can answer is not offered at all.
    top6 = _names(exposed_specs(reg, ToolExposurePolicy("query_match", k=6), query=SEL5_ASK))
    assert SEL5_NEEDED not in top6 and len(top6) == 6
    # At k=7 it appears — ranked last, confirming "strictly lowest", not "absent
    # from the registry".
    top7 = _names(exposed_specs(reg, ToolExposurePolicy("query_match", k=7), query=SEL5_ASK))
    assert top7[-1] == SEL5_NEEDED
    # Under the shipped default it is offered: the guard's no-harm end.
    offered = _names(exposed_specs(reg, ToolExposurePolicy("all"), query=SEL5_ASK))
    assert SEL5_NEEDED in offered


# --- verdict taxonomy ---------------------------------------------------------


def test_sel_verdicts_apply_the_g2_non_answer_taxonomy():
    """Contract §5's decomposition, carried into every reply-shaped SEL oracle the
    same way the sibling CTX suite carried it: a reply that never attempted the
    answer (carbon's truncation marker and nothing else, or a tool-syntax leak)
    is ``not_attempted``, never a selection failure — and never a pass. An
    answered-but-wrong reply, a decoy-quoting reply, and a reply-side sentinel
    with no tool-result backing all stay plain ``fail``: each is a real verdict
    about the axis, not a non-answer."""
    from runner.tasks.cluster_g import G2_TRUNCATION_MARKER
    from runner.tasks.cluster_s import (
        SEL2_CODE,
        SEL3_DUE_CODE,
        SEL3_SUBTOTAL_CODE,
        SEL4_CODE,
        SEL5_COUNT,
        sel2_verdict,
        sel3_verdict,
        sel4_verdict,
        sel5_verdict,
    )

    marker_only = f"  {G2_TRUNCATION_MARKER}"
    leak = "<|tool_call|>consignment_routing"

    # SEL-2: dual conjunct — the reply-side code alone (no tool-result backing)
    # is the spoof shape the D1 pattern exists to refuse.
    assert sel2_verdict(f"The code is {SEL2_CODE}.", True) == (True, "pass", None)
    assert sel2_verdict(SEL2_CODE, False)[:2] == (False, "fail")
    assert sel2_verdict("RTG-0000-XXX", True)[:2] == (False, "fail")
    assert sel2_verdict(marker_only, False)[1] == "not_attempted"
    assert sel2_verdict(leak, False)[1] == "not_attempted"
    # A reply that recalled the code attempted the answer, whatever else it holds.
    assert sel2_verdict(f"{SEL2_CODE} {leak}", False)[:2] == (False, "fail")

    # SEL-3: the decoy's value must not ride along, as the full code OR bare digits.
    assert sel3_verdict(f"amount due {SEL3_DUE_CODE}", True) == (True, "pass", None)
    assert sel3_verdict(f"subtotal {SEL3_SUBTOTAL_CODE}", True)[:2] == (False, "fail"), (
        "a confidently reported near-twin value is the failure this guard exists for"
    )
    assert sel3_verdict(f"{SEL3_DUE_CODE} (subtotal {SEL3_SUBTOTAL_CODE})", True)[:2] == (
        False,
        "fail",
    ), "the right code beside the decoy's is not a clean answer"
    assert sel3_verdict(f"{SEL3_DUE_CODE}, i.e. 17 250 plus charges", True)[:2] == (
        False,
        "fail",
    ), "the decoy's bare digits must be caught through spacing, not just the full code"
    assert sel3_verdict(marker_only, False)[1] == "not_attempted"

    # SEL-4: same dual-conjunct shape as SEL-2.
    assert sel4_verdict(f"window {SEL4_CODE}", True) == (True, "pass", None)
    assert sel4_verdict("WIN-XX-9999-0000", True)[:2] == (False, "fail")
    assert sel4_verdict(SEL4_CODE, False)[:2] == (False, "fail")
    assert sel4_verdict(marker_only, False)[1] == "not_attempted"

    # SEL-5: the D3 last-int anchor — a quoted intermediate number is not the answer.
    assert sel5_verdict(f"The depot holds {SEL5_COUNT} units.", True) == (True, "pass", None)
    assert sel5_verdict(f"{SEL5_COUNT} was last year; now 5000", True)[:2] == (False, "fail")
    assert sel5_verdict("no idea", True)[:2] == (False, "fail")
    assert sel5_verdict(marker_only, False)[1] == "not_attempted"
    assert sel5_verdict(leak, False)[1] == "not_attempted"


# --- cross-task hygiene -------------------------------------------------------


def test_sel_sentinels_are_distinct_and_absent_from_every_other_fixture():
    """No sentinel may be producible by another task's registry — a transcript
    contaminated across attempts must not be able to satisfy the wrong oracle."""
    from runner.tasks.cluster_s import (
        SEL2_CODE,
        SEL3_DUE_CODE,
        SEL3_SUBTOTAL_CODE,
        SEL4_CODE,
        SEL5_COUNT,
        sel2_registry,
        sel3_registry,
        sel4_registry,
        sel5_registry,
    )

    sentinels = {SEL2_CODE, SEL3_DUE_CODE, SEL3_SUBTOTAL_CODE, SEL4_CODE, SEL5_COUNT}
    assert len(sentinels) == 5
    everything = {
        "sel2": [reg.get(n).func() for reg in [sel2_registry()] for n in reg.names()],
        "sel3": [reg.get(n).func() for reg in [sel3_registry()] for n in reg.names()],
        "sel4": [reg.get(n).func() for reg in [sel4_registry()] for n in reg.names()],
        "sel5": [reg.get(n).func() for reg in [sel5_registry()] for n in reg.names()],
    }
    home = {
        "sel2": {SEL2_CODE},
        "sel3": {SEL3_DUE_CODE, SEL3_SUBTOTAL_CODE},
        "sel4": {SEL4_CODE},
        "sel5": {SEL5_COUNT},
    }
    for task, results in everything.items():
        foreign = sentinels - home[task]
        for text in results:
            leaked = {s for s in foreign if s in text}
            assert not leaked, f"{task}: foreign sentinel(s) {leaked} in a tool result"
