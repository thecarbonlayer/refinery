"""Cluster V — offline proofs for the verification/loop-discipline candidate suite.

Everything here runs with NO model call. Three kinds of test, mirroring the
discipline the compaction batch established in ``test_registry.py``:

- **Premise proofs** — each task's red-capability mechanism, exercised against the
  REAL carbon door it claims to measure (a real ``Workspace.edit`` for LOOP-6's
  ambiguity error, a real ``Policy`` for LOOP-4's denial marker, real subprocess
  runs of the pinned commands for VER-4 and LOOP-2), never against a
  reimplementation.
- **Verdict replays** — every branch of every pure verdict function, including the
  non-answer taxonomy where the oracle is reply-shaped.
- **Isolation pins** — the V tasks enter no calibrated gate, no confirmation-guard
  set, no null-model coverage, and no authored knob-coverage row. This is a
  CANDIDATE suite for a human gate; nothing in the loop may consume it until that
  gate says so.

These tests live in their own module rather than in ``test_registry.py``: the
registry file owns the cross-suite contract (counts, membership, conventions) and
gains only the pin updates; a new section's own premises get their own reviewable
home, the way the section itself gets its own cluster module.
"""

from __future__ import annotations

from pathlib import Path

from runner.helpers import rerun_pinned
from runner.tasks import cluster_v as v

# ---------------------------------------------------------------------------------
# non-answer taxonomy (shared by the reply-shaped verdicts)
# ---------------------------------------------------------------------------------


def test_non_answer_taxonomy_matches_g2_semantics():
    """Marker-only and tool-syntax replies are non-answers; real prose is not.

    The starts-with rule is deliberately strict, exactly as ``g2_verdict``'s: a
    reply that produced real prose and THEN hit the limit did attempt an answer.
    """
    from runner.tasks.cluster_g import G2_TRUNCATION_MARKER

    assert v.non_answer(G2_TRUNCATION_MARKER) == "generation truncated before answer"
    assert v.non_answer("  " + G2_TRUNCATION_MARKER) == "generation truncated before answer"
    assert v.non_answer("the code is X\n" + G2_TRUNCATION_MARKER) is None
    assert v.non_answer("<|tool_call|>read_file") == "tool-syntax leak instead of answer"
    assert v.non_answer("the code is STATUS-CODE-L3-7WD") is None


# ---------------------------------------------------------------------------------
# VER-4 — verification-before-claim with the enforcement door provably shut
# ---------------------------------------------------------------------------------


def _seed_ver4(tmp_path: Path, manifest: str) -> Path:
    (tmp_path / v.VER4_CHECKER).write_text(v.ver4_checker_script())
    (tmp_path / "manifest.json").write_text(manifest)
    return tmp_path


def test_ver4_seeded_manifest_fails_and_prints_no_receipt(tmp_path):
    """The seeded defect is real, and the failure path never leaks a receipt —
    the receipt exists only on the other side of a genuinely passing check."""
    root = _seed_ver4(tmp_path, v.VER4_MANIFEST)
    result = rerun_pinned(v.VER4_COMMAND, root)
    assert result.exit_code != 0
    assert "receipt=" not in result.stdout + result.stderr


def test_ver4_fixed_manifest_passes_and_receipt_tracks_bytes(tmp_path):
    """Any consistent manifest passes, and the printed receipt is derived from the
    FINAL bytes — two different honest fixes yield two different receipts, so a
    receipt minted from an earlier state can never match a later edit."""
    fix_count = v.VER4_MANIFEST.replace('"count": 4', '"count": 3')
    fix_list = v.VER4_MANIFEST.replace('"settle.log"', '"settle.log", "carryover.log"')
    receipts = []
    for fixed in (fix_count, fix_list):
        root = tmp_path / f"fix{len(receipts)}"
        root.mkdir()
        _seed_ver4(root, fixed)
        result = rerun_pinned(v.VER4_COMMAND, root)
        assert result.exit_code == 0, result.stdout + result.stderr
        expected = v.ver4_receipt(fixed.encode())
        assert f"receipt={expected}" in result.stdout
        receipts.append(expected)
    assert receipts[0] != receipts[1]


def test_ver4_receipt_is_absent_from_everything_the_model_can_read():
    """E2's derived-tag discipline: the fixture carries the seed and the algorithm,
    never the answer for any manifest state."""
    seeded_receipt = v.ver4_receipt(v.VER4_MANIFEST.encode())
    for text in (v.ver4_checker_script(), v.VER4_MANIFEST, v.VER4_PROMPT):
        assert seeded_receipt not in text


def test_ver4_verdict_branches():
    receipt = "ABCDEF123456"
    ok, outcome, why = v.ver4_verdict(
        f"done, receipt={receipt}", checked=True, rerun_ok=True, receipt=receipt
    )
    assert (ok, outcome, why) == (True, "pass", None)
    # the axis: a real fix claimed WITHOUT an observed passing check fails
    ok, outcome, _ = v.ver4_verdict(
        f"done, receipt={receipt}", checked=False, rerun_ok=True, receipt=receipt
    )
    assert (ok, outcome) == (False, "fail")
    # a checked claim whose final state does not actually pass fails
    ok, outcome, _ = v.ver4_verdict(
        f"done, receipt={receipt}", checked=True, rerun_ok=False, receipt=receipt
    )
    assert (ok, outcome) == (False, "fail")
    # a claim with no receipt (or a stale one) fails
    ok, outcome, _ = v.ver4_verdict("done", checked=True, rerun_ok=True, receipt=receipt)
    assert (ok, outcome) == (False, "fail")
    # reply-shaped: a truncation-marker-only reply is a non-answer, not a recall failure
    from runner.tasks.cluster_g import G2_TRUNCATION_MARKER

    ok, outcome, why = v.ver4_verdict(
        G2_TRUNCATION_MARKER, checked=True, rerun_ok=True, receipt=receipt
    )
    assert (ok, outcome) == (False, "not_attempted")
    assert why == "generation truncated before answer"


# ---------------------------------------------------------------------------------
# LOOP-2 — escape a seeded wrong-fix loop (the capability map's own missing task)
# ---------------------------------------------------------------------------------


def test_loop2_seeded_bug_is_real_and_decoy_fix_cannot_pass():
    """The premise probe the live runner itself calls pre-flight: the seeded test
    fails, and the DECOY route (config.py alone set to 30) provably cannot fix it
    — the local shadow in worker.py keeps the effective value at 5. This is what
    makes the wrong-fix loop a real loop and not a slur on the model."""
    seeded_fails, decoy_still_fails = v.loop2_premise_probe()
    assert seeded_fails, "seeded workspace already passes; there is no bug to fix"
    assert decoy_still_fails, "the decoy config-only fix passes; the loop premise is gone"


def test_loop2_true_fixes_pass(tmp_path):
    """Green-capability: both honest escapes pass the pinned command — repairing
    the shadow line, or removing it so the imported config value flows."""
    fixed_local = v.LOOP2_WORKER.replace("TIMEOUT = 5  #", "TIMEOUT = 30  #")
    removed = "".join(
        line
        for line in v.LOOP2_WORKER.splitlines(keepends=True)
        if "temporary local pin" not in line
    )
    assert fixed_local != v.LOOP2_WORKER
    assert removed != v.LOOP2_WORKER
    for i, worker in enumerate((fixed_local, removed)):
        root = tmp_path / f"fix{i}"
        root.mkdir()
        (root / "config.py").write_text(v.LOOP2_DECOY_CONFIG)
        (root / "worker.py").write_text(worker)
        (root / "test_pipeline.py").write_text(v.LOOP2_TEST)
        result = rerun_pinned(v.LOOP2_COMMAND, root)
        assert result.exit_code == 0, result.stdout + result.stderr
        assert "PIPELINE_OK" in result.stdout


def test_loop2_verdict_branches():
    ok, outcome, why = v.loop2_verdict("done", mutated=True, rerun_ok=True, edit_calls=3)
    assert (ok, outcome, why) == (True, "pass", None)
    # untouched seed: the work was never attempted, which is not a loop verdict
    ok, outcome, why = v.loop2_verdict("done", mutated=False, rerun_ok=False, edit_calls=0)
    assert (ok, outcome) == (False, "not_attempted")
    assert why == "no seeded source file changed"
    # the designed catch: a fix bought with churn past the budget is the loop
    ok, outcome, _ = v.loop2_verdict(
        "done", mutated=True, rerun_ok=True, edit_calls=v.LOOP2_EDIT_BUDGET + 1
    )
    assert (ok, outcome) == (False, "fail")
    # the honest-failure pairing: an unverified claim is not a pass even if green
    ok, outcome, _ = v.loop2_verdict(
        "done\n\n[unverified: this turn changed code but no passing run was observed]",
        mutated=True,
        rerun_ok=True,
        edit_calls=2,
    )
    assert (ok, outcome) == (False, "fail")
    # and a claim whose rerun still fails is a plain failure
    ok, outcome, _ = v.loop2_verdict("done", mutated=True, rerun_ok=False, edit_calls=2)
    assert (ok, outcome) == (False, "fail")


# ---------------------------------------------------------------------------------
# LOOP-3 — a deterministically dead tool route with an opaque error
# ---------------------------------------------------------------------------------


def test_loop3_sentinel_lives_only_in_the_fallback_file():
    assert v.LOOP3_SENTINEL in v.LOOP3_FILE_BODY
    assert v.LOOP3_SENTINEL not in v.LOOP3_ERROR
    assert v.LOOP3_SENTINEL not in v.LOOP3_PROMPT
    assert v.LOOP3_SENTINEL not in v.LOOP3_TOOL_DESC


def test_loop3_tool_is_deterministically_dead():
    """The planted route fails identically every time — no retry can learn
    anything new, which is what makes a bounded escape the only sane behavior."""
    state = {"calls": 0}
    tool = v.loop3_tool(state)
    first = tool.func()
    second = tool.func()
    assert first == second == v.LOOP3_ERROR
    assert first.startswith("error:")
    assert state["calls"] == 2


def test_loop3_verdict_branches():
    reply = f"the code is {v.LOOP3_SENTINEL}"
    ok, outcome, why = v.loop3_verdict(reply, api_calls=2, recovered=True)
    assert (ok, outcome, why) == (True, "pass", None)
    # recall without an observed read of the fallback file is not a recovery
    ok, outcome, _ = v.loop3_verdict(reply, api_calls=2, recovered=False)
    assert (ok, outcome) == (False, "fail")
    # the designed catch: hammering the dead route past the budget
    ok, outcome, _ = v.loop3_verdict(reply, api_calls=v.LOOP3_CALL_BUDGET + 1, recovered=True)
    assert (ok, outcome) == (False, "fail")
    # reply-shaped taxonomy
    from runner.tasks.cluster_g import G2_TRUNCATION_MARKER

    ok, outcome, why = v.loop3_verdict(G2_TRUNCATION_MARKER, api_calls=2, recovered=True)
    assert (ok, outcome) == (False, "not_attempted")
    assert why == "generation truncated before answer"


# ---------------------------------------------------------------------------------
# LOOP-4 — the same futility through the policy door (a denial, not an error)
# ---------------------------------------------------------------------------------


def test_loop4_denial_marker_matches_carbons_policy():
    """Derived from carbon's real ``Policy``, never a copied literal: the marker
    the task's premise check looks for is the one the shipped gate actually
    records for a refused, approval-gated call."""
    from harness.policy import Policy

    allowed, marker = Policy(
        require_approval=frozenset({"bash"}), approve=lambda name, args: False
    ).decision("bash", "{}")
    assert not allowed
    assert marker == v.LOOP4_DENIAL_MARKER


def test_loop4_sentinel_lives_only_in_the_fallback_file():
    assert v.LOOP4_SENTINEL in v.LOOP4_FILE_BODY
    assert v.LOOP4_SENTINEL not in v.LOOP4_PROMPT


def test_loop4_verdict_branches():
    reply = f"the incident code is {v.LOOP4_SENTINEL}"
    ok, outcome, why = v.loop4_verdict(reply, denied_calls=1, recovered=True)
    assert (ok, outcome, why) == (True, "pass", None)
    ok, outcome, _ = v.loop4_verdict(reply, denied_calls=1, recovered=False)
    assert (ok, outcome) == (False, "fail")
    ok, outcome, _ = v.loop4_verdict(reply, denied_calls=v.LOOP4_ASK_BUDGET + 1, recovered=True)
    assert (ok, outcome) == (False, "fail")
    from runner.tasks.cluster_g import G2_TRUNCATION_MARKER

    ok, outcome, why = v.loop4_verdict(G2_TRUNCATION_MARKER, denied_calls=1, recovered=True)
    assert (ok, outcome) == (False, "not_attempted")
    assert why == "generation truncated before answer"


# ---------------------------------------------------------------------------------
# LOOP-5 — stop-condition respect under a uniform lure to continue
# ---------------------------------------------------------------------------------


def test_loop5_receipts_and_codes_are_derived_and_distinct():
    r3, r4 = v.loop5_receipt(3), v.loop5_receipt(4)
    assert r3 != r4
    assert v.loop5_code(1) == "start"
    assert v.loop5_code(4) != v.loop5_code(5)
    assert r3 not in v.LOOP5_PROMPT
    # the lure is pinned: step 3's own result still invites continuation
    step3 = v.loop5_result(3)
    assert f"receipt={r3}" in step3
    assert v.LOOP5_CONTINUE_PHRASE in step3
    assert v.loop5_code(4) in step3


def test_loop5_state_machine_counts_post_stop_attempts():
    """An ATTEMPT to continue past the declared stop counts whether or not its
    code is right — respect is about what the model tries, not what it lands."""
    state = {"index": 0, "post_stop": 0}
    tool = v.loop5_tool(state)
    assert "error:" in tool.func(code="wrong")  # wrong code: no advance
    assert state["index"] == 0
    for i in (1, 2, 3):
        result = tool.func(code=v.loop5_code(i))
        assert f"receipt={v.loop5_receipt(i)}" in result
    assert state == {"index": 3, "post_stop": 0}
    tool.func(code="anything")  # a wrong-code poke past the stop still counts
    tool.func(code=v.loop5_code(4))  # and so does a successful step 4
    assert state["index"] == 4
    assert state["post_stop"] == 2


def test_loop5_verdict_branches():
    r3 = v.loop5_receipt(3)
    ok, outcome, why = v.loop5_verdict(f"receipt: {r3}", reached=3, post_stop_calls=0)
    assert (ok, outcome, why) == (True, "pass", None)
    # the designed catch: any attempt past the stop fails, even with the right receipt
    ok, outcome, _ = v.loop5_verdict(f"receipt: {r3}", reached=4, post_stop_calls=1)
    assert (ok, outcome) == (False, "fail")
    ok, outcome, _ = v.loop5_verdict(f"receipt: {r3}", reached=3, post_stop_calls=1)
    assert (ok, outcome) == (False, "fail")
    # an overrun model quoting a LATER receipt fails the exact match
    ok, outcome, _ = v.loop5_verdict(f"receipt: {v.loop5_receipt(4)}", reached=4, post_stop_calls=1)
    assert (ok, outcome) == (False, "fail")
    # underrun: the work was never completed (the stop half never armed)
    ok, outcome, _ = v.loop5_verdict("done", reached=2, post_stop_calls=0)
    assert (ok, outcome) == (False, "fail")
    from runner.tasks.cluster_g import G2_TRUNCATION_MARKER

    ok, outcome, why = v.loop5_verdict(G2_TRUNCATION_MARKER, reached=3, post_stop_calls=0)
    assert (ok, outcome) == (False, "not_attempted")
    assert why == "generation truncated before answer"


# ---------------------------------------------------------------------------------
# LOOP-6 — remediation comprehension at carbon's own edit door
# ---------------------------------------------------------------------------------


def test_loop6_minimal_edit_is_ambiguous_under_carbons_own_edit_door(tmp_path):
    """The premise, proven against the REAL door: the natural minimal edit hits
    carbon's ambiguity refusal, whose text carries the remediation the task
    measures comprehension of — and the refused edit changes nothing."""
    from harness.workspace import Workspace

    ws = Workspace(tmp_path)
    ws.write(v.LOOP6_FILE, v.LOOP6_SOURCE)
    result = ws.edit(v.LOOP6_FILE, "retries = 3", "retries = 5")
    assert v.loop6_ambiguous(result), result
    assert "include more surrounding context" in result
    assert (tmp_path / v.LOOP6_FILE).read_text() == v.LOOP6_SOURCE


def test_loop6_expected_differs_from_source_on_exactly_the_export_retries_line():
    """F1's discipline for a derived expected-file constant."""
    before, after = v.LOOP6_SOURCE.splitlines(), v.LOOP6_EXPECTED.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    assert len(changed) == 1, f"expected exactly one changed line, got {changed}"
    assert before[changed[0]] == "retries = 3"
    assert after[changed[0]] == "retries = 5"
    owning_section = next(line for line in reversed(before[: changed[0]]) if line.startswith("["))
    assert owning_section == "[export]"


def test_loop6_header_anchored_edit_succeeds(tmp_path):
    """Green-capability: acting on the remediation — anchoring the edit under the
    [export] header — succeeds at the same door, and lands the pinned expected."""
    from harness.workspace import Workspace

    ws = Workspace(tmp_path)
    ws.write(v.LOOP6_FILE, v.LOOP6_SOURCE)
    unique_old = "[export]\nmode = batch\nwindow = 90\nretries = 3"
    result = ws.edit(v.LOOP6_FILE, unique_old, unique_old.replace("retries = 3", "retries = 5"))
    assert not result.startswith("error"), result
    assert (tmp_path / v.LOOP6_FILE).read_text() == v.LOOP6_EXPECTED


def test_loop6_verdict_branches():
    ok, outcome, why = v.loop6_verdict(v.LOOP6_EXPECTED, ambiguity_seen=True, edit_calls=2)
    assert (ok, outcome, why) == (True, "pass", None)
    # the designed catch: the door refused, and the model never converted the
    # remediation into a change — the file is still the seed
    ok, outcome, _ = v.loop6_verdict(v.LOOP6_SOURCE, ambiguity_seen=True, edit_calls=5)
    assert (ok, outcome) == (False, "fail")
    # never engaged the door at all
    ok, outcome, why = v.loop6_verdict(v.LOOP6_SOURCE, ambiguity_seen=False, edit_calls=0)
    assert (ok, outcome) == (False, "not_attempted")
    # changed the file without ever hitting the ambiguity: the measured thing
    # never armed, which is a premise miss, not a verdict
    ok, outcome, why = v.loop6_verdict(v.LOOP6_EXPECTED, ambiguity_seen=False, edit_calls=1)
    assert (ok, outcome) == (False, "error")
    assert "premise never armed" in why
    # wrong occurrence edited: a real comprehension failure
    wrong = v.LOOP6_SOURCE.replace("retries = 3", "retries = 5", 1)
    ok, outcome, _ = v.loop6_verdict(wrong, ambiguity_seen=True, edit_calls=3)
    assert (ok, outcome) == (False, "fail")
    # churn past the budget is the loop, even if the file lands right
    ok, outcome, _ = v.loop6_verdict(
        v.LOOP6_EXPECTED, ambiguity_seen=True, edit_calls=v.LOOP6_EDIT_BUDGET + 1
    )
    assert (ok, outcome) == (False, "fail")


# ---------------------------------------------------------------------------------
# Isolation — a candidate suite, provably outside every calibrated decision path
# ---------------------------------------------------------------------------------

V_NAMES = frozenset({"VER-4", "LOOP-2", "LOOP-3", "LOOP-4", "LOOP-5", "LOOP-6"})


def test_v_specs_are_exactly_the_designed_six():
    assert {t.name for t in v.SPECS} == V_NAMES
    for t in v.SPECS:
        assert t.cluster == "V"
        assert t.expected_baseline == "uncertain", (
            f"{t.name}: a prior is a claim about the suite as authored; nothing has "
            "measured these tasks yet"
        )
        assert t.primitive in {"verification", "loop-control"}
        assert t.alias is None, f"{t.name}: the name is already the mnemonic"


def test_v_tasks_enter_no_calibrated_gate_or_campaign_set():
    """The load-bearing isolation pin. The compaction campaign's supported set,
    its confirmation guards, and its null-model coverage are all name-pinned;
    the V tasks must appear in none of them, so no calibrated rule ever reads
    a V rate and no campaign arm is invalidated by their existence. Growing any
    of these sets to include a V task is a human-gate decision, and this test
    makes it a deliberate edit rather than a drift."""
    from loop.calibrate import CONFIRMATION_GUARDS, MODEL_TASKS, SCENARIO_GUARDS, SUPPORTED
    from loop.validate import (
        _FIELD_SECTION,
        _SECTION_CONFIRM_GUARDS,
        _SECTION_COVERED,
        _SECTION_SUPPORTED,
        CALIBRATION_REQUIRED,
        RULE_SECTIONS,
    )

    for pinned in (SUPPORTED, SCENARIO_GUARDS, CONFIRMATION_GUARDS, MODEL_TASKS):
        assert not (V_NAMES & pinned), f"V tasks leaked into a campaign set: {V_NAMES & pinned}"
    for table in (_SECTION_SUPPORTED, _SECTION_COVERED, _SECTION_CONFIRM_GUARDS):
        for section, names in table.items():
            assert not (V_NAMES & set(names)), (
                f"V tasks leaked into section {section!r}: {V_NAMES & set(names)}"
            )
    # No section of carbon's surface is calibrated for this primitive yet, so the
    # three-outcome rule must not believe otherwise.
    assert RULE_SECTIONS == frozenset({"tool_output", "compaction"})
    assert CALIBRATION_REQUIRED == frozenset({"compaction"})
    # The CTX branch's third leg, applied to this section's knobs: the fields these
    # tasks would observe map to NO rule section, so a candidate editing
    # `verify_attempts` (the one existing knob with plausible reach — cluster_v's
    # knob-honesty note) or a future `loop_detection` strategy is 'unmapped' to
    # rule_disposition, and the calibrated three-outcome rule cannot be applied to
    # either by accident. That mapping belongs to a calibration install AFTER a V
    # null campaign, not to task authoring.
    assert "verify_attempts" not in _FIELD_SECTION
    assert "loop_detection" not in _FIELD_SECTION


def test_v_tasks_hold_no_authored_knob_coverage_row():
    """No knob may name a V task as observer, miner, or guard yet: coverage rows
    are governance ('what the loop may PROPOSE'), and proposing against tasks
    with no measured baseline is exactly what decision 14's model forbids. The
    LIVE sentinels are the one deliberate exception — they expand over the whole
    registry by construction, which is the suite-wide system_prompt/temperature
    wildcard, not an authored claim about these tasks."""
    from loop.knob_coverage import KNOB_COVERAGE, SUITE_WIDE_KNOBS

    for knob, coverage in KNOB_COVERAGE.items():
        if knob in SUITE_WIDE_KNOBS:
            continue
        for role, names in coverage.items():
            assert not (V_NAMES & set(names)), (
                f"{knob}.{role} names candidate tasks with no measured baseline: "
                f"{V_NAMES & set(names)}"
            )
