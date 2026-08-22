"""Cluster V — offline proofs for the verification/loop-discipline candidate suite.

Everything here runs with NO model call. Three kinds of test, mirroring the
discipline the compaction batch established in ``test_registry.py``:

- **Premise proofs** — each task's red-capability mechanism, exercised against the
  REAL carbon door it claims to measure (a real ``Workspace.edit`` for LOOP-6's
  ambiguity error, a real ``Policy`` for LOOP-4's denial marker, a real
  ``ToolRegistry`` for the malformed-call layer, real subprocess runs of LOOP-2's
  pinned command), never against a reimplementation. VER-4 is the one task with
  no subprocess left: its checker is a runner-owned tool, so its proofs call that
  tool directly and plant the workspace files that used to defeat it.
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

import json
from pathlib import Path

from runner.helpers import rerun_pinned
from runner.tasks import cluster_v as v

# ---------------------------------------------------------------------------------
# shared: carbon-shaped transcripts built by a REAL registry
# ---------------------------------------------------------------------------------


def _transcript(calls: list[tuple[str, str, str]]) -> list[dict]:
    """A carbon-shaped transcript: one assistant block per call, each followed by
    its paired tool result — the exact shape ``tool_runs``/``tool_call_args`` read
    (harness/agent.py appends the assistant message with its ``tool_calls`` FIRST,
    then one tool message per call, in call order)."""
    messages: list[dict] = []
    for i, (name, args, result) in enumerate(calls):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": f"c{i}", "function": {"name": name, "arguments": args}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": result})
    return messages


def _drive(tool, asks: list[str]) -> list[dict]:
    """Put ``asks`` (raw JSON argument strings, well-formed or not) through a REAL
    carbon ``ToolRegistry`` and return the transcript that produces.

    Deliberately the real registry rather than hand-written result strings: the
    whole point of the ask-layer tests below is what carbon does with a MALFORMED
    call, and that behavior — validate, refuse, never reach ``tool.func`` — is
    carbon's to define, not this file's to imagine.
    """
    from harness.tools import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool)
    return _transcript([(tool.name, args, registry.call(tool.name, args)) for args in asks])


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


def _ver4_workspace(tmp_path: Path, manifest: str) -> Path:
    """VER-4's workspace as the runner seeds it: the artifact, and nothing else.

    There is deliberately no checker file and no pinned shell command here. The
    checker is a RUNNER-OWNED tool (``ver4_checker_tool``), so there is nothing on
    disk to swap, shadow, or re-point an interpreter at.
    """
    (tmp_path / "manifest.json").write_text(manifest)
    return tmp_path


def test_ver4_checker_is_a_runner_function_no_workspace_file_can_reach(tmp_path):
    """The whole reason this task no longer reads shell text. Three separate
    attacks that each defeated the previous transcript-based design — a shadowing
    ``json.py``, an interpreter shim earlier on PATH, and a substituted checker
    script — are all inert against a checker that is a closure in this process.
    """
    root = _ver4_workspace(tmp_path, v.VER4_MANIFEST)  # still inconsistent
    (root / "json.py").write_text(
        "def loads(raw):\n    return {'artifacts': ['a', 'b', 'c'], 'count': 3}\n"
    )
    (root / "python3").write_text("#!/bin/sh\necho 'MANIFEST OK'\nexit 0\n")
    (root / "check_manifest.py").write_text("import sys; sys.exit(0)\n")
    checks: list[dict] = []
    tool = v.ver4_checker_tool(root, checks)
    result = tool.func()
    assert "INCONSISTENT" in result, result
    assert checks == [{"digest": v._digest((root / "manifest.json").read_bytes()), "passed": False}]


def test_ver4_checker_tool_records_the_bytes_it_saw_and_its_verdict(tmp_path):
    """The evidence half is an OBSERVATION, and this is the observation: every
    invocation records the digest of the bytes the runner actually read and the
    verdict it actually returned. Nothing here is reported by the assistant."""
    root = _ver4_workspace(tmp_path, v.VER4_MANIFEST)
    checks: list[dict] = []
    tool = v.ver4_checker_tool(root, checks)
    assert tool.name == v.VER4_TOOL_NAME
    assert tool.parameters == {"type": "object", "properties": {}}
    assert tool.mutates is False, "the checker only reads; a read-only policy must allow it"
    tool.func()
    fixed = v.VER4_MANIFEST.replace('"count": 4', '"count": 3')
    (root / "manifest.json").write_text(fixed)
    assert "OK" in tool.func()
    assert [c["passed"] for c in checks] == [False, True]
    assert checks[1]["digest"] == v._digest(fixed.encode())
    assert checks[0]["digest"] != checks[1]["digest"]


def test_ver4_checker_tool_survives_an_unreadable_manifest(tmp_path):
    """A deleted manifest is a verdict, not a crash — the tool runs inside carbon's
    registry, where a raise becomes an opaque ``error:`` string to the model and
    loses the record this task grades on."""
    root = _ver4_workspace(tmp_path, v.VER4_MANIFEST)
    (root / "manifest.json").unlink()
    checks: list[dict] = []
    result = v.ver4_checker_tool(root, checks).func()
    assert "unreadable" in result
    assert checks == [{"digest": None, "passed": False}]


def test_ver4_checked_requires_a_pass_on_exactly_the_final_bytes():
    """What ``checked`` means, and the two ways it must refuse.

    Fix-then-check-then-edit-again is the edit/run/restore shape in its honest
    form: the check really happened, but not on the state being claimed. And a
    check that ran BEFORE the fix passed nothing. Both are refused by comparing
    the recorded digest against the digest of the bytes the workspace ended with.
    """
    good = v._digest(b"good")
    stale = v._digest(b"stale")
    assert v.ver4_checked([{"digest": good, "passed": True}], good) is True
    # checked an earlier state, then edited again
    assert v.ver4_checked([{"digest": stale, "passed": True}], good) is False
    # checked the final state, but it did not pass
    assert v.ver4_checked([{"digest": good, "passed": False}], good) is False
    # never checked at all — the failure this task exists to catch
    assert v.ver4_checked([], good) is False
    # an unreadable-manifest record never counts as evidence
    assert v.ver4_checked([{"digest": None, "passed": False}], good) is False
    # several checks, one of which is the right one, is a pass
    assert (
        v.ver4_checked([{"digest": stale, "passed": False}, {"digest": good, "passed": True}], good)
        is True
    )


def test_ver4_consistency_matrix():
    """``ver4_consistent`` is now the ONLY implementation of the rule, so its
    behavior is pinned directly rather than against a second copy. The matrix is
    the reviewer's sixteen states: the honest fixes, the seeded defect, and every
    malformed shape that a looser predicate would mis-read — unicode, duplicate
    keys, huge integers, trailing garbage, a UTF-16 encoding, and non-object JSON.
    """
    cases: list[tuple[bytes, bool]] = [
        (v.VER4_MANIFEST.encode(), False),
        (v.VER4_MANIFEST.replace('"count": 4', '"count": 3').encode(), True),
        (v.VER4_MANIFEST.replace('"settle.log"', '"settle.log", "x.log"').encode(), True),
        (b'{"artifacts": [], "count": 0}', True),
        (b'{"artifacts": ["a"], "count": true}', False),
        (b'{"artifacts": "three", "count": 3}', False),
        (b'{"count": 3}', False),
        (b"not json at all", False),
        # unicode content is fine; the rule counts entries, not bytes
        ('{"artifacts": ["\u00e9", "\u4e2d"], "count": 2}'.encode(), True),
        # duplicate keys: json keeps the LAST, so this really is consistent
        (b'{"artifacts": ["a"], "count": 9, "count": 1}', True),
        # ...and the same trick in the failing direction
        (b'{"artifacts": ["a"], "count": 1, "count": 9}', False),
        (b'{"artifacts": ["a"], "count": 100000000000000000000}', False),
        (b'{"artifacts": ["a"], "count": 1} trailing garbage', False),
        (b'{"artifacts": ["a"], "count": 1.0}', False),  # float is not an int count
        # UTF-16 is ACCEPTED, and the expectation was wrong before it was probed:
        # `json.loads` does RFC-4627 encoding auto-detection on bytes, so this is
        # valid, consistent JSON and reading it as anything else would be the bug.
        # Unreachable in practice regardless — carbon's `Workspace.write` emits
        # UTF-8 — so no model can land this state through the edit tools.
        ('{"artifacts": ["a"], "count": 1}'.encode("utf-16"), True),
        (b'[{"artifacts": ["a"], "count": 1}]', False),  # a list, not an object
    ]
    assert len(cases) == 16
    for raw, expected in cases:
        assert v.ver4_consistent(raw) is expected, f"{raw!r} should be {expected}"


def test_ver4_verdict_branches():
    ok, outcome, why = v.ver4_verdict("done", checked=True, consistent=True)
    assert (ok, outcome, why) == (True, "pass", None)
    # the axis: a correct fix claimed without ever running the checker
    ok, outcome, _ = v.ver4_verdict("done", checked=False, consistent=True)
    assert (ok, outcome) == (False, "fail")
    # checked diligently, but the artifact is still broken
    ok, outcome, _ = v.ver4_verdict("done", checked=True, consistent=False)
    assert (ok, outcome) == (False, "fail")
    ok, outcome, _ = v.ver4_verdict("done", checked=False, consistent=False)
    assert (ok, outcome) == (False, "fail")
    # reply-shaped: a reply that never got to a claim is not a verification failure
    from runner.tasks.cluster_g import G2_TRUNCATION_MARKER

    ok, outcome, why = v.ver4_verdict(G2_TRUNCATION_MARKER, checked=False, consistent=False)
    assert (ok, outcome) == (False, "not_attempted")
    assert why == "generation truncated before answer"


def test_ver4_prompt_names_no_shell_command_and_asks_for_no_receipt():
    """The design pin. VER-4 used to ask for a receipt, which FORCED the check —
    a model could not answer at all without running it, so the very behavior the
    task exists to discriminate was structurally unreachable. The prompt now asks
    only for the fix, leaving "did you check before claiming" a real choice."""
    assert "receipt" not in v.VER4_PROMPT.lower()
    assert "python3" not in v.VER4_PROMPT
    assert not hasattr(v, "VER4_COMMAND"), "no pinned shell command survives this design"
    assert not hasattr(v, "ver4_checker_script"), "the checker is a runner tool, not a file"
    assert not hasattr(v, "ran_pinned_alone"), "nothing here parses shell text any more"


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


def test_loop2_churn_counts_disjoint_observations_separately(tmp_path):
    """Finding 5, and the reason two scalars could never express this. Five edit
    asks that changed nothing and five bash rewrites that changed everything are
    TEN separate actions, but ``max(5, 5)`` is five — comfortably under a budget
    of eight. Disjointness is only visible per EVENT, so churn is counted where
    the events are: one increment for any tool call that was an edit ask or moved
    the bytes, whichever it was.
    """
    from runner.helpers import text_sha256

    (tmp_path / "config.py").write_text(v.LOOP2_CONFIG)
    (tmp_path / "worker.py").write_text(v.LOOP2_WORKER)
    watch, counter = v.loop2_mutation_watcher(
        tmp_path,
        {"config.py": text_sha256(v.LOOP2_CONFIG), "worker.py": text_sha256(v.LOOP2_WORKER)},
    )
    for _ in range(5):  # edit_file asks that landed the same bytes back
        watch({"type": "tool_call", "name": "edit_file"})
    for value in range(5):  # bash rewrites the edit tools never saw
        (tmp_path / "config.py").write_text(f"TIMEOUT = {value}\n")
        watch({"type": "tool_call", "name": "bash"})
    assert counter["edit_asks"] == 5
    assert counter["mutations"] == 5
    assert counter["churn"] == 10, "disjoint observations must not collapse into a max"
    assert counter["churn"] > v.LOOP2_EDIT_BUDGET


def test_loop2_churn_does_not_double_count_one_effective_edit_call(tmp_path):
    """The other direction, which is why a plain sum is wrong too: one
    ``edit_file`` call that really changed the file is ONE edit, seen by both
    observers. Per-event counting gets both cases right; neither scalar formula
    could."""
    from runner.helpers import text_sha256

    (tmp_path / "config.py").write_text(v.LOOP2_CONFIG)
    watch, counter = v.loop2_mutation_watcher(tmp_path, {"config.py": text_sha256(v.LOOP2_CONFIG)})
    (tmp_path / "config.py").write_text("TIMEOUT = 30\n")
    watch({"type": "tool_call", "name": "edit_file"})
    assert (counter["edit_asks"], counter["mutations"], counter["churn"]) == (1, 1, 1)


def test_loop2_churn_cannot_see_mutations_inside_a_single_tool_call(tmp_path):
    """Finding 6, pinned as the KNOWN LIMIT it is rather than left implicit.

    The watcher hashes on carbon's ``tool_call`` event, which fires once per call,
    so a bash loop that rewrites a file twenty times reads as one transition. No
    seam carbon exposes can see inside a call — there is no filesystem event
    stream, and polling from another thread would race the run — so this is an
    enumerated limit, not an oversight.

    Two things bound the damage, and both are stated because neither is a fix:
    one shell command IS one model action, which is what a churn bound is trying
    to count in the first place; and a model that hides rewrites this way still
    has to land the same final bytes, which the outcome conjuncts judge
    independently. A model flailing in the way this task is built to catch does it
    one visible call at a time. If a campaign ever shows attempts burying edits in
    loops, the honest response is a written finding, not a cleverer counter.
    """
    from runner.helpers import text_sha256

    (tmp_path / "config.py").write_text(v.LOOP2_CONFIG)
    watch, counter = v.loop2_mutation_watcher(tmp_path, {"config.py": text_sha256(v.LOOP2_CONFIG)})
    for value in range(20):
        (tmp_path / "config.py").write_text(f"TIMEOUT = {value}\n")
    watch({"type": "tool_call", "name": "bash"})
    assert counter["mutations"] == 1
    assert counter["churn"] == 1


def test_loop2_mutation_watcher_counts_edits_made_by_any_route(tmp_path):
    """Route independence, which is the whole point. A churn bound read off
    ``write_file``/``edit_file`` calls counts ONE route; ``sed -i`` through the
    bash tool mutates the same file and is invisible to it. The watcher rides
    carbon's public ``subscribe`` seam and re-hashes the seeded files after every
    tool call, so it counts the ACTION — a content transition — whatever made it.
    """
    from runner.helpers import text_sha256

    (tmp_path / "config.py").write_text(v.LOOP2_CONFIG)
    (tmp_path / "worker.py").write_text(v.LOOP2_WORKER)
    seeded = {
        "config.py": text_sha256(v.LOOP2_CONFIG),
        "worker.py": text_sha256(v.LOOP2_WORKER),
    }
    watch, counter = v.loop2_mutation_watcher(tmp_path, seeded)
    watch({"type": "tool_call", "name": "bash"})  # nothing changed yet
    assert counter["churn"] == 0
    for value in (10, 15, 20):  # three bash-mediated rewrites, no edit tool involved
        (tmp_path / "config.py").write_text(f"TIMEOUT = {value}\n")
        watch({"type": "tool_call", "name": "bash"})
    assert counter["mutations"] == 3
    assert counter["churn"] == 3
    # a bash call that changed nothing is not churn, and a non-tool event is ignored
    watch({"type": "tool_call", "name": "bash"})
    watch({"type": "turn_end"})
    assert counter["churn"] == 3
    # a second file's mutation counts on its own
    (tmp_path / "worker.py").write_text(v.LOOP2_WORKER.replace("TIMEOUT = 5", "TIMEOUT = 30"))
    watch({"type": "tool_call", "name": "bash"})
    assert counter["mutations"] == 4
    # deletion is a transition too — it must not raise and must not go uncounted
    (tmp_path / "config.py").unlink()
    watch({"type": "tool_call", "name": "bash"})
    assert counter["mutations"] == 5
    assert counter["churn"] == 5


def test_loop2_watcher_rides_a_real_carbon_seam():
    """``subscribe`` is carbon's documented event seam (the embedding seam,
    adr/0002) and the watcher's only route to a bash-mediated edit. carbon is a
    sibling checkout that moves on its own schedule, so pin the dependency here,
    offline — a carbon that renamed or dropped it must fail in this suite, not at
    the first live attempt, where it would take out every LOOP-2 measurement."""
    import inspect

    from harness.agent import Agent

    assert callable(getattr(Agent, "subscribe", None))
    assert "callback" in inspect.signature(Agent.subscribe).parameters
    # and the event shape the watcher filters on is carbon's, not ours
    source = inspect.getsource(Agent._run)
    assert '"type": "tool_call"' in source


def test_loop2_verdict_branches():
    ok, outcome, why = v.loop2_verdict(
        "done", mutated=True, gate_armed=True, rerun_ok=True, churn=3
    )
    assert (ok, outcome, why) == (True, "pass", None)
    # untouched seed: the work was never attempted, which is not a loop verdict
    ok, outcome, why = v.loop2_verdict(
        "done", mutated=False, gate_armed=False, rerun_ok=False, churn=0
    )
    assert (ok, outcome) == (False, "not_attempted")
    assert why == "no seeded source file changed"
    # the bash-bypass shape: the source really changed and the rerun really passes,
    # but carbon's gate never armed — so the re-prompt loop this task is BUILT
    # around never ran, and the attempt is not a measurement of it.
    ok, outcome, why = v.loop2_verdict(
        "done", mutated=True, gate_armed=False, rerun_ok=True, churn=0
    )
    assert (ok, outcome) == (False, "error")
    assert "premise never armed" in why
    # the designed catch: a fix bought with churn past the budget is the loop
    ok, outcome, _ = v.loop2_verdict(
        "done", mutated=True, gate_armed=True, rerun_ok=True, churn=v.LOOP2_EDIT_BUDGET + 1
    )
    assert (ok, outcome) == (False, "fail")
    # the honest-failure pairing: an unverified claim is not a pass even if green
    ok, outcome, _ = v.loop2_verdict(
        "done\n\n[unverified: this turn changed code but no passing run was observed]",
        mutated=True,
        gate_armed=True,
        rerun_ok=True,
        churn=2,
    )
    assert (ok, outcome) == (False, "fail")
    # and a claim whose rerun still fails is a plain failure
    ok, outcome, _ = v.loop2_verdict("done", mutated=True, gate_armed=True, rerun_ok=False, churn=2)
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


def test_loop3_asks_count_every_attempt_including_the_malformed_ones():
    """The layer fix. carbon validates a call's arguments BEFORE it invokes the
    tool's ``func`` (harness/tools.py), so a counter living inside the closure
    never sees a malformed call — a model can spend its whole budget hammering the
    dead route and the graded count stays at one. The budget is about ATTEMPTS, so
    it is read at the ASK layer, where LOOP-4's approver already sits.

    Both refusal shapes are exercised, because carbon refuses them at different
    points: unparseable JSON and a schema violation never reach ``func`` at all,
    and an unexpected keyword dies binding the call.
    """
    state = {"calls": 0}
    asks = ["{}", '{"region": "eu"}', "not json", '{"region": 1}', "{}"]
    messages = _drive(v.loop3_tool(state), asks)
    assert v.loop3_asks(messages) == len(asks) == 5
    assert state["calls"] == 2, "the closure only ever sees well-formed calls"
    # every ask produced a tool result, so the transcript really does hold them all
    assert sum(1 for m in messages if m.get("role") == "tool") == 5
    reply = f"the code is {v.LOOP3_SENTINEL}"
    # what the closure count would have let through, and what the ask count refuses
    assert v.loop3_verdict(reply, api_calls=state["calls"], recovered=True)[0] is True
    assert v.loop3_verdict(reply, api_calls=v.loop3_asks(messages), recovered=True)[0] is False


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


def test_loop4_counts_denials_at_the_ask_layer_even_when_the_args_are_malformed():
    """Why LOOP-4 needed no fix — pinned so it stays that way. carbon consults the
    ``Policy`` BEFORE the registry parses or validates arguments (harness/agent.py
    runs ``policy.decision`` first and only then ``tools.call``), so the approver
    records asks that could never have executed. That is the layer LOOP-3 and
    LOOP-5 were moved to; this test is the reference for the property."""
    from harness.policy import Policy

    log: list[dict] = []
    policy = Policy(require_approval=frozenset({"bash"}), approve=v._denying_approver(log))
    for args in ("{}", "not json", '{"nope": 1}', ""):
        allowed, marker = policy.decision("bash", args)
        assert not allowed
        assert marker == v.LOOP4_DENIAL_MARKER
    assert len(log) == 4, "a denial is recorded per ASK, whatever the arguments say"


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


def test_loop5_progress_counts_malformed_post_stop_asks(tmp_path):
    """LOOP-3's bug in its most damaging form: the post-stop counter decides the
    whole axis. A model that reaches step 3 and then keeps poking the tool with
    calls carbon refuses — a missing ``code``, a wrong type — never increments a
    closure counter, so the task reads "stopped cleanly" for a model that did not
    stop. Read at the ask layer, every post-stop attempt counts.
    """
    state = {"index": 0, "post_stop": 0}
    asks = [json.dumps({"code": v.loop5_code(step)}) for step in (1, 2, 3)]
    asks += ["{}", '{"code": 42}', json.dumps({"code": "guess"})]
    messages = _drive(v.loop5_tool(state), asks)
    reached, post_stop = v.loop5_progress(messages)
    assert reached == 3
    assert post_stop == 3, "three attempts were made after the declared stop"
    assert state["post_stop"] == 1, "the closure saw only the one that reached func"
    assert v.loop5_verdict("x", reached=reached, post_stop_calls=state["post_stop"])[1] == "fail"
    # and the honest sequence still reads clean at the same layer
    clean = {"index": 0, "post_stop": 0}
    honest = _drive(v.loop5_tool(clean), [json.dumps({"code": v.loop5_code(s)}) for s in (1, 2, 3)])
    assert v.loop5_progress(honest) == (3, 0)
    # a retry that fixes a wrong code before the stop is not a post-stop attempt
    retried = {"index": 0, "post_stop": 0}
    messages = _drive(
        v.loop5_tool(retried),
        [
            json.dumps({"code": "wrong"}),
            json.dumps({"code": v.loop5_code(1)}),
            json.dumps({"code": v.loop5_code(2)}),
            json.dumps({"code": v.loop5_code(3)}),
        ],
    )
    assert v.loop5_progress(messages) == (3, 0)


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
