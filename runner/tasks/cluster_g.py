"""Cluster G — incomplete answers, repeated compaction, and workspace inheritance.

The compaction half of this module now covers three axes a fix aimed at "remember
the recent thing" could otherwise overfit straight past, and each is a separate
task rather than a variant of G2/G4 because each fails for a different reason:

- CMP-5 (supersession): the ask is two ROLES, approved and retired, and the
  retirement fact is stated only in the superseding message — so a strategy that
  keeps the early decision and drops that message cannot answer it at all.
- CMP-6 (meaning): the fact carries no sentinel at all, so nothing can be matched
  on; the verdict comes from the pinned judge, and refuses outright when that
  judge has no committed validation behind it.
- CMP-7 (buried facts): the fact competes with bulky tool output at carbon's
  DEFAULT window, which is the shape a real coding session has.
"""

from __future__ import annotations

import re

from runner.carbon_env import make_provider
from runner.helpers import (
    agent_metrics,
    neutral_dir,
    scripted_approver,
    tool_call_args,
    tool_texts,
    workspace_kwargs,
)
from runner.judge import judged_equivalent, validation_status
from runner.spec import Attempt, TaskSpec

G1_SENTINEL = "END-OF-HANDOFF-G1-8VR"
G2_FACT_A = "EARLY-DECISION-G2-3LK"
G2_FACT_B = "LATE-DECISION-G2-6QW"
G3_SENTINEL = "WORKTREE-STATE-G3-5TZ"

# G4 is the HELD-IN repeated-compaction miner. G2 measures the same door but is
# held-out, and mining a held-out task spends the generalization claim it exists to
# make — the coverage table named G2 a miner, which was that rule contradicting
# itself. G4 takes the mining role; G2 keeps the guarding one.
#
# Structurally different from G2 on purpose, or the "guard" is just a second run of
# the miner: G2 carries two homogeneous decision codes and asks for both at the end,
# while G4 carries three HETEROGENEOUS kinds of harness state and a deeper window.
# The kinds are chosen to be the ones a summarizer paraphrases away first, and are
# exactly what the reference harness tracks deterministically rather than entrusting
# to summary prose — modified files, a rejected approach, and the pending next action.
G4_FILES = ("services/ledger/reconcile.py", "services/ledger/schema_v3.sql")
G4_REJECTED = "REJECTED-APPROACH-G4-2WD"
G4_NEXT = "NEXT-ACTION-G4-5HB"

# G5's files are named, not sentinel-coded: the task asks the agent what it DID, and a
# realistic path is what a real session would carry. Distinct from G4's paths so a
# checkpoint carrying one cannot be mistaken for carrying the other.
G5_FILES = ("reconcile_flags.py", "audit_flags.py")

# --- CMP-5: supersession (contract §1, redesigned per amendment 1) ---------------
# The ask is TWO ROLES, not one code, and that is what gives the task a direction it
# can actually fail. The retirement fact — that OLD-PATH is the retired approach —
# is stated in exactly ONE message, the supersession. A strategy biased toward the
# earliest content keeps the early decision, loses that message, and then cannot fill
# the `retired=` role at all.
#
# Stated honestly, because the first version of this claimed more: CMP-5 does NOT
# cover the recency direction. A strategy biased toward the LATEST content keeps the
# supersession, which names both codes in both roles, and answers correctly. G4/G5
# cover that direction — their facts are stated early and never restated.
CMP5_CURRENT = "CURRENT-PATH-C5-4RQ"
CMP5_RETIRED = "OLD-PATH-C5-9ZK"
CMP5_EARLY = (
    f"Decision: reconciliation uses approach {CMP5_CURRENT}... wait, correction "
    "pending review. Acknowledge briefly."
)
CMP5_SUPERSESSION = (
    f"Final decision: approach {CMP5_RETIRED} is retired; the approved approach is "
    f"{CMP5_CURRENT}. Acknowledge briefly."
)
CMP5_QUESTION = (
    "Which reconciliation approach code is approved, and which is retired? "
    "Reply exactly in the form: approved=<code> retired=<code>"
)
# Mechanical parse of the two roles. Values are taken up to whitespace and then
# stripped of surrounding punctuation/quoting, so `approved=`CODE`,` parses to CODE —
# a model that answers correctly inside ordinary formatting is answering correctly,
# and grading the formatting would measure obedience instead of compaction.
_CMP5_ROLE_RE = re.compile(r"\b(approved|retired)\s*[=:]\s*([^\s,;]+)", re.IGNORECASE)
_CMP5_TRIM = "`'\"*.,;:()[]<>"

# --- CMP-6: semantic preservation, judged (contract §2) -------------------------
# No sentinel anywhere, deliberately: the fact is ordinary prose with a number and
# a reason, so the only way to answer is to have kept its MEANING. `CMP6_EXPECTED`
# is the pinned statement the judge compares an answer against — it never sees the
# transcript or this task's instructions (runner/judge.py).
CMP6_FACT = (
    "Constraint: retry backoff must never exceed 30 seconds, because the upstream "
    "gateway times out at 45 and we need headroom for one full retry before it "
    "fires. Acknowledge briefly."
)
CMP6_EXPECTED = (
    "Retry backoff must never exceed 30 seconds, because the upstream gateway times "
    "out at 45 seconds and we need headroom for one full retry before it fires."
)
CMP6_QUESTION = "What is the constraint on retry backoff, and why?"

# --- CMP-7: buried facts (contract §3) ------------------------------------------
CMP7_SENTINEL = "INCIDENT-C7-7TN"
CMP7_SCRIPT = "service_log.py"
# Tuned so one run emits ~3,000 characters: bulky enough to dominate the turn it
# arrives in, and comfortably under carbon's shipped `tool_output` budget so the
# result lands INTACT. A fixture large enough to be cut at the door would make this
# a tool_output task wearing a compaction task's name.
CMP7_NOISE_LINES = 26
# What one run of that fixture puts in front of the model, MEASURED: carbon's bash tool
# returns `(stdout + stderr).strip()` (harness/sandbox.py), so this is the COMBINED
# length, 39 characters of summary on stdout and the rest of the noise on stderr. The
# setup's hard cap is derived from it, so it may not drift: a fixture edit that changes
# the noise size has to change this number in the same commit or the test goes red.
CMP7_NOISE_CHARS = 2964
# How much slack the derived cap carries over the arithmetic minimum. Two full windows
# of noise is what the premise NEEDS; the factor covers what the arithmetic cannot see —
# the summary and retained tail a compaction leaves behind, a model that answers a bulk
# turn without calling the tool, per-turn prose variation. Generous on purpose, and
# still finite: an attempt that cannot compact twice must end as `error`, not run on.
CMP7_TURN_SLACK = 3
# How many turns must have been observed before the loop trusts its own fill rate. One
# turn's growth is a sample of the model's verbosity; several are a sample of the
# fixture's weight.
CMP7_MIN_OBSERVED_TURNS = 4
CMP7_RUN_AND_NOTE = (
    f"Run `python3 {CMP7_SCRIPT} 1` with the bash tool and tell me the last line it "
    f"printed. Note for later: the incident ticket is {CMP7_SENTINEL}."
)
CMP7_QUESTION = "What is the incident ticket? Code only."


def _plain_agent(**kwargs):
    from harness.agent import DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer

    provider = make_provider()
    kwargs.setdefault("tracer", Tracer(model=provider.model))
    return Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        agents_dir=neutral_dir(),
        **kwargs,
    )


def run_g1() -> Attempt:
    a = _plain_agent()
    try:
        result = a.run(
            "Produce a plain-text incident handoff with exactly 400 numbered lines. "
            "Each line must use the form `N: system stable`, starting at 1. After line "
            f"400, put {G1_SENTINEL} on its own final line. Do not use the sentinel earlier."
        )
        reply = result.text
        numbered = sum(
            1
            for line in reply.splitlines()
            if line.partition(":")[0].strip().isdigit() and "system stable" in line
        )
        sentinel_once = reply.count(G1_SENTINEL) == 1
        sentinel_last = reply.rstrip().endswith(G1_SENTINEL)
        ok = numbered == 400 and sentinel_once and sentinel_last
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"numbered_lines={numbered} sentinel_once={sentinel_once} "
        f"sentinel_last={sentinel_last} reply_chars={len(reply)}",
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


def _filler(i: int) -> str:
    notes = " ".join(
        f"checkpoint {i}.{j} nominal, latency stable, no owner change, review pending."
        for j in range(10)
    )
    return f"Operational notes: {notes} Acknowledge in one short sentence."


# carbon's own marker for a generation cut off at the output limit — carbon appends
# it to whatever prose was produced (harness/agent.py, the `finish_reason == "length"`
# branch). It is a LITERAL there, not a constant this could import, and a carbon change
# is out of scope for this phase, so the copy is pinned against carbon's own module
# source by `tests/test_registry.py` instead: a reword goes red rather than silently
# reclassifying every truncated generation back into ordinary compaction failures.
#
# Of G2's 95 recorded round-2 attempts, 35 replies are this marker and nothing else —
# the prose was EMPTY, so the attempt never reached an answer at all — beside 21 plain
# failures, 4 tool-syntax leaks and 35 passes.
G2_TRUNCATION_MARKER = "[incomplete: the model reached its output limit before finishing]"
# Tool-call syntax leaking into a prose turn (4 of the same 95). The model emitted
# a call the harness never executed instead of answering — again not a statement
# about what compaction carried.
_TOOL_SYNTAX_RE = re.compile(r"<\|tool_call")

# The taxonomy's labels, as constants: G2, CMP-5 and CMP-6 all classify the same
# two non-answer shapes, and an analysis counting `non_answer=` reasons across
# tasks needs ONE spelling of each. Three inline copies would drift apart the
# first time one was reworded.
NON_ANSWER_TRUNCATED = "generation truncated before answer"
NON_ANSWER_TOOL_SYNTAX = "tool-syntax leak instead of answer"


def classify_non_answer(reply: str) -> str | None:
    """The shared non-answer classification (contract §5's taxonomy): the label, or
    None for a reply that did attempt an answer.

    Deterministic and FIRST in every verifier that uses it — a reply that never
    attempted the answer is not evidence about what compaction carried, and no
    later layer (mechanical parse, judge) should be consulted about it. The
    starts-with rule for truncation is deliberately strict, exactly as G2 pinned
    it: prose that produced an answer and THEN hit the output limit did attempt,
    and stays a plain fail.
    """
    if reply.lstrip().startswith(G2_TRUNCATION_MARKER):
        return NON_ANSWER_TRUNCATED
    if _TOOL_SYNTAX_RE.search(reply):
        return NON_ANSWER_TOOL_SYNTAX
    return None


def g2_verdict(reply: str, early: bool, late: bool) -> tuple[bool, str, str | None]:
    """G2's whole verdict: ``(passed, outcome, non_answer_detail)``.

    Contract §5's decomposition, ADDITIVE by construction. ``passed`` is exactly
    what it always was — both codes recalled — and every failing branch returns
    False, so the pass fraction this task contributes is byte-identical to what the
    same replies produced before (``tests/test_registry.py`` replays all 95 recorded
    replies through this function to prove it, rather than arguing it).

    What changes is only WHY a failure is recorded. Two shapes were being counted
    as "compaction lost the codes" when they are nothing of the kind:

    - a reply that is nothing but carbon's truncation marker — the generation was
      cut off before it produced anything, which is a harness defect G1 owns the
      primitive for;
    - a reply that is a tool-call fragment — the model tried to call a tool instead
      of answering.

    Both become ``not_attempted``. The starts-with test is deliberately strict: a
    reply that produced real prose AND THEN hit the limit did attempt an answer, and
    stays a plain ``fail``.

    Exclusion from the denominator was considered and rejected: it would break
    attempt parity between arms (``runner.delta`` refuses mismatched attempt
    counts), the truncations are real defects rather than measurement noise, and the
    calibrated rule already carries attribution. So the count stays, and the outcome
    says what it was.
    """
    if early and late:
        return True, "pass", None
    non_answer = classify_non_answer(reply)
    if non_answer:
        return False, "not_attempted", non_answer
    return False, "fail", None


def run_g2() -> Attempt:
    a = _plain_agent(context_limit=700)
    try:
        a.send("We are beginning a release review. Acknowledge briefly.")
        a.send(f"Record this exact early decision for later: {G2_FACT_A}. Acknowledge briefly.")
        compactions = int(a.just_compacted)
        a.send(_filler(1))
        compactions += int(a.just_compacted)
        a.send(f"Record this exact later decision for later: {G2_FACT_B}. Acknowledge briefly.")
        compactions += int(a.just_compacted)
        for i in range(2, 14):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
            if compactions >= 2:
                break
        if compactions < 2:
            return Attempt(
                False,
                "error",
                f"repeated-compaction setup did not fire twice (count={compactions})",
                turns=len(a.messages),
                # `attempted` is published on EVERY exit, this one included: a metric
                # absent from some attempts is averaged over only the attempts that
                # reported it (runner/suite.py `_collect`), so omitting it here would
                # make the mean a fraction of the attempts that got far enough rather
                # than of the attempts the task made.
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        result = a.run(
            "Reply with the two exact release decision codes, early first and late second."
        )
        reply = result.text
        has_a = G2_FACT_A.lower() in reply.lower()
        has_b = G2_FACT_B.lower() in reply.lower()
        ok, outcome, non_answer = g2_verdict(reply, has_a, has_b)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"compactions={compactions} early_recalled={has_a} "
        f"late_recalled={has_b} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        # `reply=` stays LAST: `loop.judge_validate._extract_reply` reads it with a
        # greedy match to end-of-string, so any field added after it would be swallowed
        # into the reply the judge corpus is built from.
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={**agent_metrics(a, result=result), "attempted": float(outcome != "not_attempted")},
    )


def run_g4() -> Attempt:
    """Heterogeneous harness state must survive REPEATED compaction, not just facts.

    Three kinds of state go in, spread across the trajectory rather than clustered:
    the set of modified files, an approach already tried and rejected, and the pending
    next action. All three are the sort of thing a general "be terse but lose nothing"
    summarizer drops or paraphrases first — a file list becomes "some files", a
    rejected approach becomes "explored options" — and all three are load-bearing for
    the next turn.

    Graded per property, not as one boolean, so a partial retention is visible in the
    detail string and the failure says WHICH kind of state was lost.
    """
    a = _plain_agent(context_limit=900)
    try:
        a.send("We are mid-migration on the ledger service. Acknowledge briefly.")
        a.send(
            f"Record the files changed so far, exactly: {G4_FILES[0]} and {G4_FILES[1]}. "
            "Acknowledge briefly."
        )
        compactions = int(a.just_compacted)
        for i in range(1, 4):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
        a.send(
            f"Record that this approach was tried and rejected: {G4_REJECTED} "
            "(it deadlocked under concurrent writes). Acknowledge briefly."
        )
        compactions += int(a.just_compacted)
        for i in range(4, 8):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
        a.send(f"Record the pending next action, exactly: {G4_NEXT}. Acknowledge briefly.")
        compactions += int(a.just_compacted)
        for i in range(8, 18):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
            if compactions >= 2:
                break
        if compactions < 2:
            return Attempt(
                False,
                "error",
                f"repeated-compaction setup did not fire twice (count={compactions})",
                turns=len(a.messages),
                metrics=agent_metrics(a),
            )
        result = a.run(
            "Report three things exactly as recorded: the files changed so far, the "
            "approach that was tried and rejected, and the pending next action."
        )
        reply = result.text
        low = reply.lower()
        has_files = all(f.lower() in low for f in G4_FILES)
        has_rejected = G4_REJECTED.lower() in low
        has_next = G4_NEXT.lower() in low
        ok = has_files and has_rejected and has_next
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compactions={compactions} files_recalled={has_files} "
        f"rejected_recalled={has_rejected} next_action_recalled={has_next} "
        f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


def run_g5() -> Attempt:
    """Which files did I actually change? — asked after repeated compaction.

    G4's sibling, and deliberately not a duplicate of it. G4 STATES its facts in
    conversation, so what it measures is whether a summarizer keeps a stated fact. G5
    never states anything: the agent calls ``write_file`` itself, and the file list
    exists only as a property of the tool calls it made. That is the difference between
    "did the summary remember" and "does the harness know", and only the second is
    something the harness can guarantee rather than hope for.

    This is the observer for deterministic file tracking. A strategy that extracts the
    file list from tool calls and re-attaches it verbatim passes here even when the
    summarizer's prose drops it; a strategy that entrusts the list to that prose does
    not. Measured on ``write_file`` because an edit needs matching text to already
    exist, which adds a failure mode that has nothing to do with compaction.

    Setup failures are reported as ``error``, never ``fail``: if the model never made
    the writes, the task never reached the thing it measures.
    """
    from harness.agent import APPROVAL_TOOLS
    from harness.tools import ToolRegistry
    from harness.workspace import Workspace, write_file_tool

    ws = Workspace()
    approvals = [{"tool": "write_file", "decision": "approve"}] * 12
    a = _plain_agent(
        context_limit=900,
        approve=scripted_approver(approvals),
        approval_required=APPROVAL_TOOLS,
    )
    tools = ToolRegistry()
    tools.register(write_file_tool(ws))
    a.tools = tools

    observed: list[str] = []

    def _write(path: str, line: str) -> int:
        """Ask for one file, re-prompting once if the model answered instead of acting.

        A local model sometimes replies "I have created it" without calling anything.
        One retry keeps that from being recorded as a compaction failure — the task
        measures what survives the door, and it cannot measure it if the writes that
        produce the state never happened. Still bounded: two asks, then the setup
        guard reports `error`.

        The observation is recorded HERE, as the call happens, and never re-derived
        from the final transcript. That is not a style preference: the first version
        scanned `a.messages` at the end and saw only the second file, because
        compaction had already summarized the first write's tool call away. Reading
        setup state out of a compacted transcript measures the door, not the setup —
        and it is the same blindness this task exists to expose.
        """
        fired = 0
        for _ in range(2):
            before = len(tool_call_args(a.messages, ("write_file",)))
            a.send(
                f"Create a file named {path} containing exactly the line: {line}. "
                "Use the write_file tool. Do not describe the file — write it."
            )
            fired += int(a.just_compacted)
            new = tool_call_args(a.messages, ("write_file",))[before:]
            if any(path in args for args in new):
                observed.append(path)
                break
        return fired

    try:
        compactions = _write(G5_FILES[0], "reconcile = True")
        for i in range(1, 4):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
        compactions += _write(G5_FILES[1], "audit = True")
        for i in range(4, 18):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
            if compactions >= 2:
                break

        made_both = set(observed) == set(G5_FILES)
        if compactions < 2 or not made_both:
            return Attempt(
                False,
                "error",
                f"setup incomplete: compactions={compactions} observed={observed}",
                approvals=approvals,
                turns=len(a.messages),
                metrics=agent_metrics(a),
            )
        result = a.run("List every file you have created or modified in this session.")
        reply = result.text
        low = reply.lower()
        recalled = [f for f in G5_FILES if f.lower() in low]
        ok = len(recalled) == len(G5_FILES)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compactions={compactions} observed={observed} "
        f"recalled={recalled} reply={reply[:240]!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


def run_g3() -> Attempt:
    """A delegated worker must inspect the same workspace as its parent."""
    from harness.subagents import delegate_tool
    from harness.tools import ToolRegistry, read_file_tool
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("worker-state.txt", f"current-state={G3_SENTINEL}\n")
    provider = make_provider()
    a = _plain_agent()
    worker_tools = ToolRegistry()
    # scratch_root is the PARENT's own session scratch — carbon's own
    # _coding_tools.worker_tools() threads the same value for the same reason
    # (harness/agent.py): a worker asked to read back a scratch:// ref the
    # PARENT's own truncation door produced must resolve it against the scratch
    # the parent actually owns, not an empty/absent one.
    worker_tools.register(read_file_tool(str(ws.root), scratch_root=a.session_env.scratch_root))
    tools = ToolRegistry()
    # The parent binds the worker's tools to its own workspace through
    # delegate_tool(tools=...). That is a CODE seam in carbon (harness/subagents.py),
    # not part of the editable config surface — refinery measures it but cannot tune
    # it, which is the whole grader/graded distinction. The seam exists today, so the
    # prior is `pass` and this task proves the worker reads the intended snapshot
    # rather than a fresh, empty workspace of its own.
    tools.register(delegate_tool(model=provider.model, tools=worker_tools))
    a.tools = tools
    try:
        result = a.run(
            "Delegate this exact subtask to a worker: read worker-state.txt and return its "
            "current-state value exactly. Then reply with just that value."
        )
        reply = result.text
        delegated_results = tool_texts(a.messages)
        worker_saw_state = any(G3_SENTINEL in text for text in delegated_results)
        recalled = G3_SENTINEL.lower() in reply.lower()
        ok = worker_saw_state and recalled
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"delegate_called={bool(delegated_results)} worker_saw_state={worker_saw_state} "
        f"reply_recalled={recalled} reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


def cmp5_verdict(reply: str) -> tuple[bool, str | None, str | None]:
    """``(passed, approved_code, retired_code)`` — contract amendment 1's verifier.

    Both roles, mechanically parsed, both exactly right. The two-role form is what
    replaced the original one-sided check, which a review showed passed BOTH bias
    directions trivially: "reply with the approved code only" is answered by any
    strategy that kept either message, since the approved code appears in both.

    Asking which is retired cannot be answered from the early message at all, and
    swapping the roles fails — a reply that knows the codes but not which one is live
    has not carried the decision, which is the thing the next turn would act on.

    Unparsed roles come back as ``None`` (recorded in the detail, never silently
    treated as a wrong code): "the model did not answer in the form" and "the model
    named the wrong approach" are different facts about a strategy.
    """
    found: dict[str, str] = {}
    for role, value in _CMP5_ROLE_RE.findall(reply):
        found.setdefault(role.lower(), value.strip(_CMP5_TRIM))
    approved = found.get("approved")
    retired = found.get("retired")
    ok = (approved or "").upper() == CMP5_CURRENT and (retired or "").upper() == CMP5_RETIRED
    return ok, approved, retired


def cmp5_outcome(reply: str) -> tuple[bool, str, str | None]:
    """CMP-5's whole verdict: ``(passed, outcome, non_answer_detail)``.

    Contract §5's principle, which is not G2's alone: a reply that never attempted the
    answer must not be recorded as a compaction failure. CMP-5 asks for a two-role form,
    and a reply that ignores it entirely — including one whose PROSE is correct — says
    nothing about what compaction carried. It is ``not_attempted``.

    This matters more here than it would on a miner, because CMP-5's pooled rate would
    become its own GATE: a drop past its null quantile would REJECT a candidate, and
    formatting outcomes in that denominator make such a gate a measurement of obedience.

    WHICH IS WHAT THEY STILL ARE, corrected here because the paragraph above once read
    as if this classification fixed it. It does not. ``not_attempted`` is a LABEL on the
    record; ``TaskResult.pass_fraction`` (runner/run.py) divides passes by EVERY
    recorded attempt, so a non-answer is a non-pass in the published rate exactly like a
    wrong answer. What the label buys is that the two are TELLABLE APART afterwards --
    which is how the 29%/74% decomposition in ``run_cmp5`` was recoverable at all. It
    buys nothing for the gate, and until the Phase 3 rework recorded in ``run_cmp5``
    lands, this task does not gate candidates.

    The line is drawn at the parse, and only there. Neither role parsed means the reply
    never entered the form; once EITHER role parses, the model answered, and a wrong or
    missing code is a real failure, exactly as before.

    Known consequence, stated rather than hidden: an explicit denial ("I don't have
    those codes") parses no roles either, so it lands in ``not_attempted`` even though it
    IS a compaction failure in substance. That is a deliberate simplification of the rule
    as specified — the alternative is a denial-detection heuristic
    (``loop/judge_validate`` has one) inside a mechanical verifier, which is a different
    decision from this one. The direction is conservative for the gate: denials leave the
    pooled rate rather than depressing it, so the guard cannot fire on them.
    """
    ok, approved, retired = cmp5_verdict(reply)
    if ok:
        return True, "pass", None
    if approved is None and retired is None:
        return False, "not_attempted", "did not answer in the requested form"
    return False, "fail", None


def cmp5_supersession_pending(messages: list[dict]) -> bool:
    """Is the supersession message still sitting in the live transcript, verbatim?

    The premise this task needs is that the supersession has been THROUGH the
    compaction door, and neither a turn count nor a compaction count can establish
    that. Two carbon facts, not opinions:

    - ``Agent.run`` compacts BEFORE appending the turn's own message
      (harness/agent.py), so ``just_compacted`` read right after sending the
      supersession reports a compaction that ran when that message did not yet exist;
    - ``keep_tail`` carries the newest messages through verbatim, so the supersession
      survives the next compactions untouched while a counter happily credits them.

    A review found the first version of this task crediting exactly those. So the
    task observes the transcript instead: filler continues while this returns True.
    Config-robust by construction — it holds for any ``keep_tail``,
    ``trigger_fraction`` or window, and it is the message OBJECT that must leave, so
    a checkpoint that quotes the text verbatim still counts as having processed it.
    """
    return any(m.get("role") == "user" and m.get("content") == CMP5_SUPERSESSION for m in messages)


def run_cmp5() -> Attempt:
    """Which decision is live, and which was retired — asked after the supersession
    has itself gone through the compaction door.

    G2 and G4 ask whether a stated fact SURVIVES. This one asks whether the STATUS of
    two competing decisions survives: an early decision, later explicitly retired, and
    the approved one that replaced it. A summarizer that keeps "the decisions
    discussed" without their status passes G2 and fails here.

    The direction it genuinely covers is the earliest-bias one. The retirement fact is
    stated only in the supersession message, so a strategy that keeps the early
    decision and drops that message cannot fill the ``retired=`` role at all. It does
    NOT cover the recency direction — the supersession names both codes in both roles,
    so a latest-content bias answers correctly; G4/G5 carry that direction.

    The premise is enforced by OBSERVATION (``cmp5_supersession_pending``), not by
    counting compactions: carbon compacts before appending the turn's own message and
    ``keep_tail`` carries recent messages through verbatim, so a count credits
    compactions that never saw the supersession. Filler continues until the message
    has actually left the transcript; if it never leaves, the attempt is ``error`` —
    the task never reached the thing it measures.

    WHAT THIS TASK'S RATE ACTUALLY MEASURES, stated plainly because an earlier version
    of this docstring implied otherwise. ``not_attempted`` is a LABEL and nothing more:
    ``TaskResult.pass_fraction`` (runner/run.py) is passes over EVERY recorded attempt,
    so a non-answer stays in the denominator exactly like a wrong answer. The
    classification separates the two in the record; it does not protect the rate.

    So the published rate is a PRODUCT of two unrelated things. Over the ten committed
    Phase 2c arms: 23 of 79 attempts answered in the requested two-role form (29%), and
    17 of those 23 named both codes correctly (74%). (23/79) x (17/23) is exactly the
    17/79 the null model carries. Most of the movement available to this "guard" is
    movement in template obedience, not in what compaction carried.

    RECORDED PHASE 3 ITEM: format-robustness rework — replace the regex role parse with
    judge-based answer extraction (the same judge seam CMP-6 already uses), so the
    verdict reads what the reply MEANS rather than whether it matched a template. Until
    that lands, this task reports; it does not gate candidates.
    """
    a = _plain_agent(context_limit=900)
    try:
        a.send("We are beginning a reconciliation review. Acknowledge briefly.")
        a.send(CMP5_EARLY)
        compactions = int(a.just_compacted)
        for i in range(1, 4):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
        a.send(CMP5_SUPERSESSION)
        compactions += int(a.just_compacted)
        for i in range(4, 24):
            if not cmp5_supersession_pending(a.messages) and compactions >= 2:
                break
            a.send(_filler(i))
            compactions += int(a.just_compacted)
        superseded = not cmp5_supersession_pending(a.messages)
        if not superseded or compactions < 2:
            return Attempt(
                False,
                "error",
                f"supersession never left the live transcript (compactions={compactions} "
                f"still_verbatim={not superseded})",
                turns=len(a.messages),
                # Published on EVERY exit, this one included — see `run_g2` for why a
                # metric that skips an exit reports a mean over a shrunken denominator.
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        result = a.run(CMP5_QUESTION)
        reply = result.text
        approved, retired = cmp5_verdict(reply)[1:]
        ok, outcome, non_answer = cmp5_outcome(reply)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"compactions={compactions} approved_answer={approved!r} "
        f"retired_answer={retired!r} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        # `reply=` stays LAST, the same convention run_g2 keeps: a trailing-match
        # extractor reads everything after it as the reply.
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={**agent_metrics(a, result=result), "attempted": float(outcome != "not_attempted")},
    )


def run_cmp6() -> Attempt:
    """Compaction must preserve MEANING, not just tokens — judged, never matched.

    Every other compaction task in this suite plants a sentinel, which makes the
    verifier mechanical and the answer copyable. That is the right trade for a
    guard, and it also means the whole cluster measures one thing: whether an exact
    string came back. A summarizer could keep every sentinel and paraphrase away the
    constraint they were attached to, and nothing here would notice.

    So this task states an ordinary constraint with an ordinary reason, asks for it
    back in the model's own words, and hands the pinned expectation and the answer —
    and NOTHING else — to the judge (runner/judge.py, contract §4).

    The gate comes FIRST, before a single model call: with no committed validation
    artifact for THIS judge prompt, the task returns ``error`` and says so. It never
    falls back to a substring check. A silent fallback would publish a mechanical
    measurement under a judged task's name, which is the one failure that would make
    every number this task produces uninterpretable.

    ``attempted`` is published on every exit, as it is on G2 and CMP-5. This task has no
    ``not_attempted`` outcome — the judge returns a verdict on whatever came back — so
    the metric answers a narrower question here: what fraction of attempts got past the
    judge gate and the compaction setup to a real answer at all. Emitted from every exit
    for the same reason as elsewhere: a metric a failing exit stays silent about is
    averaged over the attempts that survived, not over the attempts that were made.

    RECORDED PHASE 3 ITEM, shared with CMP-5: format-robustness rework — judge-based
    role/answer extraction across the scenario guards, so a verdict turns on what a
    reply MEANS and not on a template. CMP-6 already judges its verdict, and CMP-5's
    pooled rate is mostly a format score until it does too; neither guard gates
    candidates before that rework lands.
    """
    validated, why = validation_status()
    if not validated:
        return Attempt(
            False,
            "error",
            f"judge not validated: {why}",
            # `attempted` alone: this exit precedes the agent, so there are no agent
            # metrics to report beside it. It still reports the zero rather than
            # nothing, or the metric's mean would skip the attempts that never started.
            metrics={"attempted": 0.0},
        )
    a = _plain_agent(context_limit=900)
    try:
        a.send("We are reviewing the gateway integration. Acknowledge briefly.")
        a.send(CMP6_FACT)
        compactions = int(a.just_compacted)
        for i in range(1, 18):
            a.send(_filler(i))
            compactions += int(a.just_compacted)
            if compactions >= 2:
                break
        if compactions < 2:
            return Attempt(
                False,
                "error",
                f"repeated-compaction setup did not fire twice (count={compactions})",
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        result = a.run(CMP6_QUESTION)
        reply = result.text
        judgment = judged_equivalent(CMP6_EXPECTED, reply, make_provider())
        ok = judgment.verdict
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compactions={compactions} verifier_kind=judged judge_verdict={ok} "
        f"judge_tokens={judgment.tokens} judge_quote={judgment.quote[:160]!r} "
        f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={
            **agent_metrics(a, result=result),
            "attempted": 1.0,
            "judge_verdict": float(ok),
            # The judge is a SECOND model call per attempt. Its cost is recorded
            # beside the agent's own, or CMP-6's per-task token mean reports only
            # half of what the task actually spent (contract amendment 4).
            "judge_tokens": float(judgment.tokens),
        },
    )


def cmp7_noise_script() -> str:
    """The log-noise fixture, pinned at authoring time — cluster_e's discipline.

    A public function returning the script's exact bytes, so a test can assert on
    what the model actually runs rather than on this module's source. One run emits
    ``CMP7_NOISE_LINES`` lines of plausible service noise (~3,000 characters) and the
    section number it was given, so three runs produce three distinguishable walls of
    text rather than one repeated block a summarizer could collapse into a single
    mention.

    The ticket appears nowhere in it. The only route to the answer is the sentence
    the user attached to the FIRST run request, which is exactly the thing repeated
    compaction has to carry.

    The noise goes to STDERR and only the summary line to stdout, and that split is the
    load-bearing part. The ask says "tell me the last line it printed", and the first
    live run showed the model doing the obvious thing with it: `python3 service_log.py 1
    | tail -n 1`, which handed back 59 characters. The bulk this task is ABOUT never
    entered the window, the fill rate ran 7.5x under what the setup assumed, and the
    second compaction never came. Carbon composes its bash result as
    `(stdout + stderr).strip()` while a pipe filters stdout alone, so noise on stderr
    reaches the model whatever the model writes — without changing the ask, the
    verifier, or the pinning.
    """
    return (
        "import sys\n"
        "section = sys.argv[1] if len(sys.argv) > 1 else '1'\n"
        f"for i in range({CMP7_NOISE_LINES}):\n"
        "    print(\n"
        "        f'2026-08-20T0{i % 10}:{i:02d}:{(i * 7) % 60:02d}Z svc-ledger "
        "worker-{i % 9} INFO '\n"
        "        f'heartbeat ok queue={i % 23} lag_ms={40 + i} retries={i % 4} "
        "shard=ledger-{i % 5} section={section}',\n"
        "        file=sys.stderr,\n"
        "    )\n"
        "print(f'section {section} complete: {" + str(CMP7_NOISE_LINES) + "} records scanned')\n"
    )


def cmp7_rerun_prompt(section: int) -> str:
    """A follow-up bulk turn. It never restates the ticket — the point is that the
    ticket has to survive the noise, not be repeated alongside it."""
    return (
        f"Run `python3 {CMP7_SCRIPT} {section}` with the bash tool and tell me the last "
        "line it printed."
    )


def cmp7_trigger() -> float:
    """The window level at which carbon compacts, from carbon's own config.

    Read at CALL time, never at import: `runner/` must not bind carbon's config at
    import (`test_importing_the_task_registry_binds_no_carbon_config`), and a candidate
    that edits the window has to move this number in the same run.
    """
    from harness.harness_config import CONFIG

    return CONFIG.default_context_limit * CONFIG.compaction.trigger_fraction


def cmp7_turns_needed(window: float, per_turn: float, trigger: float, compactions: int) -> float:
    """Turns still needed to reach two compactions, at the fill rate OBSERVED so far.

    The setup's stop-loss. A rate of zero — nothing accumulating — is infinite, which is
    the honest answer and the one the live defect deserved.

    Pessimistic in one named place: with no compaction yet, the second is costed as a
    full window from zero, because the floor a compaction leaves behind is not knowable
    until one has fired. That makes the projection give up slightly early on a hopeless
    run rather than slightly late, and the healthy path never reaches it — at the
    fixture's real weight the premise arrives in single-digit turns.
    """
    import math

    if per_turn <= 0:
        return math.inf
    needed = max(0.0, trigger - window) / per_turn
    if compactions < 1:
        needed += trigger / per_turn
    return needed


def cmp7_turn_budget() -> int:
    """The HARD cap on loop turns — a stop-loss, not the operative bound.

    The first live run of this task errored 3/3 with ``count=1``, and the instrumented
    cause was not a turn count at all: the model piped the noise away
    (``| tail -n 1``), the tool result came back at 59 characters, and the window
    crawled at 101 tokens a turn against the 762 this function assumed. The fixture now
    puts its bulk on stderr so no pipe can reach it, and the loop derives what it still
    needs from the fill it actually MEASURES (``cmp7_turns_needed``) rather than from
    any assumption about the tool result.

    This number survives as the outer bound: whatever the observed rate says, an attempt
    that cannot compact twice must end as ``error`` rather than run on. It is computed
    from carbon's own window and the fixture's measured weight, so a config edit to the
    window or the trigger moves it in the direction it should move.
    """
    import math

    from harness.compaction import estimate_tokens

    trigger = cmp7_trigger()
    # One bulk turn as the window sees it: the request, and the result it comes back
    # with. A placeholder of the fixture's measured length is exact here — carbon's
    # estimator counts characters, so the content's shape cannot change the number.
    per_turn = estimate_tokens(
        [
            {"role": "user", "content": cmp7_rerun_prompt(2)},
            {"role": "tool", "content": "x" * CMP7_NOISE_CHARS},
        ]
    )
    if per_turn <= 0:  # pragma: no cover -- a turn carrying nothing cannot fill anything
        raise ValueError("CMP-7's noise turn estimates as zero tokens; the fixture is empty")
    return CMP7_TURN_SLACK * 2 * math.ceil(trigger / per_turn)


def run_cmp7() -> Attempt:
    """A fact stated next to bulky tool output, asked for after repeated compaction.

    The realistic shape of a coding session's history is not tidy prose: it is a
    handful of sentences buried in walls of command output. G2 and G4 state their
    facts in quiet turns of their own, which is the easy case. Here the fact rides
    along with a request for ~3,000 characters of log noise, more bulky tool turns
    follow, and only then is it asked for.

    How MANY more is observed, not chosen. The first live run fixed the count at three
    bulky turns plus filler and errored 3/3 with `count=1` — the second compaction never
    accumulated before the filler ran out, so the task reported a failed premise instead
    of reaching it. The premise is two compactions, so the loop keeps asking for bulk
    until it sees them. Each turn names a different section, so the turns are distinct
    evidence rather than one command repeated.

    How it decides to give up is measured too, and that is the second lesson from the
    live runs. The setup's first cut derived its budget from an ASSUMED per-turn
    contribution, and the assumption was wrong by 7.5x because the model piped the bulk
    away — a silent `count=1` with nothing in the record to explain it. So the loop
    tracks `estimate_tokens(a.messages)` across its own turns, projects what it still
    needs at the rate it is actually seeing (`cmp7_turns_needed`), and stops when that
    exceeds what the cap has left — recording the rate either way. The config-derived
    cap (`cmp7_turn_budget`) remains as the outer stop-loss.

    Run at carbon's DEFAULT context limit, deliberately. G2 pins a 700-token window
    and pays for it: 35 of its 95 recorded replies came back as nothing but the
    truncation marker, a confound baked into the fixture. Here the window is the
    shipped one, compaction is reached by real bulk, and `default_context_limit`
    stays a knob this task can actually observe — which is why CMP-7 is the one
    compaction task on that knob's coverage row.

    The opening turn is an ordinary greeting and it is load-bearing. Compaction keeps
    the first `keep_head` messages VERBATIM (harness/compaction.py), so a fact stated
    in the very first message is never summarized at all and every strategy answers
    the question alike — a guard that cannot go red. The greeting spends that
    protected slot, which is the same reason G2 and G4 open with one.
    """
    from harness.agent import APPROVAL_TOOLS
    from harness.compaction import estimate_tokens
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import ToolRegistry
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write(CMP7_SCRIPT, cmp7_noise_script())
    approvals: list[dict] = []
    a = _plain_agent(
        approve=scripted_approver(approvals),
        approval_required=APPROVAL_TOOLS,
        **workspace_kwargs(ws.root),
    )
    tools = ToolRegistry()
    # scratch_dir=: every graded agent's Sandbox carries the shell route to
    # $CARBON_SCRATCH_DIR uniformly (see cluster_e), even on a task that never
    # offloads — carbon's own footer advertises the route unconditionally.
    tools.register(
        bash_tool(
            Sandbox(trusted=True, timeout=60, scratch_dir=a.session_env.scratch_root),
            workdir=str(ws.root),
        )
    )
    a.tools = tools
    try:
        a.send("We are triaging a ledger incident. Acknowledge briefly.")
        a.send(CMP7_RUN_AND_NOTE)
        compactions = int(a.just_compacted)
        cap = cmp7_turn_budget()
        trigger = cmp7_trigger()
        window = estimate_tokens(a.messages)
        spent = grown = 0
        growth = 0.0
        while compactions < 2 and spent < cap:
            before = window
            spent += 1
            # `spent + 1`: sections 2, 3, 4... — the fact-bearing turn already ran
            # section 1, and every turn asks for a different one.
            a.send(cmp7_rerun_prompt(spent + 1))
            compactions += int(a.just_compacted)
            window = estimate_tokens(a.messages)
            if not a.just_compacted:
                # A compaction SHRINKS the window; folding that turn into the average
                # would report a negative fill rate for a run that is filling fine.
                growth += max(0.0, window - before)
                grown += 1
            fill = growth / grown if grown else 0.0
            # Trust the rate only once several turns have shown it: one turn's growth
            # is a sample of the model's verbosity, not of the fixture's weight.
            if grown >= CMP7_MIN_OBSERVED_TURNS:
                if cmp7_turns_needed(window, fill, trigger, compactions) > cap - spent:
                    break
        fill = growth / grown if grown else 0.0
        if compactions < 2:
            return Attempt(
                False,
                "error",
                f"repeated-compaction setup did not fire twice (count={compactions} "
                f"turns={spent}/{cap} window={window} fill_per_turn={fill:.0f} "
                f"trigger={trigger:.0f})",
                approvals=approvals,
                turns=len(a.messages),
                metrics=agent_metrics(a),
            )
        result = a.run(CMP7_QUESTION)
        reply = result.text
        ok = CMP7_SENTINEL.lower() in reply.lower()
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"compactions={compactions} turns={spent} fill_per_turn={fill:.0f} "
        f"ticket_recalled={ok} reply={reply[:240]!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a, result=result),
    )


SPECS = [
    TaskSpec("G1", "held_in", "G", "uncertain", primitive="response", alias="RSP-1", run=run_g1),
    TaskSpec("G2", "held_out", "G", "uncertain", primitive="compaction", alias="CMP-2", run=run_g2),
    TaskSpec("G3", "held_in", "G", "pass", primitive="subagent", alias="SUB-1", run=run_g3),
    TaskSpec("G4", "held_in", "G", "uncertain", primitive="compaction", alias="CMP-3", run=run_g4),
    TaskSpec("G5", "held_in", "G", "uncertain", primitive="compaction", alias="CMP-4", run=run_g5),
    # The Phase 2c scenario guards. `alias=None` because the NAME is already the
    # mnemonic — a second id for the same task would be one more thing to keep in
    # sync. Priors are `uncertain`: nothing has measured them yet, and a prior is a
    # claim about the suite as authored, never a reading of a baseline.
    TaskSpec(
        "CMP-5", "held_in", "G", "uncertain", primitive="compaction", alias=None, run=run_cmp5
    ),
    TaskSpec(
        "CMP-6", "held_out", "G", "uncertain", primitive="compaction", alias=None, run=run_cmp6
    ),
    TaskSpec(
        "CMP-7", "held_in", "G", "uncertain", primitive="compaction", alias=None, run=run_cmp7
    ),
]
