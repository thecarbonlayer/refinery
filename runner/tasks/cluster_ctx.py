"""CTX — the Phase 4 context-delivery candidates (file_injection's task set).

Authored from the 2026-08-21 phase-4 brief §3, AHEAD of its human gate: five new
tasks, one per remaining decision-14 axis for the `@path` injection door. CTX-1 and
CTX-2 are NOT here — they are A5 and A4 (cluster_a), which have carried those
aliases since the Phase 1 contract froze; this module adds the axes the pair
cannot see. Cluster id stays "A" for the CMP-5/6/7 reason: the cluster is the
MECHANISM family (cluster A owns the `@path` door via A4/A5), and the task NAME is
already the mnemonic, so `alias=None` throughout.

The axes, one task each (no axis doubled):

- CTX-3 — position: middle, plus token economy. THE MINER.
- CTX-4 — completeness across positions (head+middle+tail, all required).
- CTX-5 — noise density (the needle among delivered near-duplicate decoys).
- CTX-6 — small-file economy, the no-harm end.
- CTX-7 — delivery mechanism, the bypass axis (CTX-3's shape WITH file tools).

STATUS: uncalibrated candidates. These tasks join the registry ONLY. They enter no
gain set, no guard set, no null-model coverage, and no confirmation rerun list
until their own null campaign runs on the pinned serving base and a calibration
installs — `tests/test_calibrate.py` pins that isolation, and the compaction sets
it protects, by name. `file_injection` likewise maps to NO rule section yet.
Splits ALTERNATE held-in/held-out at authoring (brief §3); the final assignment is
a gate input.

Every oracle is an exact sentinel plus deterministic counts (decision 12 — no
judge anywhere), with G2's non-answer taxonomy on every reply-shaped verdict so a
truncated generation or a tool-syntax leak is never scored as a delivery failure.
Every premise is checkable offline by running carbon's own `truncate()` on the
fixture bytes (`tests/test_registry.py`), and observed LIVE per attempt through
the injected block itself (`runner.helpers.injected_blocks`).

LIVE PREMISE CHECKS, the standing rule (the CMP-7 lesson): before any of these
tasks' numbers are trusted on a new serving base, the recorded arm must show, per
task, the delivery shape its docstring states — read off the published
`injection_*`/`*_delivered` metrics, which every attempt reports. A premise that
is knob-INDEPENDENT (injection fired at all) fails closed as `error` inside the
attempt; every knob-DEPENDENT observation is recorded and never enforced, because
erroring on the live policy's behavior would veto exactly the candidates the suite
exists to measure (cluster_a's AUTHORED_CLAMP lesson, hard constraint 2).
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from runner.carbon_env import make_provider
from runner.helpers import (
    TRUNCATION_MARK,
    agent_metrics,
    injected_blocks,
    neutral_dir,
    tool_call_args,
    tool_texts,
    workspace_kwargs,
)
from runner.spec import Attempt, TaskSpec

# AUTHORED_CLAMP is cluster_a's pin (4000, config v1); AUTHORED_STRATEGY and
# AUTHORED_TAIL_FRACTION below complete the policy the fixtures were sized
# against: carbon's shipped `file_injection` value, unchanged from config v1
# through v8, pinned at AUTHORING time. Premise checks and fixture asserts compare
# against THESE, never the live config — the live value is the editable knob under
# test (cluster_a's AUTHORED_CLAMP note; the E1/E3 history).
from runner.tasks.cluster_a import AUTHORED_CLAMP

# G2's non-answer taxonomy, reused verbatim rather than re-pinned: the marker is
# carbon's literal (pinned against carbon's source in test_registry.py) and the
# tool-syntax pattern must classify identically across tasks or the same reply
# would be `not_attempted` on one task and `fail` on another. `_TOOL_SYNTAX_RE` is
# module-private in cluster_g; importing it is deliberate coupling to the ONE
# definition, not an oversight.
from runner.tasks.cluster_g import _TOOL_SYNTAX_RE, G2_TRUNCATION_MARKER

AUTHORED_STRATEGY = "head_tail"
AUTHORED_TAIL_FRACTION = 0.5

CTX3_SENTINEL = "MIDW-CUT-52TN"
# CTX-3 passes only if the needle is recalled while the INJECTED block stays under
# this share of the fixture. Without it the one route to a green CTX-3 is a budget
# that floods the window with the whole file — the flood-shaped fix the brief's
# knob_coverage analysis already priced out for the tool_output analog (E1's
# pattern: a fraction of the fixture, never a char count, so it survives resizes;
# the shipped policy injects ~0.10 of this fixture, a needle-centred window ~0.10,
# a whole-file flood 1.0).
CTX3_MAX_INJECTED_SHARE = 0.2

# CTX-4's three facts, one per position, heterogeneous labels so a reply must
# carry each code and not a summary of "three codes exist".
CTX4_FACTS = {"head": "HD-GATE-11AK", "middle": "MD-BATCH-47QE", "tail": "TL-LEDGER-83RZ"}

# CTX-5: one approved token among near-duplicate decoys — same shape, wrong
# values, each stated as a REJECTED draft. Decoy positions are chosen (and pinned
# offline) so the baseline delivery carries the real token AND decoys, while a
# legal wrong-region cut carries decoys alone.
CTX5_REAL = "RLT-4471-KQ"
CTX5_DECOYS = (
    "RLT-8302-VN",
    "RLT-1958-DW",
    "RLT-6640-PB",
    "RLT-2217-GX",
    "RLT-9083-MS",
    "RLT-5526-HY",
)
# Which filler row each decoy lands after (ctx5_body). Slot 20 sits deep inside
# the head window even under the longest real path prefix; slot 390 sits inside
# the tail window; the middle four are cut at baseline.
CTX5_DECOY_SLOTS = (20, 100, 180, 250, 320, 390)

CTX6_SENTINEL = "ARCV-SIGN-64WD"
# CTX-6's economy ceiling: the injected block may not balloon past the file's own
# size plus this slack (the path banner and any modest strategy framing). Derived
# from the fixture's OWN length at the bottom of this module, so a fixture resize
# moves the ceiling with it; the slack is authored.
CTX6_INJECTED_SLACK = 1000

CTX7_SENTINEL = "STLB-FLAG-88KP"
# The tag on the needle's own line, and on no other line: `search_text` must have a
# real route to the middle, or a recorded bypass failure would be a fixture
# artifact instead of a finding about the knob's reach.
CTX7_SEARCH_TAG = "FLAGGED"
CTX7_FILE = "settlement-ledger.txt"
# The bypass bounds. A model that pages around the injection passes ONLY when the
# workaround is cheap: a search plus a few ranged reads. Both authored; the share
# is a fraction of the fixture (E1's convention) so a paging strategy that streams
# half the file inline fails economy however it gets there.
CTX7_MAX_FOLLOW_UP_READS = 8
CTX7_MAX_DELIVERED_SHARE = 0.25


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


# --- fixtures, public so tests assert on the EXACT bytes (the a5_body/e3 rule) ----


def ctx3_body() -> str:
    """CTX-3's fixture: one escalated row astride the midpoint of an over-budget
    vendor ledger. Filler rows are deliberately uninformative and uniform, so a
    model whose delivered block lacks the needle has nothing to guess from."""
    lines = [
        f"vendor row {i:04d}: contract current, invoices reconciled, no escalation on file"
        for i in range(500)
    ]
    lines[250] = f"vendor row 0250: ESCALATED - the cutover code for this vendor is {CTX3_SENTINEL}"
    body = "\n".join(lines) + "\n"
    assert len(body) > 5 * AUTHORED_CLAMP, "authoring bug: fixture must dwarf the clamp"
    return body


def ctx4_body() -> str:
    """CTX-4's fixture: three labeled codes — head, middle, tail — separated by
    filler, sized so the authored policy delivers exactly the head and tail ones."""
    filler = [
        f"inventory row {i:04d}: shelf audit clean, counts match ledger, no variance"
        for i in range(500)
    ]
    head = f"## Access\nThe staging gateway code is {CTX4_FACTS['head']}.\n\n"
    mid = f"\n## Mid-cycle addendum\nThe reconciliation batch code is {CTX4_FACTS['middle']}.\n"
    tail = f"\n## Final sign-off\nThe closing ledger code is {CTX4_FACTS['tail']}.\n"
    body = head + "\n".join(filler[:250]) + mid + "\n".join(filler[250:]) + tail
    assert len(body) > 5 * AUTHORED_CLAMP, "authoring bug: fixture must dwarf the clamp"
    return body


def ctx5_body() -> str:
    """CTX-5's fixture: the approved token stated once at the top, six rejected
    look-alike drafts spread through an over-budget file (`CTX5_DECOY_SLOTS`)."""
    out = [f"## Rollout decision\nThe approved rollout token is {CTX5_REAL}.\n"]
    slots = dict(zip(CTX5_DECOY_SLOTS, CTX5_DECOYS, strict=True))
    for i in range(400):
        if i in slots:
            out.append(f"draft rollout token candidate {slots[i]} (rejected in review)")
        out.append(f"rollout note {i:04d}: reviewed, no change to the approved token")
    body = "\n".join(out) + "\n"
    assert len(body) > 5 * AUTHORED_CLAMP, "authoring bug: fixture must dwarf the clamp"
    return body


def ctx6_body() -> str:
    """CTX-6's fixture: a weekly summary that fits under the authored clamp whole,
    sign-off code in the MIDDLE — where a strategy that fragments or indexes even
    small files loses bytes first."""
    days = "\n".join(
        f"day {i:02d}: quiet shift, no incidents, backlog stable, handoff clean" for i in range(30)
    )
    appendix = "\n".join(f"appendix {i:02d}: routine entry, no action required" for i in range(15))
    body = (
        f"## Weekly ops summary\n{days}\n\n## Sign-off\n"
        f"The archive verification code is {CTX6_SENTINEL}.\n\n## Appendix\n{appendix}\n"
    )
    # Sized against the AUTHORED clamp with headroom for the door's path banner —
    # never the live budget, which a candidate may legally lower (and should then
    # turn CTX-6 red at measurement time, not turn this suite red offline).
    assert len(body) + 200 < AUTHORED_CLAMP, "authoring bug: fixture must fit under the clamp"
    return body


# CTX-6's derived economy ceiling (see CTX6_INJECTED_SLACK).
CTX6_MAX_INJECTED_CHARS = len(ctx6_body()) + CTX6_INJECTED_SLACK


def ctx7_body() -> str:
    """CTX-7's fixture: CTX-3's midpoint shape with its own sentinel, in a file the
    task's `search_text`/`read_file` are rooted at — the tag `FLAGGED` marks the
    needle's line alone, so paging around the injection is a real, cheap route."""
    lines = [
        f"settlement row {i:04d}: settled clean, carryover zero, ledger matched" for i in range(500)
    ]
    lines[250] = (
        f"settlement row 0250: {CTX7_SEARCH_TAG} for manual review, batch code {CTX7_SENTINEL}"
    )
    body = "\n".join(lines) + "\n"
    assert len(body) > 5 * AUTHORED_CLAMP, "authoring bug: fixture must dwarf the clamp"
    return body


# --- verdicts, pure over (reply, measured counts) so tests pin every branch -------


def _non_answer(reply: str) -> str | None:
    """G2's classification, verbatim in substance: a reply that is nothing but
    carbon's truncation marker never produced an answer, and a tool-call fragment
    answered a different question than the one asked. Callers apply it only when
    the reply also failed the recall check — a reply that recalled the sentinel
    attempted the answer whatever else it carries."""
    if reply.lstrip().startswith(G2_TRUNCATION_MARKER):
        return "generation truncated before answer"
    if _TOOL_SYNTAX_RE.search(reply):
        return "tool-syntax leak instead of answer"
    return None


def ctx3_verdict(reply: str, injected_share: float) -> tuple[bool, str, str | None]:
    """``(passed, outcome, non_answer_detail)``. Recall AND economy: the sentinel
    bought by flooding the window is a fail, not a pass (see CTX3_MAX_INJECTED_SHARE)."""
    recalled = CTX3_SENTINEL.lower() in reply.lower()
    if recalled and injected_share <= CTX3_MAX_INJECTED_SHARE:
        return True, "pass", None
    if not recalled and (why := _non_answer(reply)):
        return False, "not_attempted", why
    return False, "fail", None


def ctx4_verdict(reply: str) -> tuple[bool, str, dict[str, bool], str | None]:
    """``(passed, outcome, recalled_by_position, non_answer_detail)`` — the G4
    per-property shape: the verdict is the conjunction, the detail says WHICH
    position was lost. Non-answer only when NO position came back at all."""
    low = reply.lower()
    recalled = {pos: fact.lower() in low for pos, fact in CTX4_FACTS.items()}
    if all(recalled.values()):
        return True, "pass", recalled, None
    if not any(recalled.values()) and (why := _non_answer(reply)):
        return False, "not_attempted", recalled, why
    return False, "fail", recalled, None


def ctx5_verdict(reply: str) -> tuple[bool, str, tuple[str, ...], str | None]:
    """``(passed, outcome, decoys_in_reply, non_answer_detail)``. The pass needs
    the real token AND a decoy-free reply: a decoy beside the real token is not a
    clean answer, and a decoy alone is the confident wrong report this guard
    exists to catch — both are real failures, never non-answers."""
    low = reply.lower()
    real = CTX5_REAL.lower() in low
    decoys = tuple(d for d in CTX5_DECOYS if d.lower() in low)
    if real and not decoys:
        return True, "pass", (), None
    if not real and not decoys and (why := _non_answer(reply)):
        return False, "not_attempted", decoys, why
    return False, "fail", decoys, None


def ctx6_verdict(reply: str, injected_chars: float) -> tuple[bool, str, str | None]:
    """``(passed, outcome, non_answer_detail)``. Recall AND leanness: a strategy
    that wraps a small file in an index or scaffold it never needed fails the
    no-harm axis even when the answer survives the wrapping."""
    recalled = CTX6_SENTINEL.lower() in reply.lower()
    if recalled and injected_chars <= CTX6_MAX_INJECTED_CHARS:
        return True, "pass", None
    if not recalled and (why := _non_answer(reply)):
        return False, "not_attempted", why
    return False, "fail", None


def ctx7_verdict(
    reply: str, follow_up_reads: int, delivered_share: float
) -> tuple[bool, str, str | None]:
    """``(passed, outcome, non_answer_detail)``. Recall bounded by workaround
    cost: the route may be the injected block OR the tools, but a pass bought with
    unbounded paging (reads past the cap, or a flood of delivered bytes) is a fail."""
    recalled = CTX7_SENTINEL.lower() in reply.lower()
    bounded = (
        follow_up_reads <= CTX7_MAX_FOLLOW_UP_READS and delivered_share <= CTX7_MAX_DELIVERED_SHARE
    )
    if recalled and bounded:
        return True, "pass", None
    if not recalled and (why := _non_answer(reply)):
        return False, "not_attempted", why
    return False, "fail", None


# --- the tasks --------------------------------------------------------------------


def run_ctx3() -> Attempt:
    """THE MINER. Axis: position (middle) plus token economy, through `@path`.

    An over-budget delivered file, needle astride the midpoint, NO tools — the A4
    pattern, so the injected block is the only route and the verdict is about the
    door alone. Oracle: exact sentinel in the reply AND an inline-injection
    ceiling (`CTX3_MAX_INJECTED_SHARE`), so a flood-shaped budget cannot buy the
    pass a middle-preserving strategy is supposed to earn.

    PRESPECIFIED, written before any run: 0/N at the shipped strategy and budget.
    This is computable — `tests/test_registry.py` proves the needle unreachable by
    every legal inline cut offline — so a nonzero baseline rate is an AUTHORING
    BUG (or a hallucinated sentinel, which the exact 13-char code makes wildly
    unlikely), never noise.

    LIVE PREMISE CHECK (before trusting numbers on a new serving base): every
    attempt reports injection fired (else `error`), `injection_truncated` = 1.0
    and `needle_delivered` = 0.0 under the baseline config. `needle_delivered`
    flipping to 1.0 under a CANDIDATE is not a premise failure — it is the
    candidate working; the metrics exist so the campaign can see which."""
    d = Path(mkdtemp(prefix="ctx3-"))
    body = ctx3_body()
    notes = d / "vendor-ledger.txt"
    notes.write_text(body)
    a = _plain_agent()
    try:
        result = a.run(
            f"@{notes} One vendor row is marked escalated. What is that vendor's "
            "cutover code? Reply with just the code."
        )
        reply = result.text
        blocks = injected_blocks(a.messages)
        if not blocks:
            return Attempt(
                False,
                "error",
                "premise failed: no @path context block was injected",
                turns=len(a.messages),
                # `attempted` on EVERY exit (the G2 rule): a metric a failing exit
                # omits is averaged over the attempts that survived, not made.
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        share = sum(len(b) for b in blocks) / len(body)
        delivered = any(CTX3_SENTINEL in b for b in blocks)
        truncated = any(TRUNCATION_MARK in b for b in blocks)
        ok, outcome, non_answer = ctx3_verdict(reply, share)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"needle_in_reply={CTX3_SENTINEL.lower() in reply.lower()} "
        f"injected_share={share:.3f} (limit {CTX3_MAX_INJECTED_SHARE}) "
        f"needle_delivered={delivered} injection_truncated={truncated} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        # `reply=` stays LAST — the G2/CMP-5 convention: a trailing-match extractor
        # reads everything after it as the reply.
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={
            **agent_metrics(a, result=result),
            "attempted": float(outcome != "not_attempted"),
            "injected_share": share,
            "needle_delivered": float(delivered),
            "injection_truncated": float(truncated),
        },
    )


def run_ctx4() -> Attempt:
    """Guard by design, miner-shaped at baseline. Axis: completeness across positions.

    Three labeled codes — head, middle, tail — all required, per-property verdict
    (the G4 pattern), no tools. What it catches that CTX-3 cannot: a strategy that
    BUYS the middle by SELLING an end stays red here (and drops CTX-1/A5), so the
    completeness claim is priced from day one.

    Prior is `fail`, stated plainly: at the authored baseline the middle fact is
    deterministically undeliverable, so the conjunction cannot hold. In
    knob_coverage's role vocabulary that makes CTX-4 miner-shaped (headroom, can't
    regress) even though the brief names it a guard — the deviation is recorded in
    the authoring report; the per-property detail is where partial trades show.

    LIVE PREMISE CHECK: injection fired (else `error`); at baseline
    `head_delivered` = 1.0, `tail_delivered` = 1.0, `middle_delivered` = 0.0."""
    d = Path(mkdtemp(prefix="ctx4-"))
    body = ctx4_body()
    notes = d / "cycle-report.txt"
    notes.write_text(body)
    a = _plain_agent()
    try:
        result = a.run(
            f"@{notes} Report three codes exactly as recorded: the staging gateway "
            "code, the reconciliation batch code, and the closing ledger code."
        )
        reply = result.text
        blocks = injected_blocks(a.messages)
        if not blocks:
            return Attempt(
                False,
                "error",
                "premise failed: no @path context block was injected",
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        delivered = {pos: any(fact in b for b in blocks) for pos, fact in CTX4_FACTS.items()}
        truncated = any(TRUNCATION_MARK in b for b in blocks)
        ok, outcome, recalled, non_answer = ctx4_verdict(reply)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"recalled={recalled} delivered={delivered} injection_truncated={truncated} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={
            **agent_metrics(a, result=result),
            "attempted": float(outcome != "not_attempted"),
            "head_delivered": float(delivered["head"]),
            "middle_delivered": float(delivered["middle"]),
            "tail_delivered": float(delivered["tail"]),
            "injection_truncated": float(truncated),
        },
    )


def run_ctx5() -> Attempt:
    """Guard. Axis: noise density — the needle among delivered near-duplicates.

    The approved token arrives in the baseline delivery ALONGSIDE rejected
    look-alike drafts (pinned offline), so at baseline the task is discrimination,
    not retrieval: prior `pass`, and the guard can regress. The failure it exists
    to catch is a region-picking strategy that hands the model a decoy-bearing
    slice without the real token — the model then reports a decoy confidently,
    and the oracle (real token present AND no decoy in the reply) goes red.

    LIVE PREMISE CHECK: injection fired (else `error`); at baseline
    `real_delivered` = 1.0 and `decoys_delivered` >= 1 on every attempt."""
    d = Path(mkdtemp(prefix="ctx5-"))
    body = ctx5_body()
    notes = d / "rollout-notes.txt"
    notes.write_text(body)
    a = _plain_agent()
    try:
        result = a.run(f"@{notes} What is the approved rollout token? Reply with just the token.")
        reply = result.text
        blocks = injected_blocks(a.messages)
        if not blocks:
            return Attempt(
                False,
                "error",
                "premise failed: no @path context block was injected",
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        real_delivered = any(CTX5_REAL in b for b in blocks)
        decoys_delivered = sum(any(d5 in b for b in blocks) for d5 in CTX5_DECOYS)
        truncated = any(TRUNCATION_MARK in b for b in blocks)
        ok, outcome, decoys_in_reply, non_answer = ctx5_verdict(reply)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"real_in_reply={CTX5_REAL.lower() in reply.lower()} "
        f"decoys_in_reply={decoys_in_reply} real_delivered={real_delivered} "
        f"decoys_delivered={decoys_delivered} injection_truncated={truncated} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={
            **agent_metrics(a, result=result),
            "attempted": float(outcome != "not_attempted"),
            "real_delivered": float(real_delivered),
            "decoys_delivered": float(decoys_delivered),
            "injection_truncated": float(truncated),
        },
    )


def run_ctx6() -> Attempt:
    """Guard. Axis: small-file economy — the no-harm end of the strategy menu.

    An UNDER-budget file must arrive whole and be answered without ceremony.
    Prior `pass`: the shipped door passes an under-budget block byte-identical.
    What it catches: a strategy that indexes, pages, or scaffolds files that never
    needed it — either the middle-of-file code stops arriving (recall fails) or
    the injected block balloons past `CTX6_MAX_INJECTED_CHARS` (economy fails).
    A candidate that legally LOWERS the budget below this fixture's size turns
    this red at measurement time; the offline suite stays green (authored clamp,
    never the live one).

    LIVE PREMISE CHECK: injection fired (else `error`); at baseline
    `injection_truncated` = 0.0 and `injected_chars` within the ceiling."""
    d = Path(mkdtemp(prefix="ctx6-"))
    body = ctx6_body()
    notes = d / "weekly-summary.txt"
    notes.write_text(body)
    a = _plain_agent()
    try:
        result = a.run(
            f"@{notes} What archive verification code does the sign-off give? "
            "Reply with just the code."
        )
        reply = result.text
        blocks = injected_blocks(a.messages)
        if not blocks:
            return Attempt(
                False,
                "error",
                "premise failed: no @path context block was injected",
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        injected_chars = sum(len(b) for b in blocks)
        truncated = any(TRUNCATION_MARK in b for b in blocks)
        ok, outcome, non_answer = ctx6_verdict(reply, injected_chars)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"code_in_reply={CTX6_SENTINEL.lower() in reply.lower()} "
        f"injected_chars={injected_chars} (limit {CTX6_MAX_INJECTED_CHARS}) "
        f"injection_truncated={truncated} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={
            **agent_metrics(a, result=result),
            "attempted": float(outcome != "not_attempted"),
            "injected_chars": float(injected_chars),
            "injection_truncated": float(truncated),
        },
    )


def run_ctx7() -> Attempt:
    """Weak guard, uncertain prior. Axis: delivery mechanism — the bypass axis.

    CTX-3's midpoint shape WITH the harness's real retrieval belt (`search_text`
    plus ranged `read_file`, rooted at the workspace the fixture sits in —
    withholding a shipped tool would manufacture the gap, E1's lesson). This
    measures the recorded bypass behavior instead of leaving it as a fear: the
    scratch-outcome precedent says the model routes around offered surfaces, and
    carbon's own truncation hint tells it to "use read_file ranges". If the model
    pages around EVERY strategy here, the knob is unobservable on this shape —
    a real finding about its reach, recorded where decisions can see it.

    PRESPECIFIED, a genuine prediction (NOT computable — label it as such in the
    first campaign report): the model bypasses the injection via ranged reads in
    most attempts, i.e. passes arrive with `follow_up_reads` > 0.

    Oracle: exact sentinel bounded by workaround cost — at most
    `CTX7_MAX_FOLLOW_UP_READS` follow-up tool calls and `CTX7_MAX_DELIVERED_SHARE`
    of the fixture delivered through tool results (E1's economy convention).

    LIVE PREMISE CHECK: injection fired (else `error`); `needle_delivered` = 0.0
    in the injected block at baseline (recovery, if any, must come through the
    tools — `follow_up_reads` records the route)."""
    from harness.tools import ToolRegistry, read_file_tool, search_text_tool
    from harness.workspace import Workspace

    ws = Workspace()
    body = ctx7_body()
    ws.write(CTX7_FILE, body)
    a = _plain_agent(**workspace_kwargs(ws.root))
    tools = ToolRegistry()
    # scratch_root: the uniform capability grant (E1's note) — inert under the
    # shipped inline default, live the day a candidate points this door at a
    # spill-shaped strategy.
    tools.register(read_file_tool(str(ws.root), scratch_root=a.session_env.scratch_root))
    tools.register(search_text_tool(str(ws.root)))
    a.tools = tools
    try:
        result = a.run(
            f"@{ws.root / CTX7_FILE} One settlement row is flagged for manual "
            "review. What is that row's batch code? Reply with just the code."
        )
        reply = result.text
        blocks = injected_blocks(a.messages)
        if not blocks:
            return Attempt(
                False,
                "error",
                "premise failed: no @path context block was injected",
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        follow_up_reads = len(tool_call_args(a.messages, ("read_file", "search_text")))
        delivered_share = sum(len(t) for t in tool_texts(a.messages)) / len(body)
        needle_delivered = any(CTX7_SENTINEL in b for b in blocks)
        truncated = any(TRUNCATION_MARK in b for b in blocks)
        ok, outcome, non_answer = ctx7_verdict(reply, follow_up_reads, delivered_share)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"needle_in_reply={CTX7_SENTINEL.lower() in reply.lower()} "
        f"follow_up_reads={follow_up_reads} (limit {CTX7_MAX_FOLLOW_UP_READS}) "
        f"delivered_share={delivered_share:.3f} (limit {CTX7_MAX_DELIVERED_SHARE}) "
        f"needle_delivered={needle_delivered} injection_truncated={truncated} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={
            **agent_metrics(a, result=result),
            "attempted": float(outcome != "not_attempted"),
            "follow_up_reads": float(follow_up_reads),
            "delivered_share": delivered_share,
            "needle_delivered": float(needle_delivered),
            "injection_truncated": float(truncated),
        },
    )


# Priors are claims about the suite AS AUTHORED, never a reading of a baseline:
# CTX-3/CTX-4 `fail` because the middle is deterministically undeliverable at the
# authored policy (proven offline); CTX-5/CTX-6 `pass` because their answers are in
# the baseline delivery by construction; CTX-7 `uncertain` because its verdict
# turns on live bypass behavior nothing has measured. Splits alternate at
# authoring, miner held-in (mining spends held-in evidence only — the G4 rule);
# the final assignment is a gate input.
SPECS = [
    TaskSpec(
        "CTX-3", "held_in", "A", "fail", primitive="context-delivery", alias=None, run=run_ctx3
    ),
    TaskSpec(
        "CTX-4", "held_out", "A", "fail", primitive="context-delivery", alias=None, run=run_ctx4
    ),
    TaskSpec(
        "CTX-5", "held_in", "A", "pass", primitive="context-delivery", alias=None, run=run_ctx5
    ),
    TaskSpec(
        "CTX-6", "held_out", "A", "pass", primitive="context-delivery", alias=None, run=run_ctx6
    ),
    TaskSpec(
        "CTX-7",
        "held_in",
        "A",
        "uncertain",
        primitive="context-delivery",
        alias=None,
        run=run_ctx7,
    ),
]
