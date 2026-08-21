"""Cluster I — session onboarding / instructions (ONB-1..ONB-5).

Authored 2026-08-21 as preparation for the Phase 4 human gate, not an activation.
Decision 14's design model applied to the `instructions` primitive: distinct
real-use scenario axes, one task per axis, and any future strategy judged across
ALL of them. B1 already proves the loaded layer is obeyed when it is the only
voice in the room, and it passes 1.0 in every committed arm — saturated, so it
discriminates nothing. Every task here therefore seeds a way the loaded layer can
LOSE, and each is red-capable for a stated reason.

WHAT CARBON'S MACHINERY ACTUALLY DOES (harness/instructions.py, harness/agent.py),
because every axis below must be grounded in it:

- ``load_agents_md(agents_dir)`` reads exactly ``<agents_dir>/AGENTS.md``. No
  parent walk, no subdirectory discovery, no merge.
- ``_system_text()`` layers built-in system prompt + AGENTS.md + skills menu into
  ONE undifferentiated system block; ``_payload()`` rebuilds it per MODEL CALL, so
  the file is re-read from disk on every call and a mid-session edit takes effect
  on the next call.
- The layer is injected WHOLE. Unlike tool results (``truncate_tool_result``) and
  ``@path`` blocks (``deliver`` -> ``truncate``), the instructions door has no
  strategy, no budget, no clamp.
- Compaction rewrites ``self.messages`` only; the system head is rebuilt fresh
  each call, so the layer can never be compacted away. That is why simple
  obedience (B1) saturates, and why discrimination here must come from conflict,
  distraction, indirection, and timing instead of from distance-to-start.

THE AXES, one task each (premise / oracle / designed to catch):

- ONB-1, pointer-following discovery (held-in). The loaded layer points at an
  unmentioned file (``docs/release.md``); the answer exists only there; read-only
  discovery tools are registered. Oracle: exact sentinel in the reply AND the
  sentinel observed in a tool result (an unattributed arrival is refused — E4's
  conservative-credit rule). Designed to catch: the model treats the loaded layer
  as the whole of onboarding and answers from priors or declines instead of
  following the pointer. Why the shipped behavior can produce it: the harness
  loads only the root file and nothing nudges discovery; the dual-interface
  scratch outcome showed this model does not adopt offered routes reliably.
- ONB-2, precedence within the layered instructions (held-out). One layer, two
  rules: a general reply signature stated first, a specific incident override
  stated later, the file itself saying the general line must not appear. Oracle:
  override sentinel present AND general sentinel absent. Designed to catch:
  first-match obedience — the model applies the general rule it saw first. Why
  shipped behavior can produce it: the whole file arrives as one undifferentiated
  system block; nothing marks the override as more specific.
- ONB-3, instruction-following under distractor content (held-in). The operative
  approval code sits mid-file in a layer BIGGER than the authored ``@path`` clamp,
  with a fenced example transcript later carrying a look-alike placeholder code
  explicitly disclaimed as not real. Oracle: operative code present AND placeholder
  absent. Designed to catch: template-copying — the model quotes the example's
  code because it is the most answer-shaped string in view. Why shipped behavior
  can produce it: the door has no budget, so the whole 4.6k block is in context
  every call and the operative rule competes with its own documentation; the
  fenced example is exactly the shape few-shot-primed models copy.
- ONB-4, stale-vs-current instruction conflict (held-out). The loaded layer
  carries the current token (with its rotation date); a stale README in the
  workspace carries the old one (with an older review date); the prompt instructs
  reading the README. Oracle: current present AND stale absent, gated on the
  conflict having become live (the stale bytes actually observed in a tool
  result — CMP-5's observed-premise discipline). Designed to catch: recency
  capture — the freshly-read file outranks the always-present layer. Why shipped
  behavior can produce it: tool results arrive as fresh turns close to the
  answer; the harness attaches no provenance cue that AGENTS.md outranks
  workspace files.
- ONB-5, instruction application timing (held-in). Ask once under v1 (the model's
  own reply seeds the stale echo), rewrite AGENTS.md to v2 mid-session — exactly
  one changed line — then ask again. Oracle: v2 present AND v1 absent. Designed
  to catch: transcript memory beating the current layer — the model repeats its
  own earlier answer. Why shipped behavior can produce it: the earlier Q&A is
  verbatim in the history and directly answers the question, while the changed
  system block carries no "this changed" cue; that the layer is re-read per call
  is real shipped behavior (``_payload`` -> ``_system_text`` -> ``load_agents_md``)
  that nothing measures today.

AXES THE CODE CANNOT EXPRESS — knob proposals, not tasks, stated per decision 14:

- Nested/subdirectory instruction discovery and merge precedence. The loader
  reads one file, unconditionally; there is no ``onboarding`` field. The
  capability map's ``none / dir_map / dir_map_toolscan`` menu is the proposal;
  ONB-1 measures the tools-only baseline of exactly the capability that knob
  would buy, so its rate is the "why now" evidence a future proposal needs.
- An instruction-layer budget or injection strategy. The layer is the one
  delivery door with no clamp (ONB-3's offline probe proves the operative rule
  would have died at the authored ``@path`` policy). Budgeting or summarizing the
  layer would be a new editable field; until it exists, layer size is a fixture
  choice, not a settable value.
- Precedence between the BUILT-IN system prompt and AGENTS.md. Deliberately not
  tasked: ``system_prompt`` is editable, so an oracle keyed to its text would
  couple a verifier to the editable surface — the exact defect H2 had with
  ``compaction_prompt`` and the reason its detection went structural.

LIVE PREMISE CHECKS (what must be observed on the pinned serving base before any
number from this section is trusted — the CMP-7 lesson):

- All tasks: the attempt re-reads the layer through carbon's own
  ``load_agents_md`` and errors on any mismatch with the authored bytes — the
  input side of the door proven at attempt time, not assumed.
- ONB-1: a passing attempt must carry the token in a tool result (in-verifier);
  campaign-level, the attempted-rate metric must show replies were produced.
- ONB-4: the stale README's sentinel must appear in a tool result before the
  final answer, or the attempt is ``error`` ("conflict never became live") — a
  vacuous pass is never recorded as discrimination.
- ONB-5: turn 1 must echo the v1 code (the stale trace is otherwise absent and
  the final ask is trivially clean); the rewrite is verified through the loader
  before the second ask.
- ONB-2/ONB-3: reply-shaped only; the premise is fixture-borne and pinned by the
  offline probes (both rules / both codes verbatim in the loaded layer).

REGISTRATION. Own section: cluster "I", primitive ``instructions``, names are the
mnemonic (alias None, the CMP-5/6/7 convention), priors ``uncertain`` (nothing
has measured them). Provably outside the compaction campaign suite and every
calibrated gate — ``tests/test_registry.py``'s isolation test holds the proof.
No knob row names these tasks as miners; there is no onboarding knob to cover.
ONB-1/ONB-4 join ``tool_output``'s observer/guard row with MEASURED break budgets
(their file reads are tool results a legal budget can cut), and ``_TOOL_USERS``.
No AGENTS.md here declares ``## Testing``, so the verification gate never arms
and cluster B remains the only cluster that ships a test command.

Verifiers are exact sentinels, deterministic, no judge (decision 12: nothing here
asks a meaning question). Reply-shaped tasks share G2's non-answer taxonomy by
IMPORT (``onb_verdict``): a truncated generation or a tool-syntax leak is
``not_attempted`` — a statement about the harness, never about what the layer
carried — and ``attempted`` is published on every exit.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

from runner.carbon_env import make_provider
from runner.helpers import agent_metrics, tool_texts, workspace_kwargs
from runner.spec import Attempt, TaskSpec
from runner.tasks.cluster_a import AUTHORED_CLAMP
from runner.tasks.cluster_g import _TOOL_SYNTAX_RE, G2_TRUNCATION_MARKER

# --- ONB-1: pointer-following discovery ---------------------------------------------
ONB1_SENTINEL = "RELGATE-ONB1-K7P4"
ONB1_AGENTS_MD = (
    "# release-tools project\n"
    "\n"
    "This file is deliberately short. The project's conventions live in the repo,\n"
    "not here: the release process, including the release gate token, is\n"
    "documented in docs/release.md. Read that file before answering release\n"
    "questions.\n"
)
ONB1_DOC = (
    "# Release process\n"
    "\n"
    "Releases ship weekly from the release branch. Every release needs a green\n"
    "suite and a sign-off recorded in the tracker before the gate.\n"
    "\n"
    f"The release gate token is {ONB1_SENTINEL}. Quote it exactly when a\n"
    "checklist step asks for the gate token.\n"
    "\n"
    "After the gate: tag the release, push the tag, and post the announcement.\n"
)
# Light chaff so discovery happens in a repo, not a single-file gimme. Token-free
# by the offline probe.
ONB1_CHAFF = {
    "README.md": (
        "# release-tools\n"
        "\n"
        "Small helpers for cutting weekly releases. See AGENTS.md for the project\n"
        "rules and pointers into docs/.\n"
    ),
    "src/version.py": 'VERSION = "3.2.1"\n',
}
ONB1_PROMPT = "What is this project's release gate token? Reply with just the token."

# --- ONB-2: precedence within the layered instructions --------------------------------
ONB2_GENERAL_SIG = "SIG-GEN-ONB2-3FA"
ONB2_INCIDENT_SIG = "SIG-INC-ONB2-9QD"
ONB2_AGENTS_MD = (
    "# support-desk project\n"
    "\n"
    "## Reply signature\n"
    f"End every reply with the signature line {ONB2_GENERAL_SIG} on its own line.\n"
    "\n"
    "## Incident replies (specific override)\n"
    f"For a reply about an active incident, end with the line {ONB2_INCIDENT_SIG}\n"
    "instead. The general signature line must not appear on incident replies.\n"
)
ONB2_PROMPT = (
    "Active incident: the payments service is returning 502 errors. "
    "Acknowledge the incident in one short sentence."
)

# --- ONB-3: instruction-following under distractor content ----------------------------
ONB3_CURRENT = "APPROVE-ONB3-5WT"
ONB3_EXAMPLE = "APPROVE-ONB3-EX0A"
ONB3_PROMPT = "Approve the staging deploy. Reply with just the deploy approval code."


def onb3_agents_md() -> str:
    """ONB-3's bulky instruction layer, as a function so tests assert on THE bytes.

    Sized past the authored ``@path`` clamp on purpose — the instructions door
    injects it whole, which the offline probe proves is the only reason the
    operative code (buried mid-file, inside the region the authored clamp policy
    would cut) reaches the model at all. The look-alike placeholder sits later, in
    a fenced example explicitly disclaimed as not real.
    """
    env_rows = "\n".join(
        f"- env-{i:02d}: region r{i % 4}, tier {'a' if i % 2 else 'b'}, config reviewed, "
        "no open incidents, owner rotation unchanged this cycle."
        for i in range(18)
    )
    style_rows = "\n".join(
        f"- style note {i:02d}: prefer small modules, name things plainly, keep functions "
        "short, and let the tests document the behavior they pin."
        for i in range(14)
    )
    body = (
        "# deploy-desk project\n"
        "\n"
        "Operational handbook for the deploy desk. Environments, code style, the\n"
        "deploy approval rule, and a worked example live below; read the section\n"
        "you need rather than skimming for something answer-shaped.\n"
        "\n"
        "## Environments\n"
        f"{env_rows}\n"
        "\n"
        "## Deploy approval\n"
        "Deploy approvals are code-gated, and the code is rotated out of band, so\n"
        "it is stated here once rather than derived from anything else in this\n"
        f"file. The deploy approval code is {ONB3_CURRENT}. When asked to approve\n"
        "a deploy, reply with exactly this code and nothing else.\n"
        "\n"
        "## Code style\n"
        f"{style_rows}\n"
        "\n"
        "## Example transcript (illustrative only)\n"
        "The code below is a documentation placeholder, never a real approval\n"
        "code:\n"
        "\n"
        "    user: approve the deploy\n"
        f"    assistant: approved — code {ONB3_EXAMPLE}\n"
        "\n"
        "## Troubleshooting\n"
        "- A stuck deploy usually means the previous one never released its lock.\n"
        "- Check the environment row above before paging anyone.\n"
    )
    assert len(body) > AUTHORED_CLAMP, "authoring bug: layer must exceed the authored clamp"
    assert body.index(ONB3_CURRENT) > 1500, "authoring bug: rule drifted into the head window"
    return body


# --- ONB-4: stale-vs-current instruction conflict -------------------------------------
ONB4_CURRENT = "ESCAL-ONB4-6MV2"
ONB4_STALE = "ESCAL-ONB4-OLD1JC"
ONB4_AGENTS_MD = (
    "# ops-runbook project\n"
    "\n"
    f"Current on-call escalation channel token: {ONB4_CURRENT} (rotated 2026-08).\n"
    "This file is maintained; older documents in this repo may lag behind it.\n"
)
ONB4_README = (
    "# ops runbook notes (last reviewed 2025-01)\n"
    "\n"
    "Team notes for the on-call rotation. Paging goes through the escalation\n"
    f"channel; the on-call escalation channel token is {ONB4_STALE}.\n"
    "Rotation swaps happen on Mondays. Update this file when the process changes.\n"
)
ONB4_PROMPT = (
    "Read README.md, then tell me the on-call escalation channel token currently "
    "in effect. Reply with just the token."
)

# --- ONB-5: instruction application timing --------------------------------------------
ONB5_V1 = "WINDOW-ONB5-V1-4HT"
ONB5_V2 = "WINDOW-ONB5-V2-8RC"
ONB5_V1_AGENTS_MD = (
    "# maintenance-window project\n"
    "\n"
    f"The current maintenance window code is {ONB5_V1}. Give exactly this\n"
    "code when asked for the maintenance window.\n"
)
# Derived with str.replace (the F1 discipline): the rewrite changes the code line
# and nothing else, so the mid-session diff the model faces is exactly one fact.
ONB5_V2_AGENTS_MD = ONB5_V1_AGENTS_MD.replace(ONB5_V1, ONB5_V2)
ONB5_ASK1 = "What is the maintenance window code? Reply with just the code."
ONB5_ASK2 = (
    "Double-check the project instructions, then reply with the maintenance "
    "window code currently in effect. Reply with just the code."
)


def onb_verdict(reply: str, ok: bool) -> tuple[bool, str, str | None]:
    """The section's whole reply-shaped verdict: ``(passed, outcome, non_answer)``.

    G2's decomposition (contract §5), shared by IMPORT rather than by copy — the
    truncation marker is cluster_g's pinned object, which ``tests/test_registry.py``
    pins against carbon's own source, so there is exactly one copy to keep true. A
    reply that is nothing but the marker, or a tool-call fragment, never stated
    what the instruction layer carried: ``not_attempted``, not a recall failure.
    Prose that attempted an answer and then hit the limit stays a plain ``fail``
    (strict starts-with, G2's rule)."""
    if ok:
        return True, "pass", None
    if reply.lstrip().startswith(G2_TRUNCATION_MARKER):
        return False, "not_attempted", "generation truncated before answer"
    if _TOOL_SYNTAX_RE.search(reply):
        return False, "not_attempted", "tool-syntax leak instead of answer"
    return False, "fail", None


def _loaded_layer_mismatch(directory: Path | str, expected: str) -> str | None:
    """The shared live premise check: what carbon's OWN loader returns for the
    fixture directory must be the authored bytes — the input side of the
    instructions door proven at attempt time, through ``load_agents_md`` itself,
    never a reimplementation or an assumption."""
    from harness.instructions import load_agents_md

    got = load_agents_md(directory)
    if got != expected:
        return f"loaded layer differs from authored fixture ({len(got)} vs {len(expected)} chars)"
    return None


def _plain_agent(agents_dir: str, **kwargs):
    """The cluster's no-tools agent. ``agents_dir`` is the SEEDED fixture dir —
    deliberately not the neutral dir every other cluster uses, because loading the
    fixture's AGENTS.md is the primitive under test."""
    from harness.agent import DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer

    provider = make_provider()
    kwargs.setdefault("tracer", Tracer(model=provider.model))
    return Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        agents_dir=agents_dir,
        **kwargs,
    )


def _agent(root: Path):
    """The cluster's discovery agent: read-only trio only (read_file, list_files,
    search_text — the tools carbon's own system prompt tells the model to prefer).
    No bash, no write, no edit: nothing can rewrite the layer or the seeded files
    mid-run, so oracle integrity holds by construction rather than by post-run
    hash checks. ``agents_dir`` and ``workspace_root`` are BOTH the seeded root."""
    from harness.tools import ToolRegistry, list_files_tool, read_file_tool, search_text_tool

    a = _plain_agent(str(root), **workspace_kwargs(root))
    tools = ToolRegistry()
    tools.register(read_file_tool(str(root), scratch_root=a.session_env.scratch_root))
    tools.register(list_files_tool(str(root)))
    tools.register(search_text_tool(str(root)))
    a.tools = tools
    return a


def _seed(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def run_onb1() -> Attempt:
    root = Path(mkdtemp(prefix="onb1-"))
    _seed(root, {"AGENTS.md": ONB1_AGENTS_MD, "docs/release.md": ONB1_DOC, **ONB1_CHAFF})
    a = _agent(root)
    try:
        mismatch = _loaded_layer_mismatch(root, ONB1_AGENTS_MD)
        if mismatch:
            return Attempt(
                False,
                "error",
                mismatch,
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        result = a.run(ONB1_PROMPT)
        reply = result.text
        in_reply = ONB1_SENTINEL.lower() in reply.lower()
        # Attribution, E4's conservative rule: the token must have ARRIVED through a
        # workspace read. The registry is read-only and root-confined and the token
        # exists only in docs/release.md, so a tool result carrying it IS a read of
        # the pointer target; a reply carrying it without one is refused.
        read_back = any(ONB1_SENTINEL in text for text in tool_texts(a.messages))
        ok, outcome, non_answer = onb_verdict(reply, in_reply and read_back)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"token_in_reply={in_reply} token_read_from_workspace={read_back} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        # `reply=` stays LAST — the trailing-match extractor convention (see run_g2).
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={**agent_metrics(a, result=result), "attempted": float(outcome != "not_attempted")},
    )


def run_onb2() -> Attempt:
    d = Path(mkdtemp(prefix="onb2-"))
    (d / "AGENTS.md").write_text(ONB2_AGENTS_MD)
    a = _plain_agent(str(d))
    try:
        mismatch = _loaded_layer_mismatch(d, ONB2_AGENTS_MD)
        if mismatch:
            return Attempt(
                False,
                "error",
                mismatch,
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        result = a.run(ONB2_PROMPT)
        reply = result.text
        has_override = ONB2_INCIDENT_SIG.lower() in reply.lower()
        has_general = ONB2_GENERAL_SIG.lower() in reply.lower()
        # The conjunction is the discriminator: obeying the general rule instead of
        # the override — the failure this task is built to catch — produces the
        # general sentinel, which is observable, not merely an absence.
        ok, outcome, non_answer = onb_verdict(reply, has_override and not has_general)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"override_sig={has_override} general_sig={has_general} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={**agent_metrics(a, result=result), "attempted": float(outcome != "not_attempted")},
    )


def run_onb3() -> Attempt:
    d = Path(mkdtemp(prefix="onb3-"))
    layer = onb3_agents_md()
    (d / "AGENTS.md").write_text(layer)
    a = _plain_agent(str(d))
    try:
        mismatch = _loaded_layer_mismatch(d, layer)
        if mismatch:
            return Attempt(
                False,
                "error",
                mismatch,
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        result = a.run(ONB3_PROMPT)
        reply = result.text
        has_current = ONB3_CURRENT.lower() in reply.lower()
        has_example = ONB3_EXAMPLE.lower() in reply.lower()
        ok, outcome, non_answer = onb_verdict(reply, has_current and not has_example)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"operative_code={has_current} placeholder_code={has_example} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={**agent_metrics(a, result=result), "attempted": float(outcome != "not_attempted")},
    )


def run_onb4() -> Attempt:
    root = Path(mkdtemp(prefix="onb4-"))
    _seed(root, {"AGENTS.md": ONB4_AGENTS_MD, "README.md": ONB4_README})
    a = _agent(root)
    try:
        mismatch = _loaded_layer_mismatch(root, ONB4_AGENTS_MD)
        if mismatch:
            return Attempt(
                False,
                "error",
                mismatch,
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        result = a.run(ONB4_PROMPT)
        reply = result.text
        # The observed premise (CMP-5's discipline): the conflict is live only once
        # the stale bytes actually reached the model. A reply produced without the
        # README read never faced the conflict this task measures — recorded as a
        # failed premise, never as a vacuous pass or a recall failure.
        stale_seen = any(ONB4_STALE in text for text in tool_texts(a.messages))
        if not stale_seen:
            return Attempt(
                False,
                "error",
                "conflict never became live: the stale README was not read before answering "
                f"(reply={reply[:160]!r})",
                turns=len(a.messages),
                metrics={**agent_metrics(a, result=result), "attempted": 0.0},
            )
        has_current = ONB4_CURRENT.lower() in reply.lower()
        has_stale = ONB4_STALE.lower() in reply.lower()
        ok, outcome, non_answer = onb_verdict(reply, has_current and not has_stale)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"current_token={has_current} stale_token={has_stale} stale_seen={stale_seen} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={**agent_metrics(a, result=result), "attempted": float(outcome != "not_attempted")},
    )


def run_onb5() -> Attempt:
    d = Path(mkdtemp(prefix="onb5-"))
    (d / "AGENTS.md").write_text(ONB5_V1_AGENTS_MD)
    a = _plain_agent(str(d))
    try:
        mismatch = _loaded_layer_mismatch(d, ONB5_V1_AGENTS_MD)
        if mismatch:
            return Attempt(
                False,
                "error",
                mismatch,
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        first = a.send(ONB5_ASK1)
        # The seeded conflict: the model's OWN v1 answer is the stale trace the
        # final ask must beat. Without it the transcript carries nothing of v1 (the
        # system head is rebuilt per call, so the rewrite erases v1 from everything
        # except this echo) and the final ask is trivially clean.
        if ONB5_V1.lower() not in first.lower():
            return Attempt(
                False,
                "error",
                f"stale echo never entered the transcript: turn 1 did not answer with the "
                f"v1 code (reply={first[:160]!r})",
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        (d / "AGENTS.md").write_text(ONB5_V2_AGENTS_MD)
        mismatch = _loaded_layer_mismatch(d, ONB5_V2_AGENTS_MD)
        if mismatch:
            return Attempt(
                False,
                "error",
                f"mid-session rewrite not visible to the loader: {mismatch}",
                turns=len(a.messages),
                metrics={**agent_metrics(a), "attempted": 0.0},
            )
        result = a.run(ONB5_ASK2)
        reply = result.text
        has_v2 = ONB5_V2.lower() in reply.lower()
        has_v1 = ONB5_V1.lower() in reply.lower()
        ok, outcome, non_answer = onb_verdict(reply, has_v2 and not has_v1)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=f"current_code={has_v2} superseded_code={has_v1} "
        + (f"non_answer={non_answer!r} " if non_answer else "")
        + f"reply={reply[:240]!r}",
        turns=len(a.messages),
        metrics={**agent_metrics(a, result=result), "attempted": float(outcome != "not_attempted")},
    )


SPECS = [
    # Priors are `uncertain` — a claim about the suite as authored, never a reading
    # of a baseline (the CMP-5/6/7 convention). Splits alternate held-in/held-out at
    # authoring; final assignment is a Phase 4 gate input.
    TaskSpec(
        "ONB-1", "held_in", "I", "uncertain", primitive="instructions", alias=None, run=run_onb1
    ),
    TaskSpec(
        "ONB-2", "held_out", "I", "uncertain", primitive="instructions", alias=None, run=run_onb2
    ),
    TaskSpec(
        "ONB-3", "held_in", "I", "uncertain", primitive="instructions", alias=None, run=run_onb3
    ),
    TaskSpec(
        "ONB-4", "held_out", "I", "uncertain", primitive="instructions", alias=None, run=run_onb4
    ),
    TaskSpec(
        "ONB-5", "held_in", "I", "uncertain", primitive="instructions", alias=None, run=run_onb5
    ),
]
