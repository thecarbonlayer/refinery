"""Cluster E — information never reaches the model.

The original A2/A4 pair was written against a 4k PREFIX clamp, which could hide a
fact near the end of a moderately large item. That is no longer the shipped policy:
`head_tail` retains a tail, so A2's sentinel survives truncation and A2 now measures
1.000 — its old `fail` prior was never caused by `tool_output`.

These two tasks make the requirement explicit, and each pins a DIFFERENT distinction
so the strategy menu can be compared rather than merely scaled:

- E2's tag sits at the very end, so a tail-preserving policy keeps it and a head-only
  policy loses it, at a FIXED budget. That is the shape a strategy test should have.
- E1's needle sits in the middle, which no shipped strategy reaches at any sane
  budget, and it must be retrieved without flooding the window. So E1 is not a
  setting to tune — it is a standing request for a middle-preserving strategy.

The region no task covers yet is a fact in the HEAD window, between the head a
tail-preserving policy keeps and the larger head a head-only policy keeps. Without it
nothing rewards head-only, so switching away from it looks free while silently losing
every fact that lives there.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import Path

from runner.carbon_env import make_provider
from runner.helpers import (
    agent_metrics,
    bash_runs,
    neutral_dir,
    scripted_approver,
    tool_runs,
    tool_texts,
    workspace_kwargs,
)
from runner.spec import Attempt, TaskSpec

E1_SENTINEL = "RANGE-ACCESS-E1-7QK"
# E1 passes only if the needle is retrieved while delivering less than this share of
# the fixture into the window. Without it the sole way through is raising the clamp
# until a naive whole-file read succeeds, which takes 24% of a 114 KB file into
# context — precisely the "flooding" the task exists to rule out. A fraction of the
# fixture, never a char count, so it survives any fixture resize.
E1_MAX_DELIVERED_SHARE = 0.10

# E2's tag is COMPUTED, never written into the script. The literal form was readable
# in the source, and the assistant duly read it and reported the answer without ever
# running the command — so the task graded "did you obey" instead of "did the tail
# survive". A sha the model cannot evaluate by inspection makes execution the only
# route, and leaves the tag's tail POSITION, which is what E2 actually measures.
E2_TAG_SEED = "e2-root-cause-seed-4417"
E2_SENTINEL = hashlib.sha256(E2_TAG_SEED.encode()).hexdigest()[:12].upper()
E2_PASS_COUNT = 3200

# E3 mirrors E2's derived-tag construction — the script carries the seed and the
# algorithm, never the answer — but plants the tag at the MIDPOINT of the stream
# rather than the tail, which is the position no shipped truncation strategy keeps.
E3_TAG_SEED = "e3-reconciliation-seed-9182"
E3_SENTINEL = hashlib.sha256(E3_TAG_SEED.encode()).hexdigest()[:12].upper()
E3_LINE_COUNT = 3200
E3_NEEDLE_INDEX = E3_LINE_COUNT // 2

# E4 is E3 at the scale where recovery, not delivery, is the only question left.
# Same derived-tag discipline, but the needle sits astride the exact midpoint of a
# stream so large that an inline excerpt reaching it would have to carry HALF the
# stream — so E3 asks whether the shipped cut reaches the middle, and E4 asks
# whether the middle is still recoverable after the cut has done its work.
E4_TAG_SEED = "e4-settlement-seed-7305"
E4_SENTINEL = hashlib.sha256(E4_TAG_SEED.encode()).hexdigest()[:12].upper()
E4_LINE_COUNT = 20_000
E4_NEEDLE_INDEX = E4_LINE_COUNT // 2
E4_SCRIPT = "run_settlement.py"
E4_STAMP = ".settlement-consumed"


def _plain_agent(*, workspace_root: Path | str, approvals: list[dict] | None = None):
    """Every E task's agent. ``workspace_root`` is required, not defaulted.

    E4's whole subject is a path the model has to be able to open, and the two
    directories in play are easy to conflate: ``agents_dir`` stays neutral so no
    stray AGENTS.md reaches the system prompt, while the workspace is the tree the
    tools are bound to and the only tree a task's ``read_file`` can reach. A
    default here would let a later task inherit the mismatch silently, which is
    the failure mode ``workspace_kwargs`` exists to end — so each caller names the
    workspace it wired its tools to.

    Agent-first, tools-after (the canonical shape, task-8/task-7): built with NO
    ``session_env``, so ``__init__`` creates one and owns it — ``close()`` then
    really ends its lifecycle. Tools are the CALLER's job, registered after this
    returns and bound via ``agent.tools = tools``, so a ``read_file`` needing
    ``scratch_root`` can pull it from the agent this just built (``E4`` is the
    task that needs it; the others simply don't reach for it).
    """
    from harness.agent import APPROVAL_TOOLS, DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer

    provider = make_provider()
    return Agent(
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        agents_dir=neutral_dir(),
        **workspace_kwargs(workspace_root),
        approve=scripted_approver(approvals) if approvals is not None else None,
        approval_required=APPROVAL_TOOLS if approvals is not None else set(),
        tracer=Tracer(model=provider.model),
    )


def _large_reference() -> str:
    lines = [
        f"record {i:04d}: component-{i % 17} healthy, owner=team-{i % 9}, status=nominal"
        for i in range(1800)
    ]
    lines[1537] = f"record 1537: release authorization code={E1_SENTINEL}"
    body = "\n".join(lines) + "\n"
    assert len(body) > 100_000
    assert body.find(E1_SENTINEL) > 80_000
    return body


def run_e1() -> Attempt:
    """A late fact in a 100k file, retrieved WITHOUT flooding the window.

    Recalling the code is necessary but not sufficient: economy is part of the verdict,
    because a budget large enough to carry a quarter of the file passes the letter of
    the task and defeats its purpose.

    The belt is the harness's real retrieval belt — ``search_text`` to locate and
    ranged ``read_file`` to confirm. An earlier revision withheld ``search_text`` and
    read the resulting failure as evidence that the harness lacked indexed retrieval;
    it does not, and a task may not manufacture the gap it reports. Withholding a
    shipped tool measures the fixture, not the harness.

    What stays measured is retrieval *economy* — locate the needle without streaming
    the haystack. Middle-of-output retention, which this task used to stand in for, is
    a genuinely different question and is measured on opaque output by E3, where no
    query tool can apply.
    """
    from harness.tools import ToolRegistry, read_file_tool, search_text_tool
    from harness.workspace import Workspace

    body = _large_reference()
    ws = Workspace()
    ws.write("release-records.txt", body)
    a = _plain_agent(workspace_root=ws.root)
    tools = ToolRegistry()
    tools.register(read_file_tool(str(ws.root), scratch_root=a.session_env.scratch_root))
    tools.register(search_text_tool(str(ws.root)))
    a.tools = tools
    try:
        reply = a.send(
            "Find the release authorization code in release-records.txt. "
            "Reply with just the exact code."
        )
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    called = any(m.get("role") == "tool" for m in a.messages)
    recalled = E1_SENTINEL.lower() in reply.lower()
    delivered = sum(len(t) for t in tool_texts(a.messages))
    share = delivered / len(body)
    economical = share <= E1_MAX_DELIVERED_SHARE
    ok = called and recalled and economical
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"read_called={called} sentinel_recalled={recalled} "
        f"delivered_share={share:.3f} (limit {E1_MAX_DELIVERED_SHARE}) "
        f"economical={economical} reply={reply[:200]!r}",
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def _long_test_output() -> str:
    prefix = "\n".join(
        f"PASS test_component_{i:04d} ({(i % 13) + 1}ms)" for i in range(E2_PASS_COUNT)
    )
    body = prefix + f"\nFAILED integration: root-cause-tag={E2_SENTINEL}\n"
    assert len(body) > 80_000
    assert body.rstrip().endswith(E2_SENTINEL)
    return body


def e2_script() -> str:
    """The script E2 asks the model to run — and which the model can freely read.

    The tag is DERIVED here rather than written in, so the text carries the seed and
    the algorithm but not the answer. In the literal form the model simply read the
    tag out of the source and replied without running anything, which made the task
    grade obedience instead of tail survival. A public function so a test can assert
    on the exact bytes the model sees, rather than on this module's source, where the
    seed appears only as an identifier.
    """
    return (
        "import hashlib\n"
        f"for i in range({E2_PASS_COUNT}):\n"
        "    print(f'PASS test_component_{i:04d} ({(i % 13) + 1}ms)')\n"
        f"tag = hashlib.sha256({E2_TAG_SEED!r}.encode()).hexdigest()[:12].upper()\n"
        "print(f'FAILED integration: root-cause-tag={tag}')\n"
    )


E2_POST_PROCESSORS = ("tail", "head", "grep", "sed", "awk")
_STDOUT_REDIRECTS = (">", ">>")


def _shell_trims_output(cmd: str) -> bool:
    """Would the SHELL cut this command's output before carbon's door sees it?

    Matched on TOKENS, never on substrings. Substring matching over the whole command
    was a live false-negative source, and every miss scored a real run 0.000 and
    charged it to truncation policy:

      - ``python3 run_tests.py 2>&1`` — excluded for containing ``>``. Merging stderr
        into stdout trims nothing, and it is a very common idiom.
      - ``cd ahead && python3 run_tests.py`` — excluded because "head" sits inside
        "ahead". Same for any path with "overhead", "headers", "grepped".
      - ``# read the header first`` + a plain run — excluded on a comment.

    Redirect handling is file-descriptor aware for the same reason: ``2> err.txt``
    and ``2>&1`` leave stdout whole, while ``> out.txt`` and ``>> out.txt`` divert it.

    An unparsable command counts as trimming. The alternative is to guess about a
    command whose shape we cannot read, and this predicate's job is to protect what
    E2/E3/E4 mean — an over-permissive read grades a shell-side rescue as evidence
    about the truncation policy.
    """
    lexer = shlex.shlex(cmd, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return True
    for i, token in enumerate(tokens):
        previous = tokens[i - 1] if i else None
        following = tokens[i + 1] if i + 1 < len(tokens) else None
        if token == "|":
            return True
        if token in _STDOUT_REDIRECTS and previous != "2":
            return True
        if token == ">&" and following != "1":  # `2>&1` merges; `>&file` diverts
            return True
        if token in E2_POST_PROCESSORS:
            return True
    return False


def _ran_script_plainly(cmd: str) -> bool:
    """Did this command run the script, without the SHELL doing the trimming?

    Two failure modes to avoid, and the obvious predicate hits both. Requiring the
    command to *start with* the invocation reports a false negative whenever the model
    wraps it — a `cd` first, or two lines in one call. That is what happened: the model
    sent ``ls run_tests.py`` and ``python3 run_tests.py`` as a single two-line command,
    ran the script, reported the right tag, and was recorded as never having run it.
    E2 read 0.000 for a reason that had nothing to do with truncation.

    Matching anywhere in the command fixes that but opens the other hole: a
    ``... | tail -5`` invocation preserves the tail by shell, not by policy, which is
    the very thing the task is trying to measure. So those are excluded instead.
    """
    return "python3 run_tests.py" in cmd and not _shell_trims_output(cmd)


def _ran_audit_plainly(cmd: str) -> bool:
    """E3's counterpart to ``_ran_script_plainly`` — same two failure modes, same fix.

    The exclusion matters more here than in E2: a middle needle is exactly what a
    ``| sed -n`` or ``| grep`` recovers trivially, and a shell-side rescue would grade
    the model's resourcefulness instead of the truncation policy under test.
    """
    return "python3 run_audit.py" in cmd and not _shell_trims_output(cmd)


def run_e2() -> Attempt:
    """The failure at the tail of noisy command output must survive door control."""
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import ToolRegistry
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("run_tests.py", e2_script())
    approvals: list[dict] = []
    a = _plain_agent(workspace_root=ws.root, approvals=approvals)
    tools = ToolRegistry()
    tools.register(bash_tool(Sandbox(trusted=True, timeout=60), workdir=str(ws.root)))
    a.tools = tools
    try:
        reply = a.send(
            "Run `python3 run_tests.py` and report the exact root-cause-tag from the failed test. "
            "Reply with just the tag."
        )
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    runs = bash_runs(a.messages, _ran_script_plainly)
    called = bool(runs)
    # A `... | tail` call may let a capable model recover despite the harness, so
    # post-processed invocations are excluded above rather than counted here: the
    # mechanism under test is whether the ORDINARY run kept the useful tail.
    first_result_kept_tail = called and E2_SENTINEL in runs[0][1]
    recalled = E2_SENTINEL.lower() in reply.lower()
    ok = called and first_result_kept_tail and recalled
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"bash_called={called} first_result_kept_tail={first_result_kept_tail} "
        f"sentinel_recalled={recalled} reply={reply[:240]!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def e3_script() -> str:
    """The script E3 asks the model to run. Derived tag, same reasoning as E2's."""
    return (
        "import hashlib\n"
        f"tag = hashlib.sha256({E3_TAG_SEED!r}.encode()).hexdigest()[:12].upper()\n"
        f"for i in range({E3_LINE_COUNT}):\n"
        f"    if i == {E3_NEEDLE_INDEX}:\n"
        "        print(f'AUDIT ANOMALY: reconciliation-tag={tag}')\n"
        "    else:\n"
        "        print(f'audit row {i:05d}: ledger balanced, delta=0.00')\n"
    )


def run_e3() -> Attempt:
    """A fact in the MIDDLE of opaque command output must survive door control.

    E2 puts its needle at the tail, so ``head_tail`` carries it by construction. This
    task puts one at the midpoint, where neither shipped truncation strategy reaches:
    ``keep_head`` takes a prefix and ``head_tail`` takes both ends. No budget short of
    delivering the whole stream retrieves it, and delivering the whole stream is the
    flooding these doors exist to prevent.

    The output is *opaque* on purpose. E1's fixture was a file, so ``search_text``
    answers it and the honest verdict there is retrieval economy. A command's stdout
    has no index to query — the only thing standing between the middle of the stream
    and the model is the truncation policy itself. That makes this, and not E1, the
    task that legitimately reports a missing middle-preserving strategy.

    Consequence, stated plainly: no value of any current setting passes this task. It
    is a Carbon capability request, not a number to tune, and it stays red until a
    middle-preserving strategy exists.
    """
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import ToolRegistry
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write("run_audit.py", e3_script())
    approvals = [{"tool": "bash", "decision": "approve"}]
    a = _plain_agent(workspace_root=ws.root, approvals=approvals)
    tools = ToolRegistry()
    tools.register(bash_tool(Sandbox(trusted=True, timeout=60), workdir=str(ws.root)))
    a.tools = tools
    try:
        reply = a.send(
            "Run `python3 run_audit.py` and report the reconciliation tag it prints. "
            "Reply with just the exact tag."
        )
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    runs = bash_runs(a.messages, _ran_audit_plainly)
    ran = bool(runs)
    # The FIRST plain run is the one under test: a later, narrowed re-run would be the
    # model routing around the door rather than the door holding.
    delivered = ran and E3_SENTINEL in runs[0][1]
    recalled = E3_SENTINEL.lower() in reply.lower()
    ok = ran and delivered and recalled
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"ran_plainly={ran} sentinel_delivered={delivered} "
        f"recalled={recalled} reply={reply[:200]!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


def e4_script() -> str:
    """The script E4 asks the model to run. Derived tag, per E2/E3 — plus a stamp.

    The stamp makes the stream ONE-SHOT: the first run writes a consumed marker and
    any later run refuses with a single line instead of replaying. E2 and E3 keep
    shell-side rescues out by excluding post-processed invocations at the verifier,
    but E4 cannot: recovery through LATER commands is the very thing it measures, so
    a re-run piped through ``grep`` would be indistinguishable from the recovery it
    exists to detect. The fixture closes that route instead — a stream that cannot
    be regenerated is also the honest shape, because an output nobody can cheaply
    reproduce is exactly the output worth writing to disk.
    """
    return (
        "import hashlib, os\n"
        f"if os.path.exists({E4_STAMP!r}):\n"
        "    print('settlement stream already consumed; a re-run cannot regenerate it')\n"
        "    raise SystemExit(1)\n"
        f"with open({E4_STAMP!r}, 'w') as fh:\n"
        "    fh.write('consumed')\n"
        f"tag = hashlib.sha256({E4_TAG_SEED!r}.encode()).hexdigest()[:12].upper()\n"
        f"for i in range({E4_LINE_COUNT}):\n"
        f"    if i == {E4_NEEDLE_INDEX}:\n"
        "        print(f'SETTLEMENT ANOMALY: settlement-tag={tag}')\n"
        "    else:\n"
        "        print(f'ledger row {i:05d}: settled, carryover=0.00')\n"
    )


def _ran_settlement_plainly(cmd: str) -> bool:
    """E4's counterpart to ``_ran_script_plainly`` — same wrapping tolerance, same
    post-processor exclusions, for the same reasons. Only the FIRST plain run is
    the door under test; everything after it is recovery, judged separately."""
    return f"python3 {E4_SCRIPT}" in cmd and not _shell_trims_output(cmd)


# A replay is an INVOCATION of the script, not any mention of its name: the offload
# file's own name is harness-chosen and could embed the generating command, so a
# bare substring match would let a legitimate `grep` of the artifact taint the very
# file it reads. Requiring interpreter-then-whitespace excludes hyphen/slug forms.
_E4_REPLAY_RE = re.compile(rf"python[\w.]*\s+\S*{re.escape(E4_SCRIPT)}")
_E4_DERIVE_MARKS = (E4_TAG_SEED, "hashlib")


def _e4_regenerates_or_derives(call_text: str) -> bool:
    """Does this tool call MAKE the tag rather than read it back?

    Two make-routes, both real: re-running the generator (the stamp refuses it, but
    the refusal must not depend on the stamp file surviving an agent-writable
    workspace), and computing ``sha256(seed)`` directly — the derived-tag
    discipline necessarily leaves seed and algorithm readable in the script, so a
    model that reads the source can mint the tag without ever touching the
    offloaded file. Any command that types the seed, or reaches for ``hashlib``,
    is derivation-shaped; a read of an on-disk artifact never needs either.

    Enumerated residuals, not oversights: hashing binaries other than python
    (``shasum``, ``openssl``) escape when the seed itself was laundered through a
    file first, and a multi-step rename chain can wash the taint off a regenerated
    dump. Both take a model deliberately optimizing against a verifier it cannot
    see — the graded party is a CONFIG value, and no legal config value can induce
    either sequence."""
    return bool(_E4_REPLAY_RE.search(call_text)) or any(m in call_text for m in _E4_DERIVE_MARKS)


def _e4_recovered_from_disk(messages: list[dict], scratch_root: Path) -> bool:
    """Did the tag reach the model by READING an on-disk artifact it can name?

    The offload contract this task measures is outcome-shaped on purpose: the
    complete raw result lands in a file in the session's private scratch
    (harness/session_env.py), at a ``scratch://`` ref the marker makes
    discoverable — never a workspace path. The marker's wording, the offload
    directory name, and the filename are all carbon's to choose and to change, so
    nothing here parses a marker or pins a path. Attribution is positive instead:
    some file under scratch contains the sentinel, and some tool call both names
    a component of that file's path and came back with the sentinel in its
    result. A `grep`/`sed` names the file or globs its directory; `read_file`
    names it in its args (a ``scratch://offload/<name>`` ref, whose parts —
    ``offload``, ``<name>`` — still match a path component the same way a
    workspace-relative path would); `find | xargs` names neither and is NOT
    credited — the conservative direction, since a missed legitimate read
    under-reports the strategy and never over-reports it.

    The taint rule closes the fabrication route: a file whose path is named by any
    replay- or derivation-shaped command (a regenerated ``> dump.txt``, a minted
    ``> tag.txt``) is disqualified as an attribution target, so making the tag and
    then reading it back does not count as recovering it."""
    calls = tool_runs(messages, ("bash", "read_file"))
    marker_texts = [args for _name, args, _result in calls if _e4_regenerates_or_derives(args)]
    for path in sorted(scratch_root.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if E4_SENTINEL not in text:
            continue
        parts = set(path.relative_to(scratch_root).parts)
        if any(part in marker for marker in marker_texts for part in parts):
            continue  # named by a command that makes the tag: fabricated, not offloaded
        for _name, args, result in calls:
            if E4_SENTINEL in result and any(part in args for part in parts):
                return True
    return False


def run_e4() -> Attempt:
    """A fact cut away at the door must be RECOVERABLE from disk afterwards.

    E3 established that no positional strategy reaches the middle of opaque command
    output; this task states the requirement the E-cluster was building toward: the
    door may withhold the middle, but it must not DESTROY it. The stream here is
    ~840KB with the needle astride the exact midpoint, so an inline excerpt reaches
    it only when the budget exceeds half the stream — at that size the excerpt is
    not a truncation policy but a flood, and the verifier refuses it separately
    (``withheld``). At every honest budget, under every tail split, the inline text
    is dark; the only route to the tag is out-of-band: notice that the result was
    truncated, follow the path the door left behind, page or search the artifact,
    and read the tag back. A strategy that offloads the full result to a file
    passes; a strategy that merely excerpts has nothing on disk to find, which is
    why this task is red by design until such a strategy is configured.

    Four conjuncts, each carrying one claim: ``ran`` — the plain, un-postprocessed
    first run is the door under test (E2/E3's discipline); ``withheld`` — the
    needle was NOT delivered inline, so a flood-sized budget cannot buy the pass
    E1 exists to price out; ``recovered`` — the tag came back from an on-disk
    artifact the transcript actually names (see ``_e4_recovered_from_disk``; bare
    tag presence is not enough, because the derived-tag discipline leaves the tag
    computable from the readable script source, and a computed tag would grade
    model cleverness under every strategy alike); ``recalled`` — it reached the
    final answer. The fixture is one-shot (``e4_script``), so a failed door cannot
    be papered over by regenerating the stream into a shell filter.
    """
    from harness.sandbox import Sandbox, bash_tool
    from harness.tools import ToolRegistry, read_file_tool
    from harness.workspace import Workspace

    ws = Workspace()
    ws.write(E4_SCRIPT, e4_script())
    approvals: list[dict] = []
    # The workspace is named explicitly so the script lands in the tree `bash`
    # above runs in; the recovery artifact itself lands in the agent's OWN
    # scratch (below), never the workspace — the door spills there, not here.
    a = _plain_agent(workspace_root=ws.root, approvals=approvals)
    tools = ToolRegistry()
    tools.register(bash_tool(Sandbox(trusted=True, timeout=60), workdir=str(ws.root)))
    # The harness's own paging belt rides along: the offload convention hands back
    # a scratch:// ref meant for ranged reads, and withholding the tool that reads
    # ranges would manufacture the failure this task exists to measure. scratch_root
    # is THIS agent's own session_env — it must own it (Agent-first, tools-after
    # above) so `read_file` resolves the same scratch the door spills into.
    tools.register(read_file_tool(str(ws.root), scratch_root=a.session_env.scratch_root))
    a.tools = tools
    try:
        reply = a.send(
            f"Run `python3 {E4_SCRIPT}` and report the settlement tag from the anomaly "
            "line. The stream is generated once — a re-run will not replay it. "
            "Reply with just the exact tag."
        )
        plain = bash_runs(a.messages, _ran_settlement_plainly)
        ran = bool(plain)
        withheld = ran and E4_SENTINEL not in plain[0][1]
        # Must run BEFORE close(): the recovery artifact lives in scratch, and
        # close() (in the finally below) removes it — read while it still exists.
        recovered = _e4_recovered_from_disk(a.messages, a.session_env.scratch_root)
        recalled = E4_SENTINEL.lower() in reply.lower()
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    ok = ran and withheld and recovered and recalled
    return Attempt(
        passed=ok,
        outcome="pass" if ok else "fail",
        detail=f"ran_plainly={ran} needle_withheld_inline={withheld} "
        f"recovered_from_disk={recovered} recalled={recalled} reply={reply[:200]!r}",
        approvals=approvals,
        turns=len(a.messages),
        metrics=agent_metrics(a),
    )


SPECS = [
    TaskSpec("E1", "held_in", "E", "uncertain", run_e1),
    TaskSpec("E2", "held_out", "E", "pass", run_e2),
    TaskSpec("E3", "held_in", "E", "fail", run_e3),
    TaskSpec("E4", "held_out", "E", "fail", run_e4),
]
