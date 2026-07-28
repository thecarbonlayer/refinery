import inspect
import json
import os
import re
from pathlib import Path

import pytest

from runner.helpers import (
    AGENT_METRIC_ATTRS,
    TRACER_TOTAL_KEYS,
    absolute_paths_outside,
    agent_metrics,
    bash_runs,
    environ_guard,
    exit_code_of,
    file_sha256,
    neutral_dir,
    scripted_approver,
    snapshot_tree,
    tool_texts,
    tree_changes,
)


def _probe_agent():
    from harness.agent import Agent
    from harness.observability import Tracer
    from model import Provider

    provider = Provider("fake://attrs", "attr-probe", responder=lambda messages, **kw: None)
    return Agent(
        provider=provider,
        model=provider.model,
        agents_dir=neutral_dir(),
        tracer=Tracer(model=provider.model),
    )


def test_agent_metric_attrs_still_exist_on_a_real_carbon_agent():
    """Each attribute read in ``agent_metrics`` is a getattr with a zero default,
    so a rename inside carbon would silently zero that metric forever and no
    baseline would look wrong. Pin the names against a real agent instead."""
    agent = _probe_agent()
    missing = [attr for attr in AGENT_METRIC_ATTRS if not hasattr(agent, attr)]
    assert not missing, f"carbon no longer exposes {missing} — agent_metrics would report 0"


def test_tracer_total_keys_still_exist_on_a_real_tracer():
    """The other half of the same hole, and the more expensive half: `tokens` and
    `cost` are read out of ``Tracer.totals()`` with a zero default and are exactly
    what the PR body publishes as cost evidence. A key rename there is every bit
    as silent as an attribute rename."""
    from harness.observability import Tracer

    totals = Tracer(model="attr-probe").totals()
    missing = [key for key in TRACER_TOTAL_KEYS if key not in totals]
    assert not missing, f"carbon's Tracer.totals() no longer reports {missing}"


def test_metric_name_lists_match_the_reads_in_agent_metrics():
    """A hand-maintained pin list that drifts from the function body protects
    nothing: renaming a read without touching the list leaves both green. Tie them
    together by reading the source of the function under test."""
    source = inspect.getsource(agent_metrics)
    attrs = set(re.findall(r"getattr\(agent, \"([^\"]+)\"", source))
    attrs |= set(re.findall(r"getattr\(agent, \"([^\"]+)\", None\)", source))
    totals_keys = set(re.findall(r"totals\.get\(\"([^\"]+)\"", source))
    assert attrs == set(AGENT_METRIC_ATTRS), (
        f"AGENT_METRIC_ATTRS is stale: source reads {sorted(attrs)}"
    )
    assert totals_keys == set(TRACER_TOTAL_KEYS), (
        f"TRACER_TOTAL_KEYS is stale: source reads {sorted(totals_keys)}"
    )


class _FakeAgent:
    """Stand-in for carbon's Agent, for VALUE-level assertions.

    The attribute names are pinned against a real agent above; what was untested was
    what `agent_metrics` computes from them. Four of nine fields survived mutation:
    the `role == "tool"` check, the error prefix, the `tool_calls` fallback, and the
    incomplete-response comparison could all be inverted or deleted with 154 tests
    still green — and these are the numbers the PR body publishes as evidence.
    """

    def __init__(self, **overrides):
        self.tracer = None
        self.messages = []
        self._turn_model_calls = 0
        self._stop_reason = "stop"
        self.compaction_count = 0
        self.retry_count = 0
        self.__dict__.update(overrides)


def test_agent_metrics_counts_only_tool_role_errors_and_falls_back_for_calls():
    agent = _FakeAgent(
        messages=[
            {"role": "assistant", "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "content": "error: no such file: x"},
            {"role": "tool", "content": "ok, 3 rows"},
            # An assistant message that merely starts with the word must NOT count.
            {"role": "user", "content": "error: this is not a tool result"},
        ]
    )
    recorded = agent_metrics(agent, include_cost=False)
    assert recorded["tool_errors"] == 1.0
    assert recorded["tool_calls"] == 2.0  # no tracer totals -> counts the messages


def test_agent_metrics_reads_each_tracer_total_into_the_right_field():
    """No test ever handed ``agent_metrics`` a tracer with NON-ZERO totals.

    So swapping the `tokens` and `cost` reads left every test green — `0.0 == 0.0` on
    both sides — and those are the two numbers the PR body publishes as cost evidence
    for a human vote. The key-set tests cannot see a value swap; only distinct values
    can. Same class as the defect the `_FakeAgent` tests were added to close, in the
    same function.
    """

    class _StubTracer:
        def totals(self):
            return {"llm_calls": 2, "tool_calls": 5, "tokens": 4567, "cost": 1.23, "seconds": 9.9}

    recorded = agent_metrics(_FakeAgent(tracer=_StubTracer()))
    assert recorded["tokens"] == 4567.0
    assert recorded["cost"] == 1.23
    assert recorded["llm_calls"] == 2.0
    assert recorded["tool_calls"] == 5.0  # tracer totals win over the message fallback


def test_agent_metrics_flags_only_an_incomplete_stop_reason():
    assert (
        agent_metrics(_FakeAgent(_stop_reason="incomplete_response"))["incomplete_responses"] == 1
    )
    for reason in ("stop", "tool_budget", ""):
        assert agent_metrics(_FakeAgent(_stop_reason=reason))["incomplete_responses"] == 0.0


def test_agent_metrics_reports_plain_attribute_counters():
    recorded = agent_metrics(_FakeAgent(compaction_count=3, retry_count=2, _turn_model_calls=7))
    assert recorded["compactions"] == 3.0
    assert recorded["retries"] == 2.0
    assert recorded["model_attempts"] == 7.0


def test_agent_metrics_omits_only_cost_fields_when_asked():
    """Assert both key sets explicitly. Defining the expectation in terms of the
    full set makes any SYMMETRIC change pass — deleting a metric from both sides
    would go unnoticed."""
    agent = _probe_agent()
    mechanism = {
        "llm_calls",
        "model_attempts",
        "tool_calls",
        "compactions",
        "tool_errors",
        "incomplete_responses",
        "retries",
    }
    assert set(agent_metrics(agent)) == mechanism | {"tokens", "cost"}
    assert set(agent_metrics(agent, include_cost=False)) == mechanism


def _bash_call(call_id: str, command: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


def test_exit_code_of():
    assert exit_code_of("[exit 0 via trusted]\nGATE_OK") == 0
    assert exit_code_of("[exit 1 via trusted]\nAssertionError") == 1
    assert exit_code_of("[exit 124 via docker]\n") == 124
    assert exit_code_of("[denied by approval gate]") is None
    assert exit_code_of("plain text") is None


def test_bash_runs_pairs_calls_with_results():
    messages = [
        _bash_call("c1", "python3 test_gate.py"),
        {"role": "tool", "tool_call_id": "c1", "content": "[exit 1 via trusted]\nCI token missing"},
        _bash_call("c2", "ls"),
        {"role": "tool", "tool_call_id": "c2", "content": "[exit 0 via trusted]\nfix_me.py"},
    ]
    runs = bash_runs(messages, lambda cmd: "python3 test_gate.py" in cmd)
    assert len(runs) == 1
    cmd, result = runs[0]
    assert cmd == "python3 test_gate.py"
    assert exit_code_of(result) == 1


def test_bash_runs_pairs_structurally_when_ids_are_empty():
    # carbon stores tc.get("id", ""), so empty (duplicate) ids are reachable with
    # local models — a dict lookup by id would let the decoy's passing result
    # overwrite the gate's failing one.
    messages = [
        _bash_call("", "python3 test_gate.py"),
        {"role": "tool", "tool_call_id": "", "content": "[exit 1 via trusted]\nGATE_FAIL"},
        _bash_call("", "echo decoy"),
        {"role": "tool", "tool_call_id": "", "content": "[exit 0 via trusted]\nGATE_OK"},
    ]
    runs = bash_runs(messages, lambda cmd: "test_gate.py" in cmd)
    assert runs == [("python3 test_gate.py", "[exit 1 via trusted]\nGATE_FAIL")]


def test_tool_texts_collects_only_tool_messages():
    messages = [
        {"role": "user", "content": "SENTINEL-IN-PROMPT"},
        {"role": "tool", "tool_call_id": "x", "content": "tool says SENTINEL-IN-TOOL"},
        {"role": "assistant", "content": "reply"},
    ]
    texts = tool_texts(messages)
    assert texts == ["tool says SENTINEL-IN-TOOL"]


def test_scripted_approver_logs_and_approves():
    log: list[dict] = []
    approve = scripted_approver(log)
    assert approve("bash", '{"command": "ls"}') is True
    assert log == [{"tool": "bash", "args": '{"command": "ls"}'}]


def test_environ_guard_sets_unsets_and_restores():
    os.environ["EG_KEEP"] = "orig"
    os.environ["EG_DROP"] = "present"
    with environ_guard(set_vars={"EG_NEW": "v", "EG_KEEP": "shadow"}, unset=("EG_DROP",)):
        assert os.environ["EG_NEW"] == "v"
        assert os.environ["EG_KEEP"] == "shadow"
        assert "EG_DROP" not in os.environ
    assert "EG_NEW" not in os.environ
    assert os.environ["EG_KEEP"] == "orig"
    assert os.environ["EG_DROP"] == "present"
    del os.environ["EG_KEEP"], os.environ["EG_DROP"]


def test_file_sha256_and_tree_snapshot(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("hello")
    h1 = file_sha256(f)
    before = snapshot_tree(tmp_path)
    assert before == {"a.txt": h1}
    f.write_text("changed")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("new")
    changed = tree_changes(tmp_path, before)
    assert sorted(changed) == ["a.txt", "sub/b.txt"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read chmod-000 files")
def test_snapshot_tree_marks_unreadable_files(tmp_path: Path):
    f = tmp_path / "locked.txt"
    f.write_text("secret")
    f.chmod(0o000)
    try:
        snap = snapshot_tree(tmp_path)
        assert snap["locked.txt"] == "<unreadable>"
    finally:
        f.chmod(0o644)


def test_absolute_paths_outside(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    # inside-root absolute path is fine; /etc/hosts is not; relative paths are fine
    assert absolute_paths_outside(f"cat {root}/config/app.conf", root) == []
    assert absolute_paths_outside("cat /etc/hosts", root) == ["/etc/hosts"]
    assert absolute_paths_outside('{"path": "config/app.conf"}', root) == []
    # arithmetic like 4/2 must not be flagged as a path
    assert absolute_paths_outside("echo $((4/2))", root) == []
    # URLs must not be flagged either
    assert absolute_paths_outside("curl http://example.com/path", root) == []


def test_absolute_paths_outside_accepts_unresolved_root(tmp_path: Path):
    # macOS: mkdtemp hands out /var/folders/... while resolve() canonicalizes
    # to /private/var/folders/... — the literal (unresolved) workspace path the
    # agent actually receives must not be flagged as an escape.
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real_dir)
    ws = link / "ws"
    ws.mkdir()
    assert absolute_paths_outside(f"cat {ws}/file.txt", ws) == []
    assert absolute_paths_outside("cat /etc/hosts", ws) == ["/etc/hosts"]


def test_absolute_paths_outside_allows_write_only_devices(tmp_path: Path):
    """`> /dev/null` (and stdout/stderr/tty) is shell plumbing, not an escape —
    but the allowlist must not launder a real outside path in the same command."""
    root = tmp_path / "ws"
    root.mkdir()
    assert absolute_paths_outside("ls config > /dev/null", root) == []
    assert absolute_paths_outside("cmd 2> /dev/stderr 1> /dev/stdout < /dev/tty", root) == []
    assert absolute_paths_outside("cat /etc/hosts > /dev/null", root) == ["/etc/hosts"]
    # only EXACT tokens are allowlisted — prefixes/extensions are still flagged
    assert absolute_paths_outside("ls /dev/nullish", root) == ["/dev/nullish"]
    assert absolute_paths_outside("cat /dev/null/../shm/x", root) == ["/dev/null/../shm/x"]


def test_arg_texts_decodes_json_string_values(tmp_path: Path):
    """JSON slash-escaping must not hide an absolute path: the raw string
    misses \\/etc\\/hosts, so the decoded values must be scanned too."""
    from runner.helpers import arg_texts

    root = tmp_path / "ws"
    root.mkdir()
    raw = '{"path": "\\/etc\\/hosts"}'
    texts = arg_texts(raw)
    assert raw in texts  # raw string always scanned
    assert any(absolute_paths_outside(t, root) == ["/etc/hosts"] for t in texts), (
        "decoded /etc/hosts must be visible to the scan"
    )


def test_arg_texts_decodes_json_string_keys(tmp_path: Path):
    """A slash-escaped path in KEY position must not evade the scan —
    '{"\\/etc\\/hosts": true}' has no string values at all."""
    from runner.helpers import arg_texts

    root = tmp_path / "ws"
    root.mkdir()
    raw = '{"\\/etc\\/hosts": true}'
    texts = arg_texts(raw)
    assert any(absolute_paths_outside(t, root) == ["/etc/hosts"] for t in texts), (
        "decoded key /etc/hosts must be visible to the scan"
    )


def test_arg_texts_handles_nested_json_and_non_json():
    from runner.helpers import arg_texts

    nested = '{"cmd": {"args": ["\\/etc\\/passwd", 3]}, "note": "x"}'
    joined = " ".join(arg_texts(nested))
    assert "/etc/passwd" in joined
    # non-JSON falls back to the raw string only
    assert arg_texts("cat /etc/hosts") == ["cat /etc/hosts"]


def test_absolute_paths_outside_home_and_single_segment(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    # single-segment absolute paths are visible
    assert absolute_paths_outside("cat /etc", root) == ["/etc"]
    # ~ and $HOME expansions are always outside the workspace root
    assert absolute_paths_outside("cat ~/secrets/keys.txt", root) == ["~/secrets/keys.txt"]
    assert absolute_paths_outside("ls $HOME/private", root) == ["$HOME/private"]
