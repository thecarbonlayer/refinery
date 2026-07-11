import json
import os
from pathlib import Path

import pytest

from runner.helpers import (
    absolute_paths_outside,
    bash_runs,
    environ_guard,
    exit_code_of,
    file_sha256,
    scripted_approver,
    snapshot_tree,
    tool_texts,
    tree_changes,
)


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
    # gemma stores tc.get("id", ""), so empty (duplicate) ids are reachable with
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


def test_absolute_paths_outside_home_and_single_segment(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    # single-segment absolute paths are visible
    assert absolute_paths_outside("cat /etc", root) == ["/etc"]
    # ~ and $HOME expansions are always outside the workspace root
    assert absolute_paths_outside("cat ~/secrets/keys.txt", root) == ["~/secrets/keys.txt"]
    assert absolute_paths_outside("ls $HOME/private", root) == ["$HOME/private"]
