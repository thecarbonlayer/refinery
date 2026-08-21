"""No machine-identifying path may sit in a committed file — enforced, not remembered.

This repo is public and results are the committed record. The prescribed pre-commit
grep covered `/Users/` and `/home/` and was blind to `/var/folders`, so two committed
candidate logs carried per-run temp paths for weeks — and a baseline log was found
holding an entire `$PATH` dump, home directory included, quoted inside a task detail.

This test is the recurrence-killer. Rows are born scrubbed now: `runner.run.
write_record` scrubs every string at serialization, after verifiers have already read
the raw text. This module remains the enforcement regardless — nothing covered here
may carry a machine path, whether the cause is a bug in that scrub, a record resumed
from before it existed, or a file dropped in by hand — so a red suite the moment an
unscrubbed artifact lands still makes the scrub impossible to forget.
`loop/scrub_results.py` remains the repair tool for anything older: it fixes what this
test finds, and proves it changed nothing but the offending strings.

WHAT IT COVERS, and why that grew. The gate began at `results/*.json*` — the files the
scrubber writes — and a plan document under `docs/` sat in the public history for days
with a home directory in three shell commands. The leak was never in the record; it was
in the prose beside it, which nothing checked. `results/*.log` joins for the same
reason: a run log is machine-generated evidence the scrubber never touches.

TWO GATES, because the material differs. Machine-generated files (`results/`) are held
to the bare patterns: nothing there has any business naming a home directory or a temp
dir, in any form. Prose (`docs/`) is held to the same patterns anchored on a following
path segment, and exempts a line that appears verbatim in this repo's own test sources
— a document that NAMES `/Users/` while stating the rule, or quotes a committed test
fixture, is not leaking a machine path, and a gate that cannot tell the difference is
one people learn to route around.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DOCS = REPO_ROOT / "docs"
TESTS = REPO_ROOT / "tests"

FORBIDDEN = (
    re.compile(r"/var/folders/"),
    re.compile(r"/Users/"),
    re.compile(r"/home/\w"),
)
# The same three, each requiring a path segment to follow: what separates a real
# machine path from prose about the shape of one.
FORBIDDEN_IN_PROSE = (
    re.compile(r"/var/folders/\w"),
    re.compile(r"/Users/\w"),
    re.compile(r"/home/\w"),
)


def _test_source_lines() -> set[str]:
    """Every line of this repo's test sources, stripped.

    A documented plan that quotes a committed test fixture reproduces its paths
    verbatim, and those paths are literals in a test rather than anything a machine
    produced. Matching on the WHOLE line, not on the path alone, keeps the exemption
    narrow: a new path smuggled into a doc has to already exist, character for
    character, in a test file to be waved through.
    """
    return {
        line.strip()
        for path in sorted(TESTS.rglob("*.py"))
        for line in path.read_text().splitlines()
        if line.strip()
    }


def test_no_results_file_carries_a_machine_path():
    offenders: dict[str, list[str]] = {}
    for path in sorted(RESULTS.glob("*.json*")) + sorted(RESULTS.glob("*.log")):
        text = path.read_text()
        hits = [p.pattern for p in FORBIDDEN if p.search(text)]
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        f"machine paths in results/ — run `uv run python -m loop.scrub_results`: {offenders}"
    )


def test_no_committed_document_carries_a_machine_path():
    """The gap that let a home directory sit in public history: docs were not covered.

    Every `.md` under `docs/`, at any depth — plans included, because a plan is where
    someone pastes the command they actually ran.
    """
    exempt = _test_source_lines()
    offenders: dict[str, list[str]] = {}
    for path in sorted(DOCS.rglob("*.md")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if line.strip() in exempt:
                continue
            hits = [p.pattern for p in FORBIDDEN_IN_PROSE if p.search(line)]
            if hits:
                rel = path.relative_to(REPO_ROOT)
                offenders.setdefault(str(rel), []).append(f"line {number}: {hits}")
    assert not offenders, (
        "machine paths in committed documents — replace them with <HOME>/<TMPDIR> "
        f"placeholders, preserving what the line was saying: {offenders}"
    )


def test_the_document_gate_would_catch_a_real_leak(tmp_path):
    """The gate above passes; this proves it passes because the documents are clean.

    A gate over a clean tree is indistinguishable from a gate that matches nothing, so
    the patterns are run against a line that IS a leak — the exact shape the scrub above
    removed from the r9 plan — and against the two shapes that must stay allowed.
    """
    leak = 'cd /Users/someone/Projects/refinery && node "/Users/someone/.claude/x.mjs"'
    assert any(p.search(leak) for p in FORBIDDEN_IN_PROSE)
    assert any(p.search("ran under /var/folders/qq/zz/T/w-1") for p in FORBIDDEN_IN_PROSE)
    # Prose naming the patterns is not a leak.
    prose = "No absolute filesystem paths (`/Users/`, `/home/`, `/var/folders/`)."
    assert not any(p.search(prose) for p in FORBIDDEN_IN_PROSE)
    # And the exemption is keyed to lines that really are in the test sources.
    exempt = _test_source_lines()
    assert leak.strip() not in exempt
    assert '"detail": "saw /private/var/folders/qq/zz/T/w-1/x.txt",' in exempt


def test_the_scrubber_changes_strings_by_substitution_and_nothing_else(tmp_path):
    """The property that makes scrubbing committed evidence defensible at all.

    Every string is in scope, not just `detail` — the first version scrubbed details
    only and its own report exposed the gap: the paths also live inside `approvals`
    entries, which quote tool arguments verbatim, and one committed log carries a
    TRUNCATED path (cut mid-directory by the runner's clamp) that a /T-anchored
    pattern never matched.
    """
    import json

    from loop.scrub_results import scrub_file

    detail = (
        "reply='@/private/var/folders/qh/x0/T/a5-1/notes.txt "
        "in /Users/someone/w and someone said hi'"
    )
    row = {
        "task": "T",
        "attempt": 0,
        "passed": False,
        "outcome": "fail",
        "runner_sha": "abc123",
        "duration_s": 41.5,
        "detail": detail,
        "approvals": [{"tool": "bash", "command": "cat /var/folders/qh/x0/T/w-1/f.txt"}],
        "metrics": {"tool_calls": 2.0},
    }
    truncated = {
        "task": "U",
        "attempt": 0,
        "passed": True,
        "outcome": "pass",
        "detail": "ran 'cd /private/var/folders/qh/83'",
        "metrics": {},
    }
    f = tmp_path / "run.jsonl"
    f.write_text(json.dumps(row) + "\n" + json.dumps(truncated) + "\n")

    assert scrub_file(f, username="someone") is True
    out, out2 = (json.loads(line) for line in f.read_text().splitlines())
    assert out["detail"] == "reply='@<TMPDIR>/a5-1/notes.txt in <HOME>/w and <USER> said hi'"
    assert out["approvals"] == [{"tool": "bash", "command": "cat <TMPDIR>/w-1/f.txt"}]
    assert out2["detail"] == "ran 'cd <TMPDIR>'", "the truncated path must vanish too"
    # Everything that is measurement is identical.
    for k in ("task", "attempt", "passed", "outcome", "runner_sha", "duration_s", "metrics"):
        assert out[k] == row[k]
    # Idempotent: a second pass changes nothing.
    assert scrub_file(f, username="someone") is False


def test_rows_are_scrubbed_at_generation_time(tmp_path):
    """The durable fix the module docstring promised: runner emits clean rows."""
    from runner.run import write_record  # the extracted serialization seam (Step 2)

    f = tmp_path / "run.jsonl"
    rec = {
        "task": "T",
        "attempt": 0,
        "passed": False,
        "outcome": "fail",
        "detail": "saw /private/var/folders/qq/zz/T/w-1/x.txt",
        "approvals": [{"tool": "bash", "args": '{"command":"cat /var/folders/qq/zz/T/w-1/x.txt"}'}],
        "metrics": {},
    }
    write_record(f, rec)
    import json

    out = json.loads(f.read_text())
    assert "<TMPDIR>" in out["detail"] and "/var/folders" not in f.read_text()
    # Non-string/structure fields must survive untouched — a scrubber that rewrote
    # keys or coerced values would pass the string check above and still be broken.
    assert out["passed"] is False and out["attempt"] == 0 and out["metrics"] == {}


def test_truncated_at_exactly_var_folders_is_scrubbed():
    """A detail clamp once cut a path at the directory name itself — no trailing slash.

    The partial pattern must catch the boundary case.
    """
    from runner.scrub import scrub_text

    assert scrub_text("CWD: /private/var/folders'") == "CWD: <TMPDIR>'"
    assert scrub_text("CWD: /var/folders") == "CWD: <TMPDIR>"
