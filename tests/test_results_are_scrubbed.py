"""No machine-identifying path may sit under results/ — enforced, not remembered.

This repo is public and results are the committed record. The prescribed pre-commit
grep covered `/Users/` and `/home/` and was blind to `/var/folders`, so two committed
candidate logs carried per-run temp paths for weeks — and a baseline log was found
holding an entire `$PATH` dump, home directory included, quoted inside a task detail.

This test is the recurrence-killer, placed OUTSIDE `runner/` on purpose: the durable
fix (scrub at generation) means touching the runner and therefore `runner_sha`, which
invalidates every recorded baseline — queued for the next invalidation window instead.
A red suite the moment an unscrubbed artifact lands makes the scrub impossible to
forget without costing a re-measurement today. `loop/scrub_results.py` fixes what this
test finds, and proves it changed nothing but the offending strings.
"""

from __future__ import annotations

import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

FORBIDDEN = (
    re.compile(r"/var/folders/"),
    re.compile(r"/Users/"),
    re.compile(r"/home/\w"),
)


def test_no_results_file_carries_a_machine_path():
    offenders: dict[str, list[str]] = {}
    for path in sorted(RESULTS.glob("*.json*")):
        text = path.read_text()
        hits = [p.pattern for p in FORBIDDEN if p.search(text)]
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        f"machine paths in results/ — run `uv run python -m loop.scrub_results`: {offenders}"
    )


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
