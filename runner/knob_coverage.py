"""Which independent tasks can see each editable Carbon knob.

This is a governance contract, not a suggestion list. A newly editable knob
must name at least one diagnostic or explicitly use the full suite before the
loop is allowed to tune it.
"""

from __future__ import annotations

KNOB_COVERAGE: dict[str, dict[str, tuple[str, ...]]] = {
    "system_prompt": {"miners": ("*",), "guards": ("B1", "B2", "B3", "D1", "D2")},
    "max_tool_steps": {"miners": ("F2",), "guards": ("D1", "D2")},
    "default_context_limit": {"miners": ("A1", "A3", "G2"), "guards": ("B1", "B2")},
    "verify_attempts": {"miners": ("B1", "B2", "B3"), "guards": ("B2", "B3")},
    "file_injection": {"miners": ("A4",), "guards": ("A2",)},
    "tool_output": {"miners": ("A2", "E2"), "guards": ("D1", "D2", "F2")},
    "compaction": {"miners": ("A1", "A3", "G2"), "guards": ("B1", "B2")},
    "compaction_prompt": {"miners": ("A1", "A3", "G2"), "guards": ("B1", "B2")},
    "temperature": {"miners": ("*",), "guards": ("B1", "B2", "B3", "D1", "D2")},
    "max_tokens": {"miners": ("G1",), "guards": ("D1", "D2")},
    "retry": {"miners": ("H1", "H2"), "guards": ("H3",)},
}
