"""Run both repos' offline suites at every legal point on the editable surface.

The harness gate answers "did THIS candidate break the harness?". It cannot answer
"can any legal candidate break the harness?" — and that second question is the one
that has been wrong four times running. Each time the shape was the same: a test
pinned the behaviour of whatever value happened to be shipped, so a different legal
value turned the suite red with nothing actually broken. Since the gate rejects a
candidate whose suites are red, a false red is not a cosmetic annoyance — it vetoes a
legal candidate and reports it as ``HARNESS GATE FAILED``, which reads like the
candidate's fault.

Known instances, in order found:
  - carbon ``test_checked_in_strategy_defaults_are_quality_oriented`` pinned a literal
    on a field the loop is allowed to edit (asserting "the loop has never accepted
    anything").
  - refinery H2's recovery rule assumed one strategy's call ordering.
  - refinery E3/E4 premise probes were built on a ``tail_fraction`` the surface never
    permitted.
  - refinery ``test_h_verdicts_are_not_stuck_true`` assumed the shipped retry policy,
    reddening 6 of the 10 legal ``(strategy, max_attempts)`` pairs.

Each was fixed at its own site. This module is the same fix stated once, mechanically:
walk the surface, and fail if any legal value is red.

Points are DERIVED from carbon's ``config_schema()`` (adr/0002), never listed here, so
the day carbon adds a fourth compaction strategy the sweep covers it without an edit
on this side — and the sweep, not a reviewer, is what notices.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loop.config_edit import config_path, known_knobs
from runner.carbon_env import CARBON_ROOT

EDITOR_ROOT = Path(__file__).resolve().parents[1]

# A bounded float is probed just inside each end rather than at it: the ends are
# exclusive on the surface, so probing AT them tests a value no candidate may
# propose — the exact defect that made two premise probes pass on illegal input
# until carbon grew a validation door.
_EPSILON = 0.001


@dataclass(frozen=True)
class Point:
    """One legal value of one field, as the surgical editor would write it."""

    field: str
    label: str  # human-readable, e.g. "tool_output.strategy=keep_head"
    value: object  # the whole field value, since the editor replaces whole fields


def _bounded_floats(spec: dict) -> list[float]:
    """Both ends of a bounded float, respecting exclusivity."""
    out: list[float] = []
    if "exclusive_min" in spec:
        out.append(spec["exclusive_min"] + _EPSILON)
    elif "min" in spec:
        out.append(float(spec["min"]))
    if "exclusive_max" in spec:
        out.append(spec["exclusive_max"] - _EPSILON)
    elif "max" in spec:
        out.append(float(spec["max"]))
    return out


def _bounded_ints(spec: dict) -> list[int]:
    out: list[int] = []
    if "min" in spec:
        out.append(int(spec["min"]))
    if "max" in spec:
        out.append(int(spec["max"]))
    return out


def enumerate_points(current: dict, schema: dict[str, dict] | None = None) -> list[Point]:
    """Every legal point worth probing, derived from carbon's published schema.

    Scope is deliberate, not exhaustive: whole menus (every strategy) and the ENDS of
    every bounded parameter. Those are where a pinned-to-today's-value test bites.
    Interiors are not swept — a test that holds at both ends and at the shipped value
    but fails in between would be a different animal, and the surface is not a product
    space anyone can walk in finite time.

    ``retry`` is the one field swept as a CROSS product (strategy x max_attempts),
    because its two parameters interact: ``fail_fast`` makes exactly one provider call
    whatever ``max_attempts`` says, and that interaction is what a pinned test missed.

    The shipped value is skipped — the gate's plain run already covers it.
    """
    schema = schema if schema is not None else known_knobs()
    points: list[Point] = []
    for name, field in sorted(schema.items()):
        if name not in current:
            continue
        live = current[name]
        params = field.get("parameters") or {}
        strategies = field.get("strategies") or []

        if name == "retry":
            for strategy in strategies:
                for attempts in _bounded_ints(params.get("max_attempts", {})):
                    value = {**live, "strategy": strategy, "max_attempts": attempts}
                    if value != live:
                        points.append(
                            Point(name, f"retry={strategy}/{attempts}", value),
                        )
            for delay in _bounded_ints(params.get("base_delay_ms", {})):
                value = {**live, "base_delay_ms": delay}
                if value != live:
                    points.append(Point(name, f"retry.base_delay_ms={delay}", value))
            continue

        for strategy in strategies:
            value = {**live, "strategy": strategy}
            if value != live:
                points.append(Point(name, f"{name}.strategy={strategy}", value))
        for param, spec in sorted(params.items()):
            if spec.get("type") != "float":
                continue
            for probe in _bounded_floats(spec):
                value = {**live, param: probe}
                if value != live:
                    points.append(Point(name, f"{name}.{param}={probe}", value))
    return points


def _surgical_write(path: Path, text: str, field: str, old: object, new: object) -> str:
    """Replace one whole field in place, same contract as ``apply_candidate``.

    Whole-file rewriting is not an option even here: this repo's own editor requires
    each field on a single line, and a reformatted file fails two of its tests for a
    reason that has nothing to do with the value being probed.
    """
    old_text = f'"{field}": {json.dumps(old)}'
    new_text = f'"{field}": {json.dumps(new)}'
    if text.count(old_text) != 1:
        raise ValueError(
            f"surface sweep: cannot surgically edit {field!r} — {old_text!r} occurs "
            f"{text.count(old_text)} times in {path}"
        )
    return text.replace(old_text, new_text)


def _run_suites(carbon_root: Path, editor_root: Path) -> tuple[bool, dict]:
    """Both offline suites at one point on the surface.

    ``pytest`` for carbon, deliberately NOT ``uv run verify`` — the gate's own carbon
    check is the full ``verify``, and the difference is on purpose: ruff and mypy never
    read ``harness_config.json``, so their verdict cannot vary with the value being
    probed. Running them 16 more times would buy no signal. Anything that DOES read the
    config runs under pytest, which is what the sweep is asking about.
    """
    detail: dict = {}
    ok = True
    for name, cwd in (("carbon", carbon_root), ("refinery", editor_root)):
        proc = subprocess.run(
            ["uv", "run", "pytest", "-q"], cwd=cwd, capture_output=True, text=True
        )
        passed = proc.returncode == 0
        detail[name] = {"passed": passed, "exit_code": proc.returncode}
        if not passed:
            ok = False
            failed = [
                line
                for line in (proc.stdout + proc.stderr).splitlines()
                if line.startswith("FAILED") or line.startswith("ERROR")
            ]
            detail[name]["failed"] = failed[:12]
    return ok, detail


def sweep(
    carbon_root: Path = CARBON_ROOT,
    editor_root: Path = EDITOR_ROOT,
    run_suites: Callable[[Path, Path], tuple[bool, dict]] = _run_suites,
    only: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Apply each legal point, run both suites, revert. Report every red.

    Restores the EXACT bytes it read, then verifies the restore — a sweep that left
    carbon's config edited would poison every measurement after it. Deliberately not a
    ``git checkout``: the gate runs this with a candidate already applied to the working
    tree, and checking out would silently discard the candidate and leave the caller
    measuring the committed config instead of the one it proposed.
    """
    path = config_path(carbon_root)
    original = path.read_text()
    current = json.loads(original)
    points = [p for p in enumerate_points(current) if only is None or p.field == only]
    out: dict = {"passed": True, "probed": 0, "points": {}}
    log(f"surface sweep: {len(points)} legal points")
    try:
        for point in points:
            path.write_text(
                _surgical_write(path, original, point.field, current[point.field], point.value)
            )
            ok, detail = run_suites(carbon_root, editor_root)
            out["probed"] += 1
            out["points"][point.label] = {"passed": ok, **detail}
            if not ok:
                out["passed"] = False
                broken = [n for n, d in detail.items() if not d["passed"]]
                log(f"  RED  {point.label} ({', '.join(broken)})")
                for name in broken:
                    for line in detail[name].get("failed", []):
                        log(f"         {line}")
            else:
                log(f"  ok   {point.label}")
    finally:
        path.write_text(original)
        if path.read_text() != original:
            raise RuntimeError(f"surface sweep: failed to restore {path}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="restrict to one field, e.g. tool_output")
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = parser.parse_args(argv)
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("surface sweep shells out to pytest; it must not run inside pytest")
    report = sweep(only=args.only)
    if args.json:
        print(json.dumps(report, indent=2))
    reds = [label for label, r in report["points"].items() if not r["passed"]]
    print(f"\nsurface sweep: {report['probed']} probed, {len(reds)} red")
    for label in reds:
        print(f"  RED {label}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
