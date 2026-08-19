"""Startup contract: is the sibling carbon checkout the base refinery expects?

Refinery is built against the carbon base named in <repo>/carbon-base.json.
A fresh clone of both default branches is NOT an operable pair today — carbon
main lacks symbols the suite imports — and that used to surface as a bare
ImportError mid-collection. This turns it into one loud, early, explanatory
failure.

Deliberately OUTSIDE runner/: runner's content hash versions every baseline,
and this check must be able to evolve without invalidating measurements.
Commit drift is a warning, not an error: iteration work legitimately moves the
checkout, and baseline reuse is decided by behavior key, not SHA.
"""

from __future__ import annotations

import importlib
import json
import subprocess
from pathlib import Path

from runner.carbon_env import CARBON_ROOT

PIN_FILE = Path(__file__).resolve().parents[1] / "carbon-base.json"


class CarbonBaseError(RuntimeError):
    """The sibling carbon checkout cannot run this refinery."""


def load_pin(pin_file: Path = PIN_FILE) -> dict:
    if not pin_file.is_file():
        raise CarbonBaseError(f"missing pin file: {pin_file}")
    return json.loads(pin_file.read_text())


def _carbon_head(root: Path = CARBON_ROOT) -> str:
    out = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def require_carbon_base(pin_file: Path = PIN_FILE) -> list[str]:
    """Raise CarbonBaseError (with remediation) on missing symbols; return warnings."""
    pin = load_pin(pin_file)
    missing = []
    for module_name, attr in pin["required_symbols"]:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            missing.append(f"{module_name} (module missing)")
            continue
        if not hasattr(module, attr):
            missing.append(f"{module_name}.{attr}")
    if missing:
        raise CarbonBaseError(
            f"carbon checkout at {CARBON_ROOT} is not the base this refinery is "
            f"built against.\nMissing: {', '.join(missing)}.\n"
            f"Fix: git -C {CARBON_ROOT} checkout {pin['carbon_branch']}\n"
            f"(pinned base: {pin['carbon_branch']} @ {pin['carbon_commit'][:9]}; "
            f"see carbon-base.json)"
        )
    warnings = []
    head = _carbon_head()
    if head != pin["carbon_commit"]:
        warnings.append(
            f"carbon HEAD {head[:9]} != pinned {pin['carbon_commit'][:9]} "
            "(allowed: baseline reuse is decided by behavior key, not SHA)"
        )
    return warnings
