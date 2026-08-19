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
import re
import subprocess
from pathlib import Path

from runner.carbon_env import CARBON_ROOT

PIN_FILE = Path(__file__).resolve().parents[1] / "carbon-base.json"
CARBON_BASE_EXIT_CODE = 7  # Distinct from pytest's reserved 0-5


class CarbonBaseError(RuntimeError):
    """The sibling carbon checkout cannot run this refinery."""


_REQUIRED_KEYS = ("carbon_branch", "carbon_commit", "required_symbols")


def load_pin(pin_file: Path = PIN_FILE) -> dict:
    if not pin_file.is_file():
        raise CarbonBaseError(f"missing pin file: {pin_file}")
    try:
        pin = json.loads(pin_file.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CarbonBaseError(f"unreadable pin file {pin_file}: {exc}") from exc

    if not isinstance(pin, dict):
        raise CarbonBaseError(f"{pin_file} must contain a JSON object, got {type(pin).__name__}")

    missing_keys = [key for key in _REQUIRED_KEYS if key not in pin]
    if missing_keys:
        raise CarbonBaseError(f"{pin_file} is missing required key(s): {', '.join(missing_keys)}")

    for key in ("carbon_branch", "carbon_commit"):
        if not isinstance(pin[key], str):
            raise CarbonBaseError(
                f"{pin_file}'s {key!r} must be a string, got {type(pin[key]).__name__}"
            )

    if not pin["carbon_branch"]:
        raise CarbonBaseError(f"{pin_file}'s 'carbon_branch' must not be empty")

    if not re.fullmatch(r"[0-9a-f]{40}", pin["carbon_commit"]):
        raise CarbonBaseError(
            f"{pin_file}'s 'carbon_commit' must be a 40-character lowercase hex SHA, "
            f"got {pin['carbon_commit']!r}"
        )

    required_symbols = pin["required_symbols"]
    if not isinstance(required_symbols, list) or not required_symbols:
        raise CarbonBaseError(
            f"{pin_file} has an empty or non-list required_symbols; "
            "an empty list would make the compatibility check a silent no-op"
        )
    for entry in required_symbols:
        is_well_formed = (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and all(isinstance(part, str) for part in entry)
        )
        if not is_well_formed:
            raise CarbonBaseError(
                f"{pin_file} has a malformed required_symbols entry: {entry!r} "
                "(expected a 2-item [module, attr] list of strings)"
            )

    return pin


def _carbon_head(root: Path = CARBON_ROOT) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def require_carbon_base(pin_file: Path = PIN_FILE) -> list[str]:
    """Raise CarbonBaseError (with remediation) on missing symbols; return warnings."""
    pin = load_pin(pin_file)
    missing = []
    for module_name, attr in pin["required_symbols"]:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # exc.name is the module that was actually not found. When it
            # matches module_name, module_name itself is absent. When it
            # doesn't, module_name exists but one of ITS OWN imports (a
            # dependency) failed — that must not read the same as "module
            # missing", or a dependency problem looks like a checkout problem.
            if exc.name == module_name:
                missing.append(f"{module_name} (module missing)")
            else:
                missing.append(f"{module_name} (import failed: {type(exc).__name__})")
            continue
        except Exception as exc:
            # A syntax-broken module, a junk module name (e.g. ""), a plain
            # ImportError raised by the module itself, or any other
            # import-time failure must still fold into the one loud
            # CarbonBaseError below, not crash raw mid-collection.
            missing.append(f"{module_name} (import failed: {type(exc).__name__})")
            continue
        try:
            has_attr = hasattr(module, attr)
        except Exception as exc:
            # hasattr() only swallows AttributeError. A module-level
            # __getattr__ (PEP 562) can raise anything on attribute access;
            # that must fold into the loud error too, not crash raw.
            missing.append(f"{module_name}.{attr} (attribute check failed: {type(exc).__name__})")
            continue
        if not has_attr:
            missing.append(f"{module_name}.{attr}")
    if missing:
        raise CarbonBaseError(
            f"carbon checkout at {CARBON_ROOT} is not the base this refinery is "
            f"built against.\nMissing: {', '.join(missing)}.\n"
            f"Fix: git -C {CARBON_ROOT} checkout {pin['carbon_branch']}\n"
            f"(already on that branch? advance it to the pinned commit)\n"
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
