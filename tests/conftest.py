"""Test isolation: give the whole suite its own scratch parent (mirrors carbon's
own ``tests/conftest.py`` fix).

refinery's tests build real carbon ``Agent``/``SessionEnvironment`` objects (e.g.
``test_helpers.py``'s ``_probe_agent``, ``test_registry.py``'s ``_build_c_agent``) to
pin behavior against the live harness rather than a mock. Each one that reaches
construction makes a real ``mkdtemp()`` scratch directory under the OS temp dir.

The previous version of this fixture snapshot-diffed the REAL OS temp dir once per
test: anything matching ``carbon-scratch-*`` that appeared since the last ``yield``
was assumed to be this test's own leak and removed. That cannot tell "this test's
leak" from "a different process's directory that happened to appear in the same
window" — refinery's own purpose is running live measurements against carbon
(``runner/``), and a live measurement running concurrently on the same machine loses
its scratch mid-attempt, which surfaces as a fabricated C3 mechanical security
failure (see ``runner/tasks/cluster_c.py``), not as what it actually is: a test
sweep with no way to tell the two apart. Carbon's own test suite had the identical
P1, fixed the same way; this mirrors that fix.

Fixed by OWNERSHIP, not diffing: replace ``harness.session_env``'s own
``scratch_parent_dir`` — the ONE function ``local_session_env`` and ``scavenge()``'s
default root both call for "where does ephemeral scratch live" (see that module) —
for the life of this pytest process, so every session env built anywhere, any way,
during the whole suite lands under a throwaway root nothing else on the machine ever
writes to. A concurrent process (a live measurement, a different pytest run) runs its
OWN Python process with its OWN imported copy of ``harness.session_env`` — this
patches only the copy loaded into THIS process — so its scratch keeps landing in the
real OS temp dir, structurally out of reach of anything below: not merely excluded by
a check that could be wrong, but never glob-reachable from the redirected root at all.

A monkeypatched FUNCTION, deliberately — not an env var. An earlier version of
carbon's own fixture used an env var, which a review caught as reachable from
PRODUCTION: an env var crosses via ``.env`` too (carbon's ``model/provider.py``
``Provider.from_env()`` calls ``os.environ.setdefault`` for every key in that file,
and every production entrypoint resolves a ``Provider`` before constructing its first
``Agent``), so a single stray line in the file carbon tells users to edit for their
model endpoint would have been enough to silently redirect a REAL session's scratch
(into the repo if the value were ``.``) and silently stop ``scavenge()`` from ever
sweeping the real temp dir again. A swapped FUNCTION has zero production surface:
nothing outside this fixture's own process can ever replace it, so there is no file,
environment variable, or subprocess boundary left for a stray value to cross.

``pytest.MonkeyPatch()`` is used directly, not the ordinary function-scoped
``monkeypatch`` fixture (which pytest does not allow at session scope) — the patch is
undone explicitly in ``finally`` instead of automatically at a function's end.

Session-scoped, not per-test: the old fixture's snapshot-diff globbed the real temp
dir twice per test regardless of whether that test ever built an Agent. Redirecting
once, up front, costs one function swap for the whole session; nothing per-test
remains to glob at all.

The root's own name still starts with ``SCRATCH_PREFIX``
(``carbon-scratch-pytest-session-<random>``, not an unrelated prefix): a run killed
hard enough to skip this fixture's own ``finally`` (SIGKILL, an OOM-kill — the one
failure mode no process-local cleanup can guard against) leaves the whole root
behind, but named this way it stays glob-reachable by a FUTURE, unrelated
``scavenge()`` sweep of the real temp dir — reaped as one unit after
``SCAVENGE_AGE_S``, the same backstop every ordinary abandoned session already relies
on, rather than orphaned forever under a prefix nothing will ever look for again.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(scope="session", autouse=True)
def _isolated_scratch_root():
    import harness.session_env as session_env_mod

    root = Path(tempfile.mkdtemp(prefix=f"{session_env_mod.SCRATCH_PREFIX}pytest-session-"))
    mp = pytest.MonkeyPatch()
    mp.setattr(session_env_mod, "scratch_parent_dir", lambda: root)
    try:
        yield root
    finally:
        mp.undo()
        shutil.rmtree(root, ignore_errors=True)


def pytest_sessionstart(session):
    """Fail the whole run early — remediation, not an ImportError spray — when
    the sibling carbon checkout is not the pinned base (see carbon-base.json).

    ``import loop.compat`` first imports the parent ``loop`` package, whose
    ``__init__.py`` already runs the guard (``require_carbon_base()``) and
    prints any commit-drift warnings to stderr — so this import is the ONE
    guard run for the whole pytest session. Calling ``require_carbon_base()``
    again here would run it a second time for no benefit.

    ``pytest_sessionstart``, not ``pytest_configure``: pytest.exit's
    returncode only becomes the process status inside ``wrap_session``
    (``_pytest/main.py``), and informational commands like ``--help`` and
    ``--markers`` call ``config._do_configure()`` OUTSIDE it — a guard raising
    pytest.exit from pytest_configure escaped those commands as a raw
    ``_pytest.outcomes.Exit`` traceback, exit 1. Every session-running
    invocation (a normal run, ``--collect-only``, ``--fixtures``) fires
    pytest_sessionstart inside ``wrap_session``, BEFORE collection — so the
    guard still precedes any import of test modules — and ``wrap_session``
    both honors the returncode and prints the message. ``--help``/``--markers``
    never start a session and never touch the carbon pair, so they now
    correctly print their output and succeed instead of tracebacking.

    The except clause matches ``RuntimeError`` and then checks the class name
    AND its defining module by string, rather than importing and checking
    ``isinstance(exc, CarbonBaseError)``: the import that raised the error is
    the very ``loop.compat`` import in the try block, so importing
    ``CarbonBaseError`` to test with would require the module to already be
    importable — chicken-and-egg. ``CarbonBaseError`` subclasses
    ``RuntimeError``, so this still narrows out everything else. The module
    check matters because name alone is spoofable: an unrelated
    ``RuntimeError`` subclass defined elsewhere and also named
    ``CarbonBaseError`` would match on name, get swallowed into
    ``pytest.exit``, and lose its real traceback. Any OTHER exception (a
    genuine bug elsewhere in the import, or a same-named class from a
    different module) re-raises with its real traceback instead of being
    swallowed.

    ``CARBON_BASE_EXIT_CODE`` is read from ``sys.modules`` AFTER the failure,
    never imported inside the ``try``: the guard raises while
    ``loop/__init__.py`` executes, so ``import loop.compat`` never returns and
    a ``from loop.compat import CARBON_BASE_EXIT_CODE`` placed after it never
    binds. A previous version did exactly that, and the handler then hit
    UnboundLocalError on the constant — pytest INTERNALERROR, exit 3 — on the
    one path this guard exists for. ``loop.compat`` itself IS fully
    initialized and present in ``sys.modules`` at that point:
    ``loop/__init__.py`` imports it before calling the guard, and a failed
    package init rolls back ``loop``, not the already-imported submodule. If
    the constant is ever NOT recoverable, the error re-raises with its real
    traceback rather than exiting with a guessed code.
    """
    try:
        import loop.compat  # noqa: F401
    except RuntimeError as exc:
        if type(exc).__name__ == "CarbonBaseError" and type(exc).__module__ == "loop.compat":
            exit_code = getattr(sys.modules.get("loop.compat"), "CARBON_BASE_EXIT_CODE", None)
            if exit_code is None:
                raise
            pytest.exit(f"\n{exc}", returncode=exit_code)
        raise
