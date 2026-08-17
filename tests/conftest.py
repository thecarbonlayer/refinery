"""Test isolation: sweep leaked carbon scratch directories (mirrors carbon's
``tests/conftest.py``).

refinery's tests build real carbon ``Agent``/``SessionEnvironment`` objects (e.g.
``test_helpers.py``'s ``_probe_agent``, ``test_registry.py``'s ``_build_c_agent``) to
pin behavior against the live harness rather than a mock. Each one that reaches
construction makes a real ``mkdtemp()`` scratch directory under the OS temp dir. Most
of those tests close what they own — the ownership/lifecycle tests
(``test_build_c_agent_binds_recording_wrapped_tools_and_scratch_root`` and friends)
assert exactly that in-body, and this fixture changes nothing about what THEY prove,
since the sweep below only runs after the test function has already returned. But a
future test that forgets its ``close()``, or crashes before reaching one, would
otherwise abandon a ``carbon-scratch-*`` directory for carbon's own ``scavenge()`` to
find — up to 24h later (see carbon's ``harness/session_env.py``), and every stray
directory in the temp dir slows every later ``Agent()`` construction's glob+stat, not
just the leaking test's.

Snapshot-before/remove-only-new-after, same as carbon: never touch a directory this
test did not create, so a directory another process (or a test that legitimately
keeps its env alive past its own yield) owns is left alone. Explicit ``close()``/
``cleanup()`` calls remain the contract; this is only the net underneath it.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _sweep_scratch_dirs_this_test_leaked():
    from harness.session_env import SCRATCH_PREFIX

    root = Path(tempfile.gettempdir())
    before = set(root.glob(f"{SCRATCH_PREFIX}*"))
    yield
    for stray in set(root.glob(f"{SCRATCH_PREFIX}*")) - before:
        shutil.rmtree(stray, ignore_errors=True)
