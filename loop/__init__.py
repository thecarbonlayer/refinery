"""The mine -> propose -> validate -> PR pipeline (the self-improvement loop).

Role split: the MINING and PROPOSAL
steps are reasoning, performed directly by the proposer model (Fable,
in-session, for iteration 1) — they are not code here. Their conclusions are
written to fixed JSON artifacts (see ``artifacts.py``) so the code half of the
loop consumes a stable contract and never cares who produced a candidate.
What IS code: applying a candidate to the editable surface (``config_edit.py``),
validating it against the task suite via the runner (``validate.py``), and
turning an accepted edit into a reviewable pull request (``prpipe.py``).
"""

import sys as _sys

from loop.compat import require_carbon_base as _require_carbon_base

for _warning in _require_carbon_base():
    print(f"carbon-base warning: {_warning}", file=_sys.stderr)
