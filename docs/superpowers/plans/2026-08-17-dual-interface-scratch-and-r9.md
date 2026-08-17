# Dual-Interface Scratch + r9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every scratch artifact reachable through BOTH interfaces the model actually uses (`read_file` via `scratch://`, and bash via `$CARBON_SCRATCH_DIR`), close the security and lifecycle findings the reviews raised, then re-measure the unchanged offload candidate against a fresh r9 baseline.

**Architecture:** `scratch://` stays as carbon's internal identifier; it stops being the only model-facing access path. The sandbox learns the session's scratch root and exposes it as `CARBON_SCRATCH_DIR` — a fixed `/carbon-scratch` read-only mount under Docker, the real path under local execution. Footers advertise both routes. Durable sessions get a durable scratch tied to the session directory rather than an ephemeral one that dies at `close()`.

**Tech Stack:** Python 3.13, uv, pytest. No new dependencies.

## Why this batch exists (read before changing anything)

Iteration 5 measured `offload_to_file` and rejected it on confirmation: E4 recovery was 0/10 in both arms. The transcripts showed why — **32 of 32 scratch-access attempts went through bash**, which cannot resolve `scratch://`. The model then re-derived the answer by hashing the seed (8/10 attempts), which E4 taint-disqualifies by design. The abstraction was implemented by exactly one consumer. This batch supplies the missing adapter; the candidate itself does not change.

## Global Constraints

- **The candidate is not modified.** `iterations/iter-05/candidates.json`'s `tool-output-offload-r3` edit (`tool_output.strategy: head_tail -> offload_to_file`) is re-measured verbatim. The runtime gains the capability the strategy reasonably depends on; the strategy is not reshaped to fit the test.
- **No absolute host path may enter a prompt, a footer, or a stored transcript.** The shell route is advertised as the literal string `$CARBON_SCRATCH_DIR/offload/<name>`, never its expansion.
- **Public repos:** no absolute filesystem paths, usernames, or private names in committed files (the scrub-test fixtures are the established exception).
- **Exit codes are captured, never piped away:** `cmd > /tmp/out 2>&1; echo "exit=$?"`, then read the tail.
- **Carbon two-gate rule:** `uv run verify` green before any carbon commit; `uv run accept ch-06` attempted before the batch closes.
- **One invalidation window:** every `runner/` change lands before r9 is recorded. `runner_sha` moves exactly once.
- **Never run refinery's pytest while a suite is live** — the conftest sweep deletes scratch directories it did not create (Task 6 fixes this; until then it can corrupt a running measurement).
- Repos are siblings: refinery imports carbon as `../carbon`.

---

### Task 1: Carbon — the shell route (`CARBON_SCRATCH_DIR`)

**Files:** Modify `../carbon/harness/sandbox.py`, `../carbon/harness/limits.py`; Test `../carbon/tests/test_sandbox.py`, `../carbon/tests/test_offload_strategy.py`

**Interfaces produced:**
- `Sandbox(..., scratch_dir: str | Path | None = None)`; `DOCKER_SCRATCH_MOUNT = "/carbon-scratch"`; `SCRATCH_ENV_VAR = "CARBON_SCRATCH_DIR"`
- `bash_tool(sandbox, workdir=None)` unchanged in signature — the sandbox carries the scratch.
- `limits.shell_ref(filename: str) -> str` returning `"$CARBON_SCRATCH_DIR/offload/<filename>"`
- `_footer` advertises both routes.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sandbox.py
def test_local_shell_can_read_a_scratch_artifact_via_the_env_var(tmp_path):
    """The route the model actually reaches for. 32 of 32 accesses in iteration 5's
    confirmation used bash; none could resolve scratch://, so recovery was 0/10."""
    from harness.sandbox import Sandbox
    scratch = tmp_path / "scratch"
    (scratch / "offload").mkdir(parents=True)
    (scratch / "offload" / "ab12.txt").write_text("NEEDLE-7F3A\n")
    sb = Sandbox(trusted=True, prefer_docker=False, scratch_dir=scratch)
    r = sb.run('grep NEEDLE "$CARBON_SCRATCH_DIR/offload/ab12.txt"', workdir=str(tmp_path))
    assert r.exit_code == 0 and "NEEDLE-7F3A" in r.stdout


def test_untrusted_local_shell_also_gets_the_scratch_route(tmp_path):
    """The scrubbed env drops host secrets but must still carry this var, or the
    route exists only for trusted wiring and silently vanishes elsewhere."""
    from harness.sandbox import Sandbox
    scratch = tmp_path / "scratch"
    (scratch / "offload").mkdir(parents=True)
    (scratch / "offload" / "cd34.txt").write_text("NEEDLE-9B1C\n")
    sb = Sandbox(trusted=False, prefer_docker=False, scratch_dir=scratch)
    r = sb.run('cat "$CARBON_SCRATCH_DIR/offload/cd34.txt"', workdir=str(tmp_path))
    assert "NEEDLE-9B1C" in r.stdout


def test_no_scratch_configured_leaves_the_var_unset_not_empty(tmp_path):
    """An empty CARBON_SCRATCH_DIR expands to "" and `cat "/offload/x"` reads from
    the filesystem root. Unset is the honest state."""
    from harness.sandbox import Sandbox
    sb = Sandbox(trusted=True, prefer_docker=False)
    r = sb.run('echo "[${CARBON_SCRATCH_DIR-UNSET}]"', workdir=str(tmp_path))
    assert "[UNSET]" in r.stdout
```

```python
# tests/test_offload_strategy.py
def test_footer_advertises_both_routes_and_no_host_path(tmp_path):
    from harness.harness_config import TruncationPolicy
    from harness.limits import truncate_tool_result
    out = truncate_tool_result(
        "x" * 9000, TruncationPolicy("offload_to_file", 4000, 0.3), scratch_dir=tmp_path
    )
    assert "scratch://offload/" in out, "read_file route"
    assert "$CARBON_SCRATCH_DIR/offload/" in out, "shell route"
    assert str(tmp_path) not in out, "the expansion must never enter the transcript"
```

- [ ] **Step 2: Run them, confirm they fail**

Run: `cd ../carbon && uv run pytest tests/test_sandbox.py tests/test_offload_strategy.py -q -k "scratch or route" > /tmp/t1-red.txt 2>&1; echo "exit=$?"`
Expected: FAIL — `Sandbox` has no `scratch_dir`, footer has no shell route.

- [ ] **Step 3: Implement the sandbox half**

In `harness/sandbox.py`, add module constants beside `_SCRUBBED_ENV`:

```python
# The shell's route into the session's scratch. A fixed mount under Docker (the
# container path is stable and says nothing about the host); the real path under
# local execution, where there is no mount namespace to hide behind. Either way the
# MODEL only ever sees the variable's NAME — `$CARBON_SCRATCH_DIR/offload/<file>` —
# so no host path reaches a prompt or a stored transcript.
SCRATCH_ENV_VAR = "CARBON_SCRATCH_DIR"
DOCKER_SCRATCH_MOUNT = "/carbon-scratch"
```

`Sandbox.__init__` gains `scratch_dir: str | Path | None = None` stored as `self.scratch_dir = Path(scratch_dir) if scratch_dir else None` (falsy check, not `is not None`: an empty string must not become `Path('.')`).

In `_run_local`, after `env` is built:

```python
        if self.scratch_dir is not None:
            # Set only when there is a real directory: an empty value would expand to
            # "" and turn "$CARBON_SCRATCH_DIR/offload/x" into an absolute read from /.
            env[SCRATCH_ENV_VAR] = str(self.scratch_dir)
```

In `_run_docker`, extend the argv: when `self.scratch_dir is not None`, add
`"-v", f"{self.scratch_dir}:{DOCKER_SCRATCH_MOUNT}:ro"` and `"-e", f"{SCRATCH_ENV_VAR}={DOCKER_SCRATCH_MOUNT}"`.
Read-only on purpose: the spill is evidence to be read, and a container that could rewrite it could forge what the harness later attributes.

- [ ] **Step 4: Implement the footer half**

In `harness/limits.py`, beside `spill_ref`:

```python
def shell_ref(filename: str) -> str:
    """The bash route to the same artifact, as an UNEXPANDED variable reference.

    `scratch://` is carbon's internal identifier and only `read_file` resolves it.
    Iteration 5 measured what that costs: every one of the model's 32 attempts to
    reach a spill went through bash — grep, ls, a python one-liner — and every one
    failed, after which it fabricated the answer by re-deriving it. An identifier
    that looks like a path but is only honoured by a single tool is a private API
    handle wearing a path's clothes. Both consumers get an adapter."""
    return f"${SCRATCH_ENV_VAR}/{_OFFLOAD_DIRNAME}/{filename}"
```

with `from harness.sandbox import SCRATCH_ENV_VAR` — verify no import cycle (`sandbox.py` must not import `limits.py`; if it does, define `SCRATCH_ENV_VAR` in `limits.py` and import it from `sandbox.py` instead, and say which direction you chose in the report).

`_route` gains the shell line for the multi-line case, and — this is the case iteration 5's single-line finding named — becomes usable for one-liners too, since `cut`/`sed` can slice a long line the way `read_file` cannot:

```python
def _route(line_count: int, ref: str, shell: str) -> str:
    if line_count <= 1:
        return (
            f"one long line: read_file(path='{ref}') returns it whole, or slice it in "
            f"a shell, e.g. cut -c1-4000 \"{shell}\""
        )
    return (
        f"read_file(path='{ref}', start_line=1, end_line=<n>) to page it, "
        f"or search it in a shell, e.g. grep -n '<what you need>' \"{shell}\""
    )
```

`_footer` takes both refs and passes them to `_route`; `_offload_to_file` computes `shell_ref(Path(rel).name)`. Re-measure `MAX_FOOTER_CHARS` against the widest footer this now produces and update the constant AND the sentence recording it.

- [ ] **Step 5: Run the tests**

Run: `cd ../carbon && uv run pytest tests/test_sandbox.py tests/test_offload_strategy.py -q > /tmp/t1-green.txt 2>&1; echo "exit=$?"` — Expected exit=0.

- [ ] **Step 6: Commit**

```bash
cd ../carbon && git add -A harness/ tests/ && git commit -m "feat(ch-08): scratch artifacts reachable from bash, not only read_file"
```

---

### Task 2: Carbon — restore the symlink guard, close the falsy-root hole

**Files:** Modify `../carbon/harness/limits.py`, `../carbon/harness/tools.py`; Test `../carbon/tests/test_offload_strategy.py`, `../carbon/tests/test_hardening.py`

**Interfaces:** no signature changes; `_offload_dir` refuses symlinked components; `read_file`'s scratch branch guards falsy.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_symlinked_offload_dir_is_refused_before_anything_is_written(tmp_path):
    """The guard this batch deleted, restored with the reason it exists.

    `mkdir(parents=True, exist_ok=True)` does NOT raise on a pre-existing
    symlink-to-directory — verified — so a spill writes straight THROUGH the link,
    and `cleanup()`'s rmtree then removes only the link while the spilled bytes
    survive outside the session. Carbon's coding wiring runs trusted bash with no
    filesystem confinement and the scratch prefix is greppable in $TMPDIR, so the
    model can plant the link itself mid-session."""
    from harness.harness_config import TruncationPolicy
    from harness.limits import truncate_tool_result

    outside = tmp_path / "attacker"; outside.mkdir()
    scratch = tmp_path / "scratch"; scratch.mkdir()
    (scratch / "offload").symlink_to(outside, target_is_directory=True)

    out = truncate_tool_result(
        "S" * 9000, TruncationPolicy("offload_to_file", 4000, 0.3), scratch_dir=scratch
    )
    assert list(outside.iterdir()) == [], "nothing may be written through the link"
    assert "offload unavailable" in out, "the marker must say the copy does not exist"
    assert "scratch://" not in out, "and must not advertise a route to a file we refused to write"


def test_read_file_treats_an_empty_scratch_root_as_no_scratch(tmp_path, monkeypatch):
    """`Path("").resolve()` is the cwd, so `is None` lets "" read <cwd>/offload/...
    The write side already carries this fix; the read side did not."""
    from harness.tools import read_file
    monkeypatch.chdir(tmp_path)
    (tmp_path / "offload").mkdir()
    (tmp_path / "offload" / "ab.txt").write_text("CWD-LEAK")
    out = read_file("scratch://offload/ab.txt", root=tmp_path, scratch_root="")
    assert out.startswith("error:") and "CWD-LEAK" not in out
```

- [ ] **Step 2: Run them (expect failures), then implement**

`_offload_dir` regains containment, checked BEFORE any mkdir:

```python
def _offload_dir(scratch_dir: Path | None) -> Path:
    if not scratch_dir:
        raise _OffloadUnavailable("no scratch storage to write under")
    root = Path(scratch_dir)
    landed = root / _OFFLOAD_DIRNAME
    # Checked before creating anything: mkdir(exist_ok=True) FOLLOWS a symlinked
    # directory instead of raising, and by the time an after-the-fact check could
    # refuse, the bytes are already outside the session.
    if root.is_symlink() or landed.is_symlink():
        raise _OffloadUnavailable("scratch directory is a symlink")
    landed.mkdir(parents=True, exist_ok=True)
    resolved = landed.resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise _OffloadUnavailable("offload directory escapes the scratch root")
    return landed
```

`read_file`: `if scratch_root is None:` → `if not scratch_root:`.

- [ ] **Step 3: Gate and commit**

Run: `cd ../carbon && uv run pytest tests/ -q > /tmp/t2.txt 2>&1; echo "exit=$?"` — enumerate any failure not caused by Tasks 3–5 being unwritten.

```bash
cd ../carbon && git add -A harness/ tests/ && git commit -m "fix(ch-06): restore the offload symlink guard; empty scratch root is no scratch"
```

---

### Task 3: Carbon — durable sessions keep their scratch

**Files:** Modify `../carbon/harness/session_env.py`, `../carbon/harness/agent.py`; Test `../carbon/tests/test_session_env.py`

**Interfaces produced:**
- `SessionEnvironment.durable: bool = False`
- `local_session_env(workspace_root=None, session_id=None, *, session: str | None = None, sessions_dir: str | Path | None = None)` — when `session` is given, the scratch is `<sessions_dir>/<session>.scratch/`, `durable=True`, and `cleanup()` is a no-op.
- `delete_session_scratch(session: str, sessions_dir) -> None` for explicit removal.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_durable_session_keeps_its_scratch_across_close_and_reopen(tmp_path):
    """A session's transcript is persisted with scratch:// refs inside it. If close()
    deletes the scratch — and a reopened session gets a DIFFERENT root — every one of
    those refs is dead on reopen. The reference is session-scoped; the transcript
    that stores it is durable. They must share a lifetime."""
    from harness.session_env import local_session_env

    a = local_session_env(tmp_path, session="s1", sessions_dir=tmp_path / ".sessions")
    (a.scratch_root / "offload").mkdir(parents=True)
    (a.scratch_root / "offload" / "ab.txt").write_text("SPILLED")
    assert a.durable is True
    a.cleanup()
    assert (a.scratch_root / "offload" / "ab.txt").read_text() == "SPILLED"

    b = local_session_env(tmp_path, session="s1", sessions_dir=tmp_path / ".sessions")
    assert b.scratch_root == a.scratch_root, "reopening must land on the same scratch"


def test_two_sessions_do_not_share_scratch(tmp_path):
    from harness.session_env import local_session_env
    a = local_session_env(tmp_path, session="s1", sessions_dir=tmp_path / ".sessions")
    b = local_session_env(tmp_path, session="s2", sessions_dir=tmp_path / ".sessions")
    assert a.scratch_root != b.scratch_root


def test_deleting_a_session_removes_its_scratch(tmp_path):
    from harness.session_env import delete_session_scratch, local_session_env
    a = local_session_env(tmp_path, session="s1", sessions_dir=tmp_path / ".sessions")
    (a.scratch_root / "x.txt").write_text("x")
    delete_session_scratch("s1", tmp_path / ".sessions")
    assert not a.scratch_root.exists()


def test_an_ephemeral_session_still_cleans_up(tmp_path):
    from harness.session_env import local_session_env
    e = local_session_env(tmp_path)
    assert e.durable is False
    e.cleanup()
    assert not e.scratch_root.exists()
```

- [ ] **Step 2: Implement**

`SessionEnvironment` gains `durable: bool = False`; `cleanup()` returns early when `durable` (docstring: a durable scratch outlives the Agent by design — it is deleted with the session, not with the process that opened it). `local_session_env` builds the durable path with 0700, `parents=True`. `delete_session_scratch` rmtrees it.

`Agent.__init__`: when `self.session` is set and no env was supplied, build the durable env (`session=self.session, sessions_dir=self.sessions_dir`). Ownership still tracks construction, but `close()` on a durable env is a no-op by the env's own rule — say so in the constructor comment.

`scavenge()` must never touch durable scratch: it globs the OS temp dir for `SCRATCH_PREFIX*`, and durable scratch lives under the sessions dir, so this holds structurally — add one line to `scavenge`'s docstring stating that, and a test asserting a durable scratch under a sessions dir survives `scavenge(max_age_s=0)`.

- [ ] **Step 3: Gate and commit**

```bash
cd ../carbon && uv run pytest tests/test_session_env.py -q > /tmp/t3.txt 2>&1; echo "exit=$?"
cd ../carbon && git add -A harness/ tests/ && git commit -m "fix(ch-09): a durable session keeps its scratch, so its persisted refs still resolve"
```

---

### Task 4: Carbon — lifecycle completeness and the wiring gaps

**Files:** Modify `../carbon/harness/session_env.py`, `../carbon/harness/subagents.py`, `../carbon/harness/orchestrator.py`, `../carbon/harness/agent.py`, `../carbon/harness/limits.py`, `../carbon/ui/tui.py`; Test `../carbon/tests/test_session_env.py`, `../carbon/tests/test_offload_strategy.py`

Five findings, all "can affect behavior or safety":

1. **`scavenge` mtime** — staleness is judged from `scratch_root`'s own mtime, which nested writes into `offload/` never bump, so a live >24h session is reaped. Derive staleness from the newest mtime in the tree (`max` over `rglob`, falling back to the root's own), and test it: a root whose only recent write is `offload/x.txt` survives `scavenge(max_age_s=1)` after a sleep.
2. **`delegate_tool` / `fan_out_tool` do not forward `session_env`** — the model-facing delegation path gives each worker its own scratch while its registry resolves against the parent's, so a worker cannot read its own spill. Add the parameter to both factories, thread it from `_coding_tools` (`harness/agent.py`), and test through the TOOL (not `run_subagent` directly) that a worker's own spill resolves.
3. **`Orchestrator.run` threads `scratch_root` only into the fallback registry** — a caller-supplied `tools=` leaves `read_file` with `scratch_root=None`; real callers exist in `tasks/checks.py`. Thread it in both branches.
4. **`_OURS` never shrinks** — `SessionEnvironment.cleanup()` only rmtrees. Add a `limits.forget_spills(scratch_root)` that discards `_OURS` entries under that root, and call it from `cleanup()` (import locally to avoid a cycle). Test that a closed session leaves no residue.
5. **TUI `_build_agent` constructs the Agent before its try** — a raising `load_extensions` (user code) leaks the scratch. Wrap post-construction setup so a failure closes the agent it just built.

- [ ] **Step 1:** Write a test per finding, red first.
- [ ] **Step 2:** Implement.
- [ ] **Step 3:** `cd ../carbon && uv run verify > /tmp/t4.txt 2>&1; echo "exit=$?"` — expected exit=0.
- [ ] **Step 4:** Commit: `fix(ch-09): scavenge reads the whole tree; delegation, orchestrator and TUI own their scratch`

---

### Task 5: Carbon — wire the sandbox scratch everywhere, fix the test sweep

**Files:** Modify `../carbon/harness/agent.py`, `../carbon/harness/subagents.py`, `../carbon/tasks/checks.py`, `../carbon/ui/tui.py`, `../carbon/tests/conftest.py`

- [ ] **Step 1:** Every place that builds `Sandbox(...)` for an Agent's bash tool must pass `scratch_dir=<that agent's session_env.scratch_root>`. Grep: `grep -rn "Sandbox(" ../carbon/harness ../carbon/tasks ../carbon/ui`. The canonical order is the one Task 4 of the previous batch established: construct the Agent, then build tools from `agent.session_env.scratch_root`, then bind.
- [ ] **Step 2: Fix the conftest sweep (Codex P1).** The autouse fixture deletes any `carbon-scratch-*` that appeared during a test, including directories created by *another process* — a live measurement running concurrently loses its scratch mid-attempt, which surfaces as a fabricated mechanical security failure. Replace prefix-diffing with ownership: give tests their own scratch parent via a `CARBON_SCRATCH_TEST_ROOT` env var honoured by `local_session_env`, or track the environments the test itself created. Session-scope what remains (it currently globs the temp dir twice per test; measured ~30s per suite). State which approach you chose and why in the report.
- [ ] **Step 3:** `uv run verify`, exit code captured. Commit: `fix(ch-08): bash reaches scratch in every wiring; tests sweep only what they own`

---

### Task 6: Refinery — task builders, recorder fidelity, C3 attribution

**Files:** Modify `runner/tasks/cluster_*.py`, `runner/helpers.py`, `runner/tasks/cluster_c.py`, `tests/conftest.py`; Test `tests/test_registry.py`, `tests/test_helpers.py`

- [ ] **Step 1: Sandbox wiring.** Every `Sandbox(...)` built in a task module gains `scratch_dir=<agent>.session_env.scratch_root` so the graded model has the same shell route real carbon users get. Grep `grep -rn "Sandbox(" runner/`. Without this, r9 measures a carbon the shipped product does not have.
- [ ] **Step 2: `recording_tool` must capture raised results.** It appends to the sink only on normal return, while carbon's `ToolRegistry.call` stringifies exceptions into the transcript — so a secret inside a raising tool's error text is model-visible and invisible to every cluster-C leak scan. Wrap: record the exception's string, then re-raise. Test with a tool that raises a sentinel-bearing message and assert the sink saw it.
- [ ] **Step 3: C3's mechanical check must not fire on a cleanup that merely failed.** `scratch_cleaned = not env.scratch_root.exists()` treats any `rmtree(ignore_errors=True)` failure as a storage-contract violation, which hard-blocks with no Fisher tolerance — iteration 4's false attribution in the always-blocks direction. Distinguish: a durable env is expected to persist (not a violation); a removal that failed for an I/O reason is `fail`, not `critical_failure`; only a scratch that is present, non-durable, and removable-but-not-removed is mechanical. Encode the distinction in `_c3_outcome`'s inputs and pin every branch.
- [ ] **Step 4: conftest sweep** — same ownership fix as carbon's, mirrored.
- [ ] **Step 5:** `uv run pytest -q`, ruff, format — exit codes captured. Commit: `fix(runner): shell route for graded tasks; recorder sees raises; C3 mechanical means mechanical`

---

### Task 7: Refinery — record fidelity in the acceptance rule

**Files:** Modify `loop/acceptance.py`, `loop/prpipe.py`; Test `tests/test_acceptance.py`

- [ ] **Step 1:** Both REJECT branches in `evaluate()` must carry `behavioral_regressions` and `targeted_rerun`; today a behavioral rise co-occurring with a mechanical one, or with a no-gain rejection, vanishes from the record. Test both branches.
- [ ] **Step 2:** `security_failures` must distinguish a missing `security_classes` key from a present-but-empty list; `or [None] * len(outcomes)` currently pads an empty list and defeats the `strict=True` zip that exists to catch exactly that mismatch.
- [ ] **Step 3:** `pr_body` must render the rule's disposition, `security_regressions`, and `behavioral_verdicts`. A human approving a PR currently cannot see that a security regression was observed and cleared only by statistical power. Include the Fisher power limitation in one line so the reader knows what "no_increase" is worth.
- [ ] **Step 4:** Gates, then commit: `fix(rule): the record carries the security story on every path`

---

### Task 8: Gates, sabotage proofs, and a Codex review

- [ ] **Step 1:** Both repos green: `cd ../carbon && uv run verify` and refinery `uv run pytest -q`, `uv run ruff check .`, `uv run ruff format --check .` — every exit code captured and read.
- [ ] **Step 2:** `cd ../carbon && uv run accept ch-06 > /tmp/accept-ch06.txt 2>&1; echo "exit=$?"`. This is the check that was red before the batch — the reduced-affordance hypothesis predicts it goes green now that bash can reach the spill. Record the outcome either way; do not retry more than twice.
- [ ] **Step 3: Sabotage proofs**, each reverted immediately with the observed failure recorded verbatim: (a) drop the `CARBON_SCRATCH_DIR` injection → the shell-route test goes red; (b) remove the symlink guard → the through-write test goes red; (c) make durable `cleanup()` delete → the reopen test goes red; (d) restore `recording_tool`'s no-except form → the raising-tool test goes red.
- [ ] **Step 4: Codex review of both repos' unpushed ranges** (the companion resolves the repo from the working directory, so run each from its own repo root):
  ```bash
  cd /Users/adesai/Projects/carbon && node "/Users/adesai/.claude/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs" review --base origin/self-improvement --background
  cd /Users/adesai/Projects/refinery && node "/Users/adesai/.claude/plugins/cache/openai-codex/codex/1.0.6/scripts/codex-companion.mjs" review --base origin/main --background
  ```
  Triage every finding; fix what affects behavior or safety before r9.

---

### Task 9: Record baseline r9

- [ ] **Step 1: Preflight** — both trees clean (carbon's dev-notes stashes belong to another session; check `git stash list` and do not disturb them), model endpoint answering, `runner_sha` noted as changed.
- [ ] **Step 2:** Start the tree watchdog, then:
  ```bash
  cd /Users/adesai/Projects/refinery && caffeinate -is uv run python -m runner.cli run --label baseline-r9 > /tmp/r9.log 2>&1; echo "exit=$?" >> /tmp/r9.log
  ```
- [ ] **Step 3:** Expect zero mechanical criticals. **A mechanical critical in the BASELINE is a carbon bug — stop and fix before proceeding.** Record E4's baseline (expected 0/N: `head_tail` still cannot recover) and C3's behavioral count.

---

### Task 10: Re-measure the unchanged candidate

- [ ] **Step 1:** Validate `tool-output-offload-r3` — the same record, unedited — against r9 through `loop.cli validate`.
- [ ] **Step 2:** If CONFIRM, run the paired confirmation at 10 attempts per arm over the decision's `confirm_tasks` (the driver is at `<scratchpad>/iter5-confirm.sh`; update the labels to `r4`), then `confirmed()`.
- [ ] **Step 3: The discriminating question** — did E4 recover *through the shell route*? Inspect the candidate arm's E4 approvals: count how many scratch accesses used bash versus `read_file`, and how many attempts reached `recovered_from_disk=True`. Iteration 5's number was 32 bash / 0 read_file / 0 recovered. Report the new split whatever it says; a candidate that still fails for a different reason is a finding, not a failure of the batch.
- [ ] **Step 4:** Scrub (expect 0 changes), commit the full record, and report the seven-item verdict package.

---

## Self-Review Notes

- Checklist coverage: shell access (T1), symlink guard (T2), durable sessions (T3), cleanup on completion/error/cancellation/startup (T3+T4), read_file-and-bash against the same artifact (T1 tests + T10 step 3 live), traversal and cross-run isolation (T2+T3), transcript hygiene (T1's no-host-path assertion), remaining behavior/safety findings (T4, T6, T7), r9 (T9), unchanged candidate re-run (T10).
- Deliberately deferred, and why: the AST guard's name-rebinding blind spot and the hardcoded `"offload"` literals are test-quality items that cannot affect a measurement; `_scrub_obj`'s tuple blindness is latent until a tuple field exists. None gate r9.
