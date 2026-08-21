"""The written record beside `model-r2.json`, checked against the artifact it describes.

Two documents live in `iterations/calibration-compaction/`: a README that says what
state the compaction calibration is in, and a CORRECTION that fixes a wrong diagnosis in
a commit message. Both make claims that are checkable — how many arms, which check
refused, how many attempts errored and why — and a document whose numbers nobody checks
drifts away from the record it sits next to. Every claim asserted here is re-derived from
the committed evidence first, and only then looked for in the prose.

This is deliberately not a spell-check of the documents. It pins the load-bearing facts:
if a number here goes red, either the record changed or the document is now wrong, and
those are the only two states worth interrupting someone for.
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = REPO_ROOT / "iterations" / "calibration-compaction"
README = CALIBRATION / "README.md"
CORRECTION = CALIBRATION / "CORRECTION.md"
MODEL = CALIBRATION / "model-r2.json"
RESULTS = REPO_ROOT / "results"

P2C_ARMS = tuple([f"p2c-null-full-{c}" for c in "abc"] + [f"p2c-null-cmp-{c}" for c in "abcdefg"])


def _model() -> dict:
    return json.loads(MODEL.read_text())


def _records(task: str | None = None) -> list[dict]:
    """Every attempt recorded across the ten Phase 2c arms, optionally one task's."""
    out = []
    for label in P2C_ARMS:
        for line in (RESULTS / f"{label}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if task is None or record["task"] == task:
                out.append(record)
    return out


def _normalized(path: Path) -> str:
    """The document as one whitespace-normalized line, so a claim's presence never
    turns on where a paragraph happened to wrap."""
    return " ".join(path.read_text().split())


# ---------------------------------------------------------------------------
# The README: the state the section is actually in
# ---------------------------------------------------------------------------


def test_the_readme_describes_the_artifact_that_is_actually_committed():
    """Every count the README states, re-derived from the artifact beside it."""
    model = _model()
    assert tuple(p["label"] for p in model["provenance"]) == P2C_ARMS
    assert len(model["null_model"]) == 7
    assert model["fitness"]["fit"] is False
    assert model["fitness"]["stability"]["pass"] is False

    doc = _normalized(README)
    assert "ten arms" in doc, "the pooling's size"
    assert "seven" in doc, "and the number of tasks it rates"
    assert "fit: false" in doc.lower() or "fit=false" in doc.lower()
    assert "stability" in doc, "and which check refused"
    # The round-1 history is KEPT, not overwritten by the current state.
    assert "analysis-r1-unfit.json" in doc
    assert "withdrawn" in doc.lower()
    # And round 2 is marked superseded rather than silently dropped.
    assert "superseded" in doc.lower()


def test_the_readme_states_what_a_round_three_measurement_needs():
    """A refusal with no stated route out is a dead end dressed as a finding.

    The three things a round-3 measurement has to change are named: bounds computed at
    the attempt counts judgments actually use; more held-in attempts (the auditor's
    finding — at n=3 the grain is 1/9 and adding ARMS alone buys a PASS without buying
    knowledge); and re-measurement on a pinned provider and quantization.
    """
    doc = _normalized(README)
    assert "1/9" in doc, "the grain the held-in attempt count pins"
    assert "n=3" in doc or "three attempts" in doc
    assert "held-in attempts" in doc, "raising them is what buys resolution"
    assert "OpenRouter" in doc
    assert "quantization" in doc and "provider" in doc

    # The grain claim is the artifact's own, not a remembered one.
    grain = _model()["fitness"]["grain"]["held_in"]
    assert Fraction(grain["grain"]) == Fraction(1, 9)
    assert set(grain["attempts"].values()) == {3}


def test_no_document_beside_the_artifact_cites_a_document_that_does_not_live_here():
    """AGENTS.md's rule, applied to the record's own prose.

    `test_p2b_closing.py` scans `loop/` and `tests/` for citations of the private
    program contract. The README cited it too and nothing looked, so a reader of the
    committed record was pointed at a file that does not exist in this repo.
    """
    # Assembled from halves so this file is not its own first offender.
    markers = ("contracts/" + "phase2-calibration-contract", "contracts/" + "phase2b-calibration")
    offenders = []
    for path in sorted(CALIBRATION.glob("*.md")):
        text = path.read_text()
        offenders.extend(f"{path.name}: {m}" for m in markers if m in text)
    assert not offenders, offenders


# ---------------------------------------------------------------------------
# The CORRECTION: what the six error attempts actually were
# ---------------------------------------------------------------------------


def test_the_campaign_errors_were_serving_faults_not_premise_failures():
    """The correction's central claim, re-derived from the tracebacks themselves.

    Commit `8de800c` recorded two CMP-7 errors as "setup errors — premise mostly but not
    always fires", i.e. as the task failing to reach the thing it measures. Every one of
    the six is an `httpx.HTTPStatusError: 400` from the local model endpoint. A serving
    fault and a premise that did not fire are different facts about the suite, and only
    one of them is evidence about compaction.
    """
    errors = [r for r in _records() if r["outcome"] == "error"]
    assert len(errors) == 6, [(r["task"], r["attempt"]) for r in errors]
    for record in errors:
        assert "HTTPStatusError" in record["detail"], (record["task"], record["attempt"])
        # `Client error '400`, not the whole phrase: `runner/run.py` clamps a traceback
        # to eight frames and one of these six is cut mid-sentence at "'400 B". A test
        # that demanded the full string would be asserting the clamp, not the fault.
        assert "Client error '400" in record["detail"], (record["task"], record["attempt"])
        # Not the premise guard: those exit through `Attempt(False, "error", ...)` with a
        # written reason, never with a traceback.
        assert "Traceback" in record["detail"]
        assert "did not fire" not in record["detail"]
        assert "never left the live transcript" not in record["detail"]

    # Three of the six died inside compaction's own summarizer call, which is NOT routed
    # through the retry wrapper the agent's own calls use.
    summarizer = [
        r for r in errors if "compaction.py" in r["detail"] and "_summarize" in r["detail"]
    ]
    assert len(summarizer) == 3, [(r["task"], r["attempt"]) for r in summarizer]
    retried = [r for r in errors if "model_call_with_recovery" in r["detail"]]
    assert len(retried) == 3
    assert not [r for r in summarizer if "model_call_with_recovery" in r["detail"]]

    doc = _normalized(CORRECTION)
    assert "8de800c" in doc, "the correction must name the commit it corrects"
    assert "400" in doc and "HTTP" in doc
    assert "summarizer" in doc.lower()


def test_the_correction_states_the_denominator_effect_the_record_shows():
    """What the diagnosis changes: not a verdict, a denominator.

    A serving fault is not evidence about compaction, but the runner counts a visible
    error as a recorded non-pass, so it sits in the published pooled rate anyway. Both
    numbers belong in the record — the one the artifact publishes and the one the task
    would have shown had the endpoint stayed up — and neither replaces the other.
    """
    published = {
        task: Fraction(_model()["null_model"][task]["null_rate"]) for task in ("CMP-7", "G5")
    }
    for task, excluding in (("CMP-7", (67, 74)), ("G5", (49, 78))):
        records = _records(task)
        passes = sum(1 for r in records if r["passed"])
        errors = sum(1 for r in records if r["outcome"] == "error")
        assert (passes, len(records) - errors) == excluding, (task, passes, len(records), errors)
        # The published rate keeps the errors in the denominator: that is the runner's
        # visible-error policy (`runner/run.py`'s `TaskResult.pass_fraction`), not a
        # rounding difference.
        assert published[task] == Fraction(passes, len(records))
        assert published[task] < Fraction(*excluding)

    doc = _normalized(CORRECTION)
    assert "67/74" in doc and "90.5" in doc
    assert "49/78" in doc
    assert "pass_fraction" in doc or "visible-error" in doc, (
        "the correction must say WHY the published rates still include the faults"
    )


def test_the_correction_does_not_edit_the_record_it_corrects():
    """iter-04's convention: the record stands, the prose beside it corrects the
    narrative. The six error rows keep their tracebacks and their `error` outcome, and
    the artifact keeps the rates it published."""
    records = _records()
    errors = [r for r in records if r["outcome"] == "error"]
    assert len(errors) == 6
    assert all(r["passed"] is False for r in errors)
    doc = _normalized(CORRECTION)
    assert re.search(r"stands|unedited|not (been )?edited", doc, re.I), doc[:200]
    # The denominator the correction quotes for "six of N attempts".
    assert len(records) == 835
    assert "6 of 835" in doc or "Six of 835" in doc
