"""The meaning-equivalence judge (contract §4) — CMP-6's verifier.

A pinned, minimal LLM call: given an EXPECTED fact and an extracted ANSWER,
decide whether the answer conveys the same meaning. Used by CMP-6's judged
verifier and by ``loop.judge_validate``, which measures how often the judge
agrees with mechanical ground truth on real (expected, answer) pairs pulled
from the round-2 null campaign.

The judge sees ONLY the expected fact and the answer text — never the
transcript, never the task's own instructions (contract §4, hard constraint).
A judge that saw the transcript could rubber-stamp a task by pattern-matching
the setup prose instead of judging the answer, and a judge briefed on the
task's own instructions could be steered by wording a self-editing loop
controls. Keeping the payload to two fields makes that structurally
impossible rather than a convention someone could forget — see
``judged_equivalent``'s signature: there is nowhere to put a transcript.

Output is constrained to two lines (``VERDICT: YES|NO`` then ``QUOTE: ...``)
and parsed strictly: anything else is a parse failure, and a parse failure
fails CLOSED (verdict False) rather than guessing. A judge that free-forms is
a judge that is not actually pinned, and quietly accepting near-miss formats
would let the effective judging behavior drift without ``JUDGE_PROMPT_SHA``
ever changing to say so.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

JUDGE_PROMPT = """You judge whether an ANSWER conveys the same meaning as an EXPECTED FACT.

You will be given:
  EXPECTED FACT: a short statement of fact.
  ANSWER: a separate piece of text that may or may not restate that fact.

Judge YES only if the ANSWER states the EXPECTED FACT's meaning, even if worded
differently, abbreviated, or embedded in a longer reply. Judge NO if the ANSWER:
  - denies having the information (e.g. "I don't have that", "not provided")
  - states a different or contradictory value
  - is empty, off-topic, garbled, or a tool-call fragment instead of prose
  - only partially matches (a required detail is missing or wrong)

Respond with EXACTLY two lines and nothing else:
VERDICT: YES
QUOTE: <the exact span of ANSWER that supports your verdict>

or:
VERDICT: NO
QUOTE: <the exact span of ANSWER that contradicts it, or its closest attempt>
"""

# Pinned so a prompt edit is LOUD (tests/test_judge.py asserts this equality) —
# the judge's behavior is only as trustworthy as iterations/judge-validation/
# agreement.json says it is, and that artifact is only trustworthy for THIS
# exact prompt. Changing JUDGE_PROMPT without re-running validation must be a
# visible, deliberate act, never a silent drift.
JUDGE_PROMPT_SHA = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()


# The validation artifact (contract §4) and, with it, the judge's ACTIVATION
# gate. The path lives here rather than in ``loop.judge_validate`` — which
# writes the file — because the reader is a TASK (``runner/tasks/cluster_g.py``
# CMP-6) and ``runner/`` may not import ``loop/``: that package's ``__init__``
# runs the carbon-base guard and the dependency would invert the layering. One
# constant, imported by the writer, so the two can never name different files.
AGREEMENT_PATH = (
    Path(__file__).resolve().parents[1] / "iterations" / "judge-validation" / "agreement.json"
)


def validation_status(path: Path | None = None) -> tuple[bool, str]:
    """Is this judge validated for THIS prompt? ``(ok, reason)``.

    Contract §4: CMP-6's activation is gated on
    ``iterations/judge-validation/agreement.json`` existing, recording
    ``pass: true``, AND carrying the CURRENT ``JUDGE_PROMPT_SHA``. Any other
    state returns False with the reason, and CMP-6 turns that into an
    ``error`` outcome — never a mechanical fallback, which would silently
    replace a meaning check with a substring check and report the result under
    the same task name.

    The sha comparison is the half that is easy to forget and the one that
    matters most: the artifact is a measurement of a SPECIFIC prompt's
    agreement with mechanical ground truth, and a prompt edit leaves the file
    on disk, passing, describing a judge that no longer exists.

    Fails CLOSED on every unreadable state (missing file, bad JSON, an OSError)
    — the same discipline as ``_parse_judgment``.
    """
    path = AGREEMENT_PATH if path is None else path
    if not path.is_file():
        return False, "no validation artifact on disk (run `python -m loop.judge_validate`)"
    try:
        artifact = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return False, f"validation artifact unreadable ({type(exc).__name__})"
    if not isinstance(artifact, dict):
        return False, "validation artifact is not a JSON object"
    recorded_sha = artifact.get("judge_prompt_sha")
    if recorded_sha != JUDGE_PROMPT_SHA:
        return False, (
            f"validation artifact was measured for judge_prompt_sha={str(recorded_sha)[:12]!r}, "
            f"this judge is {JUDGE_PROMPT_SHA[:12]!r} — re-run the validation"
        )
    if artifact.get("pass") is not True:
        return False, f"validation artifact records pass={artifact.get('pass')!r}"
    return True, ""


@dataclass(frozen=True)
class Judgment:
    """One judge call's verdict (contract §4).

    ``quote`` is the answer span the judge cites (parsed from its own QUOTE
    line, never re-derived); ``raw`` is the judge's full, unparsed output —
    kept even on a parse failure so a human can see what actually came back.

    ``tokens`` is what the judge call itself cost, from the provider's OWN
    reported usage. A judged task makes a SECOND model call per attempt, and
    an unrecorded one is cost the suite's per-task means silently understate.
    Zero when the provider reported no usage (a scripted provider, a transport
    failure) — never an estimate, which would put a fabricated number in the
    same field as measured ones.
    """

    verdict: bool
    quote: str
    raw: str
    tokens: int = 0


def _usage_tokens(usage: dict) -> int:
    """The judge call's total tokens from an OpenAI-style usage dict, or 0.

    ``total_tokens`` when the provider reported one; otherwise the prompt/completion
    split summed, which is the only other shape carbon's own accounting produces
    (model/pricing.py takes the same two routes). Absent or unreadable usage is 0 —
    a call whose cost nobody reported is recorded as unmeasured, never estimated.
    """
    if not usage:
        return 0
    try:
        total = int(usage.get("total_tokens", 0) or 0)
        if total:
            return total
        return int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _parse_judgment(raw: str) -> Judgment:
    """Strict two-line parse: first line ``VERDICT: YES`` or ``VERDICT: NO``
    exactly, second line starting ``QUOTE:``. Anything else fails CLOSED
    (verdict False, quote "") with ``raw`` preserved. Lines after the second
    are ignored — the contract pins the first two, not the total length."""
    lines = raw.splitlines()
    if len(lines) < 2:
        return Judgment(False, "", raw)
    verdict_line = lines[0].strip()
    quote_line = lines[1].strip()
    if verdict_line not in ("VERDICT: YES", "VERDICT: NO"):
        return Judgment(False, "", raw)
    if not quote_line.startswith("QUOTE:"):
        return Judgment(False, "", raw)
    quote = quote_line[len("QUOTE:") :].strip()
    return Judgment(verdict_line == "VERDICT: YES", quote, raw)


def judged_equivalent(expected: str, answer: str, provider) -> Judgment:
    """Ask the judge whether ``answer`` means the same thing as ``expected``.

    ``provider`` is the same seam every refinery task uses
    (``runner.carbon_env.make_provider`` live, or a scripted one in tests) —
    injected, never constructed here, so this stays offline-testable and the
    live path is entirely the caller's choice of provider.

    The payload sent is built from ONLY ``expected`` and ``answer`` (contract
    §4) — no transcript, no task instructions; there is nothing else in this
    function's signature for either to come from.

    Fail-closed guarantee covers both output parsing AND transport: provider
    exceptions (network, auth, rate-limit, service unavailable) return
    Judgment(False, "", "<provider error: ...>") so one flaky call degrades
    the pair rather than crashing the validation. The runner's own catch-all
    remains the outer net.
    """
    from model import chat  # lazy: runner/ modules never bind carbon at import time

    payload = f"EXPECTED FACT:\n{expected}\n\nANSWER:\n{answer}"
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": payload},
    ]
    try:
        response = chat(messages, provider=provider, temperature=0.0, max_tokens=512)
        judgment = _parse_judgment(response.content or "")
        return replace(judgment, tokens=_usage_tokens(response.usage))
    except Exception as exc:
        error_msg = f"<provider error: {exc}>"
        return Judgment(False, "", error_msg)
