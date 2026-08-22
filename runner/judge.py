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

A YES must also be GROUNDED. The format check alone let two passes through that
no span of the answer supported: ``QUOTE:`` with nothing after it, and a quote
the judge composed rather than copied. Both read as a well-formed YES, and CMP-6
turns a YES straight into a task pass — so a judge having a bad minute could hand
this suite a pass built on a sentence the model never wrote.
``quote_is_grounded`` closes that: a YES stands only when its quote appears
verbatim in the answer, compared case-insensitively with whitespace collapsed.
Nothing looser — no punctuation stripping, no markdown stripping, no fuzzy match
— because every loosening starts accepting reconstructions, which is the exact
thing being refused. NO verdicts are untouched: the prompt asks a NO to cite "its
closest attempt", which by construction need not be a real span, and this rule may
only ever turn a pass into a failure, never the reverse.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
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


# --- transport: bounded retry for transient serving faults ----------------------
#
# The judge is a model call that does NOT go through carbon's ``Agent``, so none of
# carbon's retry policy reaches it: ``judged_equivalent`` calls ``chat()`` directly
# and every provider fault was terminal for its pair. This is the same defect class
# carbon fixed in its compaction summarizer (the one unretried model call in a turn),
# and it cost the same way — measured 2026-08-21 on the pinned OpenRouter/Novita
# base, a live re-validation of the 635-pair corpus lost 58 pairs to HTTP 429 and
# scored 0.899 agreement against a 0.95 threshold. Delivered-only agreement was
# 0.9896 with zero false approvals: the judge was sound, the delivery was not.
#
# The policy below is carbon's, restated: same transient classes, same exponential
# backoff, same "max_attempts bounds TOTAL tries, not retries". It is a COPY, not an
# import, on purpose — refinery grades carbon, and a grader that took its own
# transport correctness from the graded harness would be silently retuned by a new
# pinned base (or, in principle, by an accepted candidate). The copy's drift alarm is
# ``tests/test_judge.py``, which compares this classifier against carbon's own on a
# shared probe list.

# Transient status codes matched as standalone numbers, never substrings: carbon's
# own regression — "requested 15020 tokens" contains "502", and substring matching
# classified a context-overflow message as a transient gateway fault, spending the
# whole retry budget on a payload that could never fit.
_TRANSIENT_STATUS = re.compile(r"\b(?:429|502|503|504)\b")

_TRANSIENT_MARKERS = (
    "rate limit",
    "timeout",
    "timed out",
    "temporarily unavailable",
    "connection reset",
    "connection refused",
)


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded recovery for transient provider failures on the judge call.

    ``max_attempts`` bounds TOTAL tries, not retries — 5 means four waits (2s, 4s,
    8s, 16s at the shipped base) and then a give-up, which is carbon's reading of the
    same field name. ``max_delay_s`` caps a single wait; carbon has no such cap
    because its computed backoff never reaches one, and neither does this one (16 <
    30). It exists for the server-supplied ``Retry-After``, which is not ours to
    bound by construction: an endpoint asking for an hour must not stall the run.
    """

    max_attempts: int = 5
    base_delay_ms: int = 2000
    max_delay_s: float = 30.0


# Refinery's own values, deliberately equal to the ones carbon ships today
# (harness_config.json: backoff, 5, 2000). They are NOT read from carbon's config:
# that file is the editable surface this repo's loop rewrites, and a candidate edit
# must never be able to reach into the grader's transport.
JUDGE_RETRY = RetryPolicy()

# The wait, injectable. Tests replace this so an offline suite never spends a real
# 2/4/8/16-second backoff (``tests/conftest.py``), and so a test can assert on the
# delays the policy asked for rather than on wall-clock time.
_sleep = time.sleep


def transient_fault(exc: Exception) -> bool:
    """Is ``exc`` a serving fault worth trying again? Carbon's rule, restated.

    Public because ``tests/test_judge.py`` pins it against carbon's
    ``Agent._transient_error`` — the drift alarm the copy above needs.
    """
    text = str(exc).lower()
    if any(marker in text for marker in _TRANSIENT_MARKERS):
        return True
    # Status codes go through the word-boundary pattern, not ``in`` — see
    # ``_TRANSIENT_STATUS`` for the token-count false match that prevents.
    return _TRANSIENT_STATUS.search(text) is not None


def fault_class(exc: Exception) -> str:
    """A short, groupable label for one delivery fault, for the artifact's record.

    Only the KNOWN transient statuses are read out of the message, by the same
    word-boundary pattern the classifier uses; anything else is labelled by exception
    type. A loose three-digit scrape would have relabelled "you requested 15020
    tokens" as an HTTP 502 in the very diagnostics a human reads to find the cause.
    """
    status = _TRANSIENT_STATUS.search(str(exc))
    if status:
        return f"http_{status.group(0)}"
    lowered = str(exc).lower()
    for marker in _TRANSIENT_MARKERS:
        if marker in lowered:
            return marker.replace(" ", "_")
    return type(exc).__name__


def _retry_after_seconds(exc: Exception) -> float | None:
    """The server's own ``Retry-After``, in seconds, when the exception carries one.

    Duck-typed through ``exc.response.headers`` rather than typed against httpx: the
    provider seam is carbon's, refinery does not depend on its HTTP client, and a
    provider that raises something else simply has no header to offer. The delta-
    seconds form is honored; the HTTP-date form returns None and the computed backoff
    stands, because parsing a date against a possibly-skewed local clock to decide a
    2-vs-4-second wait buys nothing.

    A 429 that names its own window is the one authoritative number in this whole
    policy — ours are guesses about an undocumented ceiling, and continuing to guess
    over an explicit instruction is how a client keeps hammering a closed window.
    """
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except (AttributeError, TypeError):
        return None
    if raw is None:  # a response that carries headers but not this one
        return None
    try:
        seconds = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return seconds if seconds >= 0 else None


def _next_delay(attempt: int, exc: Exception, policy: RetryPolicy) -> float | None:
    """Seconds to wait before the NEXT try, or None when there is no next try.

    ``attempt`` is 1-indexed and already incremented, so ``max_attempts`` bounds
    total tries: at max_attempts=5 a delay is offered after tries 1-4 and refused
    after try 5.
    """
    if attempt >= policy.max_attempts or not transient_fault(exc):
        return None
    server_hint = _retry_after_seconds(exc)
    if server_hint is not None:
        return min(server_hint, policy.max_delay_s)
    return min(policy.base_delay_ms * (2 ** (attempt - 1)) / 1000, policy.max_delay_s)


# The validation artifact (contract §4) and, with it, the judge's ACTIVATION
# gate. The path lives here rather than in ``loop.judge_validate`` — which
# writes the file — because the reader is a TASK (``runner/tasks/cluster_g.py``
# CMP-6) and ``runner/`` may not import ``loop/``: that package's ``__init__``
# runs the carbon-base guard and the dependency would invert the layering. One
# constant, imported by the writer, so the two can never name different files.
AGREEMENT_PATH = (
    Path(__file__).resolve().parents[1] / "iterations" / "judge-validation" / "agreement.json"
)


# The version of the SCORING COMPUTATION that turns per-pair judge outputs into
# the artifact's agreement numbers and its ``pass`` verdict — the fourth pin of
# the artifact's identity, beside the prompt sha, the parser version, and the
# model. It lives HERE, not in ``loop.judge_validate`` where the computation
# runs, for ``AGREEMENT_PATH``'s reason: the reader (this gate) may not import
# ``loop``, and one constant imported by the writer keeps the stamp and the
# check from drifting apart.
#
# BUMP THIS, in the same commit, on any change to ``run_validation``'s scoring
# or pass logic that could alter agreement, the clean-denial gate, or ``pass``
# for the same judge outputs. The gate refuses an artifact stamped with any
# other version — or with none, which is what every artifact written before
# this pin carries. History: 1 = verdict-equality agreement (an undelivered
# judgment's fail-closed False could count as a correct NO); 2 = delivered-
# verdict agreement (undelivered pairs never agree; delivered/undelivered
# counts recorded per artifact, ``ran`` per record).
VALIDATION_COMPUTATION_VERSION = 2


def validation_status(path: Path | None = None, *, judge_model: str) -> tuple[bool, str]:
    """Is this judge validated for THIS prompt, parser, and model? ``(ok, reason)``.

    Contract §4: the judged verifiers' activation (CMP-6's verdict, CMP-5's
    extraction lane) is gated on
    ``iterations/judge-validation/agreement.json`` existing, recording
    ``pass: true``, AND carrying the CURRENT ``JUDGE_PROMPT_SHA``,
    ``JUDGE_PARSER_VERSION``, ``VALIDATION_COMPUTATION_VERSION``, and the model
    the live judge will actually run on (``judge_model`` — required, so no
    caller can forget the binding). Any
    other state returns False with the reason, and the tasks turn that into an
    ``error`` outcome — never a mechanical fallback, which would silently
    replace a meaning check with a substring check and report the result under
    the same task name.

    The identity comparisons are the half that is easy to forget and the one
    that matters most: the artifact is a measurement of a SPECIFIC prompt read
    by a SPECIFIC parser on a SPECIFIC model, and a change to any of the three
    leaves the file on disk, passing, describing a judge that no longer
    exists. Serving identity beyond the model string (provider, quantization)
    is NOT bound yet, deliberately: those fields do not exist on the provider
    seam today — they arrive with the queued fingerprint extension for the new
    serving base, and the artifact and this check must grow them in that same
    change rather than pretend to bind what is not recorded.

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
    # The parser half of the identity, same discipline as the sha: an artifact is a
    # measurement of a (prompt, parser) PAIR. A missing key is an artifact measured
    # before the parser was versioned at all — refused for the same reason a
    # mismatch is, because "the parser it describes no longer exists" covers both.
    recorded_parser = artifact.get("judge_parser_version")
    if recorded_parser != JUDGE_PARSER_VERSION:
        return False, (
            f"validation artifact was measured for judge_parser_version={recorded_parser!r}, "
            f"this parser is {JUDGE_PARSER_VERSION} — re-run the validation"
        )
    # The scoring-computation half: an artifact scored under another rule (or
    # before scoring was versioned at all — the missing-key case) may carry a
    # ``pass: true`` the current rule would refuse.
    recorded_computation = artifact.get("validation_computation_version")
    if recorded_computation != VALIDATION_COMPUTATION_VERSION:
        return False, (
            "validation artifact was scored under "
            f"validation_computation_version={recorded_computation!r}, this scorer is "
            f"{VALIDATION_COMPUTATION_VERSION} — re-run the validation"
        )
    # The model half of the identity: agreement measured on one model says nothing
    # about another. A missing key refuses too — an artifact that never said which
    # model it measured is not evidence about any.
    recorded_model = artifact.get("model")
    if recorded_model != judge_model:
        return False, (
            f"validation artifact was measured with judge model {recorded_model!r}, "
            f"this run's judge is {judge_model!r} — re-run the validation"
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

    ``ran`` separates two kinds of False verdict. True means a verdict actually
    came back in the pinned two-line format — a real NO, or a YES the grounding
    rule refused (the judge decided; the decision failed the pair). False means
    NO decision exists: the provider call failed or the output never parsed.
    Both fail closed to ``verdict=False``, but a verifier that recorded the
    second kind as a task failure would be blaming the strategy under test for
    a judge outage — ``ran`` is what lets it refuse (outcome ``error``) instead.

    ``attempts`` and ``faults`` are the DELIVERY record: how many tries this one
    judgment took, and a label per fault that cost a try (``fault_class``). Both
    describe the transport and neither can move a verdict — ``attempts`` is 1 and
    ``faults`` empty on every call that succeeded first time, which is nearly all of
    them. They exist so a validation run can state what the transport did instead of
    leaving a human to re-derive it from hundreds of raw error strings, which is
    exactly how the 2026-08-21 throttling was diagnosed.
    """

    verdict: bool
    quote: str
    raw: str
    ran: bool = True
    tokens: int = 0
    attempts: int = 1
    faults: tuple[str, ...] = ()


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


# The parser's behavior version, pinned by the activation gate BESIDE the prompt
# sha. The sha pins what the judge is ASKED; it cannot see a change in how the
# reply is READ — and that reading is the three functions below (``_normalized``,
# ``quote_is_grounded``, ``_parse_judgment``). The validation artifact records the
# version it was measured under, and ``validation_status`` refuses any artifact
# measured under a different one, so a parser change makes the judge loudly
# unvalidated instead of silently re-scoring history under a new rule.
#
# BUMP THIS, in the same commit, on any change to those functions that could alter
# a verdict or quote for the same raw judge output — a new refusal rule, a
# loosened or tightened match, a reordered check. Additive metadata that leaves
# every verdict and quote untouched does not bump. ``tests/test_judge.py`` pins
# the three functions' source by digest, so no edit can skip this decision
# silently. History: 1 = the strict two-line parse; 2 = the grounding rule (a YES
# must cite a verbatim span of the answer).
JUDGE_PARSER_VERSION = 2


def _normalized(text: str) -> str:
    """``text`` with every whitespace run collapsed to one space, case-folded.

    The two differences a real quote is allowed to have from the answer it was
    copied out of: the model re-wrapped a line, or it re-cased a word. Both leave
    the SPAN intact. Everything else — different punctuation, dropped markdown,
    a synonym — means the judge produced the span instead of copying it, and that
    is exactly what the grounding check exists to catch.
    """
    return " ".join(text.split()).casefold()


def quote_is_grounded(quote: str, answer: str) -> bool:
    """Does ``quote`` appear verbatim in ``answer`` (case-insensitive, whitespace
    normalized)?

    Public because it is applied in two places that must never disagree: here, at
    judgment time, and in the offline re-scoring of
    ``iterations/judge-validation/agreement.json`` (``tests/test_judge.py``), which
    is how the committed validation's numbers are checked against this rule without
    a live re-run. A second copy of "verbatim" would let the gate and the evidence
    for the gate drift apart.

    An empty (or whitespace-only) quote is never grounded: a YES that cites nothing
    cites nothing, and the two-line format alone used to let it through.
    """
    normalized = _normalized(quote)
    return bool(normalized) and normalized in _normalized(answer)


def _parse_judgment(raw: str, answer: str) -> Judgment:
    """Strict two-line parse plus the grounding rule: first line ``VERDICT: YES``
    or ``VERDICT: NO`` exactly, second line starting ``QUOTE:``. Anything else
    fails CLOSED (verdict False, quote "", and ``ran=False`` — no verdict ever
    came back) with ``raw`` preserved. Lines after the second are ignored — the
    contract pins the first two, not the total length.

    A parsed YES then has to earn its verdict: its quote must be grounded in
    ``answer``. An ungrounded YES fails closed the same way a malformed one does,
    except the quote is KEPT — a reader has to see what the judge claimed to be
    citing — and the reason is appended to ``raw`` beneath the judge's own output,
    which is preserved intact. The reason lands in ``raw`` rather than in a new
    field because ``raw`` is what ``loop.judge_validate`` already records per pair
    and what a human reads when a verdict looks wrong.
    """
    lines = raw.splitlines()
    if len(lines) < 2:
        return Judgment(False, "", raw, ran=False)
    verdict_line = lines[0].strip()
    quote_line = lines[1].strip()
    if verdict_line not in ("VERDICT: YES", "VERDICT: NO"):
        return Judgment(False, "", raw, ran=False)
    if not quote_line.startswith("QUOTE:"):
        return Judgment(False, "", raw, ran=False)
    quote = quote_line[len("QUOTE:") :].strip()
    if verdict_line == "VERDICT: NO":
        return Judgment(False, quote, raw)
    if not quote.strip():
        return Judgment(
            False, quote, f"{raw}\n<ungrounded: the QUOTE is empty, so this YES cites nothing>"
        )
    if not quote_is_grounded(quote, answer):
        return Judgment(
            False,
            quote,
            f"{raw}\n<ungrounded: the QUOTE does not appear in the ANSWER, so this YES "
            "rests on a span the answer never contained>",
        )
    return Judgment(True, quote, raw)


def judged_equivalent(
    expected: str, answer: str, provider, *, retry: RetryPolicy = JUDGE_RETRY
) -> Judgment:
    """Ask the judge whether ``answer`` means the same thing as ``expected``.

    ``provider`` is the same seam every refinery task uses
    (``runner.carbon_env.make_provider`` live, or a scripted one in tests) —
    injected, never constructed here, so this stays offline-testable and the
    live path is entirely the caller's choice of provider.

    The payload sent is built from ONLY ``expected`` and ``answer`` (contract
    §4) — no transcript, no task instructions; there is nothing else in this
    function's signature for either to come from.

    Fail-closed guarantee covers output parsing, GROUNDING, and transport: a
    malformed reply, a YES whose quote is not a verbatim span of ``answer``, and a
    provider exception (network, auth, rate-limit, service unavailable) all return
    a verdict of False, so one flaky call — or one invented citation — degrades the
    pair rather than crashing the validation or passing a task. The runner's own
    catch-all remains the outer net.

    A TRANSIENT provider fault is now tried again first, under ``retry`` (see
    ``RetryPolicy``): 429/502/503/504 and the timeout/rate-limit/connection markers,
    with exponential backoff and the server's own ``Retry-After`` when it sends one.
    The bound is hard, and exhausting it changes nothing about the outcome — a
    persistent fault still returns ``ran=False`` with the last error in ``raw``, so a
    throttled judge degrades the pair exactly as before rather than quietly becoming
    a verdict.

    What is NOT retried is the line this fix must not cross: a reply that ARRIVED and
    did not parse returns immediately. That is a fact about the judge's behavior, not
    about the transport, and re-rolling it would hand a free-forming judge extra
    chances at the pinned format — changing what the validation artifact measures.
    Only a call that produced no output at all is repeated, so the same raw output
    still parses and scores identically; the retry can turn a NON-verdict into a
    verdict, never one verdict into another.

    ``answer`` therefore reaches the parser as well as the payload. It is the same
    string in both places, by construction: there is one ``answer`` in this
    function, and nothing between here and the parse can substitute another.
    """
    from model import chat  # lazy: runner/ modules never bind carbon at import time

    payload = f"EXPECTED FACT:\n{expected}\n\nANSWER:\n{answer}"
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": payload},
    ]
    faults: list[str] = []
    attempt = 0
    while True:
        attempt += 1
        try:
            # The guarded region is the whole original one — call, parse, and usage —
            # so nothing that used to fail closed now escapes into the caller.
            response = chat(messages, provider=provider, temperature=0.0, max_tokens=512)
            judgment = _parse_judgment(response.content or "", answer)
            return replace(
                judgment,
                tokens=_usage_tokens(response.usage),
                attempts=attempt,
                faults=tuple(faults),
            )
        except Exception as exc:
            faults.append(fault_class(exc))
            delay = _next_delay(attempt, exc, retry)
            if delay is None:
                error_msg = f"<provider error: {exc}>"
                return Judgment(
                    False, "", error_msg, ran=False, attempts=attempt, faults=tuple(faults)
                )
            if delay:
                _sleep(delay)
