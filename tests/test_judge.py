"""Offline tests for the meaning-equivalence judge and its validation harness
(contract §4). Every provider here is scripted — no live model call anywhere
in this file."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from model.fake import fake
from model.provider import LLMResponse, Provider

from loop import judge_validate
from runner.judge import JUDGE_PROMPT, JUDGE_PROMPT_SHA, Judgment, judged_equivalent

# --- runner.judge ---------------------------------------------------------------


# The judge prompt's digest, as a LITERAL. Derived from `JUDGE_PROMPT` it was a
# tautology: `JUDGE_PROMPT_SHA` is itself computed by that same expression, so the
# assertion re-ran the definition and could not go red for any prompt edit. Written
# out, it goes red on every one — which is the whole point of pinning a prompt whose
# measured agreement lives in a separate file.
PINNED_JUDGE_PROMPT_SHA = "88fe0e70592e7a6ee72797f032156afd6f747cbe5c8d6a95fd127b941ef07663"


def test_judge_prompt_sha_is_pinned():
    """A prompt edit must be a LOUD, deliberate act (contract §4): this test goes red
    the instant JUDGE_PROMPT changes, and stays red until the validation artifact is
    re-measured for the new prompt and this literal is updated to match."""
    assert JUDGE_PROMPT_SHA == PINNED_JUDGE_PROMPT_SHA
    # And the constant really is the digest of the prompt beside it, so a hand-edited
    # `JUDGE_PROMPT_SHA` cannot satisfy the pin above on its own.
    assert hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest() == PINNED_JUDGE_PROMPT_SHA


def test_the_committed_agreement_artifact_was_measured_for_this_prompt():
    """The pin's reason, made checkable: `agreement.json` is a measurement OF this
    prompt, and CMP-6's activation gate compares the two shas. If they ever separate,
    the artifact describes a judge that no longer exists."""
    from runner.judge import AGREEMENT_PATH

    artifact = json.loads(AGREEMENT_PATH.read_text())
    assert artifact["judge_prompt_sha"] == PINNED_JUDGE_PROMPT_SHA


# The parsing code's own digest, a LITERAL like the prompt sha above and for the same
# reason: `_normalized` + `quote_is_grounded` + `_parse_judgment` ARE the judge's
# reading of its model's output, and the prompt sha cannot see them change. This pin
# goes red on ANY edit to those three functions. The discipline it enforces: decide,
# per edit, whether the change can alter a verdict or quote for the same raw output —
# if yes, bump JUDGE_PARSER_VERSION in the same commit (the activation gate then
# refuses the stale validation artifact until a re-run); if no (comments, formatting),
# update this digest alone and say so in the commit.
PINNED_JUDGE_PARSER_SOURCE_SHA = "97a91f9ae6892c13b3ff92eaa0445f07a0eea75143998428c43020f965ffe6b5"


def _parser_source() -> str:
    import inspect

    from runner import judge

    return "".join(
        inspect.getsource(f)
        for f in (judge._normalized, judge.quote_is_grounded, judge._parse_judgment)
    )


def test_judge_parser_version_is_pinned_against_the_parsing_source():
    """Handoff item: the prompt sha alone cannot see parser-behavior changes. The
    version constant is the artifact-facing half (the activation gate compares it);
    this source pin is the discipline-facing half — it makes forgetting the bump
    impossible by turning every parser edit into a deliberate decision."""
    from runner.judge import JUDGE_PARSER_VERSION

    assert JUDGE_PARSER_VERSION == 2  # 1 = strict two-line parse; 2 = grounding rule
    digest = hashlib.sha256(_parser_source().encode()).hexdigest()
    assert digest == PINNED_JUDGE_PARSER_SOURCE_SHA, (
        "the judge parser's source changed: if the change can alter any verdict or "
        "quote for the same raw judge output, bump JUDGE_PARSER_VERSION in the same "
        "commit and queue a re-validation; either way update the pinned digest "
        "deliberately, never reflexively"
    )


def test_validation_status_requires_the_artifact_to_match_the_parser_version(tmp_path):
    """The activation gate's new conjunct: an artifact is a measurement of a
    (prompt, parser) PAIR, and either half changing leaves it describing a judge
    that no longer exists. Missing and mismatched versions both refuse — a missing
    key is an artifact measured before the parser was versioned at all."""
    from runner.judge import (
        JUDGE_PARSER_VERSION,
        JUDGE_PROMPT_SHA,
        VALIDATION_COMPUTATION_VERSION,
        validation_status,
    )

    good = tmp_path / "agreement.json"
    good.write_text(
        json.dumps(
            {
                "pass": True,
                "judge_prompt_sha": JUDGE_PROMPT_SHA,
                "judge_parser_version": JUDGE_PARSER_VERSION,
                "validation_computation_version": VALIDATION_COMPUTATION_VERSION,
                "model": "m",
            }
        )
    )
    assert validation_status(good, judge_model="m") == (True, "")

    for version in (None, JUDGE_PARSER_VERSION + 1, "2"):
        artifact = {
            "pass": True,
            "judge_prompt_sha": JUDGE_PROMPT_SHA,
            "validation_computation_version": VALIDATION_COMPUTATION_VERSION,
            "model": "m",
        }
        if version is not None:
            artifact["judge_parser_version"] = version
        stale = tmp_path / "stale.json"
        stale.write_text(json.dumps(artifact))
        ok, why = validation_status(stale, judge_model="m")
        assert ok is False
        assert "judge_parser_version" in why and "re-run" in why


def test_validation_status_binds_the_artifact_to_the_live_judge_model(tmp_path):
    """The third identity pin, same construction as the prompt sha and the parser
    version: agreement is a measurement of a specific JUDGE, and the judge is a
    model as much as it is a prompt and a parser. Validate with model A, flip the
    provider to model B, and the artifact used to keep activating CMP-5/6 —
    describing a judge that is not the one running. The gate now takes the live
    judge's model identity and refuses on mismatch; a missing key refuses too (an
    artifact that never said which model it measured is not evidence about any)."""
    from runner.judge import (
        JUDGE_PARSER_VERSION,
        JUDGE_PROMPT_SHA,
        VALIDATION_COMPUTATION_VERSION,
        validation_status,
    )

    artifact = {
        "pass": True,
        "judge_prompt_sha": JUDGE_PROMPT_SHA,
        "judge_parser_version": JUDGE_PARSER_VERSION,
        "validation_computation_version": VALIDATION_COMPUTATION_VERSION,
        "model": "model-A",
    }
    path = tmp_path / "agreement.json"
    path.write_text(json.dumps(artifact))
    assert validation_status(path, judge_model="model-A") == (True, "")

    ok, why = validation_status(path, judge_model="model-B")
    assert ok is False
    assert "model-A" in why and "model-B" in why and "re-run" in why

    del artifact["model"]
    path.write_text(json.dumps(artifact))
    ok, why = validation_status(path, judge_model="model-A")
    assert ok is False and "model" in why


def test_validation_status_requires_the_current_validation_computation(tmp_path):
    """The reopened half of the undelivered-verdict hole, closed at the GATE.

    Fixing run_validation alone protected only artifacts the fixed scorer writes.
    An artifact produced by the OLD scorer — no ``ran`` records, no delivery
    counts — can carry the current prompt sha, parser version, and model with
    ``pass: true`` stamped while its judge timed out on every negative, and the
    gate would activate it: the exact hole, reintroduced through the artifact
    store. So the artifact's identity gains a fourth pin, the same construction
    as the parser version: ``run_validation`` stamps the version of the scoring
    computation it ran, and the gate refuses any artifact stamped with another —
    or with none, which is precisely what every pre-fix artifact carries. A
    version pin rather than a keys-exist check, deliberately: the presence of
    ``delivered_count`` proves a key exists, not that ``pass`` was computed under
    the delivered-verdict rule — and the pin covers the NEXT scoring change with
    a one-line bump, which a schema sniff never would."""
    from runner.judge import (
        JUDGE_PARSER_VERSION,
        JUDGE_PROMPT_SHA,
        VALIDATION_COMPUTATION_VERSION,
        validation_status,
    )

    identity = {
        "pass": True,
        "judge_prompt_sha": JUDGE_PROMPT_SHA,
        "judge_parser_version": JUDGE_PARSER_VERSION,
        "model": "m",
    }
    path = tmp_path / "agreement.json"

    # The current computation, stamped: accepted.
    path.write_text(
        json.dumps({**identity, "validation_computation_version": VALIDATION_COMPUTATION_VERSION})
    )
    assert validation_status(path, judge_model="m") == (True, "")

    # No stamp — every artifact the pre-fix scorer wrote looks like this.
    path.write_text(json.dumps(identity))
    ok, why = validation_status(path, judge_model="m")
    assert ok is False
    assert "validation_computation_version" in why and "re-run" in why

    # A stale stamp: the old verdict-equality scoring.
    path.write_text(json.dumps({**identity, "validation_computation_version": 1}))
    ok, why = validation_status(path, judge_model="m")
    assert ok is False and "validation_computation_version" in why


def test_the_committed_agreement_artifact_is_refused_until_revalidated_for_this_parser():
    """The honest current state, pinned on purpose so it cannot look accidental.

    The committed artifact was measured before the parser carried a version (and
    before the grounding rule existed). The offline re-scoring below shows its
    NUMBERS survive the stricter rule — but the gate's job is identity, not
    arithmetic: an artifact measured under a different parser is not evidence about
    this one. So CMP-5 and CMP-6 refuse (outcome `error`) until the queued live
    re-validation on the new serving base writes an artifact stamped with the
    current version. Fail closed is the designed state here, not a regression."""
    from runner.judge import validation_status

    # Any live model: the parser-version refusal comes first regardless.
    ok, why = validation_status(judge_model="any-model")
    assert ok is False
    assert "judge_parser_version" in why


def test_yes_verdict_parses_quote():
    raw = "VERDICT: YES\nQUOTE: never exceed 30 seconds"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent(
        "retry backoff must never exceed 30 seconds",
        "the backoff should never exceed 30 seconds between retries",
        provider,
    )
    assert judgment == Judgment(True, "never exceed 30 seconds", raw)


# --- Grounding: a YES stands only on a verbatim span of the ANSWER ---------------


def test_yes_with_an_empty_quote_fails_closed():
    """A YES that cites nothing cites nothing. The two-line format was satisfied, so
    the old parser handed back verdict=True with an empty quote — a pass no span of
    the answer supports."""
    raw = "VERDICT: YES\nQUOTE:"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("expected fact", "an answer that states the fact", provider)
    assert judgment.verdict is False
    assert raw in judgment.raw, "the judge's own output is preserved"
    assert "ungrounded" in judgment.raw


def test_yes_with_a_quote_absent_from_the_answer_fails_closed():
    """The attack this closes: a judge that INVENTS the supporting span. The quote is
    fluent, on-topic and nowhere in the answer, and the verdict rode on it."""
    raw = "VERDICT: YES\nQUOTE: the deploy key rotates every Tuesday"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent(
        "the deploy key rotates every Tuesday",
        "I believe there is a rotation policy of some kind.",
        provider,
    )
    assert judgment.verdict is False
    assert judgment.quote == "the deploy key rotates every Tuesday", (
        "the quote is still reported — a reader has to see what the judge claimed"
    )
    assert raw in judgment.raw and "ungrounded" in judgment.raw


def test_yes_with_a_grounded_quote_stands():
    raw = "VERDICT: YES\nQUOTE: TUESDAY-KEY-9X"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("TUESDAY-KEY-9X", "The key is `TUESDAY-KEY-9X`.", provider)
    assert judgment == Judgment(True, "TUESDAY-KEY-9X", raw)


def test_grounding_is_case_insensitive_and_whitespace_normalized():
    """A model that re-wraps or re-cases a span it really did copy is still quoting it.
    Anything looser than this (stripping punctuation, markdown, or stopwords) would
    start accepting reconstructions, which is the thing being refused."""
    raw = "VERDICT: YES\nQUOTE: Never   Exceed\n30 Seconds"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent(
        "backoff never exceeds 30 seconds",
        "the backoff will never exceed 30 seconds, per the runbook",
        provider,
    )
    assert judgment.verdict is True


def test_a_no_verdict_is_never_regrounded():
    """NO verdicts are unaffected: the contract asks a NO to cite the closest attempt,
    which by construction may not be a verbatim span at all. Fail-closed only ever
    turns a pass into a failure, never the reverse."""
    raw = "VERDICT: NO\nQUOTE: nothing in the answer says this"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("expected fact", "unrelated reply", provider)
    assert judgment == Judgment(False, "nothing in the answer says this", raw)


def test_a_no_verdict_with_an_empty_quote_is_still_just_a_no():
    raw = "VERDICT: NO\nQUOTE:"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", raw)


def test_the_committed_agreement_artifact_survives_the_grounding_rule():
    """Re-validation of the judge, done OFFLINE against the record it already wrote.

    A live re-run is a Phase 3 item — it needs model calls. But the artifact stores
    every pair's recorded verdict and quote, and `build_corpus()` rebuilds the same
    corpus from the same committed JSONLs, so the recorded judge outputs can be
    RE-SCORED under the grounding rule without asking the judge anything. The join is
    positional and verified field by field, never assumed.

    What this pins: the agreement gate (>= 95%) and the zero-YES-on-clean-denial gate
    both still hold once every YES has to cite a real span. If a future prompt or rule
    change breaks either, this goes red on committed evidence rather than on a rerun
    nobody scheduled.
    """
    from runner.judge import AGREEMENT_PATH, quote_is_grounded

    artifact = json.loads(AGREEMENT_PATH.read_text())
    corpus = judge_validate.build_corpus()
    records = artifact["records"]
    assert len(records) == len(corpus) == 635

    agree = 0
    denial_yes = 0
    regrounded = 0
    for record, pair in zip(records, corpus, strict=True):
        assert (record["task"], record["fact"], record["source_file"], record["attempt"]) == (
            pair.task,
            pair.fact,
            pair.source_file,
            pair.attempt,
        ), "the artifact's records must line up with the corpus they were measured on"
        assert record["expected"] == pair.expected
        assert record["ground_truth"] == pair.ground_truth
        verdict = record["judge_verdict"]
        if verdict and not quote_is_grounded(record["quote"], pair.answer):
            verdict = False
            regrounded += 1
        agree += int(verdict == pair.ground_truth)
        denial_yes += int(pair.is_clean_denial and verdict)

    # Exactly one recorded YES loses its verdict: a G4 reply whose quote dropped the
    # markdown emphasis around it, so the span was reconstructed rather than copied.
    assert regrounded == 1
    assert denial_yes == 0, "the clean-denial gate still holds under the stricter rule"
    assert agree / len(records) >= judge_validate.AGREEMENT_THRESHOLD
    assert artifact["pass"] is True


def test_no_verdict_parses_quote():
    raw = "VERDICT: NO\nQUOTE: the answer says something else entirely"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("expected fact", "wrong answer", provider)
    assert judgment == Judgment(False, "the answer says something else entirely", raw)


def test_garbage_output_fails_closed():
    raw = "I think the answer is probably related to the topic."
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("expected fact", "some answer", provider)
    assert judgment == Judgment(False, "", raw, ran=False)


def test_empty_output_fails_closed():
    provider = fake(scripted=lambda messages: "")
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", "", ran=False)


def test_missing_quote_line_fails_closed():
    raw = "VERDICT: YES"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", raw, ran=False)


def test_wrong_case_fails_closed():
    raw = "verdict: yes\nquote: something"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", raw, ran=False)


def test_second_line_not_quote_prefixed_fails_closed():
    raw = "VERDICT: YES\nnever exceed 30 seconds"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", raw, ran=False)


def test_trailing_lines_after_quote_are_ignored():
    raw = "VERDICT: NO\nQUOTE: xyz\nsome extra reasoning the model appended anyway"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("e", "a", provider)
    assert judgment.verdict is False
    assert judgment.quote == "xyz"


def test_payload_carries_only_expected_and_answer():
    """Contract §4: the judge sees ONLY the expected fact and the extracted
    answer -- never a transcript, never task instructions. Assert directly on
    the scripted provider's captured payload rather than trusting the
    docstring."""
    captured = {}

    def responder(messages):
        captured["messages"] = messages
        return "VERDICT: YES\nQUOTE: q"

    provider = fake(scripted=responder)
    expected = "the constraint on retry backoff"
    answer = "the answer text extracted from the transcript"
    judged_equivalent(expected, answer, provider)

    assert captured["messages"] == [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"EXPECTED FACT:\n{expected}\n\nANSWER:\n{answer}"},
    ]


def test_every_cmp6_pass_in_the_campaign_carries_a_grounded_quote():
    """The other half of the offline re-validation: the campaign's own CMP-6 passes.

    `agreement.json` measures the judge against MECHANICAL ground truth on A1/G2/G4/G5.
    CMP-6 is the task where a judge YES IS the pass, with no mechanical check beside
    it, so the grounding rule's real exposure is there. Every CMP-6 pass across the ten
    Phase 2c arms is re-checked here against the reply it was made on; both strings are
    in the recorded `detail`.

    Read honestly: `runner/tasks/cluster_g.py` clamps the recorded quote to 160
    characters and the recorded reply to 240. No reply in this campaign reached the
    reply clamp, so the answers are whole; 15 of the 17 quotes sit exactly at the quote
    clamp, so what is verified for those is that the first 160 characters of the quote
    are a verbatim span of the full reply. The clamp is a limit of this retrospective
    check only — the live rule sees both strings unclamped.
    """
    import ast
    import re

    from runner.judge import quote_is_grounded

    results = Path(__file__).resolve().parents[1] / "results"
    passes = []
    for path in sorted(results.glob("p2c-null-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("task") == "CMP-6" and record.get("passed"):
                passes.append((path.name, record))

    assert len(passes) == 17, "the campaign's CMP-6 pass count, from the record itself"
    ungrounded = []
    clamped = 0
    for name, record in passes:
        detail = record["detail"]
        quote = ast.literal_eval(re.search(r"judge_quote=(.*?) reply=", detail, re.DOTALL).group(1))
        reply = ast.literal_eval(re.search(r"reply=(.*)$", detail, re.DOTALL).group(1))
        clamped += int(len(quote) == 160)
        if not quote_is_grounded(quote, reply):
            ungrounded.append((name, record["attempt"], quote))
    assert not ungrounded, (
        "a CMP-6 pass whose judge quote is not a span of the reply it judged: the "
        f"recorded rate would change under the grounding rule — {ungrounded}"
    )
    assert clamped == 15, "and the clamp's reach, stated rather than assumed"


# --- loop.judge_validate: corpus construction, real fixture lines ---------------

# Copied verbatim from results/r2-null-cmp-a.jsonl (contract §4's labeled
# corpus source) so parsing is exercised against real production formatting,
# not a hand-written approximation of it -- including the backtick-wrapped
# reply and the escaped-newline reply that are easy to get wrong.

A1_PASS_LINE = (
    '{"task": "A1", "split": "held_in", "cluster": "A", "expected_baseline": "uncertain", '
    '"primitive": "compaction", "alias": "CMP-1", "attempt": 0, "passed": true, "outcome": '
    '"pass", "security_class": null, "detail": "compacted=True sentinel_recalled=True '
    'reply=\'`TUESDAY-KEY-9X`\'", "approvals": [], "turns": 11, "metrics": {}, '
    '"duration_s": 45.2, "gemma_sha": "31d74ca3ba7489ee0f7d14b6c5efe0a18da9a304", '
    '"gemma_dirty": false, "dirty_sha": null, "config_version": 8, '
    '"model": "google/gemma-4-26b-a4b", "runner_sha": "0d7ea47", "behavior_key": "c1011a82a588"}'
)

A1_DENY_LINE = (
    '{"task": "A1", "split": "held_in", "cluster": "A", "expected_baseline": "uncertain", '
    '"primitive": "compaction", "alias": "CMP-1", "attempt": 4, "passed": false, "outcome": '
    '"fail", "security_class": null, "detail": "compacted=True sentinel_recalled=False '
    "reply='I do not have access to information regarding a deploy key rotation schedule "
    'in our current conversation context.\'", "approvals": [], "turns": 11, "metrics": {}, '
    '"duration_s": 49.8, "gemma_sha": "31d74ca3ba7489ee0f7d14b6c5efe0a18da9a304", '
    '"gemma_dirty": false, "dirty_sha": null, "config_version": 8, '
    '"model": "google/gemma-4-26b-a4b", "runner_sha": "0d7ea47", "behavior_key": "c1011a82a588"}'
)

G2_PASS_LINE = (
    '{"task": "G2", "split": "held_out", "cluster": "G", "expected_baseline": "uncertain", '
    '"primitive": "compaction", "alias": "CMP-2", "attempt": 1, "passed": true, "outcome": '
    '"pass", "security_class": null, "detail": "compactions=2 early_recalled=True '
    "late_recalled=True reply='EARLY-DECISION-G2-3LK, LATE-DECISION-G2-6QW'\", "
    '"approvals": [], "turns": 9, "metrics": {}, "duration_s": 65.2, '
    '"gemma_sha": "31d74ca3ba7489ee0f7d14b6c5efe0a18da9a304", "gemma_dirty": false, '
    '"dirty_sha": null, "config_version": 8, "model": "google/gemma-4-26b-a4b", '
    '"runner_sha": "0d7ea47", "behavior_key": "c1011a82a588"}'
)

G4_PARTIAL_LINE = (
    '{"task": "G4", "split": "held_in", "cluster": "G", "expected_baseline": "uncertain", '
    '"primitive": "compaction", "alias": "CMP-3", "attempt": 0, "passed": false, "outcome": '
    '"fail", "security_class": null, "detail": "compactions=5 files_recalled=False '
    "rejected_recalled=False next_action_recalled=True reply='Files changed so far: None "
    "recorded.\\\\nApproach tried and rejected: None recorded.\\\\nPending next action: "
    'NEXT-ACTION-G4-5HB.\'", "approvals": [], "turns": 9, "metrics": {}, '
    '"duration_s": 172.9, "gemma_sha": "31d74ca3ba7489ee0f7d14b6c5efe0a18da9a304", '
    '"gemma_dirty": false, "dirty_sha": null, "config_version": 8, '
    '"model": "google/gemma-4-26b-a4b", "runner_sha": "0d7ea47", "behavior_key": "c1011a82a588"}'
)

G5_FAIL_LINE = (
    '{"task": "G5", "split": "held_in", "cluster": "G", "expected_baseline": "uncertain", '
    '"primitive": "compaction", "alias": "CMP-4", "attempt": 2, "passed": false, "outcome": '
    '"fail", "security_class": null, "detail": "compactions=2 '
    "observed=['reconcile_flags.py', 'audit_flags.py'] recalled=['audit_flags.py'] "
    "reply='In this session, I have created:\\\\n* `audit_flags.py`'\", "
    '"approvals": [], "turns": 10, "metrics": {}, "duration_s": 111.9, '
    '"gemma_sha": "31d74ca3ba7489ee0f7d14b6c5efe0a18da9a304", "gemma_dirty": false, '
    '"dirty_sha": null, "config_version": 8, "model": "google/gemma-4-26b-a4b", '
    '"runner_sha": "0d7ea47", "behavior_key": "c1011a82a588"}'
)

# Setup-failure detail carries no reply= field at all -- must be skipped, not judged.
A1_SETUP_ERROR_LINE = (
    '{"task": "A1", "split": "held_in", "cluster": "A", "expected_baseline": "uncertain", '
    '"primitive": "compaction", "alias": "CMP-1", "attempt": 2, "passed": false, "outcome": '
    '"error", "security_class": null, "detail": "compaction never fired after 22 filler '
    'turns (last_tokens=1200)", "approvals": [], "turns": 24, "metrics": {}, '
    '"duration_s": 12.0, "gemma_sha": "x", "gemma_dirty": false, "dirty_sha": null, '
    '"config_version": 8, "model": "google/gemma-4-26b-a4b", "runner_sha": "0d7ea47", '
    '"behavior_key": "c1011a82a588"}'
)


def test_a1_pass_line_yields_one_pair():
    record = json.loads(A1_PASS_LINE)
    pairs = judge_validate.pairs_from_record(record, "r2-null-cmp-a.jsonl")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.task == "A1"
    assert p.expected == "TUESDAY-KEY-9X"
    assert p.answer == "`TUESDAY-KEY-9X`"
    assert p.ground_truth is True
    assert p.is_clean_denial is False
    assert p.attempt == 0


def test_a1_deny_line_is_clean_denial():
    record = json.loads(A1_DENY_LINE)
    pairs = judge_validate.pairs_from_record(record, "r2-null-cmp-a.jsonl")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.ground_truth is False
    assert p.is_clean_denial is True
    assert p.answer.startswith("I do not have access")


def test_g2_pass_line_splits_into_two_pairs():
    record = json.loads(G2_PASS_LINE)
    pairs = judge_validate.pairs_from_record(record, "r2-null-cmp-a.jsonl")
    assert len(pairs) == 2
    by_fact = {p.fact: p for p in pairs}
    assert set(by_fact) == {"early", "late"}
    assert by_fact["early"].expected == "EARLY-DECISION-G2-3LK"
    assert by_fact["late"].expected == "LATE-DECISION-G2-6QW"
    assert by_fact["early"].ground_truth is True
    assert by_fact["late"].ground_truth is True
    # Both pairs share the one recorded reply.
    assert by_fact["early"].answer == by_fact["late"].answer
    assert by_fact["early"].is_clean_denial is False


def test_g4_partial_line_splits_into_three_pairs_with_mixed_ground_truth():
    record = json.loads(G4_PARTIAL_LINE)
    pairs = judge_validate.pairs_from_record(record, "r2-null-cmp-a.jsonl")
    assert len(pairs) == 3
    by_fact = {p.fact: p for p in pairs}
    assert set(by_fact) == {"files", "rejected", "next"}
    assert by_fact["files"].ground_truth is False
    assert by_fact["rejected"].ground_truth is False
    assert by_fact["next"].ground_truth is True
    # G4's own reply text has a real embedded newline once un-repr'd -- the
    # escaped-newline decoding must round-trip correctly.
    assert "\n" in by_fact["files"].answer
    assert "NEXT-ACTION-G4-5HB" in by_fact["next"].answer
    # G4's "None recorded" phrasing is a different failure style than A1/G2's
    # explicit refusal and must NOT be flagged as a clean denial.
    assert by_fact["files"].is_clean_denial is False


def test_g5_fail_line_yields_one_pair():
    record = json.loads(G5_FAIL_LINE)
    pairs = judge_validate.pairs_from_record(record, "r2-null-cmp-a.jsonl")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.ground_truth is False  # only one of the two files was recalled
    assert "audit_flags.py" in p.answer
    assert p.is_clean_denial is False


def test_setup_error_line_yields_no_pairs():
    """A setup failure has no reply -- nothing to judge, so it must not
    silently become a (expected, "") pair with a fabricated ground truth."""
    record = json.loads(A1_SETUP_ERROR_LINE)
    assert judge_validate.pairs_from_record(record, "r2-null-cmp-a.jsonl") == []


def test_unrelated_task_yields_no_pairs():
    record = {"task": "H1", "detail": "reply='x'", "passed": True}
    assert judge_validate.pairs_from_record(record, "f.jsonl") == []


def test_build_corpus_reads_and_aggregates_across_files(tmp_path):
    f1 = tmp_path / "r2-null-cmp-x.jsonl"
    f1.write_text(A1_PASS_LINE + "\n" + G2_PASS_LINE + "\n")
    f2 = tmp_path / "r2-null-cmp-y.jsonl"
    f2.write_text(G4_PARTIAL_LINE + "\n" + A1_SETUP_ERROR_LINE + "\n")

    corpus = judge_validate.build_corpus(files=[f1, f2])

    # 1 (A1) + 2 (G2) + 3 (G4) = 6; the setup-error line contributes nothing.
    assert len(corpus) == 6
    assert {p.source_file for p in corpus} == {"r2-null-cmp-x.jsonl", "r2-null-cmp-y.jsonl"}


def test_real_corpus_clean_denial_counts_match_contract():
    """Contract §4 names this exact set: "the 15 G2 + 10 A1 explicit 'I don't
    have it' replies". G2 splits into 2 pairs/reply (early+late), so those 15
    replies surface as 30 clean-denial PAIRS; A1 is single-pair, so its 10
    replies are 10 pairs. G4/G5 deny in a structurally different way ("None
    recorded") and must contribute zero. Offline: reads the eleven committed
    files directly, no model call."""
    corpus = judge_validate.build_corpus()
    denial_pairs = Counter(p.task for p in corpus if p.is_clean_denial)
    assert denial_pairs["A1"] == 10
    assert denial_pairs["G2"] == 30
    assert denial_pairs["G4"] == 0
    assert denial_pairs["G5"] == 0


# --- loop.judge_validate: run_validation / main, tiny synthetic corpus ----------


def _pair(task="X1", fact="sentinel", expected="E", answer="A", ground_truth=True, denial=False):
    return judge_validate.CorpusPair(task, fact, expected, answer, ground_truth, denial, "f", 0)


def test_run_validation_computes_agreement_and_disagreements():
    corpus = [
        _pair(answer="agrees-yes", ground_truth=True),
        _pair(answer="agrees-no", ground_truth=False),
        _pair(answer="disagrees", ground_truth=True),
    ]

    def responder(messages):
        text = messages[1]["content"]
        if "agrees-yes" in text:
            return "VERDICT: YES\nQUOTE: agrees-yes"  # grounded, so the YES stands
        if "agrees-no" in text:
            return "VERDICT: NO\nQUOTE: q"
        return "VERDICT: NO\nQUOTE: q"  # disagrees with ground_truth=True

    provider = fake(scripted=responder)
    result = judge_validate.run_validation(corpus, provider)

    assert result["total_pairs"] == 3
    assert result["agree_count"] == 2
    assert result["overall_agreement"] == 2 / 3
    assert result["disagree_count"] == 1
    assert result["disagreements"][0]["expected"] == "E"
    assert result["judge_prompt_sha"] == JUDGE_PROMPT_SHA
    # The artifact records the (prompt, parser, scoring) identity it measured —
    # the activation gate refuses it the moment any of the three moves. The
    # scoring stamp matters doubly: if the WRITER dropped it, every fresh
    # artifact would refuse at the gate while this suite stayed green.
    from runner.judge import JUDGE_PARSER_VERSION, VALIDATION_COMPUTATION_VERSION

    assert result["judge_parser_version"] == JUDGE_PARSER_VERSION
    assert result["validation_computation_version"] == VALIDATION_COMPUTATION_VERSION
    assert result["model"] == provider.model


def test_run_validation_fails_below_agreement_threshold():
    corpus = [_pair(answer="a", ground_truth=True), _pair(answer="b", ground_truth=True)]
    provider = fake(scripted=lambda messages: "VERDICT: NO\nQUOTE: q")  # 0% agreement
    result = judge_validate.run_validation(corpus, provider)
    assert result["overall_agreement"] == 0.0
    assert result["pass"] is False


def test_run_validation_fails_on_yes_for_clean_denial():
    """The contract's second gate: even at 100% overall agreement, a single
    judge=YES on a clean-denial pair must fail the whole artifact."""
    corpus = [
        _pair(answer="agrees", ground_truth=True, denial=False),
        _pair(answer="clean-denial-reply", ground_truth=False, denial=True),
    ]

    def responder(messages):
        text = messages[1]["content"]
        if "clean-denial-reply" in text:
            # Wrongly says YES on a denial -- and cites a real span of it, so the
            # grounding rule cannot mask the failure this gate exists to catch.
            return "VERDICT: YES\nQUOTE: clean-denial-reply"
        return "VERDICT: YES\nQUOTE: agrees"

    provider = fake(scripted=responder)
    result = judge_validate.run_validation(corpus, provider)
    # Both pairs "agree" by construction (ground_truth=True/verdict=True for
    # the first pair) except the denial one, which disagrees AND trips the
    # clean-denial gate.
    assert result["clean_denial_yes_count"] == 1
    assert result["pass"] is False


def test_run_validation_never_counts_an_undelivered_verdict_as_agreement():
    """The review's probe, kept as a test: a judge that delivers YES on positives and
    times out on negatives used to score 100% agreement with a clean-denial gate pass
    — and not one NO was ever delivered, because an undelivered judgment fails closed
    to verdict=False, which happens to equal every negative ground truth. 306 of the
    635 real corpus pairs are negative, so that hole was material.

    A pair with no verdict is not evidence of agreement. It counts as NOT agreed, it
    lands in the disagreement list, and the artifact says how many verdicts were
    actually delivered — so a validation run the judge slept through cannot stamp
    pass: true."""
    corpus = [
        _pair(answer="works", ground_truth=True),
        _pair(answer="neg-denial", ground_truth=False, denial=True),
        _pair(answer="neg-plain", ground_truth=False),
    ]

    def responder(messages):
        text = messages[1]["content"]
        if "works" in text:
            return "VERDICT: YES\nQUOTE: works"
        raise RuntimeError("timeout")  # ran=False: no verdict ever delivered

    result = judge_validate.run_validation(corpus, fake(scripted=responder))

    assert result["agree_count"] == 1
    assert result["overall_agreement"] == 1 / 3
    assert result["pass"] is False
    assert result["delivered_count"] == 1
    assert result["undelivered_count"] == 2
    # The undelivered pairs are visible per record and in the disagreement list.
    assert [r["ran"] for r in result["records"]] == [True, False, False]
    assert result["disagree_count"] == 2


def test_run_validation_empty_corpus_does_not_vacuously_pass():
    provider = fake(scripted=lambda messages: "VERDICT: YES\nQUOTE: q")
    result = judge_validate.run_validation([], provider)
    assert result["total_pairs"] == 0
    assert result["pass"] is False


def test_main_writes_agreement_artifact(tmp_path):
    output_path = tmp_path / "agreement.json"
    provider = fake(scripted=lambda messages: "VERDICT: YES\nQUOTE: q")

    result = judge_validate.main(provider=provider, output_path=output_path)

    assert output_path.exists()
    on_disk = json.loads(output_path.read_text())
    assert on_disk == result
    assert on_disk["judge_prompt_sha"] == JUDGE_PROMPT_SHA
    from runner.judge import JUDGE_PARSER_VERSION, VALIDATION_COMPUTATION_VERSION

    assert on_disk["judge_parser_version"] == JUDGE_PARSER_VERSION
    assert on_disk["validation_computation_version"] == VALIDATION_COMPUTATION_VERSION
    assert on_disk["model"] == provider.model
    assert on_disk["total_pairs"] == len(judge_validate.build_corpus())


# --- ran: a False verdict vs the absence of a verdict ----------------------------


def test_a_judgment_says_whether_a_verdict_actually_came_back():
    """``ran`` separates two kinds of False verdict that must not be conflated.

    A judge that answered ``VERDICT: NO`` made a decision about the answer; a judge
    whose call failed or whose output never parsed made NO decision at all. Both fail
    closed to ``verdict=False`` — that stays — but a verifier that treats the second
    kind as evidence about the strategy under test is recording a judge outage as a
    task failure. ``ran`` is what lets such a caller refuse (outcome ``error``)
    instead of blaming the strategy.
    """
    # Delivered verdicts, YES and NO: the judge decided.
    yes = judged_equivalent(
        "E", "the answer states E", fake(scripted=lambda m: "VERDICT: YES\nQUOTE: E")
    )
    assert yes.ran is True and yes.verdict is True
    no = judged_equivalent("E", "unrelated", fake(scripted=lambda m: "VERDICT: NO\nQUOTE: q"))
    assert no.ran is True and no.verdict is False

    # An ungrounded YES also RAN: the judge delivered a verdict and the grounding
    # rule refused it — a real (policy) failure of the pair, not a judge outage.
    ungrounded = judged_equivalent(
        "E", "nothing supporting it", fake(scripted=lambda m: "VERDICT: YES\nQUOTE: invented span")
    )
    assert ungrounded.ran is True and ungrounded.verdict is False

    # No verdict came back: garbage, empty, and a provider exception.
    for scripted in (lambda m: "I think so, probably.", lambda m: ""):
        judgment = judged_equivalent("E", "a", fake(scripted=scripted))
        assert judgment.ran is False and judgment.verdict is False

    def boom(messages):
        raise RuntimeError("service unavailable")

    failed = judged_equivalent("E", "a", fake(scripted=boom))
    assert failed.ran is False and failed.verdict is False


# --- Transport/provider failures (fail-closed) ----------------------------------


def test_provider_exception_fails_closed():
    """If the provider raises an exception (network, auth, API error), the judge
    should catch it and return Judgment(False, "", "<provider error: ...>") so one
    flaky call degrades that pair instead of crashing the validation."""

    def failing_responder(messages):
        raise RuntimeError("Model service unavailable")

    provider = fake(scripted=failing_responder)
    judgment = judged_equivalent("expected fact", "some answer", provider)
    assert judgment.verdict is False
    assert judgment.quote == ""
    assert "provider error" in judgment.raw
    assert "Model service unavailable" in judgment.raw


def test_run_validation_guards_provider_exception_per_pair():
    """Each pair's judge call should be guarded independently so one flaky call
    records a disagreement (pair recorded with the error in raw) instead of
    aborting the whole validation run."""

    def responder_with_failure(messages):
        text = messages[1]["content"]
        if "fails" in text:
            raise ValueError("API rate limit exceeded")
        return "VERDICT: YES\nQUOTE: works"  # grounded in both surviving answers

    corpus = [
        _pair(answer="works", ground_truth=True),
        _pair(answer="fails", ground_truth=True),
        _pair(answer="works-too", ground_truth=True),
    ]

    provider = fake(scripted=responder_with_failure)
    result = judge_validate.run_validation(corpus, provider)

    # All three pairs should be present in the records, no exception raised.
    assert result["total_pairs"] == 3
    # The second pair should disagree (expected True, got False due to exception).
    assert result["records"][1]["judge_verdict"] is False
    assert "provider error" in result["records"][1]["raw"]
    assert result["disagree_count"] >= 1


# --- Clean denial / ground truth contradiction (loud assertion) ----------------


def test_pairs_from_record_asserts_no_clean_denial_with_ground_truth():
    """A corpus pair that is both a clean denial AND has ground_truth=True is a
    contradiction: the mechanical verifier said the reply was correct, but the
    reply explicitly says "I don't have that". This is an empirically-safe-today
    case that a future corpus append might break. Assert loudly instead of
    silently misclassifying."""

    # Build a synthetic record where the reply is a clean denial (matches _DENIAL_RE)
    # but the mechanical verifier scored it as True (passed).
    reply_text = "I do not have access to this information"
    synthetic_record = {
        "task": "A1",
        "passed": True,  # ground_truth will be True
        "detail": f"compacted=True sentinel_recalled=True reply='{reply_text}'",
        "attempt": 0,
    }

    # This should raise an AssertionError with a clear message.
    with pytest.raises(AssertionError, match="clean denial.*ground_truth"):
        judge_validate.pairs_from_record(synthetic_record, "synthetic.jsonl")


# --- Transport: bounded retry and pacing (delivery, never verdicts) --------------
#
# Measured 2026-08-21 on the pinned OpenRouter/Novita base: a live re-validation
# scored 0.899 agreement (571/635) and failed its gate. All 58 undelivered pairs
# were HTTP 429, spread from index 11 to 634 in 37 bursts (longest 6), at a
# sustained ~50 requests/minute. Delivered-only agreement was 0.9896 with zero
# false approvals — the judge was sound, the transport was not. These tests pin
# the two halves of the fix and, just as importantly, its boundary: retry may
# turn a NON-verdict into a verdict, and must never turn one verdict into another.

_HTTP_429 = (
    "Client error '429 Too Many Requests' for url 'https://openrouter.ai/api/v1/chat/completions'"
)


def _flaky(faults: list[Exception], reply: str = "VERDICT: YES\nQUOTE: a"):
    """A responder that raises ``faults`` in order, then answers ``reply``."""
    remaining = list(faults)

    def responder(messages):
        if remaining:
            raise remaining.pop(0)
        return reply

    return responder


def test_a_transient_fault_is_retried_and_can_still_deliver(slept):
    """The defect, inverted. One 429 used to be terminal for its pair: judged_equivalent
    caught the provider exception and returned ran=False, so the pair scored as an
    undelivered non-verdict. It now retries under a bounded backoff and delivers."""
    provider = fake(scripted=_flaky([RuntimeError(_HTTP_429)]))

    judgment = judged_equivalent("E", "a", provider)

    assert judgment.ran is True and judgment.verdict is True
    assert judgment.attempts == 2
    assert judgment.faults == ("http_429",)
    assert slept == [2.0]  # carbon's policy: base_delay_ms=2000, doubling


def test_a_persistent_fault_still_records_undelivered(slept):
    """The fail-closed contract does not change. Retries are BOUNDED: a fault that
    outlives the budget still records ran=False, so a throttled judge degrades the
    pair rather than silently becoming a verdict."""
    from runner.judge import JUDGE_RETRY

    def always_429(messages):
        raise RuntimeError(_HTTP_429)

    judgment = judged_equivalent("E", "a", fake(scripted=always_429))

    assert judgment.ran is False and judgment.verdict is False
    assert judgment.quote == ""
    assert "provider error" in judgment.raw and "429" in judgment.raw
    assert judgment.attempts == JUDGE_RETRY.max_attempts == 5
    assert judgment.faults == ("http_429",) * 5
    assert slept == [2.0, 4.0, 8.0, 16.0]  # bounded: four waits, then it gives up


def test_a_persistent_fault_still_counts_as_undelivered_in_the_artifact():
    """And the same thing seen from the scorer: an exhausted retry budget is still a
    pair with no verdict — not agreement, in the disagreement list, and counted in
    ``undelivered_count``. Scoring is untouched by the retry."""

    def always_429(messages):
        raise RuntimeError(_HTTP_429)

    corpus = [_pair(answer="a", ground_truth=False)]  # False == the fail-closed verdict
    result = judge_validate.run_validation(corpus, fake(scripted=always_429))

    assert result["records"][0]["ran"] is False
    assert result["agree_count"] == 0  # a non-verdict never agrees, even by accident
    assert result["undelivered_count"] == 1
    assert result["disagree_count"] == 1
    assert result["pass"] is False


def test_a_non_transient_fault_is_not_retried(slept):
    """The budget is for serving faults, not for defects. A 401 will fail identically
    five times; spending the budget on it delays the run and hides the real cause."""

    def unauthorized(messages):
        raise RuntimeError("Client error '401 Unauthorized' for url 'https://x/y'")

    judgment = judged_equivalent("E", "a", fake(scripted=unauthorized))

    assert judgment.ran is False
    assert judgment.attempts == 1
    assert judgment.faults == ("RuntimeError",)
    assert slept == []


def test_an_unparseable_reply_is_never_retried(slept):
    """The boundary that keeps this fix a DELIVERY fix. A reply that arrived and did
    not parse is a fact about the judge, not about the transport — re-rolling it would
    hand a free-forming judge extra chances and change what the artifact measures.
    Only a provider EXCEPTION (nothing arrived) is retried."""
    calls = []

    def freeforms(messages):
        calls.append(1)
        return "I think so, probably."

    judgment = judged_equivalent("E", "a", fake(scripted=freeforms))

    assert judgment.ran is False and judgment.verdict is False
    assert judgment.attempts == 1 and len(calls) == 1
    assert judgment.faults == ()
    assert slept == []


def test_the_servers_own_retry_after_beats_the_computed_backoff(slept):
    """When the 429 carries a Retry-After, the server has told us how long it wants;
    guessing over that is how a client keeps hammering a window it was asked to skip.

    The three DECISIONS about awkward values — an HTTP-date, a value longer than we
    will wait, a canonically-cased header name — have their own tests below. This one
    keeps the two plain cases: an ordinary value is honored exactly, and a response
    with no such header falls back to the computed backoff."""

    class _Throttled(RuntimeError):
        def __init__(self, headers):
            super().__init__(_HTTP_429)
            self.response = SimpleNamespace(headers=headers)

    honored = judged_equivalent("E", "a", fake(scripted=_flaky([_Throttled({"retry-after": "7"})])))
    assert honored.ran is True
    assert slept == [7.0]  # not the computed 2.0

    slept.clear()
    # A response that carries headers but not this one — the shape the 2026-08-21
    # 429s may well have had, since str(exc) captured no headers and the evidence
    # cannot say. The computed backoff stands.
    quiet = judged_equivalent("E", "a", fake(scripted=_flaky([_Throttled({})])))
    assert quiet.ran is True
    assert slept == [2.0]
    assert slept == [2.0]


def test_the_live_validation_run_paces_its_calls(slept, tmp_path):
    """The other half. 635 calls issued back to back at ~50/minute is what provoked
    the throttling; the retry recovers from it, the pace stops asking for it. Paced
    BETWEEN calls only — nothing before the first, nothing after the last."""
    from loop.judge_validate import JUDGE_PACE_SECONDS

    corpus = [_pair(answer="a"), _pair(answer="a"), _pair(answer="a")]
    provider = fake(scripted=lambda m: "VERDICT: YES\nQUOTE: a")

    judge_validate.run_validation(corpus, provider, pace_seconds=0.25)
    assert slept == [0.25, 0.25]

    # Off by default, so scripted callers (this suite) pay nothing …
    slept.clear()
    judge_validate.run_validation(corpus, provider)
    assert slept == []

    # … and on for the live entry point, which is the one that hits a real endpoint.
    slept.clear()
    result = judge_validate.main(provider=provider, output_path=tmp_path / "a.json")
    assert JUDGE_PACE_SECONDS > 0
    assert slept == [JUDGE_PACE_SECONDS] * (result["total_pairs"] - 1)
    assert result["delivery"]["pace_seconds"] == JUDGE_PACE_SECONDS


def test_the_artifact_records_what_the_transport_did():
    """Delivery diagnostics, so the next run's report can state plainly what happened
    instead of a human re-deriving it from 635 raw strings (which is how tonight's
    cause was found). Counts of retries, of pairs that recovered, and faults by class.
    """

    def one_429_then_fine(messages):
        text = messages[1]["content"]
        if "flaky" in text and not getattr(one_429_then_fine, "done", False):
            one_429_then_fine.done = True
            raise RuntimeError(_HTTP_429)
        if "dead" in text:
            raise RuntimeError(_HTTP_429)
        return "VERDICT: YES\nQUOTE: a"

    corpus = [
        _pair(answer="a-fine"),
        _pair(answer="a-flaky"),
        _pair(answer="a-dead"),
    ]
    result = judge_validate.run_validation(corpus, fake(scripted=one_429_then_fine))
    delivery = result["delivery"]

    assert delivery["pairs_retried"] == 2  # the flaky one and the dead one
    assert delivery["pairs_recovered_by_retry"] == 1  # only the flaky one came back
    assert delivery["retries_attempted"] == 1 + 4  # one, then a whole exhausted budget
    # Faults counts every try that was LOST, which is one more than the retries the
    # exhausted pair bought: 1 (flaky) + 5 (dead, every try in its budget).
    assert delivery["faults_by_class"] == {"http_429": 6}
    assert delivery["faults_total"] == 6
    assert delivery["pace_seconds"] == 0.0
    # And per record, beside the raw the human reads.
    assert [r["attempts"] for r in result["records"]] == [1, 2, 5]
    assert result["records"][2]["faults"] == ["http_429"] * 5


def test_the_judges_transient_rule_agrees_with_carbons():
    """Reuse of semantics, pinned. This is the same defect class carbon fixed in its
    compaction summarizer, and the classifier is deliberately a COPY rather than an
    import: the grader must not take its own transport correctness from the harness it
    grades (an accepted candidate, or a new pinned base, would silently retune the
    judge). A copy needs a drift alarm, which is this test.

    The probe list carries carbon's own regression: "requested 15020 tokens" contains
    "502" as a substring and must NOT be transient — matching status codes with `in`
    spent a whole retry budget on a payload that could never fit.
    """
    from harness.agent import Agent

    from runner.judge import transient_fault

    probes = [
        _HTTP_429,
        "Server error '502 Bad Gateway' for url 'https://x/y'",
        "Server error '503 Service Unavailable'",
        "Server error '504 Gateway Timeout'",
        "API rate limit exceeded",
        "Read timed out",
        "the model is temporarily unavailable",
        "connection reset by peer",
        "connection refused",
        "Client error '401 Unauthorized' for url 'https://x/y'",
        "Client error '400 Bad Request' for url 'https://x/y'",
        "this model's maximum context length is 8192 tokens; you requested 15020 tokens",
        "no such file or directory",
    ]
    mine = [transient_fault(RuntimeError(p)) for p in probes]
    theirs = [Agent._transient_error(RuntimeError(p)) for p in probes]
    assert mine == theirs
    assert mine[-2] is False  # the token-count substring trap, still shut


# --- The retry is scoped to the CALL, not to everything around it ----------------
#
# Review of the first transport fix (Codex, 2026-08-22) found the boundary it
# documented — "a retry converts a non-verdict into a verdict, never one verdict
# into another" — did not actually hold. The guarded region was the whole original
# try block: call, parse, AND usage extraction. So an exception raised AFTER a reply
# had already arrived could be classified transient and re-roll a delivered verdict
# (HIGH), or, when it was not transient, discard a delivered verdict as undelivered
# (MEDIUM). Both were reproduced. They are one defect: the retry decision was scoped
# to everything that can raise instead of to the CALL.
#
# The rule these pin: once a reply has been delivered, nothing that happens
# afterwards may re-roll it or mark it undelivered. Failing to read telemetry costs
# the telemetry number, not the verdict.


class _DeliveredThenRaises(LLMResponse):
    """A reply that ARRIVED, whose usage accessor then raises.

    The live transport cannot produce this today — httpx raises status errors before
    ``LLMResponse`` is ever constructed (carbon model/openai_compatible.py) — which
    is why the defect never fired in the field. "Cannot fire on today's transport"
    is not the same guarantee as "cannot fire", and CMP-5/CMP-6 call this same
    function live against whatever provider they are handed.
    """

    def __init__(self, content: str, exc: Exception):
        super().__init__(content=content, finish_reason="stop")
        self._exc = exc

    @property
    def usage(self):
        raise self._exc

    @usage.setter
    def usage(self, value):  # LLMResponse.__init__ assigns the field
        pass


def _responses(*items):
    """A provider that returns ``items`` in order, counting its calls."""
    calls = []

    def responder(messages, **_kw):
        calls.append(1)
        return items[min(len(calls) - 1, len(items) - 1)]

    provider = Provider(base_url="fake://local", model="probe", api_key="x", responder=responder)
    return provider, calls


def test_a_delivered_verdict_is_never_re_rolled_by_a_later_failure(slept):
    """HIGH. Reproduced against the previous fix: the first call returns a grounded
    YES, its usage accessor then raises RuntimeError("429 while reading usage"), the
    transient matcher sees 429, retries — and the SECOND call's NO becomes the
    answer. calls=2, attempts=2, verdict=False. A verdict was replaced by another
    verdict, which is exactly what the retry must never do."""
    provider, calls = _responses(
        _DeliveredThenRaises("VERDICT: YES\nQUOTE: a", RuntimeError("429 while reading usage")),
        LLMResponse(content="VERDICT: NO\nQUOTE: a", finish_reason="stop"),
    )

    judgment = judged_equivalent("E", "a", provider)

    assert len(calls) == 1, "a delivered reply must not be asked for again"
    assert judgment.attempts == 1
    assert judgment.ran is True and judgment.verdict is True  # the YES that arrived
    assert judgment.faults == ()
    assert slept == []


def test_unreadable_usage_costs_the_number_not_the_verdict():
    """MEDIUM. Same defect, non-transient half: a valid delivered YES was discarded
    as undelivered because reading its telemetry raised. Live that is a phantom
    'judge unavailable' — CMP-6 errors, CMP-5 can error when no other delivered
    verdict decides — and in the artifact it lowers delivered_count and turns the
    pair into a disagreement despite a verdict having arrived.

    The verdict now stands and ``tokens`` falls back to 0, which is the field's
    existing meaning: unmeasured, never an estimate."""
    provider, calls = _responses(
        _DeliveredThenRaises("VERDICT: YES\nQUOTE: a", ValueError("telemetry decode failed"))
    )

    judgment = judged_equivalent("E", "a", provider)

    assert len(calls) == 1
    assert judgment.ran is True and judgment.verdict is True
    assert judgment.tokens == 0
    assert "usage unreadable" in judgment.raw  # the loss is recorded, not hidden
    assert "VERDICT: YES" in judgment.raw  # beneath the judge's own output, intact


def test_an_unreadable_reply_fails_closed_without_a_re_roll(slept):
    """The third face of the same class. A reply that arrived but cannot be READ at
    all is not a transport fault — the reply is already in hand, and asking again
    would put a different question to the judge. It fails closed (ran=False, no
    verdict) exactly like an unparseable one, and it is NOT retried."""
    # `content or ""` already absorbs None, so the unreadable case is a content the
    # parser cannot read at all.
    broken = LLMResponse(content="VERDICT: YES\nQUOTE: a", finish_reason="stop")
    broken.content = object()
    provider, calls = _responses(broken, LLMResponse(content="VERDICT: NO\nQUOTE: a"))

    judgment = judged_equivalent("E", "a", provider)

    assert len(calls) == 1
    assert judgment.attempts == 1
    assert judgment.ran is False and judgment.verdict is False
    assert slept == []


def test_a_transport_fault_is_still_retried_after_the_scoping(slept):
    """And the fix must not have thrown the baby out: a fault from the CALL — where
    no reply ever arrived — is still retried and can still deliver."""
    calls = []

    def responder(messages, **_kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError(_HTTP_429)
        return LLMResponse(content="VERDICT: YES\nQUOTE: a", finish_reason="stop")

    provider = Provider(base_url="fake://local", model="probe", api_key="x", responder=responder)
    judgment = judged_equivalent("E", "a", provider)

    assert judgment.ran is True and judgment.verdict is True
    assert judgment.attempts == 2 and judgment.faults == ("http_429",)
    assert slept == [2.0]


# --- Retry-After: three decisions, made rather than inherited -------------------


def _throttled(headers: dict) -> RuntimeError:
    exc = RuntimeError(_HTTP_429)
    exc.response = SimpleNamespace(headers=headers)
    return exc


def test_retry_after_is_read_case_insensitively():
    """HTTP header names are case-insensitive and servers send ``Retry-After``
    canonically. httpx's own Headers mapping hides that; a plain dict does not, and
    this code accepts any mapping. Missing it silently discards the one authoritative
    number in the whole policy."""
    from runner.judge import _retry_after_seconds

    assert _retry_after_seconds(_throttled({"Retry-After": "7"})) == 7.0
    assert _retry_after_seconds(_throttled({"retry-after": "7"})) == 7.0
    assert _retry_after_seconds(_throttled({"RETRY-AFTER": "7"})) == 7.0
    assert _retry_after_seconds(_throttled({"x-other": "7"})) is None


def test_retry_after_http_date_is_honored():
    """DECIDED, not inherited. RFC 7231 allows an HTTP-date and the previous fix
    silently ignored it — a server instruction dropped on the floor with no trace.
    It is parsed and converted to a wait. Clock skew was the argument for ignoring
    it; the clamp already bounds skew to [0, max_delay_s], so the argument does not
    survive its own safeguard."""
    from email.utils import format_datetime

    from runner.judge import _retry_after_seconds

    soon = datetime.now(UTC) + timedelta(seconds=8)
    seconds = _retry_after_seconds(_throttled({"Retry-After": format_datetime(soon)}))
    assert seconds is not None and 6.0 <= seconds <= 9.0

    # A date already past means "you may retry now", not "wait forever".
    past = datetime.now(UTC) - timedelta(seconds=120)
    assert _retry_after_seconds(_throttled({"Retry-After": format_datetime(past)})) == 0.0

    # Unparseable stays a fallback to the computed backoff, explicitly.
    assert _retry_after_seconds(_throttled({"Retry-After": "next tuesday"})) is None


def test_a_retry_after_we_will_not_wait_for_stops_the_retry(slept):
    """DECIDED. The previous fix clamped a large Retry-After down to max_delay_s and
    retried anyway — i.e. it asked again BEFORE the server said it could, silently
    under-honoring the instruction and near-certainly burning the try. Refusing is
    the honest reading: the pair records undelivered, which the gate can see, rather
    than a retry we know is early."""
    from runner.judge import JUDGE_RETRY

    too_long = str(int(JUDGE_RETRY.max_delay_s) + 60)
    calls = []

    def responder(messages, **_kw):
        calls.append(1)
        raise _throttled({"Retry-After": too_long})

    provider = Provider(base_url="fake://local", model="probe", api_key="x", responder=responder)
    judgment = judged_equivalent("E", "a", provider)

    assert len(calls) == 1, "no early retry against an explicit instruction"
    assert judgment.ran is False and judgment.attempts == 1
    assert judgment.faults == ("http_429",)
    assert slept == []
    assert "Retry-After" in judgment.raw  # the reason is recorded, not silent

    # And a value we WILL wait for is honored exactly, not clamped away.
    slept.clear()
    at_the_limit = str(int(JUDGE_RETRY.max_delay_s))
    calls2 = []

    def responder2(messages, **_kw):
        calls2.append(1)
        if len(calls2) == 1:
            raise _throttled({"Retry-After": at_the_limit})
        return LLMResponse(content="VERDICT: YES\nQUOTE: a", finish_reason="stop")

    provider2 = Provider(base_url="fake://local", model="probe", api_key="x", responder=responder2)
    assert judged_equivalent("E", "a", provider2).ran is True
    assert slept == [JUDGE_RETRY.max_delay_s]


# --- The retry decision may never end a run -------------------------------------
#
# Review of 08d80cd (Codex, 2026-08-22). Honoring the HTTP-date form of Retry-After
# was the right call and it opened a new fail-closed escape: parsedate_to_datetime
# raises OverflowError on an extreme year, which the guard did not name. Reproduced:
# the raise escapes _retry_after_seconds, escapes judged_equivalent — past the whole
# fail-closed guarantee — and aborts run_validation outright, because
# judged_equivalent IS that loop's per-pair isolation and there is no second net.
# In task execution it becomes a generic empty-metrics error record, losing the very
# judge_attempts and refusal reason the diagnostics plumbing was added to preserve.

# The real header value the escape was found with. Kept as a probe, not a synthetic
# exception: the point is that a STRING a provider can send reaches a parser that
# raises outside the guarded set, and only the string proves that.
_EXTREME_DATE = "Fri, 31 Dec 999999999999 23:59:59 GMT"


def test_an_extreme_retry_after_date_cannot_escape_the_parse_guard():
    """The parser is handed attacker-adjacent input — a header chosen by whatever
    answered. Its failure modes are not limited to the two the guard first named."""
    from email.utils import parsedate_to_datetime

    from runner.judge import _retry_after_seconds

    # The probe is real: this is what the stdlib actually does with that string.
    with pytest.raises(OverflowError):
        parsedate_to_datetime(_EXTREME_DATE)

    # And it must not reach the caller. Unparseable means fall back, as documented.
    assert _retry_after_seconds(_throttled({"Retry-After": _EXTREME_DATE})) is None


def test_a_malformed_header_degrades_one_pair_instead_of_ending_the_run(slept):
    """The blast radius, pinned where it was measured: a 635-pair validation loop.

    One provider fault carrying a malformed header used to take the whole run with
    it. The pair now records undelivered — which is what every other transport
    failure already does — and the loop finishes."""

    def throttled_with_a_bad_date(messages):
        raise _throttled({"Retry-After": _EXTREME_DATE})

    provider = fake(scripted=throttled_with_a_bad_date)

    judgment = judged_equivalent("E", "a", provider)
    assert judgment.ran is False and judgment.verdict is False
    assert judgment.faults == ("http_429",) * 5  # fell back to the computed backoff
    assert slept == [2.0, 4.0, 8.0, 16.0]

    corpus = [_pair(answer="a"), _pair(answer="a"), _pair(answer="a")]
    result = judge_validate.run_validation(corpus, provider)
    assert result["total_pairs"] == 3
    assert result["undelivered_count"] == 3
    assert result["pass"] is False


def test_the_retry_decision_itself_cannot_end_a_run(monkeypatch, slept):
    """The CLASS, not just the one hole. The retry decision reads whatever a provider
    put on the wire, and a raise anywhere in it escapes ``judged_equivalent``
    entirely. Closing the OverflowError alone leaves the promise resting on the
    policy code being exhaustively right about every exception a malformed header can
    produce — which is the assumption that just failed. A fault in DECIDING a retry
    degrades the pair, with the reason recorded, like any other."""
    import runner.judge as judge_mod

    def exploding_policy(attempt, exc, policy):
        raise KeyError("a shape the policy did not expect")

    monkeypatch.setattr(judge_mod, "_next_delay", exploding_policy)

    def failing(messages):
        raise RuntimeError(_HTTP_429)

    judgment = judged_equivalent("E", "a", fake(scripted=failing))
    assert judgment.ran is False and judgment.verdict is False
    assert judgment.attempts == 1
    assert judgment.faults == ("http_429",)
    assert "retry policy" in judgment.raw  # the reason is recorded, not swallowed
    assert slept == []


def test_an_exception_that_cannot_describe_itself_is_still_handled():
    """The same class from the other side: every branch of the fault handler reads
    ``str(exc)``, and an exception whose own ``__str__`` raises would take the run
    down before the retry decision was ever consulted."""

    class _Unspeakable(RuntimeError):
        def __str__(self):
            raise ValueError("this exception cannot describe itself")

    def failing(messages):
        raise _Unspeakable()

    judgment = judged_equivalent("E", "a", fake(scripted=failing))
    assert judgment.ran is False and judgment.verdict is False
    assert judgment.attempts == 1  # not transient — nothing readable said it was
    assert judgment.faults == ("_Unspeakable",)
    assert "_Unspeakable" in judgment.raw


def test_a_zero_token_count_says_whether_it_was_measured():
    """Visibility, not scoring. ``_usage_tokens`` catches a malformed scalar and
    returns 0 itself, so the annotation promised by the delivered-verdict path never
    fired for the case that most needs it: zero is also a LEGITIMATE value, and an
    unannotated 0 conflated "the judge cost nothing to measure" with "the measurement
    failed". The verdict and the 0 fallback were always right; the record was silent
    about which zero it was."""
    delivered = "VERDICT: YES\nQUOTE: a"

    def responding(usage):
        response = LLMResponse(content=delivered, finish_reason="stop")
        response.usage = usage
        provider, _calls = _responses(response)
        return judged_equivalent("E", "a", provider)

    # Reported but unreadable: annotated, so the zero is visibly unmeasured.
    # (`{"prompt_tokens": {}}` is deliberately NOT here: an empty dict is falsy, so
    # `or 0` absorbs it and 0 really is the right unannotated answer.)
    for bad in ({"total_tokens": "abc"}, {"total_tokens": [1]}, {"prompt_tokens": {"a": 1}}):
        judgment = responding(bad)
        assert judgment.verdict is True and judgment.tokens == 0, bad
        assert "usage unreadable" in judgment.raw, bad
        assert judgment.raw.startswith(delivered), bad  # beneath the judge's own output

    # A provider that reported nothing is the documented, expected zero — no note.
    for quiet in ({}, None):
        judgment = responding(quiet)
        assert judgment.verdict is True and judgment.tokens == 0
        assert "usage unreadable" not in judgment.raw

    # And a real number still arrives unannotated.
    measured = responding({"total_tokens": 312})
    assert measured.tokens == 312 and "usage unreadable" not in measured.raw
