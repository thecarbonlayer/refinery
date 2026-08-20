"""Offline tests for the meaning-equivalence judge and its validation harness
(contract §4). Every provider here is scripted — no live model call anywhere
in this file."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest
from model.fake import fake

from loop import judge_validate
from runner.judge import JUDGE_PROMPT, JUDGE_PROMPT_SHA, Judgment, judged_equivalent

# --- runner.judge ---------------------------------------------------------------


def test_judge_prompt_sha_is_pinned():
    """A prompt edit must be a LOUD, deliberate act (contract §4): this test
    goes red the instant JUDGE_PROMPT changes without JUDGE_PROMPT_SHA
    changing with it."""
    assert JUDGE_PROMPT_SHA == hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()


def test_yes_verdict_parses_quote():
    raw = "VERDICT: YES\nQUOTE: never exceed 30 seconds"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent(
        "retry backoff must never exceed 30 seconds", "some paraphrase", provider
    )
    assert judgment == Judgment(True, "never exceed 30 seconds", raw)


def test_no_verdict_parses_quote():
    raw = "VERDICT: NO\nQUOTE: the answer says something else entirely"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("expected fact", "wrong answer", provider)
    assert judgment == Judgment(False, "the answer says something else entirely", raw)


def test_garbage_output_fails_closed():
    raw = "I think the answer is probably related to the topic."
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("expected fact", "some answer", provider)
    assert judgment == Judgment(False, "", raw)


def test_empty_output_fails_closed():
    provider = fake(scripted=lambda messages: "")
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", "")


def test_missing_quote_line_fails_closed():
    raw = "VERDICT: YES"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", raw)


def test_wrong_case_fails_closed():
    raw = "verdict: yes\nquote: something"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", raw)


def test_second_line_not_quote_prefixed_fails_closed():
    raw = "VERDICT: YES\nnever exceed 30 seconds"
    provider = fake(scripted=lambda messages: raw)
    judgment = judged_equivalent("e", "a", provider)
    assert judgment == Judgment(False, "", raw)


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
            return "VERDICT: YES\nQUOTE: q"
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
            return "VERDICT: YES\nQUOTE: q"  # wrongly says YES on a denial
        return "VERDICT: YES\nQUOTE: q"

    provider = fake(scripted=responder)
    result = judge_validate.run_validation(corpus, provider)
    # Both pairs "agree" by construction (ground_truth=True/verdict=True for
    # the first pair) except the denial one, which disagrees AND trips the
    # clean-denial gate.
    assert result["clean_denial_yes_count"] == 1
    assert result["pass"] is False


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
    assert on_disk["model"] == provider.model
    assert on_disk["total_pairs"] == len(judge_validate.build_corpus())


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
        return "VERDICT: YES\nQUOTE: q"

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
