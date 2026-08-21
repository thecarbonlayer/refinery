"""Offline tests for the meaning-equivalence judge and its validation harness
(contract §4). Every provider here is scripted — no live model call anywhere
in this file."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from model.fake import fake

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
