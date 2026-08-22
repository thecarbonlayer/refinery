"""The judge's validation harness (contract §4) — offline corpus, judge, agreement.

Builds a labeled corpus from the eleven committed ``results/r2-null-*.jsonl``
files: every A1/G2/G4/G5 attempt whose ``detail`` string carries a reply,
paired with the task's expected fact(s) and the MECHANICAL ground truth
already recorded in that same detail string. Ground truth is never
re-derived from the reply text here — that would just be re-running the
mechanical verifier under a different name and could never disagree with
itself.

G2 (two codes) and G4 (three properties) are multi-sentinel: their mechanical
``passed`` is an AND of several booleans, so a corpus built from ``passed``
alone would blur a real per-fact disagreement into a whole-attempt one. Both
split into one corpus pair per sentinel/property, using the fact-specific
boolean the task already recorded in its detail string
(``early_recalled``/``late_recalled``, ``files_recalled``/
``rejected_recalled``/``next_action_recalled``). A1 and G5 are single-fact
tasks for this purpose — G5's mechanical check already requires BOTH file
names to appear before it can pass at all — so they use the recorded
top-level ``passed`` as-is, one pair per attempt.

``main()`` runs the judge over that corpus and writes
``iterations/judge-validation/agreement.json`` — the artifact CMP-6 gates on
(contract §4): overall agreement >= 95%, AND zero cases of judge=YES on a
clean denial (an explicit "I don't have it" reply the mechanical check
already scored False). Contract §4 names this set as "the 15 G2 + 10 A1
explicit 'I don't have it' replies" — counted as REPLIES. Because G2 splits
into 2 corpus pairs per reply, those 15 G2 replies surface here as 30
clean-denial PAIRS (both judged NO independently, once per fact); A1 is
single-pair, so its 10 replies are 10 pairs. The provider is injectable;
tests use a scripted one, mostly over a tiny synthetic corpus, and never make
a live model call.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from runner.judge import (
    AGREEMENT_PATH,
    JUDGE_PARSER_VERSION,
    JUDGE_PROMPT_SHA,
    JUDGE_RETRY,
    VALIDATION_COMPUTATION_VERSION,
    judged_equivalent,
)
from runner.suite import RESULTS_DIR
from runner.tasks.cluster_a import A1_SENTINEL
from runner.tasks.cluster_g import G2_FACT_A, G2_FACT_B, G4_FILES, G4_NEXT, G4_REJECTED, G5_FILES

# The artifact's path is defined in ``runner.judge`` beside the gate that READS
# it (CMP-6's activation check): the reader may not import this package, so a
# second definition here could drift into naming a different file — a judge
# gated on an artifact nobody writes, or one written where nothing looks.
__all__ = ["AGREEMENT_PATH", "build_corpus", "main", "pairs_from_record", "run_validation"]

# The eleven committed files the corpus is built from (contract §4). Sorted
# for a deterministic corpus order run to run.
R2_NULL_FILES = sorted(RESULTS_DIR.glob("r2-null-*.jsonl"))

AGREEMENT_THRESHOLD = 0.95

# --- pacing: how fast this run is allowed to ask -------------------------------
#
# ``run_validation`` issues one judge call per corpus pair, back to back, and the
# corpus is 635 pairs. Measured 2026-08-21 on the pinned OpenRouter/Novita base: that
# sequence ran 23:23:07 to 23:35:53 — 766s for 635 calls, 1.21s per call, ~49.7
# requests per minute sustained — and the endpoint refused 58 of them with HTTP 429,
# a 9.1% loss, in 37 bursts spread from pair 11 to pair 634. Bursty and spread, not
# an outage: the endpoint was throttling a rate it was willing to serve more slowly.
#
# The published ceiling for that route is not something this repo knows, so the pace
# below is not tuned to it and does not pretend to be. At 1.21s of latency per call,
# a half-second wait between calls takes the offered rate from ~49.7/min to
# ~35.1/min — a 29% cut in what provokes the limiter — for about five extra minutes
# on a run that took thirteen. That trade is worth making blind; a larger one is not.
# The GUARANTEE of delivery is the retry policy beside it (``runner.judge``), which
# recovers whatever pacing fails to prevent and reports it. Pacing lowers the fault
# rate; retry is what makes the remaining faults stop costing pairs.
JUDGE_PACE_SECONDS = 0.5

# Injectable, like the retry's: an offline suite never waits (``tests/conftest.py``).
_sleep = time.sleep

_TASKS_WITH_REPLIES = {"A1", "G2", "G4", "G5"}

# Explicit-denial detector, tuned against the real corpus: phrase-level, tight
# enough that a wrong-but-attempted answer, a tool-call leak, or a truncated
# generation never matches, only an actual "I don't have that" refusal does.
# Verified against all eleven committed files: matches exactly the 15 G2 + 10
# A1 replies contract §4 names as the clean-denial set (30 + 10 corpus pairs;
# see module docstring) and nothing in G4 or G5's failing replies, which deny
# in a structurally different way ("None recorded" rather than "I don't have").
_DENIAL_RE = re.compile(
    r"(do not|don't|does not|doesn't)\s+(have|see|contain|find)"
    r"|cannot find|can not find"
    r"|(not been|were not|was not|have not been|has not been)\s+(provided|mentioned)"
    r"|\bno\s[a-z ]{0,40}were provided\b",
    re.IGNORECASE,
)


def _is_clean_denial(reply: str) -> bool:
    return bool(_DENIAL_RE.search(reply))


def _extract_reply(detail: str) -> str | None:
    """Pull the trailing ``reply=<repr>`` field every A1/G2/G4/G5 detail
    string ends with, when the attempt got far enough to produce one. A setup
    failure's detail (e.g. "repeated-compaction setup did not fire twice")
    carries no ``reply=`` at all — those attempts are skipped, not judged,
    because there is nothing to judge."""
    match = re.search(r"reply=(.*)$", detail, re.DOTALL)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(1))
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def _extract_bool(detail: str, field_name: str) -> bool | None:
    match = re.search(rf"\b{re.escape(field_name)}=(True|False)", detail)
    return match.group(1) == "True" if match else None


@dataclass(frozen=True)
class CorpusPair:
    task: str
    fact: str  # which sentinel/property within the task, e.g. "early", "files"
    expected: str
    answer: str
    ground_truth: bool
    is_clean_denial: bool
    source_file: str
    attempt: int


def pairs_from_record(record: dict, source_file: str) -> list[CorpusPair]:
    """Zero or more corpus pairs from one JSONL record: zero for a task this
    corpus doesn't cover, or for an attempt whose detail carries no reply."""
    task = record.get("task")
    if task not in _TASKS_WITH_REPLIES:
        return []
    detail = record.get("detail", "")
    reply = _extract_reply(detail)
    if reply is None:
        return []
    attempt = record.get("attempt", -1)
    denial = _is_clean_denial(reply)

    def pair(fact: str, expected: str, ground_truth: bool) -> CorpusPair:
        # Empirically safe today: a clean denial (explicit "I don't have it") paired
        # with ground_truth=True would mean the mechanical verifier scored a refusal as
        # correct. This contradicts the task design and would be a corpus error. Fail
        # loudly instead of silently misclassifying — a future corpus append might hit
        # this case and we want immediate visibility.
        assert not (denial and ground_truth), (
            f"Contradiction in corpus pair: clean denial ({reply!r}) "
            f"paired with ground_truth=True. Mechanical verifier and denial detector "
            f"disagree on this record (task={task}, fact={fact}, attempt={attempt}). "
            f"This is a corpus error — see {source_file}."
        )
        return CorpusPair(task, fact, expected, reply, ground_truth, denial, source_file, attempt)

    if task == "A1":
        return [pair("sentinel", A1_SENTINEL, bool(record.get("passed")))]

    if task == "G2":
        pairs = []
        has_a = _extract_bool(detail, "early_recalled")
        if has_a is not None:
            pairs.append(pair("early", G2_FACT_A, has_a))
        has_b = _extract_bool(detail, "late_recalled")
        if has_b is not None:
            pairs.append(pair("late", G2_FACT_B, has_b))
        return pairs

    if task == "G4":
        pairs = []
        files_expected = f"The files changed so far are {G4_FILES[0]} and {G4_FILES[1]}."
        has_files = _extract_bool(detail, "files_recalled")
        if has_files is not None:
            pairs.append(pair("files", files_expected, has_files))
        has_rejected = _extract_bool(detail, "rejected_recalled")
        if has_rejected is not None:
            pairs.append(pair("rejected", G4_REJECTED, has_rejected))
        has_next = _extract_bool(detail, "next_action_recalled")
        if has_next is not None:
            pairs.append(pair("next", G4_NEXT, has_next))
        return pairs

    # task == "G5"
    files_expected = (
        f"The files created or modified in this session are {G5_FILES[0]} and {G5_FILES[1]}."
    )
    return [pair("files", files_expected, bool(record.get("passed")))]


def build_corpus(files: list[Path] | None = None) -> list[CorpusPair]:
    """The labeled corpus, built from ``files`` (default: the eleven committed
    r2-null files)."""
    files = R2_NULL_FILES if files is None else files
    corpus: list[CorpusPair] = []
    for path in files:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                corpus.extend(pairs_from_record(record, path.name))
    return corpus


def run_validation(corpus: list[CorpusPair], provider, *, pace_seconds: float = 0.0) -> dict:
    """Run the judge over ``corpus`` and assemble the agreement artifact
    (contract §4 shape): overall agreement, the zero-YES-on-clean-denial
    check, per-pair records, a disagreement list with quotes, prompt sha,
    model, counts, and ``pass``.

    Each pair's judge call is guarded independently: if the judge call fails
    (provider exception), that pair is recorded as a disagreement with the
    error in raw, instead of aborting the whole validation run. A TRANSIENT
    fault is retried inside ``judged_equivalent`` first — the guard here is
    what happens once that bound is spent.

    ``pace_seconds`` waits between calls (never before the first, never after the
    last). It defaults to 0: a scripted provider has no rate limit and this suite
    must not sit through hundreds of real waits. ``main`` — the entry point that
    reaches a real endpoint — passes ``JUDGE_PACE_SECONDS``, whose measured
    justification is written out beside it.

    Neither pacing nor retry touches the scoring below. Both change how many
    judgments get DELIVERED; the arithmetic that turns delivered judgments into
    agreement, the clean-denial gate, and ``pass`` is byte-for-byte the rule
    ``VALIDATION_COMPUTATION_VERSION`` 2 already names, so the stamp does not move.
    What the run records about its own transport lands in ``delivery`` and per
    record, as additive metadata no gate reads.

    An UNDELIVERED verdict is never agreement. An undelivered judgment
    (``Judgment.ran`` False — provider failure, unparseable output) fails
    closed to ``verdict=False``, and that False happens to equal every
    negative pair's ground truth — so comparing verdicts alone would let a
    judge that times out on negatives score perfect agreement without ever
    delivering a NO (306 of the 635 real pairs are negative). A pair with no
    verdict counts as NOT agreed, joins the disagreement list, and the
    artifact records how many verdicts were actually delivered, so a mostly
    dead judge fails the 95% gate by construction.
    """
    records = []
    agree_count = 0
    disagreements = []
    clean_denial_yes = []
    faults_by_class: dict[str, int] = {}
    retries_attempted = 0
    pairs_retried = 0
    pairs_recovered_by_retry = 0
    for index, p in enumerate(corpus):
        if index and pace_seconds > 0:
            _sleep(pace_seconds)
        judgment = judged_equivalent(p.expected, p.answer, provider)
        retries_attempted += judgment.attempts - 1
        pairs_retried += int(judgment.attempts > 1)
        pairs_recovered_by_retry += int(judgment.attempts > 1 and judgment.ran)
        for fault in judgment.faults:
            faults_by_class[fault] = faults_by_class.get(fault, 0) + 1
        agrees = judgment.ran and judgment.verdict == p.ground_truth
        agree_count += int(agrees)
        record = {
            "task": p.task,
            "fact": p.fact,
            "source_file": p.source_file,
            "attempt": p.attempt,
            "expected": p.expected,
            "ground_truth": p.ground_truth,
            "judge_verdict": judgment.verdict,
            "ran": judgment.ran,
            "quote": judgment.quote,
            "raw": judgment.raw,
            "agree": agrees,
            "is_clean_denial": p.is_clean_denial,
            # Delivery, not verdict: how many tries this pair took and what cost
            # them. 1 and [] on a first-try call, which is nearly every pair.
            "attempts": judgment.attempts,
            "faults": list(judgment.faults),
        }
        records.append(record)
        if not agrees:
            disagreements.append(record)
        if p.is_clean_denial and judgment.verdict:
            clean_denial_yes.append(record)

    total = len(corpus)
    overall_agreement = (agree_count / total) if total else 0.0
    clean_denial_total = sum(1 for p in corpus if p.is_clean_denial)
    undelivered = sum(1 for record in records if not record["ran"])
    passes = total > 0 and overall_agreement >= AGREEMENT_THRESHOLD and not clean_denial_yes

    return {
        "pass": passes,
        "delivered_count": total - undelivered,
        "undelivered_count": undelivered,
        "judge_prompt_sha": JUDGE_PROMPT_SHA,
        # The parser half of the artifact's identity: the activation gate refuses
        # an artifact measured under any other parser version, exactly as it
        # refuses one measured for any other prompt.
        "judge_parser_version": JUDGE_PARSER_VERSION,
        # And the scoring half: the version of THIS function's agreement/pass
        # computation. The gate refuses an artifact scored under any other — a
        # pre-pin artifact (no stamp) could carry a pass:true the delivered-
        # verdict rule would refuse.
        "validation_computation_version": VALIDATION_COMPUTATION_VERSION,
        "model": getattr(provider, "model", None),
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "overall_agreement": overall_agreement,
        "total_pairs": total,
        "agree_count": agree_count,
        "disagree_count": len(disagreements),
        "clean_denial_total": clean_denial_total,
        "clean_denial_yes_count": len(clean_denial_yes),
        "clean_denial_yes": clean_denial_yes,
        # What the TRANSPORT did, so the run's report can say it plainly instead of a
        # human re-deriving it from hundreds of raw strings — which is how the
        # 2026-08-21 throttling was found, hours after the gate failed. Read by
        # people, by no gate: none of these numbers can move ``pass``.
        "delivery": {
            "pace_seconds": pace_seconds,
            "retry_max_attempts": JUDGE_RETRY.max_attempts,
            "retry_base_delay_ms": JUDGE_RETRY.base_delay_ms,
            "retries_attempted": retries_attempted,
            "pairs_retried": pairs_retried,
            "pairs_recovered_by_retry": pairs_recovered_by_retry,
            "faults_total": sum(faults_by_class.values()),
            "faults_by_class": dict(sorted(faults_by_class.items())),
        },
        "disagreements": disagreements,
        "records": records,
    }


def main(
    provider=None,
    output_path: Path = AGREEMENT_PATH,
    *,
    pace_seconds: float = JUDGE_PACE_SECONDS,
) -> dict:
    """Build the corpus from the eleven committed files, run the judge (the
    real provider by default), and write the agreement artifact.

    This is the LIVE entry point, so it is the one that paces: ``run_validation``
    defaults to no pacing because its other callers are scripted, and a rate limit
    is a property of a real endpoint.
    """
    if provider is None:
        from runner.carbon_env import make_provider

        provider = make_provider()
    corpus = build_corpus()
    result = run_validation(corpus, provider, pace_seconds=pace_seconds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
