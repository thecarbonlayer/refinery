"""Cluster S — tool exposure (the SEL section, decision 14's decoy-task suite).

Carbon's tool-exposure knob (``tool_exposure``: all | allowlist | query_match,
strategy-surface seam 3) landed with no task able to see it. The capability
map's prerequisite is explicit: land SEL-2 — one relevant tool among 30
plausible decoys — before the knob may be trusted with candidates. This module
is that suite: four distinct real-use axes, one task per axis, judged together
(decision 14: any strategy value is judged across ALL of them, never one).

Naming note. The map snapshot names SEL-2 as the decoy task; SEL-1 is not
assigned in any document this repo can cite (the existing D1-D3 already cover
the bare "reach for the right tool" axis and are this knob's cross-section
guards in ``loop/knob_coverage.py``). The suite therefore starts at SEL-2 and
counts up; the names ARE the map mnemonics, so ``alias=None`` (the CMP-5/6/7
convention).

The axes, each red-capable, with what each is designed to catch:

- **SEL-2 (held-in, THE MINER) — tool-count overload.** One relevant tool
  buried among 30 plausible decoys, the map's own shape. Designed to catch:
  the baseline agent mis-picking or stalling once the registry hits the low
  tens (the failure the roadmap predicts for ``all``), and any exposure value
  that buries or hides the needed tool. Red at baseline is the mining signal;
  whether it IS red is deliberately unmeasured here (prior ``uncertain`` —
  the D tasks pass 1.0 with tiny registries, and nothing has measured 31).
- **SEL-3 (held-out) — near-duplicate decoy that must NOT decide the answer.**
  ``invoice_subtotal`` (registered first, lexically stronger for the ask)
  against ``invoice_total`` (the only tool with the right value). Designed to
  catch: a model that answers from the near-twin's confident wrong value, and
  a rank-based exposure value that offers ONLY the twin — under
  ``query_match k=1`` the decoy is the single offered tool (proven offline
  through carbon's own selector in ``tests/test_sel_tasks.py``), so the right
  answer is unreachable and the guard goes red.
- **SEL-4 (held-in) — exposure-order sensitivity.** The needed tool is
  registered LAST behind 30 decoys whose descriptions outscore it for the ask,
  so ``all`` offers it last and ``query_match`` ranks it last at any k (and
  drops it entirely for k <= 30, a legal value — proven offline). Designed to
  catch: position-in-list cost at baseline, and a ranking value that
  re-orders the needed tool out of reach. Distinct from SEL-2: same count,
  but here the decoys are engineered to outrank, so failure is attributable
  to ORDER, not breadth.
- **SEL-5 (held-out) — vocabulary mismatch must not select the tool away.**
  A SMALL registry (7 tools) where the ask shares zero tokens with the one
  tool that can answer and at least one token with every filler. The roadmap's
  guard clause verbatim: the needed tool must never be selected away. Under
  ``all`` the model should find it by meaning; under ``query_match`` with any
  k <= 6 it is not offered at all (proven offline). Designed to catch: a
  lexical exposure value that wins SEL-2/SEL-4 by aggressive filtering and
  pays for it here.

Verifier discipline: mechanical only (decision 12 — exact sentinel codes, dual
reply+tool-result conjuncts on the D1 pattern, decoy-value exclusion). No
judge; every oracle fact is an exact string. Every reply-shaped verdict is a
pure ``sel*_verdict`` function carrying G2's non-answer taxonomy (contract §5's
decomposition, the same reuse the sibling CTX suite made): a reply that is
nothing but carbon's truncation marker, or a tool-syntax leak, is
``not_attempted`` — never scored as a selection failure, and never a pass. The
``attempted`` metric is published on every attempt so the fraction is over
attempts made, not attempts that got far enough.

Live premise, recorded per attempt: each ``run_*`` reconstructs the offered
tool set for its ask under the exposure policy actually in force, through
carbon's OWN selector (``exposed_specs`` + ``CONFIG.tool_exposure`` — the same
function and policy the agent's turn used), and records
``needed_tool_offered`` / ``needed_offered_position`` / ``offered_count``
metrics plus a premise flag in the detail. A red attempt is thereby
attributable: needed-tool-offered red is the model failing the axis;
needed-tool-hidden red is the exposure value failing the guard. This is a
reconstruction through the mechanism, not a transcript observation — carbon's
own suite pins that the mechanism and the sent payload agree byte-for-byte.

Registration: cluster "S", primitive "tool-selection", its own section —
``SECTION_TASKS`` below is the section identity a future calibration artifact
must carry. Until that campaign runs, ``tests/test_sel_tasks.py`` proves these
names appear in NO committed calibration artifact, i.e. outside every
calibrated gate (decision 20).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from runner.carbon_env import make_provider
from runner.helpers import agent_metrics, neutral_dir, tool_texts
from runner.spec import Attempt, TaskSpec

# G2's non-answer taxonomy, reused verbatim rather than re-pinned: the marker is
# carbon's literal (pinned against carbon's source in test_registry.py) and the
# tool-syntax pattern must classify identically across tasks or the same reply
# would be `not_attempted` on one task and `fail` on another. `_TOOL_SYNTAX_RE` is
# module-private in cluster_g; importing it is deliberate coupling to the ONE
# definition, not an oversight.
from runner.tasks.cluster_g import _TOOL_SYNTAX_RE, G2_TRUNCATION_MARKER

SECTION = "tool_exposure"
SECTION_TASKS = ("SEL-2", "SEL-3", "SEL-4", "SEL-5")

# --- pinned oracles (authoring-time constants, never derived at run time) -----

SEL2_NEEDED = "consignment_routing"
SEL2_CODE = "RTG-QK47-BSL"
SEL2_ASK = "What is the current routing code for consignment CN-7743? Reply with just the code."

SEL3_NEEDED = "invoice_total"
SEL3_DECOY = "invoice_subtotal"
SEL3_DUE_CODE = "TTL-19406-55-EUR"
SEL3_SUBTOTAL_CODE = "TTL-17250-40-EUR"
SEL3_ASK = (
    "What is the amount due, with tax and shipping, for invoice INV-8842? "
    "Reply with just the amount code."
)

SEL4_NEEDED = "service_maintenance"
SEL4_CODE = "WIN-TH-0200-0400"
SEL4_ASK = (
    "Which maintenance window is scheduled for the payments API this week? "
    "Reply with just the window code."
)

SEL5_NEEDED = "inventory_lookup"
SEL5_COUNT = "4127"
SEL5_ASK = (
    "How many crimson gadgets does the Basel depot currently hold? Reply with just the number."
)


def _canned(text: str) -> Callable[..., str]:
    """A tool body that returns one fixed line whatever it is called with —
    decoys must be callable with any plausible arguments and never error,
    because an error string would nudge the model onward for free."""

    def _f(**kwargs: object) -> str:
        return text

    return _f


def _make_registry(rows: list[tuple[str, str, str]]):
    """Build a ToolRegistry from (name, description, canned result) rows.

    Every tool takes one OPTIONAL string param ``query`` so any call shape the
    model produces validates; results are fixed at authoring time. Carbon is
    imported inside (the no-module-scope-carbon rule: config binds at import,
    and the loop's fresh-subprocess measurement depends on it).
    """
    from harness.tools import Tool, ToolRegistry

    reg = ToolRegistry()
    for name, description, result in rows:
        reg.register(
            Tool(
                name=name,
                description=description,
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": [],
                },
                func=_canned(result),
                mutates=False,
            )
        )
    return reg


# --- SEL-2 fixture: 30 plausible logistics decoys around one routing tool -----

_SEL2_DECOYS: list[tuple[str, str]] = [
    ("list_consignments", "List consignment ids for a customer account."),
    ("consignment_history", "Past status events for a consignment."),
    ("depot_directory", "Depot names, addresses, and opening hours."),
    ("carrier_contact", "Contact details for a contracted carrier."),
    ("customs_status", "Customs clearance state for a cross-border shipment."),
    ("shipment_weight", "Declared weight and dimensions for a shipment."),
    ("pallet_count", "Pallet count and stacking plan for a load."),
    ("fuel_surcharge", "The current fuel surcharge table."),
    ("rate_quote", "Freight rate quote between two depots."),
    ("delivery_window", "Requested delivery window for an order."),
    ("proof_of_delivery", "Signed proof-of-delivery receipt for an order."),
    ("address_validate", "Validate and normalize a delivery address."),
    ("label_reprint", "Reprint a shipping label for a parcel."),
    ("pickup_schedule", "Scheduled pickup slots for a depot."),
    ("holiday_calendar", "Depot holiday closures for the year."),
    # SEL2_NEEDED is registered here, at index 15 — buried mid-registry.
    ("temperature_log", "Cold-chain temperature log for a trailer."),
    ("driver_roster", "Driver roster for a delivery route."),
    ("vehicle_position", "Last known GPS position of a vehicle."),
    ("route_planner", "Plan a road route between two sites."),
    ("dock_availability", "Free dock doors at a depot right now."),
    ("damage_report", "File a damage report for a consignment."),
    ("insurance_quote", "Insurance quote for a declared cargo value."),
    ("return_authorization", "Authorize a return shipment."),
    ("invoice_lookup", "Invoice document for a completed shipment."),
    ("duty_calculator", "Import duty estimate for a commodity code."),
    ("packaging_specs", "Approved packaging specs by item class."),
    ("warehouse_stock", "On-hand stock at a warehouse."),
    ("dangerous_goods", "Dangerous-goods handling rules by class."),
    ("eta_history", "Historical on-time performance for a lane."),
    ("notification_prefs", "Notification preferences for an account."),
]


def sel2_registry():
    rows = [(n, d, f"{n}: no records match that request.") for n, d in _SEL2_DECOYS[:15]]
    rows.append(
        (
            SEL2_NEEDED,
            "Current routing code for a consignment id.",
            # Fixed line, echoing the PINNED id, not the caller's argument: the
            # task measures tool selection, and argument fidelity must not be a
            # second failure mode hiding inside the same verdict.
            f"consignment CN-7743: routing code {SEL2_CODE}",
        )
    )
    rows.extend((n, d, f"{n}: no records match that request.") for n, d in _SEL2_DECOYS[15:])
    return _make_registry(rows)


# --- SEL-3 fixture: near-duplicate twins plus neutral fillers -----------------


def sel3_registry():
    rows = [
        (
            # The decoy twin, registered FIRST, description engineered to outscore
            # the needed tool for the ask (one extra overlapping token: "code") —
            # under query_match k=1 it is the only tool offered.
            SEL3_DECOY,
            "Amount due subtotal for an invoice, before tax and shipping; "
            "returns the subtotal amount code.",
            f"invoice INV-8842: subtotal {SEL3_SUBTOTAL_CODE}",
        ),
        ("payment_terms", "Payment terms agreed for a customer account.", "net 30"),
        ("currency_rates", "Daily currency conversion rates.", "EUR/USD 1.09"),
        ("customer_lookup", "Customer account details by id.", "customer 2210: active"),
        (
            SEL3_NEEDED,
            "Total amount due for an invoice, including tax and shipping.",
            f"invoice INV-8842: amount due {SEL3_DUE_CODE}",
        ),
        ("dunning_status", "Dunning status for overdue accounts.", "no overdue accounts"),
        ("credit_limit", "Credit limit for a customer account.", "limit EUR 50000"),
        ("tax_rates", "VAT rates by destination country.", "CH 8.1 percent"),
    ]
    return _make_registry(rows)


# --- SEL-4 fixture: vocabulary-stuffed decoys ahead of a last-registered tool --

# Every decoy description shares at least two tokens with SEL4_ASK (most share
# three or more), and the needed tool's name+description shares exactly two
# ("maintenance", "scheduled" — its "windows" deliberately does not token-match
# the ask's "window"), so every decoy ranks at or above it and registration
# order settles ties with the needed tool LAST either way. The premise test
# proves both consequences through carbon's real selector.
_SEL4_DECOYS: list[tuple[str, str]] = [
    ("payments_api_docs", "Reference docs for the payments API."),
    ("payments_api_status", "Live status page for the payments API."),
    ("api_error_codes", "Error code reference for the payments API."),
    ("weekly_report", "The weekly payments summary report."),
    ("deployment_calendar", "Deployments scheduled for this week."),
    ("oncall_roster", "Who is on call for payments this week."),
    ("incident_history", "Past incidents for the payments API."),
    ("api_keys", "Manage API keys for the payments API."),
    ("sandbox_reset", "Reset the payments sandbox for the week."),
    ("changelog_feed", "Changelog entries for the payments API."),
    ("rate_limits", "Rate limits for the payments API."),
    ("uptime_stats", "Uptime stats for the API this week."),
    ("release_notes", "Release notes scheduled for this week."),
    ("ticket_search", "Search tickets about the payments API."),
    ("slo_dashboard", "SLO dashboard for the payments API."),
    ("billing_cycles", "Billing cycle dates for the payments API."),
    ("webhook_config", "Webhook config for the payments API."),
    ("api_gateway_logs", "Gateway logs for the payments API."),
    ("cert_expiry", "Certificate expiry dates for the API."),
    ("feature_flags", "Feature flags scheduled for the payments API."),
    ("traffic_forecast", "Traffic forecast for the API this week."),
    ("dependency_map", "Service dependency map for the payments API."),
    ("runbook_index", "Runbooks for the payments API."),
    ("audit_trail", "Audit trail for payments changes this week."),
    ("capacity_plan", "Capacity plan for the payments API."),
    ("meeting_scheduler", "Schedule a meeting about the payments API."),
    ("code_freeze_dates", "Code freeze dates for the week."),
    ("env_variables", "Environment variables for the payments API."),
    ("latency_report", "Latency report for the API this week."),
    ("team_directory", "Directory for the payments API team."),
]


def sel4_registry():
    rows = [(n, d, f"{n}: nothing relevant on file.") for n, d in _SEL4_DECOYS]
    rows.append(
        (
            SEL4_NEEDED,
            "Scheduled maintenance windows by service name.",
            f"payments API: maintenance window {SEL4_CODE}",
        )
    )
    return _make_registry(rows)


# --- SEL-5 fixture: zero-overlap needed tool among lexical-bait fillers --------


def sel5_registry():
    rows = [
        ("gadget_catalog", "Marketing copy for the gadgets we sell.", "see brochure"),
        ("depot_directory", "Addresses and hours for the Basel depot.", "Basel: Hafenstrasse 9"),
        ("colour_chart", "Colour reference chart: crimson, teal, ochre.", "chart v3"),
        (
            # The one tool that can answer, sharing NO token with the ask — a
            # lexical ranker cannot reach it; a model reading meaning can.
            SEL5_NEEDED,
            "Stock levels by item and site.",
            f"basel: crimson gadget stock {SEL5_COUNT} units",
        ),
        ("holiday_hold", "Place a holiday hold on deliveries.", "no holds active"),
        ("phone_extensions", "Phone numbers for depot staff.", "reception 100"),
        ("shift_roster", "Who is currently on shift.", "day shift"),
    ]
    return _make_registry(rows)


# --- shared drive/verify machinery --------------------------------------------


def _sel_agent(registry):
    from harness.agent import DEFAULT_SYSTEM, Agent
    from harness.observability import Tracer

    provider = make_provider()
    return Agent(
        tracer=Tracer(model=provider.model),
        system=DEFAULT_SYSTEM,
        provider=provider,
        model=provider.model,
        tools=registry,
        agents_dir=neutral_dir(),
    )


def _called_names(messages: list[dict]) -> list[str]:
    out: list[str] = []
    for m in messages:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        for tc in m["tool_calls"]:
            out.append(tc.get("function", {}).get("name", ""))
    return out


def _exposure_premise(registry, ask: str, needed: str) -> dict[str, float]:
    """The live premise: which tools the policy in force offers for this ask.

    Reconstructed through carbon's OWN selector under ``CONFIG.tool_exposure`` —
    the same function, policy, and query the agent's turn used (carbon's suite
    pins that the mechanism and the sent payload agree). Recorded as metrics so
    a red attempt is attributable: offered-and-red is the model failing the
    axis; hidden-and-red is the exposure value failing the guard.
    """
    from harness.harness_config import CONFIG
    from harness.tools import exposed_specs

    # Direct attribute access, no getattr default: a carbon base without the knob
    # must fail loudly here, never quietly reconstruct the premise under "all".
    offered = [
        s["function"]["name"] for s in exposed_specs(registry, CONFIG.tool_exposure, query=ask)
    ]
    return {
        "needed_tool_offered": float(needed in offered),
        "needed_offered_position": float(offered.index(needed)) if needed in offered else -1.0,
        "offered_count": float(len(offered)),
    }


def _last_int(text: str) -> str | None:
    """The last integer token in the text — the model's stated final answer
    (the D3 anchor: quoted intermediate numbers must not read as the answer)."""
    nums = re.findall(r"\d+", text)
    return nums[-1] if nums else None


def _non_answer(reply: str) -> str | None:
    """G2's classification, verbatim in substance: a reply that is nothing but
    carbon's truncation marker never produced an answer, and a tool-call fragment
    answered a different question than the one asked. Verdicts apply it only when
    the reply also failed its reply-side recall conjunct — a reply that recalled
    the sentinel attempted the answer whatever else it carries."""
    if reply.lstrip().startswith(G2_TRUNCATION_MARKER):
        return "generation truncated before answer"
    if _TOOL_SYNTAX_RE.search(reply):
        return "tool-syntax leak instead of answer"
    return None


def _sel3_decoy_clean(reply: str) -> bool:
    """The decoy's value must be absent BOTH as the full code and as its bare
    digits — a reply quoting the subtotal confidently is the failure SEL-3
    exists to catch, however it is formatted."""
    squeezed = reply.replace(",", "").replace(" ", "")
    return SEL3_SUBTOTAL_CODE not in reply and "17250" not in squeezed


# An explicit refusal ANYWHERE in the reply, and a negation sitting immediately
# against the count. Both were found by a review probe scoring as passes: the
# last-int anchor reads position, not meaning, so "Unable to answer; diagnostic
# code 4127" and "The stock is 9999, not 4127" both satisfied it.
#
# Blocklists are incomplete BY CONSTRUCTION and this one is no exception — it
# refuses the phrasings a real refusal actually uses, not every phrasing one
# could. The alternative (a judge) is refused by decision 12 for a fact that is
# an exact string, and a stricter mechanical rule — "the count must be the only
# integer in the final sentence" — fails legitimate answers like
# "4127 units (as of 2026)". So: a deterministic tightening against demonstrated
# spoof shapes, not a proof that no non-answer can score.
_SEL5_REFUSAL_RE = re.compile(
    r"\b(?:unable|cannot|can'?t|could\s?n[o']t|do(?:es)?\s?n[o']t know|not able|"
    r"no (?:tool|data|record|records|information|result|results|match|matches))\b",
    re.I,
)
# Window of 12 characters, deliberately short: it catches "not 4127" and
# "rather than 4127" while leaving "does not stock 9999; it holds 4127" — a
# correction that ENDS on the right answer — scoring as the answer it is.
_SEL5_NEGATED_RE = re.compile(
    rf"\b(?:not|isn'?t|aren'?t|rather than|instead of|no longer|never)\b.{{0,12}}?"
    rf"{re.escape(SEL5_COUNT)}",
    re.I,
)


def _sel5_stated(reply: str) -> bool:
    """Did the reply STATE the count as its final answer?

    The D3 anchor (last integer) plus two deterministic exclusions: an explicit
    refusal marker, and a negation immediately before the count. Without them the
    anchor credits any reply whose final integer happens to be the count,
    including one that refuses to answer and one that names the count to deny
    it."""
    if _last_int(reply) != SEL5_COUNT:
        return False
    if _SEL5_REFUSAL_RE.search(reply) or _SEL5_NEGATED_RE.search(reply):
        return False
    return True


def sel2_verdict(reply: str, code_in_tool: bool) -> tuple[bool, str, str | None]:
    """``(passed, outcome, non_answer_detail)``. Dual conjunct on the D1 pattern:
    the code in the reply AND in a tool result — a reply-side code with no
    tool-result backing is the spoof shape, a plain fail."""
    recalled = SEL2_CODE in reply
    if recalled and code_in_tool:
        return True, "pass", None
    if not recalled and (why := _non_answer(reply)):
        return False, "not_attempted", why
    return False, "fail", None


def sel3_verdict(reply: str, due_in_tool: bool) -> tuple[bool, str, str | None]:
    """``(passed, outcome, non_answer_detail)``. The dual conjunct PLUS decoy
    exclusion: the due code recalled and tool-backed, and the near-twin's value
    absent in every formatting."""
    recalled = SEL3_DUE_CODE in reply
    if recalled and due_in_tool and _sel3_decoy_clean(reply):
        return True, "pass", None
    if not recalled and (why := _non_answer(reply)):
        return False, "not_attempted", why
    return False, "fail", None


def sel4_verdict(reply: str, window_in_tool: bool) -> tuple[bool, str, str | None]:
    """``(passed, outcome, non_answer_detail)``. Same dual-conjunct shape as
    SEL-2; the axis differs (order, not breadth), the oracle does not."""
    recalled = SEL4_CODE in reply
    if recalled and window_in_tool:
        return True, "pass", None
    if not recalled and (why := _non_answer(reply)):
        return False, "not_attempted", why
    return False, "fail", None


def sel5_verdict(reply: str, count_in_tool: bool) -> tuple[bool, str, str | None]:
    """``(passed, outcome, non_answer_detail)``. The stated final answer (last
    integer) must be the count, and the count must have come through a tool."""
    if _sel5_stated(reply) and count_in_tool:
        return True, "pass", None
    if not _sel5_stated(reply) and (why := _non_answer(reply)):
        return False, "not_attempted", why
    return False, "fail", None


def _sel_attempt(
    a,
    result,
    reply: str,
    verdict: tuple[bool, str, str | None],
    why: str,
    premise: dict[str, float],
    called: list[str],
    needed: str,
) -> Attempt:
    """Assemble one SEL Attempt: shared metrics + attribution accounting.

    Each ``run_*`` constructs, drives, and closes its OWN agent (the
    close-in-a-finally shape ``test_registry.py`` sweeps structurally); only
    the read-after-close assembly is shared here."""
    ok, outcome, non_answer = verdict
    metrics = agent_metrics(a, result=result)
    metrics.update(premise)
    metrics["needed_tool_called"] = float(needed in called)
    metrics["decoy_calls"] = float(sum(1 for n in called if n and n != needed))
    # Published on every attempt (the G2 rule): a metric absent from some attempts
    # is averaged over only the attempts that reported it.
    metrics["attempted"] = float(outcome != "not_attempted")
    return Attempt(
        passed=ok,
        outcome=outcome,
        detail=(
            f"{why} needed_offered={premise['needed_tool_offered'] == 1.0} "
            f"called={called!r} "
            + (f"non_answer={non_answer!r} " if non_answer else "")
            # `reply=` stays LAST: `loop.judge_validate._extract_reply` reads it
            # with a greedy match to end-of-string.
            + f"reply={reply!r}"
        ),
        turns=len(a.messages),
        metrics=metrics,
    )


# --- the four axes -------------------------------------------------------------


def run_sel2() -> Attempt:
    reg = sel2_registry()
    a = _sel_agent(reg)
    try:
        premise = _exposure_premise(reg, SEL2_ASK, SEL2_NEEDED)
        result = a.run(SEL2_ASK)
        reply = result.text
        in_tool = any(SEL2_CODE in t for t in tool_texts(a.messages))
        called = _called_names(a.messages)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return _sel_attempt(
        a,
        result,
        reply,
        sel2_verdict(reply, in_tool),
        f"code_in_reply={SEL2_CODE in reply} code_in_tool_result={in_tool}",
        premise,
        called,
        SEL2_NEEDED,
    )


def run_sel3() -> Attempt:
    reg = sel3_registry()
    a = _sel_agent(reg)
    try:
        premise = _exposure_premise(reg, SEL3_ASK, SEL3_NEEDED)
        result = a.run(SEL3_ASK)
        reply = result.text
        in_tool = any(SEL3_DUE_CODE in t for t in tool_texts(a.messages))
        called = _called_names(a.messages)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return _sel_attempt(
        a,
        result,
        reply,
        sel3_verdict(reply, in_tool),
        f"due_in_reply={SEL3_DUE_CODE in reply} due_in_tool_result={in_tool} "
        f"decoy_absent={_sel3_decoy_clean(reply)}",
        premise,
        called,
        SEL3_NEEDED,
    )


def run_sel4() -> Attempt:
    reg = sel4_registry()
    a = _sel_agent(reg)
    try:
        premise = _exposure_premise(reg, SEL4_ASK, SEL4_NEEDED)
        result = a.run(SEL4_ASK)
        reply = result.text
        in_tool = any(SEL4_CODE in t for t in tool_texts(a.messages))
        called = _called_names(a.messages)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return _sel_attempt(
        a,
        result,
        reply,
        sel4_verdict(reply, in_tool),
        f"window_in_reply={SEL4_CODE in reply} window_in_tool_result={in_tool}",
        premise,
        called,
        SEL4_NEEDED,
    )


def run_sel5() -> Attempt:
    reg = sel5_registry()
    a = _sel_agent(reg)
    try:
        premise = _exposure_premise(reg, SEL5_ASK, SEL5_NEEDED)
        result = a.run(SEL5_ASK)
        reply = result.text
        in_tool = any(SEL5_COUNT in t for t in tool_texts(a.messages))
        called = _called_names(a.messages)
    finally:
        a.close()  # the storage contract says close ends the scratch lifecycle
    return _sel_attempt(
        a,
        result,
        reply,
        sel5_verdict(reply, in_tool),
        f"count_is_final_answer={_sel5_stated(reply)} count_in_tool_result={in_tool}",
        premise,
        called,
        SEL5_NEEDED,
    )


SPECS = [
    TaskSpec(
        "SEL-2", "held_in", "S", "uncertain", primitive="tool-selection", alias=None, run=run_sel2
    ),
    TaskSpec(
        "SEL-3", "held_out", "S", "uncertain", primitive="tool-selection", alias=None, run=run_sel3
    ),
    TaskSpec(
        "SEL-4", "held_in", "S", "uncertain", primitive="tool-selection", alias=None, run=run_sel4
    ),
    TaskSpec(
        "SEL-5", "held_out", "S", "uncertain", primitive="tool-selection", alias=None, run=run_sel5
    ),
]
