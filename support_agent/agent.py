"""The three-stage agent: classify -> draft -> route."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from .intents import (
    INTENT_NAMES, taxonomy_prompt_block, policy_defaults, HARD_ESCALATION_CUES,
)
from .llm import LLM
from .retrieve import ReplyRetriever, evidence_block


# ---------------------------------------------------------------------------
# 1. Classification
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM = """You are an intent classifier for Spectrum (a US cable \
internet and TV provider) customer support on Twitter.

Classify the customer's message into exactly one intent:
{taxonomy}

Also judge severity — how much damage is done if this sits unanswered for 4 hours:
  low    = informational, no service impact
  medium = service is degraded or the customer is annoyed
  high   = service fully out, money at stake, or the customer is furious / threatening to leave

Return ONLY a JSON object, no prose:
{{"intent": "<one of: {names}>", "confidence": <0.0-1.0>, "severity": "low|medium|high", "rationale": "<max 15 words>"}}"""

# Few-shot examples are hand-picked from the TRAINING slice only, one per
# intent, chosen to be prototypical rather than hard. D9.
CLASSIFY_FEWSHOT = [
    ("Internet has been down in <id> since 6am, third time this month. Any ETA?", "outage"),
    ("paying for 200mbps and getting 12. every single night. fix it", "connectivity_degraded"),
    ("Why did my bill jump from 79 to 134 this month? Nobody told me anything", "billing"),
    ("can't log in to my account, password reset email never arrives", "account_access"),
    ("my cable box keeps rebooting itself every 20 minutes, need a new one", "equipment"),
    ("tech was supposed to be here 8-11, it's 2pm and nobody showed up", "appointment"),
    ("where is the game? channel 30 is just a blue screen", "channel_content"),
    ("done with you guys, switching to fios tomorrow. how do I cancel", "cancellation"),
    ("thanks for getting it sorted so fast, appreciate it", "other"),
]


@dataclass
class Classification:
    intent: str
    confidence: float
    severity: str
    rationale: str


def classify(llm: LLM, message: str) -> Classification:
    system = CLASSIFY_SYSTEM.format(
        taxonomy=taxonomy_prompt_block(), names=", ".join(INTENT_NAMES)
    )
    shots = "\n".join(
        f'Message: {t}\nOutput: {{"intent": "{lab}", "confidence": 0.9, '
        f'"severity": "medium", "rationale": "prototypical"}}'
        for t, lab in CLASSIFY_FEWSHOT
    )
    user = f"{shots}\n\nMessage: {message}\nOutput:"
    out = llm.complete_json(system, user, max_tokens=200)

    intent = str(out.get("intent", "other")).strip().lower()
    if intent not in INTENT_NAMES:
        intent = "other"
    try:
        conf = float(out.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    sev = str(out.get("severity", "medium")).lower()
    if sev not in {"low", "medium", "high"}:
        sev = "medium"
    return Classification(intent, max(0.0, min(1.0, conf)), sev,
                          str(out.get("rationale", ""))[:120])


# ---------------------------------------------------------------------------
# 2. Grounded drafting
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = """You write public Twitter replies as Spectrum support (@Ask_Spectrum).

You are given real past conversations where Spectrum handled a similar issue. \
Match how Spectrum actually replies — do not invent a house style.

Hard rules:
- Under 280 characters.
- Never state a specific outage ETA, credit amount, price, or appointment time \
unless it appears verbatim in the evidence.
- Never claim you have looked at their account. You cannot see accounts.
- If the issue needs account details, ask them to DM — that is what Spectrum does.
- No hashtags. At most one apology clause. Do not open with "We apologize for \
the inconvenience" every time.

Return ONLY JSON: {"reply": "<the reply text>"}"""


def draft(llm: LLM, message: str, intent: str, evidence) -> str:
    user = (
        f"Detected intent: {intent}\n\n"
        f"Past conversations for grounding:\n{evidence_block(evidence)}\n\n"
        f"New customer message:\n{message}\n\n"
        "Write Spectrum's reply."
    )
    out = llm.complete_json(DRAFT_SYSTEM, user, max_tokens=300)
    reply = str(out.get("reply", "")).strip()
    if not reply:
        reply = ("Sorry for the trouble. Please send us a DM with your address "
                 "and we'll take a look.")
    return reply[:280]


# ---------------------------------------------------------------------------
# 3. Routing
# ---------------------------------------------------------------------------

@dataclass
class Routing:
    action: str            # "auto" | "escalate"
    reason: str
    triggers: list = field(default_factory=list)


def route(
    message: str,
    cls: Classification,
    retrieval_score: float,
    conf_threshold: float = 0.65,
    grounding_threshold: float = 0.18,
) -> Routing:
    """Rule-based routing on top of model signals.

    Four independent escalation triggers. Any one fires -> human.
    Ordered so the REASON returned is the most actionable one.
    """
    triggers = []
    low = message.lower()

    hit = next((c for c in HARD_ESCALATION_CUES if c in low), None)
    if hit:
        triggers.append(("risk_cue", f"message contains a risk phrase ('{hit}')"))

    default_action, needs_account = policy_defaults(cls.intent)
    if needs_account:
        triggers.append(
            ("needs_account_access",
             f"intent '{cls.intent}' cannot be resolved without account access")
        )

    if cls.confidence < conf_threshold:
        triggers.append(
            ("low_confidence",
             f"intent confidence {cls.confidence:.2f} below {conf_threshold:.2f}")
        )

    if retrieval_score < grounding_threshold:
        triggers.append(
            ("no_precedent",
             f"no similar past conversation (top similarity {retrieval_score:.2f})")
        )

    if cls.severity == "high" and cls.intent in {"outage", "connectivity_degraded"}:
        # A furious customer on a technically auto-handleable intent is exactly
        # the case where a canned reply goes viral. D11.
        triggers.append(("high_severity", "high-severity complaint on a public channel"))

    if triggers:
        return Routing("escalate", triggers[0][1], [t[0] for t in triggers])
    return Routing("auto", f"intent '{cls.intent}' is self-serve, "
                           f"confident ({cls.confidence:.2f}) and has precedent", [])


# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    message: str
    intent: str
    confidence: float
    severity: str
    reply: str
    action: str
    reason: str
    triggers: list
    retrieval_score: float
    evidence: list

    def to_dict(self):
        d = asdict(self)
        d["evidence"] = [e if isinstance(e, dict) else e.to_dict() for e in self.evidence]
        return d


class SupportAgent:
    def __init__(self, llm: LLM, retriever: ReplyRetriever, cfg: dict | None = None):
        self.llm = llm
        self.retriever = retriever
        cfg = cfg or {}
        self.conf_threshold = cfg.get("conf_threshold", 0.65)
        self.grounding_threshold = cfg.get("grounding_threshold", 0.18)

    def __call__(self, message: str) -> AgentOutput:
        cls = classify(self.llm, message)
        ev = self.retriever.search(message)
        top = ev[0].score if ev else 0.0
        reply = draft(self.llm, message, cls.intent, ev)
        r = route(message, cls, top, self.conf_threshold, self.grounding_threshold)
        return AgentOutput(
            message=message, intent=cls.intent, confidence=cls.confidence,
            severity=cls.severity, reply=reply, action=r.action, reason=r.reason,
            triggers=r.triggers, retrieval_score=top,
            evidence=[e.to_dict() for e in ev],
        )
