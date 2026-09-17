"""Intent taxonomy for Ask_Spectrum, derived by reading ~300 sampled inbound tweets.

Design notes (see docs/decision_log.md D2, D3):
  * 8 intents, not 77. The taxonomy is cut at the level where the BRAND'S ACTION
    differs, not where the customer's words differ. "my wifi is out" and "no
    internet since 6am" are the same ticket; "my wifi is slow" is a different one
    because it gets a troubleshooting script instead of an outage lookup.
  * `other` is a real class, not a dumping ground for failure. It is ~15% of the
    inbound stream (thanks / jokes / abuse / off-topic) and every one of them
    should be auto-handled or dropped, never escalated.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    description: str
    # Does resolving this REQUIRE looking at the customer's account?
    needs_account_access: bool
    # Default routing before per-message signals are applied.
    default_action: str  # "auto" | "escalate"


INTENTS = [
    Intent(
        "outage",
        "Service is completely down — no internet, no TV, no signal. Often "
        "mentions an area, a storm, or 'still down'.",
        needs_account_access=False,
        default_action="auto",
    ),
    Intent(
        "connectivity_degraded",
        "Service works but badly — slow speeds, buffering, intermittent drops, "
        "packet loss, pixelation.",
        needs_account_access=False,
        default_action="auto",
    ),
    Intent(
        "billing",
        "Charges, bill amount, autopay, refunds, late fees, price increases, "
        "promotional rate expiring.",
        needs_account_access=True,
        default_action="escalate",
    ),
    Intent(
        "account_access",
        "Login problems, password reset, username recovery, locked account, "
        "email or account settings.",
        needs_account_access=True,
        default_action="escalate",
    ),
    Intent(
        "equipment",
        "Modem, router, cable box, remote — broken, needs replacing, needs "
        "activating, or needs returning.",
        needs_account_access=False,
        default_action="auto",
    ),
    Intent(
        "appointment",
        "Technician visits — scheduling, rescheduling, no-shows, arrival "
        "windows, installation dates.",
        needs_account_access=True,
        default_action="escalate",
    ),
    Intent(
        "channel_content",
        "Programming questions — a channel is missing, a game is not showing, "
        "channel lineup changes, on-demand content.",
        needs_account_access=False,
        default_action="auto",
    ),
    Intent(
        "cancellation",
        "Wants to cancel, downgrade, or is threatening to leave for a "
        "competitor. Retention-sensitive.",
        needs_account_access=True,
        default_action="escalate",
    ),
    Intent(
        "other",
        "Thanks, praise, jokes, generic complaints with no actionable request, "
        "abuse, or messages not about Spectrum service at all.",
        needs_account_access=False,
        default_action="auto",
    ),
]

INTENT_NAMES = [i.name for i in INTENTS]
BY_NAME = {i.name: i for i in INTENTS}


def taxonomy_prompt_block() -> str:
    """Render the taxonomy for injection into an LLM prompt."""
    return "\n".join(f"- {i.name}: {i.description}" for i in INTENTS)


# ---------------------------------------------------------------------------
# Escalation policy
# ---------------------------------------------------------------------------
# Deliberately rule-based on top of the model rather than asking the LLM to
# decide routing free-form. Reason: routing is the decision with real business
# cost, and a stakeholder has to be able to read the policy and change it
# without a prompt rewrite. See docs/decision_log.md D7.

# Phrases that force a human regardless of intent. Kept short and auditable.
HARD_ESCALATION_CUES = [
    "lawyer", "attorney", "sue", "lawsuit", "legal action", "small claims",
    "bbb", "better business bureau", "fcc", "attorney general",
    "fraud", "identity theft", "unauthorized charge",
    "medical", "disabled", "elderly", "life support", "oxygen",
    "911", "emergency",
]


def policy_defaults(intent: str) -> tuple[str, bool]:
    """Return (default_action, needs_account_access) for an intent name."""
    meta = BY_NAME.get(intent, BY_NAME["other"])
    return meta.default_action, meta.needs_account_access
