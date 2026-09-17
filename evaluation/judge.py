"""LLM-as-judge for reply quality.

Guards against the three ways a judge lies to you:
  * It does not know which system wrote the reply (blinded).
  * It scores one reply at a time against a fixed rubric (absolute, not
    pairwise) so scores are comparable across systems without re-running.
  * It is a DIFFERENT and larger model than the generator. Same-model judging
    inflates scores for your own system's style. This is not fully solved —
    both are gpt-oss family — and the report says so. D13.
"""

from __future__ import annotations

from support_agent.llm import LLM  # noqa: F401

JUDGE_SYSTEM = """You are a strict quality reviewer for customer-support replies \
posted publicly on Twitter by an internet/TV provider.

You will see the customer's message, some real past conversations from the same \
brand, and one candidate reply. Score the candidate on four axes, 1-5.

groundedness — Does every specific claim (times, prices, credits, causes, \
appointment slots) trace to the evidence or to something universally true?
  5 = no unsupported specifics.  3 = vague but not wrong.  1 = invented facts.

helpfulness — Does it move the ticket forward? Asking for a DM counts as \
progress ONLY when the issue genuinely needs account data.
  5 = clear next step.  3 = generic but relevant.  1 = no progress or off-topic.

tone — Does it sound like this brand's support, to an angry stranger in public?
  5 = natural, warm, not robotic.  3 = canned but inoffensive.  1 = tone-deaf.

safety — Does it avoid promises the brand cannot keep, admissions of fault, \
requests for sensitive data in public, or claims to have seen the account?
  5 = safe.  3 = borderline.  1 = would need a retraction.

Be harsh. A reply that is merely inoffensive is a 3, not a 5.

Return ONLY JSON:
{"groundedness": <1-5>, "helpfulness": <1-5>, "tone": <1-5>, "safety": <1-5>, \
"overall": <1-5>, "rationale": "<max 25 words>"}"""


def judge_reply(llm, customer_message: str, reply: str, evidence_block: str) -> dict:
    user = (
        f"Customer message:\n{customer_message}\n\n"
        f"Past conversations from this brand:\n{evidence_block}\n\n"
        f"Candidate reply:\n{reply}\n\n"
        "Score it."
    )
    out = llm.complete_json(JUDGE_SYSTEM, user, max_tokens=100)

    def _clamp(k, default=3):
        try:
            v = int(round(float(out.get(k, default))))
        except (TypeError, ValueError):
            v = default
        return max(1, min(5, v))

    scores = {k: _clamp(k) for k in ("groundedness", "helpfulness", "tone", "safety")}
    # Recompute overall ourselves rather than trusting the model's arithmetic;
    # safety is a floor, not an average term.
    avg = sum(scores[k] for k in ("groundedness", "helpfulness", "tone")) / 3
    scores["overall"] = int(round(min(avg, scores["safety"])))
    scores["rationale"] = str(out.get("rationale", ""))[:200]
    return scores
