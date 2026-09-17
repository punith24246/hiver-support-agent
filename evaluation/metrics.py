"""Automated metrics.

Two opinions baked in here:

1. Report MACRO-F1 for intent, not accuracy. `other` and `connectivity_degraded`
   dominate the stream; accuracy rewards a model that ignores `cancellation`,
   which is the intent with the highest business cost per miss.

2. Routing is NOT symmetric and must not be scored with accuracy or F1.
   A false-auto (agent answers something that needed a human) is the expensive
   error; a false-escalate just costs an agent a minute. We report both rates
   separately and a cost-weighted score with the weight stated out loud. D12.
"""

from __future__ import annotations

import re

from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, cohen_kappa_score,
)

# Cost of letting a should-escalate message be auto-answered, relative to the
# cost of needlessly escalating one. Chosen, not measured — see the report.
FALSE_AUTO_COST = 5.0


def intent_metrics(y_true, y_pred, labels=None) -> dict:
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        lab: {"precision": float(p[i]), "recall": float(r[i]),
              "f1": float(f[i]), "support": int(s[i])}
        for i, lab in enumerate(labels or sorted(set(y_true)))
    }
    return {
        "macro_f1": float(macro),
        "micro_f1": float(micro),
        "per_class": per_class,
        "report": classification_report(y_true, y_pred, zero_division=0),
    }


def routing_metrics(y_true, y_pred) -> dict:
    """y_* are 'auto' / 'escalate' strings."""
    labels = ["auto", "escalate"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # rows = truth, cols = pred
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    # tp = correctly escalated, fn = FALSE AUTO (the dangerous one)
    n_escalate = tp + fn
    n_auto = tn + fp

    false_auto_rate = fn / n_escalate if n_escalate else 0.0
    false_escalate_rate = fp / n_auto if n_auto else 0.0
    automation_rate = (tn + fn) / len(y_true) if len(y_true) else 0.0

    cost = (FALSE_AUTO_COST * fn + fp) / len(y_true) if len(y_true) else 0.0

    return {
        "escalation_recall": float(tp / n_escalate) if n_escalate else 0.0,
        "escalation_precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "false_auto_rate": float(false_auto_rate),
        "false_escalate_rate": float(false_escalate_rate),
        "automation_rate": float(automation_rate),
        "cost_per_message": float(cost),
        "false_auto_cost_weight": FALSE_AUTO_COST,
        "confusion": {"tn_auto_auto": int(tn), "fp_auto_escalated": int(fp),
                      "fn_escalate_autoed": int(fn), "tp_escalate_escalate": int(tp)},
    }


# --- cheap deterministic reply checks --------------------------------------
# These catch the failures an LLM judge is bad at: hallucinated specifics.

_NUM_CLAIM = re.compile(
    r"(?:\$\s?\d+(?:\.\d{2})?"
    r"|\b\d{1,3}\s*(?:hours?|hrs?|minutes?|mins?|days?|business days?)\b"
    r"|\b\d{1,3}\s?%"
    r"|\bby\s+\d{1,2}\s?(?:am|pm)\b"
    r"|\b\d{1,2}[:/]\d{2}\s?(?:am|pm)?\b)", re.I)
_ACCOUNT_CLAIM = re.compile(
    r"\b(I (?:can )?see (?:on )?your account|looking at your account|"
    r"I've checked your account|your account shows)\b", re.I)


def reply_checks(reply: str, evidence_texts: list[str]) -> dict:
    ev_blob = " ".join(evidence_texts).lower()
    claims = _NUM_CLAIM.findall(reply)
    ungrounded = [c for c in claims if c.lower().strip() not in ev_blob]
    return {
        "length_ok": len(reply) <= 280,
        "n_specific_claims": len(claims),
        "n_ungrounded_claims": len(ungrounded),
        "ungrounded_claims": ungrounded[:5],
        "fake_account_access": bool(_ACCOUNT_CLAIM.search(reply)),
    }


def judge_agreement(human_scores, judge_scores) -> dict:
    """Agreement between the human's and the judge's overall 1-5 ratings.

    We report three numbers because each hides something:
      * exact agreement   — harsh, 5-point scales are noisy
      * within-1 agreement — the number people actually care about
      * quadratic-weighted kappa — chance-corrected and ordinal-aware
    """
    h = list(human_scores)
    j = list(judge_scores)
    n = len(h)
    exact = sum(a == b for a, b in zip(h, j)) / n
    within1 = sum(abs(a - b) <= 1 for a, b in zip(h, j)) / n
    qwk = cohen_kappa_score(h, j, weights="quadratic")
    bias = sum(j) / n - sum(h) / n
    return {
        "n": n,
        "exact_agreement": float(exact),
        "within_1_agreement": float(within1),
        "quadratic_weighted_kappa": float(qwk),
        "judge_minus_human_mean": float(bias),
    }
