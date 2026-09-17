"""Baselines. The report is worthless without these.

B0 "trivial"  — predict the majority intent, send one canned reply to everyone,
                escalate nothing. This is the number that tells you whether the
                dataset is just imbalanced.
B1 "simple"   — TF-IDF + logistic regression for intent (trained on the golden
                set with cross-validation), nearest-neighbour copy of a past
                brand reply, and a keyword rule for escalation. No LLM at all.
                This is the number the LLM agent has to beat to justify its cost.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .intents import policy_defaults, HARD_ESCALATION_CUES
from .retrieve import ReplyRetriever

CANNED = ("Sorry for the trouble! Please send us a DM with your name and "
          "service address so we can take a closer look.")


class TrivialBaseline:
    name = "B0_trivial"

    def fit(self, texts, intents, actions):
        self.majority_intent = Counter(intents).most_common(1)[0][0]
        self.majority_action = Counter(actions).most_common(1)[0][0]
        return self

    def __call__(self, message: str) -> dict:
        return {
            "intent": self.majority_intent,
            "reply": CANNED,
            "action": self.majority_action,
            "reason": "constant prediction",
        }


class SimpleBaseline:
    """No LLM. Everything here is a 40-year-old technique."""

    name = "B1_simple"

    def __init__(self, corpus_cust=None, corpus_brand=None):
        self.clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=2.0),
        )
        self.retriever = None
        if corpus_cust is not None:
            self.retriever = ReplyRetriever(k=1, min_score=0.0).fit(corpus_cust, corpus_brand)

    def fit(self, texts, intents, actions=None):
        self.clf.fit(texts, intents)
        return self

    def __call__(self, message: str) -> dict:
        intent = self.clf.predict([message])[0]
        proba = float(np.max(self.clf.predict_proba([message])))

        if self.retriever is not None:
            ev = self.retriever.search(message, k=1)
            reply = ev[0].brand_text if ev else CANNED
        else:
            reply = CANNED

        default_action, needs_account = policy_defaults(intent)
        low = message.lower()
        escalate = (
            needs_account
            or any(c in low for c in HARD_ESCALATION_CUES)
            or proba < 0.5
        )
        return {
            "intent": intent,
            "reply": reply,
            "action": "escalate" if escalate else "auto",
            "reason": "keyword + intent rule",
            "confidence": proba,
        }
