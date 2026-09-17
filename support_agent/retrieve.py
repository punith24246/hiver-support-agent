"""Grounding retriever.

TF-IDF + cosine over the HISTORICAL CUSTOMER MESSAGES, returning the brand's
reply to each match. We match customer-to-customer, not customer-to-reply,
because the two sides of a support thread share almost no vocabulary — the
customer says "no internet since 6am", the agent says "I'm sorry to hear that,
please DM us". Matching on the customer side is what actually finds analogous
tickets. D8.

Why TF-IDF and not embeddings: the messages are short, keyword-dominated
(channel names, city names, error codes), and the whole thing has to run on a
Colab CPU inside the 15-minute budget. We report the embedding comparison as
future work rather than pretending we tuned it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Evidence:
    customer_text: str
    brand_text: str
    score: float

    def to_dict(self):
        return asdict(self)


class ReplyRetriever:
    def __init__(self, k: int = 4, min_score: float = 0.12):
        self.k = k
        self.min_score = min_score
        self.vec = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=60_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self._matrix = None
        self._cust: list[str] = []
        self._brand: list[str] = []

    def fit(self, customer_texts, brand_texts):
        self._cust = list(customer_texts)
        self._brand = list(brand_texts)
        self._matrix = self.vec.fit_transform(self._cust)
        return self

    def search(self, query: str, k: int | None = None) -> list[Evidence]:
        k = k or self.k
        q = self.vec.transform([query])
        sims = cosine_similarity(q, self._matrix).ravel()
        idx = np.argsort(-sims)[:k]
        return [
            Evidence(self._cust[i], self._brand[i], float(sims[i]))
            for i in idx
            if sims[i] >= self.min_score
        ]

    def top_score(self, query: str) -> float:
        q = self.vec.transform([query])
        return float(cosine_similarity(q, self._matrix).max())


def evidence_block(ev: list[Evidence]) -> str:
    if not ev:
        return "(no similar past conversations found)"
    return "\n\n".join(
        f"[{i+1}] Customer said: {e.customer_text}\n    Spectrum replied: {e.brand_text}"
        for i, e in enumerate(ev)
    )
