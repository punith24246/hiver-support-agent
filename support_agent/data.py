"""Load the Customer Support on Twitter dump and reshape it into the two
things the agent actually needs:

  1. `inbound_messages` — first customer message of each thread to the brand.
     This is what the agent sees at inference time.
  2. `resolution_pairs`  — (customer message, brand's historical reply) used as
     the grounding corpus for retrieval.

The raw file is ~3M rows. Everything here streams or filters early so a Colab
CPU runtime does not die.
"""

from __future__ import annotations

import re
import pandas as pd

RAW_COLUMNS = [
    "tweet_id", "author_id", "inbound", "created_at",
    "text", "response_tweet_id", "in_response_to_tweet_id",
]

# --- normalisation ---------------------------------------------------------
_MENTION = re.compile(r"@\w+")
_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")


def clean_text(s: str, keep_mentions: bool = False) -> str:
    """Light normalisation only.

    We deliberately do NOT lowercase, strip punctuation, or remove emoji:
    ALL-CAPS and '!!!' carry escalation signal, and emoji carry sentiment.
    See docs/decision_log.md D4.
    """
    if not isinstance(s, str):
        return ""
    s = _URL.sub(" <url> ", s)
    if not keep_mentions:
        s = _MENTION.sub(" ", s)
    # The dataset anonymises handles as @123456; collapse leftover ids.
    s = re.sub(r"\b\d{6,}\b", " <id> ", s)
    return _WS.sub(" ", s).strip()


def load_raw(csv_path: str, usecols=None) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=usecols or RAW_COLUMNS,
        dtype={
            "tweet_id": "int64",
            "author_id": "string",
            "text": "string",
            "response_tweet_id": "string",
            "in_response_to_tweet_id": "float64",
        },
        low_memory=False,
    )
    return df


def brand_pairs(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    """Join every brand reply to the customer tweet it replied to.

    Returns columns: customer_tweet_id, customer_text, brand_text, created_at.
    """
    outbound = df[(df.author_id == brand) & (~df.inbound.astype(bool))].copy()
    outbound = outbound.dropna(subset=["in_response_to_tweet_id"])
    outbound["parent_id"] = outbound["in_response_to_tweet_id"].astype("int64")

    inbound = df[df.inbound.astype(bool)][["tweet_id", "text", "created_at", "author_id"]]
    inbound = inbound.rename(columns={
        "text": "customer_text",
        "author_id": "customer_id",
        "created_at": "customer_created_at",
    })

    merged = outbound.merge(
        inbound, left_on="parent_id", right_on="tweet_id",
        how="inner", suffixes=("_brand", "_cust"),
    )

    # Keep only the FIRST customer message of a thread: a mid-thread reply
    # ("ok DMing you now") has no standalone intent and would poison both the
    # classifier and the retrieval corpus. D5.
    merged = merged[merged["in_response_to_tweet_id"].notna()]
    first_contact = merged[~merged["parent_id"].isin(set(merged["tweet_id_brand"]))]

    out = pd.DataFrame({
        "customer_tweet_id": first_contact["parent_id"].values,
        "customer_id": first_contact["customer_id"].values,
        "customer_text": [clean_text(t) for t in first_contact["customer_text"]],
        "brand_text": [clean_text(t) for t in first_contact["text"]],
        "created_at": first_contact["customer_created_at"].values,
    })

    out = out[(out.customer_text.str.len() >= 15) & (out.brand_text.str.len() >= 15)]
    return out.drop_duplicates(subset=["customer_tweet_id"]).reset_index(drop=True)


def split_corpus_and_pool(
    pairs: pd.DataFrame, holdout_frac: float = 0.25, seed: int = 13
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by TIME, not randomly.

    The grounding corpus is the older slice; the evaluation pool is the newer
    slice. A random split would let the agent retrieve a reply written to the
    exact same customer about the exact same outage, which inflates every reply
    quality number. This is the single most important guard in the repo and it
    is discussed in the 'misleading headline number' section. D6.
    """
    p = pairs.copy()
    p["created_at"] = pd.to_datetime(p["created_at"], errors="coerce", utc=True, format="mixed")
    p = p.dropna(subset=["created_at"]).sort_values("created_at")
    cut = int(len(p) * (1 - holdout_frac))
    return p.iloc[:cut].reset_index(drop=True), p.iloc[cut:].reset_index(drop=True)


def load_brand(csv_path: str, brand: str, holdout_frac: float = 0.25):
    df = load_raw(csv_path)
    pairs = brand_pairs(df, brand)
    del df
    return split_corpus_and_pool(pairs, holdout_frac=holdout_frac)
