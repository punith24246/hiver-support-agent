"""End-to-end smoke test on synthetic twcs-shaped data. No network, no API key.

    AGENT_MOCK=1 pytest -q

This exists so a reviewer can verify the pipeline wiring in 5 seconds without
downloading 3M tweets or holding a Groq key.
"""

import os
import random

import pandas as pd
import pytest

os.environ["AGENT_MOCK"] = "1"

from support_agent.data import brand_pairs, split_corpus_and_pool, clean_text
from support_agent.retrieve import ReplyRetriever
from support_agent.agent import SupportAgent, route, Classification
from support_agent.baselines import TrivialBaseline, SimpleBaseline
from support_agent.llm import LLM, parse_json
from support_agent.intents import INTENT_NAMES
from evaluation.metrics import routing_metrics, reply_checks, judge_agreement

BRAND = "Ask_Spectrum"

CUST = [
    "internet is down again in my whole neighborhood, third time this week",
    "speeds are terrible every night, paying for 400 getting 30",
    "my bill went up 40 dollars with no warning, explain please",
    "cant log into my account, reset link never comes",
    "cable box wont turn on after the storm",
    "tech never showed up for my 8-11 window, waited all day",
    "espn is missing from my lineup since yesterday",
    "i want to cancel, switching to fios next week",
    "thanks for the quick help earlier, appreciate it",
]
REPLY = [
    "Sorry about that! We're aware of an issue in some areas. DM us your address and we'll check.",
    "That's not the experience we want. Please DM us so we can run a line test.",
    "We can look into that billing change for you. Please DM us your account details.",
    "Let's get you back in. DM us and we'll help reset your credentials.",
    "Let's troubleshoot that box. Try unplugging for 60 seconds, then DM us if it persists.",
    "We're sorry for the missed appointment. DM us and we'll reschedule right away.",
    "Channel lineups vary by area. DM us your zip and we'll confirm.",
    "We'd hate to see you go. DM us and we'll see what options we have.",
    "Happy to help! Let us know if you need anything else.",
]


def make_raw(n=240, seed=3):
    random.seed(seed)
    rows, tid = [], 1000
    for i in range(n):
        j = i % len(CUST)
        cust_id, brand_tid = tid, tid + 1
        tid += 2
        ts = f"Tue Oct {1 + i % 28} 1{i % 10}:00:00 +0000 2017"
        rows.append(dict(tweet_id=cust_id, author_id=f"user{i}", inbound=True,
                         created_at=ts, text=CUST[j] + f" (#{i})",
                         response_tweet_id=str(brand_tid),
                         in_response_to_tweet_id=float("nan")))
        rows.append(dict(tweet_id=brand_tid, author_id=BRAND, inbound=False,
                         created_at=ts, text=REPLY[j],
                         response_tweet_id="", in_response_to_tweet_id=float(cust_id)))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pieces():
    raw = make_raw()
    pairs = brand_pairs(raw, BRAND)
    corpus, pool = split_corpus_and_pool(pairs, holdout_frac=0.3)
    return pairs, corpus, pool


def test_pairs_built(pieces):
    pairs, corpus, pool = pieces
    assert len(pairs) > 100
    assert {"customer_text", "brand_text"} <= set(pairs.columns)
    assert len(corpus) > 0 and len(pool) > 0


def test_time_split_has_no_overlap(pieces):
    _, corpus, pool = pieces
    assert not set(corpus.customer_tweet_id) & set(pool.customer_tweet_id)


def test_clean_text_keeps_signal():
    out = clean_text("@115712 MY INTERNET IS OUT!!! https://t.co/x 123456789")
    assert "MY INTERNET IS OUT!!!" in out
    assert "@115712" not in out and "https" not in out


def test_retriever_finds_analogue(pieces):
    _, corpus, _ = pieces
    r = ReplyRetriever(k=3, min_score=0.0).fit(corpus.customer_text, corpus.brand_text)
    ev = r.search("my internet is down in the neighborhood")
    assert ev and ev[0].score > 0.2
    assert "DM" in ev[0].brand_text or "check" in ev[0].brand_text


def test_agent_end_to_end(pieces):
    _, corpus, pool = pieces
    r = ReplyRetriever().fit(corpus.customer_text, corpus.brand_text)
    agent = SupportAgent(LLM("mock", mock=True), r)
    out = agent(pool.customer_text.iloc[0])
    assert out.intent in INTENT_NAMES
    assert out.action in {"auto", "escalate"}
    assert out.reason and len(out.reply) <= 280
    assert isinstance(out.to_dict(), dict)


def test_routing_rules():
    # account-bound intent always escalates
    r = route("where is my bill", Classification("billing", 0.99, "low", ""), 0.9)
    assert r.action == "escalate" and "account" in r.reason

    # legal cue overrides an otherwise auto intent
    r = route("internet down, calling my lawyer",
              Classification("outage", 0.99, "low", ""), 0.9)
    assert r.action == "escalate" and r.triggers[0] == "risk_cue"

    # low confidence escalates
    r = route("hmm", Classification("outage", 0.20, "low", ""), 0.9)
    assert r.action == "escalate" and "confidence" in r.reason

    # no precedent escalates
    r = route("internet down", Classification("outage", 0.95, "low", ""), 0.01)
    assert r.action == "escalate" and r.triggers[0] == "no_precedent"

    # clean auto case
    r = route("internet down", Classification("outage", 0.95, "low", ""), 0.6)
    assert r.action == "auto" and not r.triggers


def test_baselines(pieces):
    _, corpus, pool = pieces
    texts = list(pool.customer_text)
    intents = ["outage" if "down" in t else "other" for t in texts]
    actions = ["escalate" if i == "outage" else "auto" for i in intents]

    b0 = TrivialBaseline().fit(texts, intents, actions)
    assert b0(texts[0])["intent"] in {"outage", "other"}

    b1 = SimpleBaseline(corpus.customer_text, corpus.brand_text).fit(texts, intents)
    out = b1(texts[0])
    assert out["intent"] in {"outage", "other"} and out["action"] in {"auto", "escalate"}


def test_routing_metrics_asymmetry():
    y_true = ["escalate"] * 10 + ["auto"] * 10
    all_auto = ["auto"] * 20
    m = routing_metrics(y_true, all_auto)
    assert m["false_auto_rate"] == 1.0
    assert m["escalation_recall"] == 0.0
    assert m["automation_rate"] == 1.0
    assert m["cost_per_message"] > 2.0  # the trivial system is loudly punished


def test_reply_checks_flag_hallucination():
    c = reply_checks("We'll have you back online in 2 hours and credit you $25.", [])
    assert c["n_ungrounded_claims"] >= 2
    c2 = reply_checks("I can see on your account that it's active.", [])
    assert c2["fake_account_access"]


def test_judge_agreement_math():
    a = judge_agreement([3, 4, 5, 2, 3] * 8, [3, 4, 5, 2, 3] * 8)
    assert a["exact_agreement"] == 1.0
    assert a["quadratic_weighted_kappa"] == pytest.approx(1.0)


def test_parse_json_tolerates_chatty_models():
    assert parse_json('Here you go:\n```json\n{"intent": "outage"}\n```')["intent"] == "outage"
    assert parse_json('{"a": {"b": 1}} trailing prose')["a"]["b"] == 1
    assert parse_json("no json here") == {}
