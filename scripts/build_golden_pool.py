"""Build the pool you will hand-label.

    python -m scripts.build_golden_pool --config configs/spectrum.yaml --n 220

Sampling strategy (write this into the report verbatim):

  * 60% stratified by weak keyword strata, so the rare-but-expensive intents
    (cancellation, appointment, account_access) are not sampled out of
    existence. Pure random over Spectrum's stream is ~45% connectivity
    complaints and you end up with 4 cancellation examples and a macro-F1 that
    swings +/-0.15 on one label flip.
  * 40% pure random, untouched, so we can still say something about the real
    distribution and can reweight the stratified numbers back if we want to.
  * Blind control arm: 50 examples are held out from pre-labelling entirely.
    Labelling them cold and comparing against the pre-labelled ones is how we
    measure how much the pre-labeller anchored us. D10.

Output: data/golden/to_label.jsonl with `intent` and `action` left blank
(or pre-filled + flagged), ready for scripts/label_cli.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random

import pandas as pd
import yaml

from support_agent.data import load_brand
from support_agent.llm import LLM
from support_agent.agent import classify

# Weak, deliberately imperfect keyword strata. These are for SAMPLING ONLY —
# they never touch the classifier, and using them as labels would be circular.
STRATA = {
    "outage": ["outage", "no internet", "down", "no service", "no signal"],
    "connectivity_degraded": ["slow", "buffering", "lag", "dropping", "speeds", "pixel"],
    "billing": ["bill", "charge", "payment", "refund", "price", "autopay", "fee"],
    "account_access": ["password", "log in", "login", "locked", "username", "sign in"],
    "equipment": ["modem", "router", "cable box", "remote", "equipment", "dvr"],
    "appointment": ["technician", "tech ", "appointment", "install", "no show", "window"],
    "channel_content": ["channel", "game", "espn", "on demand", "programming", "lineup"],
    "cancellation": ["cancel", "switching", "fios", "at&t", "leaving", "done with"],
}


def stratum_of(text: str) -> str | None:
    low = text.lower()
    for name, kws in STRATA.items():
        if any(k in low for k in kws):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spectrum.yaml")
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--blind", type=int, default=50,
                    help="examples held out from pre-labelling")
    ap.add_argument("--prelabel", action="store_true",
                    help="pre-fill intents with the judge model to speed up labelling")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    random.seed(args.seed)

    _corpus, pool = load_brand(
        cfg["data"]["csv_path"], cfg["brand"], cfg["data"]["holdout_frac"]
    )
    print(f"eval pool (time-held-out): {len(pool):,} messages")

    pool = pool.copy()
    pool["stratum"] = pool.customer_text.map(stratum_of)

    n_strat = int(args.n * 0.6)
    per = max(1, n_strat // len(STRATA))
    picks = []
    for name in STRATA:
        sub = pool[pool.stratum == name]
        take = min(per, len(sub))
        if take:
            picks.append(sub.sample(take, random_state=args.seed))
    strat = pd.concat(picks) if picks else pool.head(0)

    rest = pool.drop(index=strat.index, errors="ignore")
    rand = rest.sample(min(args.n - len(strat), len(rest)), random_state=args.seed)

    sel = pd.concat([strat, rand]).sample(frac=1, random_state=args.seed)
    sel = sel.reset_index(drop=True)
    sel["sample_arm"] = ["stratified"] * len(strat) + ["random"] * len(rand)
    sel = sel.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    blind_idx = set(sel.index[: args.blind])

    prelabels = {}
    if args.prelabel:
        # Pre-label with the LARGER model, never the one being evaluated.
        llm = LLM(cfg["models"]["judge"], temperature=0.0, mock=cfg.get("mock", False))
        print(f"pre-labelling {len(sel) - len(blind_idx)} examples with "
              f"{cfg['models']['judge']} ...")
        for i, row in sel.iterrows():
            if i in blind_idx:
                continue
            prelabels[i] = classify(llm, row.customer_text).intent

    os.makedirs("data/golden", exist_ok=True)
    out = "data/golden/to_label.jsonl"
    with open(out, "w") as f:
        for i, row in sel.iterrows():
            f.write(json.dumps({
                "customer_tweet_id": int(row.customer_tweet_id),
                "text": row.customer_text,
                "brand_reply_actual": row.brand_text,
                "sample_arm": row.sample_arm,
                "blind": i in blind_idx,
                "prelabel_intent": prelabels.get(i),
                "intent": None,
                "action": None,
                "action_reason": None,
            }) + "\n")

    print(f"\nwrote {len(sel)} examples -> {out}")
    print(f"  stratified: {len(strat)}   random: {len(rand)}   blind arm: {len(blind_idx)}")
    print("\nNext: python -m scripts.label_cli")


if __name__ == "__main__":
    main()
