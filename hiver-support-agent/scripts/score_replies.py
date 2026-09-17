"""Hand-score agent replies so we can measure whether the LLM judge agrees.

    python -m scripts.score_replies --run runs/<stamp> --n 60

Shows you the customer message and the reply, WITHOUT the judge's score, and
asks for 1-5. Blinding matters: if you see the judge's number first you will
anchor on it and the agreement figure becomes meaningless.

60 replies is about 20 minutes and is enough for a stable quadratic-weighted
kappa. Below ~40 the confidence interval is wide enough to be useless.
"""

from __future__ import annotations

import argparse
import json
import os
import random

RUBRIC = """
  5  ship it as-is
  4  minor edit needed
  3  generic but not wrong
  2  would annoy the customer or needs a real rewrite
  1  wrong, unsafe, or invents facts
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="data/golden/human_scores.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    preds = [json.loads(l) for l in open(os.path.join(args.run, "predictions.jsonl"))]
    random.seed(args.seed)
    sample = random.sample(preds, min(args.n, len(preds)))

    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["customer_tweet_id"] for l in open(args.out)}

    print(f"Scoring agent replies 1-5. {RUBRIC}")
    with open(args.out, "a") as f:
        for i, p in enumerate(sample, 1):
            if p["customer_tweet_id"] in done:
                continue
            print("=" * 78)
            print(f"[{i}/{len(sample)}]\nCUSTOMER: {p['text']}\n")
            print(f"AGENT REPLY: {p['agent']['reply']}\n")
            s = ""
            while s not in {"1", "2", "3", "4", "5", "q"}:
                s = input("score (1-5, q=quit)> ").strip()
            if s == "q":
                break
            f.write(json.dumps({
                "customer_tweet_id": p["customer_tweet_id"],
                "human_overall": int(s),
            }) + "\n")
            f.flush()

    print(f"\nsaved -> {args.out}. Re-run evaluation.run_eval to get the kappa.")


if __name__ == "__main__":
    main()
