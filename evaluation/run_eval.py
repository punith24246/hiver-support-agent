"""Headline results. Run from the repo root:

    python -m evaluation.run_eval --config configs/spectrum.yaml

Produces runs/<stamp>/{predictions.jsonl, metrics.json, summary.md}.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml

from support_agent.data import load_brand
from support_agent.llm import LLM
from support_agent.retrieve import ReplyRetriever, evidence_block, Evidence
from support_agent.agent import SupportAgent
from support_agent.baselines import TrivialBaseline, SimpleBaseline
from support_agent.intents import INTENT_NAMES
from evaluation.judge import judge_reply
from evaluation.metrics import (
    intent_metrics, routing_metrics, reply_checks, judge_agreement,
)


def load_golden(path: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    required = {"customer_tweet_id", "text", "intent", "action"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"golden set missing columns: {missing}")
    bad = set(df.intent) - set(INTENT_NAMES)
    if bad:
        raise ValueError(f"golden set has unknown intents: {bad}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/spectrum.yaml")
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N golden examples (smoke test)")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    outdir = os.path.join("runs", stamp)
    os.makedirs(outdir, exist_ok=True)

    print("[1/6] loading brand corpus ...")
    corpus, _pool = load_brand(
        cfg["data"]["csv_path"], cfg["brand"], cfg["data"]["holdout_frac"]
    )
    print(f"      grounding corpus: {len(corpus):,} historical pairs")

    golden = load_golden(cfg["eval"]["golden_path"])
    if args.limit:
        golden = golden.head(args.limit)
    print(f"[2/6] golden set: {len(golden)} labelled examples")

    retriever = ReplyRetriever(
        k=cfg["retrieval"]["k"], min_score=cfg["retrieval"]["min_score"]
    ).fit(corpus.customer_text, corpus.brand_text)

    gen = LLM(cfg["models"]["generator"], temperature=0.0, mock=cfg.get("mock", False))
    jud = LLM(cfg["models"]["judge"], temperature=0.0, mock=cfg.get("mock", False))
    agent = SupportAgent(gen, retriever, cfg["routing"])

    # --- baselines are fit on the golden set with a held-out split ----------
    # The simple baseline needs labels, and the golden set is the only labelled
    # data we have. We fit B1 on the first half and evaluate ALL systems on the
    # second half so the comparison is fair. Stated in the report. D14.
    half = len(golden) // 2
    fit_df, test_df = golden.iloc[:half], golden.iloc[half:]

    b0 = TrivialBaseline().fit(fit_df.text, fit_df.intent, fit_df.action)
    b1 = SimpleBaseline(corpus.customer_text, corpus.brand_text).fit(
        fit_df.text, fit_df.intent
    )

    print(f"[3/6] running agent on {len(test_df)} held-out examples ...")
    with ThreadPoolExecutor(max_workers=cfg.get("workers", 1)) as ex:
        agent_out = list(ex.map(agent, test_df.text.tolist()))

    systems = {
        "B0_trivial": [b0(t) for t in test_df.text],
        "B1_simple": [b1(t) for t in test_df.text],
        "agent": [o.to_dict() for o in agent_out],
    }

    print("[4/6] scoring intent + routing ...")
    results = {}
    for name, preds in systems.items():
        results[name] = {
            "intent": intent_metrics(
                test_df.intent.tolist(), [p["intent"] for p in preds],
                labels=INTENT_NAMES,
            ),
            "routing": routing_metrics(
                test_df.action.tolist(), [p["action"] for p in preds]
            ),
        }

    # --- judge -------------------------------------------------------------
    if not args.no_judge:
        print("[5/6] judging replies ...")
        ev_blocks = [
            evidence_block([Evidence(**e) for e in o.evidence]) for o in agent_out
        ]

        def _judge_batch(name, preds):
            def one(i):
                return judge_reply(jud, test_df.text.iloc[i], preds[i]["reply"], ev_blocks[i])
            with ThreadPoolExecutor(max_workers=cfg.get("workers", 1)) as ex:
                return list(ex.map(one, range(len(preds))))

        for name, preds in systems.items():
            scores = _judge_batch(name, preds)
            
            for p, s in zip(preds, scores):
                p["judge"] = s
            axes = ("groundedness", "helpfulness", "tone", "safety", "overall")
            results[name]["judge"] = {
                f"mean_{a}": float(sum(s[a] for s in scores) / len(scores)) for a in axes
            }
            checks = [
                reply_checks(p["reply"], [e["brand_text"] for e in p.get("evidence", [])])
                for p in preds
            ]
            results[name]["reply_checks"] = {
                "pct_over_280": 100 * sum(not c["length_ok"] for c in checks) / len(checks),
                "pct_with_ungrounded_claim":
                    100 * sum(c["n_ungrounded_claims"] > 0 for c in checks) / len(checks),
                "pct_fake_account_access":
                    100 * sum(c["fake_account_access"] for c in checks) / len(checks),
            }

        # judge vs human agreement, on whatever subset you hand-scored
        hpath = cfg["eval"].get("human_scores_path")
        if hpath and os.path.exists(hpath):
            hs = pd.read_json(hpath, lines=True)
            merged = test_df.reset_index(drop=True).assign(
                judge_overall=[p["judge"]["overall"] for p in systems["agent"]],
                reply=[p["reply"] for p in systems["agent"]],
            ).merge(hs[["customer_tweet_id", "human_overall"]], on="customer_tweet_id")
            if len(merged) >= 20:
                results["judge_agreement"] = judge_agreement(
                    merged.human_overall.tolist(), merged.judge_overall.tolist()
                )
                print(f"      judge vs human on {len(merged)} replies: "
                      f"QWK={results['judge_agreement']['quadratic_weighted_kappa']:.2f}")
            else:
                print(f"      only {len(merged)} human-scored replies matched — "
                      "need >=20 for a meaningful kappa")
        else:
            print("      no human scores found; skipping agreement "
                  "(run scripts/score_replies.py)")

    print("[6/6] writing artefacts ...")
    with open(os.path.join(outdir, "predictions.jsonl"), "w") as f:
        for i in range(len(test_df)):
            f.write(json.dumps({
                "customer_tweet_id": int(test_df.customer_tweet_id.iloc[i]),
                "text": test_df.text.iloc[i],
                "gold_intent": test_df.intent.iloc[i],
                "gold_action": test_df.action.iloc[i],
                **{name: preds[i] for name, preds in systems.items()},
            }, default=str) + "\n")

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    write_summary(os.path.join(outdir, "summary.md"), results, len(test_df))
    print(f"\nDone -> {outdir}/summary.md")
    print(open(os.path.join(outdir, "summary.md")).read())


def write_summary(path, results, n):
    lines = [f"# Results (n = {n} held-out golden examples)\n",
             "| system | intent macro-F1 | escalation recall | false-auto rate | "
             "automation rate | judge overall |",
             "|---|---|---|---|---|---|"]
    for name in ("B0_trivial", "B1_simple", "agent"):
        r = results.get(name, {})
        i, rt = r.get("intent", {}), r.get("routing", {})
        j = r.get("judge", {})
        lines.append(
            f"| {name} | {i.get('macro_f1', 0):.3f} | "
            f"{rt.get('escalation_recall', 0):.3f} | "
            f"{rt.get('false_auto_rate', 0):.3f} | "
            f"{rt.get('automation_rate', 0):.3f} | "
            f"{j.get('mean_overall', float('nan')):.2f} |"
        )
    if "judge_agreement" in results:
        a = results["judge_agreement"]
        lines += ["", "## Judge reliability",
                  f"- n = {a['n']} replies scored by both human and judge",
                  f"- exact agreement: {a['exact_agreement']:.2f}",
                  f"- within-1 agreement: {a['within_1_agreement']:.2f}",
                  f"- quadratic-weighted kappa: {a['quadratic_weighted_kappa']:.2f}",
                  f"- judge mean minus human mean: {a['judge_minus_human_mean']:+.2f}"]
    open(path, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
