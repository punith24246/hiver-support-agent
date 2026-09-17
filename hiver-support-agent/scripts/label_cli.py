"""Keyboard labelling tool. Resumes where you left off.

    python -m scripts.label_cli

Per example: press a digit for intent, then a/e for auto/escalate, then a short
reason. `s` skips, `b` goes back, `q` saves and quits.

At 200 examples this is roughly 45-70 minutes. Budget for it — the golden set is
the deliverable everything else is measured against.
"""

from __future__ import annotations

import json
import os
import sys

from support_agent.intents import INTENTS, INTENT_NAMES

IN = "data/golden/to_label.jsonl"
OUT = "data/golden/golden.jsonl"


def load(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def save(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def menu():
    print("\nIntents:")
    for i, it in enumerate(INTENTS, 1):
        print(f"  {i}. {it.name:<24} {it.description[:56]}")
    print("  s=skip  b=back  q=save&quit\n")


def main():
    if not os.path.exists(IN):
        sys.exit(f"{IN} not found — run scripts/build_golden_pool.py first")

    pool = load(IN)
    done = {r["customer_tweet_id"]: r for r in load(OUT)}
    todo = [r for r in pool if r["customer_tweet_id"] not in done]
    labelled = list(done.values())

    print(f"{len(done)} already labelled, {len(todo)} to go.")
    menu()

    i = 0
    while i < len(todo):
        row = todo[i]
        print("=" * 78)
        print(f"[{len(labelled)+1}/{len(pool)}]  arm={row['sample_arm']}"
              f"{'  (BLIND)' if row.get('blind') else ''}")
        print(f"\nCUSTOMER: {row['text']}\n")
        print(f"(what Spectrum actually replied: {row['brand_reply_actual'][:140]})")
        if row.get("prelabel_intent") and not row.get("blind"):
            print(f"\n  suggested: {row['prelabel_intent']}  [enter to accept]")

        raw = input("intent> ").strip().lower()
        if raw == "q":
            break
        if raw == "s":
            i += 1
            continue
        if raw == "b":
            if labelled:
                back = labelled.pop()
                todo.insert(i, back)
            continue
        if raw == "" and row.get("prelabel_intent") and not row.get("blind"):
            intent = row["prelabel_intent"]
        elif raw.isdigit() and 1 <= int(raw) <= len(INTENT_NAMES):
            intent = INTENT_NAMES[int(raw) - 1]
        elif raw in INTENT_NAMES:
            intent = raw
        else:
            print("  ?? try again")
            continue

        act = ""
        while act not in {"a", "e"}:
            act = input("auto(a) / escalate(e)> ").strip().lower()
        reason = input("reason (short)> ").strip()

        row.update({
            "intent": intent,
            "action": "auto" if act == "a" else "escalate",
            "action_reason": reason,
            "changed_prelabel": bool(
                row.get("prelabel_intent") and row["prelabel_intent"] != intent
            ),
        })
        labelled.append(row)
        save(labelled, OUT)
        i += 1
        if len(labelled) % 25 == 0:
            menu()

    save(labelled, OUT)
    n_pre = [r for r in labelled if r.get("prelabel_intent")]
    changed = [r for r in n_pre if r.get("changed_prelabel")]
    print(f"\nsaved {len(labelled)} -> {OUT}")
    if n_pre:
        print(f"you overrode the pre-label on {len(changed)}/{len(n_pre)} "
              f"({100*len(changed)/len(n_pre):.0f}%) — quote this in the report")


if __name__ == "__main__":
    main()
