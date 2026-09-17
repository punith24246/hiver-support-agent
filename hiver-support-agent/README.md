# Ask_Spectrum support agent

An AI support agent for **@Ask_Spectrum** built from the [Customer Support on
Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. It classifies an incoming customer tweet into one of 8 intents, drafts
a reply grounded in how Spectrum has historically handled similar tickets, and
decides whether to auto-send or hand to a human — with a stated reason.

The interesting part of this repo is not the agent. It is
[`evaluation/`](evaluation/) and the
[misleading-headline-number section](docs/report.md#5-what-is-misleading-about-my-headline-number)
of the report.

---

## Reproduce the headline numbers (< 15 min)

```bash
git clone <repo-url> && cd hiver-support-agent
pip install -r requirements.txt
export GROQ_API_KEY=...

# 1. get the data (~500MB, the only slow step)
python -m scripts.get_data

# 2. reproduce
python -m evaluation.run_eval --config configs/spectrum.yaml
```

Results land in `runs/<timestamp>/summary.md`. The golden set
(`data/golden/golden.jsonl`) and human reply scores
(`data/golden/human_scores.jsonl`) are **committed to this repo**, so you are
reproducing against exactly the labels I used.

LLM calls are cached in `runs/llm_cache.sqlite`. First run ~9 min, second ~40 s.

### No API key? No data?

```bash
AGENT_MOCK=1 pytest -q      # 11 tests, synthetic data, no network, ~2 s
```

This verifies every piece of wiring — thread reconstruction, the time-based
split, retrieval, all four routing rules, the asymmetric routing metric, the
hallucination checks, and the kappa math.

---

## Headline results

`n = <N>` held-out golden examples. Fill from `runs/<stamp>/summary.md`.

| system | intent macro-F1 | escalation recall | false-auto rate | automation rate | judge overall |
|---|---|---|---|---|---|
| B0 trivial (majority + canned) | | | | | |
| B1 simple (TF-IDF LogReg + NN reply) | | | | | |
| **agent (gpt-oss-20b + retrieval)** | | | | | |

Judge reliability vs. me, on 60 blind-scored replies: quadratic-weighted
kappa = `<k>`, within-1 agreement = `<x>`.

**Read this before the table**: macro-F1 is computed over a golden set I both
sampled and labelled, deliberately over-sampled toward rare intents. It is not
an estimate of production accuracy on Spectrum's real stream. See
[report §5](docs/report.md#5-what-is-misleading-about-my-headline-number).

---

## How it works

```
tweet ─► classify (LLM, 8-way, few-shot)      ─► intent + confidence + severity
      ─► retrieve (TF-IDF over past customer messages, k=4)
                                               ─► 4 analogous resolved threads
      ─► draft    (LLM, grounded in those threads, ≤280 chars)
      ─► route    (deterministic rules over model signals)
                                               ─► auto | escalate + reason
```

**Routing is rules, not an LLM call.** Four independent triggers, any one fires
→ human: a risk phrase (legal/medical/fraud), an intent that cannot be resolved
without account access, classifier confidence below threshold, or no similar
precedent in the corpus. A stakeholder can read and change this policy without
touching a prompt.

**The split is by time, not random.** The grounding corpus is the older 75% of
threads; everything is evaluated on the newer 25%. A random split lets the
retriever surface a reply written to the same customer about the same outage,
which inflates every reply-quality number.

## Layout

```
support_agent/   intents.py  data.py  retrieve.py  agent.py  baselines.py  llm.py
evaluation/      run_eval.py  metrics.py  judge.py
scripts/         get_data.py  build_golden_pool.py  label_cli.py  score_replies.py
docs/            report.md  decision_log.md
tests/           test_smoke.py
```

## Rebuilding the golden set from scratch

```bash
python -m scripts.build_golden_pool --n 220 --blind 50 --prelabel
python -m scripts.label_cli                              # ~60 min by hand
python -m evaluation.run_eval --config configs/spectrum.yaml
python -m scripts.score_replies --run runs/<stamp> --n 60 # ~20 min by hand
python -m evaluation.run_eval --config configs/spectrum.yaml  # now with kappa
```

## Attribution

- Dataset: Thought Vector, *Customer Support on Twitter* (Kaggle), CC BY-NC-SA 4.0.
- Models: `openai/gpt-oss-20b` (generator) and `openai/gpt-oss-120b` (judge) via the Groq API.
- Libraries: scikit-learn (TF-IDF, logistic regression, Cohen's kappa), pandas, openai-python.
- LLM-as-judge design follows the now-standard blinded absolute-rubric setup
  (Zheng et al., *Judging LLM-as-a-Judge with MT-Bench*, 2023); the rubric text,
  the safety-as-floor aggregation, and the agreement protocol are mine.
- Claude and Codex were used as coding assistants throughout. Every design
  decision in `docs/decision_log.md` is one I can defend and change live.
