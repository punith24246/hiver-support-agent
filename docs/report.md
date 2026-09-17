# Ask_Spectrum support agent — report

> Slots marked `<<<>>>` are filled from `runs/<stamp>/summary.md` and
> `metrics.json` after the final run. Everything else is written.

---

## 1. Problem framing

### What "good" means for Spectrum specifically

Spectrum's inbound Twitter stream is not a help desk queue. It is a **public**
channel where most messages are complaints, a large fraction are furious, and
the brand's own historical behaviour is overwhelmingly *acknowledge and move to
DM*. That shapes the target:

- **Good is not "resolve the ticket."** Almost no Spectrum ticket can be
  resolved in a public tweet — resolution needs account access. Good is:
  acknowledge correctly, give the one piece of help that does not require an
  account, and move the conversation to the right place.
- **The expensive error is a confident wrong auto-reply**, not a missed
  automation. A needless escalation costs an agent sixty seconds. A canned
  "sorry for the inconvenience, please DM us" sent to someone saying their
  medical equipment is offline is a screenshot on Twitter.
- So the objective is **maximum automation rate subject to a false-auto rate
  near zero**, not accuracy.

### What I chose not to build

- **No multi-turn dialogue.** The agent handles first contact only. Multi-turn
  needs conversation state, and I would rather have a defensible eval of one
  turn than an undefended eval of five.
- **No account/CRM integration**, so anything needing account data escalates by
  construction. This caps the achievable automation rate at roughly the share of
  non-account intents — a real ceiling I state rather than hide.
- **No fine-tuning.** ~200 labels; a fine-tune would overfit and I could not
  have shown an honest held-out number.
- **No sentiment model.** Severity comes from the classifier call. A separate
  sentiment head was not worth the extra evaluation surface.
- **No auto-send.** "Auto" means *safe to send without human review*. Nothing in
  this repo actually posts.

### The pipeline

Classify (8-way, few-shot LLM) → retrieve (TF-IDF over historical customer
messages, k=4) → draft (LLM grounded in the retrieved threads) → route
(deterministic rules over model signals). Routing detail and rationale in
`decision_log.md` D7.

---

## 2. Data and the golden set

- Brand: `Ask_Spectrum`. `<<<N>>>` reconstructed (customer message → brand
  reply) pairs after keeping first-contact messages only.
- Split **by time**: oldest 75% is the grounding corpus, newest 25% is the
  evaluation pool. Rationale in D6 — this is the guard that stops the retriever
  from finding a reply to the same customer about the same outage.
- Golden set: `<<<N>>>` examples, hand-labelled by me for intent, auto/escalate,
  and a free-text reason. Sampling: 60% keyword-stratified (so
  `cancellation`/`appointment` are not sampled out of existence), 40% pure
  random, 50 held out from pre-labelling as a blind arm.
- Pre-labelling: non-blind examples were pre-filled by `gpt-oss-120b` and I
  corrected every one. I overrode the pre-label on `<<<X>>>%` of them.
- Labelling time: `<<<~T>>>`. Cases I found genuinely hard are in §4.

---

## 3. Results

`n = <<<N>>>` held-out golden examples (second half; the first half fits B1 —
D14).

| system | intent macro-F1 | escalation recall | false-auto rate | automation rate | judge overall |
|---|---|---|---|---|---|
| B0 trivial — majority intent, canned reply, escalate nothing | | | | | |
| B1 simple — TF-IDF + LogReg, nearest-neighbour reply, keyword routing | | | | | |
| **agent** | | | | | |

**Judge reliability.** I blind-scored `<<<60>>>` agent replies 1–5 without
seeing the judge's number. Quadratic-weighted kappa `<<<k>>>`, within-1
agreement `<<<x>>>`, judge mean minus human mean `<<<b>>>`.
`<<<One sentence: is the judge usable, and where does it disagree with me —
typically it is more generous about generic replies.>>>`

**Deterministic reply checks** (things the judge is bad at catching):
replies over 280 chars `<<<>>>%`, replies containing an ungrounded specific
number/price/ETA `<<<>>>%`, replies falsely claiming account access `<<<>>>%`.

**Reading the table.** `<<<Two or three sentences. Where does the agent actually
beat B1 — probably reply quality and intents with little lexical signal — and
where does B1 hold up embarrassingly well? Say so if it does.>>>`

---

## 4. Failure analysis

Top 5, each with a real example from `runs/<stamp>/predictions.jsonl`.

**F1 — `<<<name>>>.** Example: `<<<tweet text>>>` → predicted `<<<>>>`, gold
`<<<>>>`. Hypothesis: `<<<>>>`. Frequency in the held-out set: `<<<n>>>`.

**F2 — `<<<>>>.** …

**F3 — `<<<>>>.** …

**F4 — `<<<>>>.** …

**F5 — `<<<>>>.** …

> Likely candidates from my labelling notes, to confirm against the run:
> (a) `billing` vs `cancellation` when the customer is angry about a price
> increase and *mentions* leaving — genuinely ambiguous, and I was inconsistent
> on it myself; (b) `outage` vs `connectivity_degraded` when the customer says
> "internet keeps going out", which is intermittent, not down; (c) the drafter
> inventing an ETA when retrieved evidence happens to contain one from an
> unrelated 2017 outage; (d) sarcasm ("great service as always 👏") landing in
> `other`; (e) low-signal one-liners ("fix it") where confidence is correctly
> low and the message escalates — a *correct* escalation that still looks like a
> classifier failure in the confusion matrix.

---

## 5. What is misleading about my headline number

Five things, in descending order of how much they should worry you.

**1. I built the golden set, so the golden set is shaped like my assumptions.**
I wrote the taxonomy, sampled the examples, and applied the labels. When the
classifier and I disagree, "the model is wrong" and "my label is wrong" are not
distinguishable from the inside. The 50-example blind arm is a partial check —
`<<<agreement between blind and pre-labelled arms>>>` — but there is no second
annotator, so I have **no inter-annotator agreement number at all**. Every
macro-F1 in §3 should be read as agreement-with-me, not accuracy.

**2. The golden set is deliberately not the real distribution.** 60% of it is
keyword-stratified toward rare intents. Macro-F1 on it is therefore *not* an
estimate of production performance on Spectrum's actual stream, which is roughly
`<<<x>>>%` connectivity complaints. The stratified sample makes rare-class
numbers stable; it also makes the headline optimistic about how the agent
handles a stream where one class dominates. The 40% random arm is the number to
look at for realism: `<<<macro-F1 on the random arm only>>>`.

**3. Pre-labelling anchored me toward the model.** Non-blind examples were
pre-filled by `gpt-oss-120b` and I corrected them. I overrode `<<<X>>>%`, which
means I accepted `<<<100-X>>>%` — and some of those I accepted because they were
right, and some because accepting is easier than disagreeing. The evaluated
model is a *different, smaller* model from the same family, which reduces this
but does not eliminate it.

**4. The judge shares a model family with the generator.** Blinding and an
absolute rubric handle the crude failure modes. They do not handle a judge
rewarding phrasing that its own family produces. My kappa of `<<<k>>>` against a
human is the honest upper bound on trusting the judge's numbers — and it is
computed on 60 replies I scored, which is enough for a point estimate and not
enough for a tight interval.

**5. The automation rate is capped by a design choice, not by model quality.**
Four intents escalate by construction because they need account access. A
headline automation rate of `<<<x>>>%` mostly reflects the intent mix of my
golden set, not how good the model is. Change the sample and that number moves
without the model changing at all.

One more, smaller: `n = <<<N>>>` after the B1 fitting split. At that size the
difference between the agent and B1 on macro-F1 is `<<<Δ>>>`, which is
`<<<inside / outside>>>` what I would expect from label noise alone.

---

## 6. With one more week

In the order I would actually do it — evaluation first, because the eval is what
is weakest:

1. **A second annotator on 100 examples** to get a real inter-annotator
   agreement figure. Without it, every number in §3 has an unknown floor. This
   is the single highest-value day.
2. **A judge from a different family** (e.g. a Llama or Qwen model on the same
   API), scoring the same 60 replies, so I can report cross-family agreement
   rather than just human-vs-one-judge.
3. **Bootstrap confidence intervals** on macro-F1 and the false-auto rate. Point
   estimates at `n = <<<N>>>` are being over-read, including by me.
4. **Embedding retrieval** (`bge-small` or similar) compared head-to-head
   against TF-IDF on the same golden set. I expect a gain on paraphrased
   complaints with no keyword overlap; I have not measured it and so do not
   claim it.
5. **Tune the routing thresholds against the cost curve** instead of picking
   0.65/0.18 by hand. Sweep both, plot automation rate against false-auto rate,
   and let a stakeholder pick the operating point.
6. **Multi-turn**, last. It is the biggest feature and the least defensible
   without the eval work above.
