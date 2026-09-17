# Decision log

Non-obvious calls, and what I traded away for each.

**D1 — Brand: Ask_Spectrum, not SpotifyCares or XboxSupport.**
I sampled ~100 inbound tweets from each. Spotify's stream is dominated by
playlist and app-bug chatter with no clean auto/escalate boundary; Xbox is
heavily gamertag- and account-bound, so nearly everything escalates and the
routing task becomes trivial. Spectrum has a genuine mix: outages and
troubleshooting are safely auto-handleable, billing and cancellation are not.
That split is the whole point of deliverable 3.

**D2 — 8 intents, cut where the brand's ACTION changes, not where the
customer's words change.** "no internet since 6am" and "wifi is out" are one
ticket. "wifi is slow" is a different one, because it gets a troubleshooting
script instead of an outage lookup. I rejected a 15-intent taxonomy after
labelling 40 examples with it and finding I could not apply it consistently to
myself.

**D3 — Did not use Banking77.** It is a different domain (retail banking) with a
different label granularity. Transferring its taxonomy would have given me 77
labels that mostly do not occur in Spectrum's stream and a classifier tuned for
the wrong distribution. Cost: no external validation of my taxonomy.

**D4 — Minimal text normalisation.** No lowercasing, no punctuation stripping,
no emoji removal. ALL-CAPS and "!!!" are escalation signal, and emoji carry
sentiment. I only strip URLs, @handles, and the dataset's numeric pseudo-IDs.

**D5 — Only the first customer message of a thread is a training or eval
example.** A mid-thread "ok DMing you now" has no standalone intent. Including
them inflates the `other` class and teaches the retriever to surface
conversational filler.

**D6 — Time-based split, not random.** Grounding corpus = oldest 75%, eval pool
= newest 25%. A random split lets the retriever find a reply written to the same
customer about the same outage the same afternoon. I measured this: under a
random split the top-1 retrieval similarity was visibly higher and judge
groundedness rose, which is leakage, not quality. This is the single most
load-bearing guard in the repo.

**D7 — Routing is deterministic rules over model signals, not an LLM decision.**
An LLM asked "should this escalate?" gives a plausible answer with no stable
threshold you can tune, and its reason is post-hoc. My four triggers are
readable, individually testable (see `tests/test_smoke.py::test_routing_rules`),
and a support manager can change `conf_threshold` without touching a prompt.
Cost: the rules cannot catch a novel escalation-worthy situation that does not
match a cue or an intent.

**D8 — Retrieval matches customer-message to customer-message, not
customer-to-reply.** The two sides of a support thread share almost no
vocabulary. Matching on the customer side finds analogous tickets; matching
against replies just finds whichever canned response has the most common words.

**D9 — Few-shot examples are drawn only from the older (corpus) slice, one per
intent, chosen as prototypical rather than hard.** Prototypes taken from the
eval slice would be direct leakage. Cost: the classifier has seen no hard cases,
which shows up in the boundary failures in report §4.

**D10 — The golden set is 60% keyword-stratified, 40% pure random, with a
50-example blind arm.** Pure random over Spectrum's stream is ~45%
connectivity complaints; you end up with 4 cancellation examples and a macro-F1
that swings ±0.15 on one label flip. The strata are weak keyword rules used for
*sampling only* — they never touch the classifier, because that would be
circular. The 40% random arm preserves a view of the true distribution. The
blind arm is labelled with no model suggestion visible, which is how I measure
how much the pre-labeller anchored me (report §5).

**D11 — High-severity outage complaints escalate even though `outage` is an
auto-handleable intent.** A furious customer on a *public* channel is exactly
the case where a correct canned reply goes viral. The trigger costs automation
rate and I think it is worth it; it is one config line to remove.

**D12 — Routing is scored with separate false-auto and false-escalate rates plus
a cost-weighted score, never accuracy or F1.** The errors are not symmetric: a
false-auto puts an unsupervised reply in front of an angry customer; a
false-escalate costs a human one minute. I weight false-auto 5× and I say so in
the metric output, because the weight is a choice I made, not a measurement.

**D13 — Judge is `gpt-oss-120b`, generator is `gpt-oss-20b`; blinded, absolute
rubric, one reply at a time.** Blinded so the judge cannot favour "the agent".
Absolute rather than pairwise so scores stay comparable across systems without
re-running every comparison. I recompute `overall` myself with safety as a floor
rather than an average term, because a 1 on safety must not be rescued by a 5 on
tone. **Unsolved:** both models are the same family, so some shared-style bias
almost certainly remains. My kappa against a human is the honest bound on how
much to trust the judge, and it is in the README.

**D14 — The simple baseline is fit on the first half of the golden set and all
three systems are evaluated on the second half.** B1 needs labels and the golden
set is the only labelled data I have. Evaluating the LLM agent on the full
golden set while B1 only saw half would be an unfair comparison in my own
favour. Cost: the headline `n` is halved, which widens every confidence
interval.

**D15 — No fine-tuning, no embeddings, no agent framework.** With one day and
~200 labels, a fine-tune would overfit and I could not have shown you an honest
held-out number. TF-IDF is defensible on short keyword-dominated text and runs
on a Colab CPU inside the time budget. Embedding retrieval is the first thing I
would try with another week — listed in report §6, not claimed here.
