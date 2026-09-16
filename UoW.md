# Unit of Work — RedRock Email Classifier

## Problem statement (in my own words)

RedRock, a financial services company, needs incoming client emails routed to
the right internal department automatically. Misrouting matters here in a way
it wouldn't for a generic inbox: this is a regulated industry, and a claim
routed to the wrong team, or a compliance-sensitive email that never reaches
anyone, can cause real delay and regulatory exposure. I was given 44 labeled
sample emails (`train/` + `train_labels.csv`) and 12 unlabeled ones (`test/`)
and asked to build a classifier over five categories — Account Management,
Investment Advisory, Loan Processing, Insurance Claims, Other — that outputs
both a predicted category and a confidence score, in a way that runs
end-to-end from raw files to a results CSV using only open-source tooling.

## Lab book

### Data first

Before picking a model I looked at what I actually had: 44 labeled emails
across 5 classes (13 / 12 / 7 / 6 / 6 — mildly imbalanced but not pathological),
and 12 unlabeled test emails. Each `.html` file is a rendered "email client"
wrapper around a `data-field`-tagged metadata block and a *nested*, independently
parseable HTML fragment holding the actual message. `src/ingestion.py` pulls
subject/sender/date out of the `data-field` markers and the message text out
of the nested `<body>`, rather than treating the whole file as flat text —
otherwise CSS, a duplicated `<title>`, and layout markup pollute the input.

**Assumption flagged:** the on-disk filename (`email_10.html`) is *not* a
stable identifier — train and test each restart numbering at 1, so
`train/email_1.html` and `test/email_1.html` would collide if I used the
filename as `email_id`. Each file also carries an internal
`data-field="email_id"` value, and I verified it is unique across all 56
files combined (1–56, no collisions). I used that internal field as the
`email_id` column in the output, not the filename, since it's the only one of
the two that is actually a valid identifier. This is exactly the kind of
spec ambiguity worth documenting rather than guessing at silently.

### Choosing the architecture

The original plan going into this was: LLM API as the primary classifier
(zero/few-shot, with the model's own confidence or an ensemble-agreement
score), with a classical TF-IDF baseline for comparison. Two things in the
actual environment changed that:

1. **No API key was available** in the environment this has to run in, and
   the spec is explicit that the solution "must be written using only open
   source tools... so we can run your solution and check your results." A
   hosted LLM call is a hard external dependency a reviewer can't satisfy
   without their own paid key — that's a real risk to "does this even run
   for the grader," not a stylistic preference.
2. **44 labeled examples is not enough to fine-tune anything.** Whatever the
   primary approach was, it needed to work well with roughly 9 examples per
   class.

So the primary approach became a **frozen sentence-embedding encoder
(`all-MiniLM-L6-v2`) + nearest-centroid classifier**: embed each labeled
email, average the embeddings per class into a centroid, and classify a new
email by cosine similarity to each centroid (softmax over similarities gives
the confidence score). This needs no training beyond averaging a handful of
vectors, captures *semantic* similarity that a bag-of-words model can't
(paraphrase, synonymy — e.g. "transfer my account" vs "move my funds to
another institution" should land near each other), and runs fully offline
after a single ~90MB model download. It's the closest offline substitute for
"LLM-based understanding" that doesn't require an API key, and it directly
answers the spec's explicit allowance for "LLMs or other NLP techniques."

**TF-IDF + Logistic Regression** stayed in as the baseline it was always
meant to be: the naive benchmark the smarter approach has to beat, and a
useful sanity check on whether the embedding classifier is earning its
extra complexity.

### What I'd have built if an LLM API were viable

Worth being explicit about the road not taken: with an API key available, a
few-shot LLM classifier (one example per category in the prompt, structured
JSON output for category + reasoning + confidence) would likely have handled
ambiguous/boundary-case emails better than either approach here, since it can
reason about content rather than just measuring similarity to a handful of
labeled examples. I'd treat it as a natural "phase 2" — see Future Work.

### Confidence scores

- **Baseline (TF-IDF + LogReg):** `predict_proba` output for the winning class.
- **Embedding classifier:** softmax over cosine similarities to each class
  centroid (temperature=0.05, chosen empirically so the softmax is peaked
  enough to be informative without being a near-one-hot).

Both are bounded in [0, 1] by construction (tested explicitly — see
`tests/test_classifier_baseline.py::test_confidence_score_within_bounds` and
the embedding equivalent).

### Evaluation methodology

44 labeled examples is too few for a single held-out validation split to be
trustworthy — a 20% split leaves ~1-2 examples per class in validation. I
used **5-fold stratified cross-validation** over the labeled set instead
(`src/evaluate.py::cross_validate`), which uses every labeled email as a
test example exactly once, and reported accuracy, macro-F1, per-class
precision/recall, and a confidence-calibration table (accuracy within each
confidence bucket).

### Results

Full numbers in `results/evaluation_results.json` and `results/comparison.csv`.
All figures are 5-fold stratified cross-validation on the 44 labeled train
emails (every labeled email scored exactly once, out-of-fold).

| Model | Accuracy | Macro-F1 | CV wall time |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.909 | 0.864 | 0.1s |
| **Sentence-embedding + nearest-centroid (primary)** | **0.932** | **0.922** | 98s |

The embedding classifier wins on both accuracy and macro-F1, and by a wider
margin on macro-F1 — meaning its advantage is concentrated in the classes
the baseline handles worst (see below), not spread evenly. The cost is
~1000x the wall-clock time (dominated by loading/encoding with the
transformer model, not by anything that scales with data size) — irrelevant
at this data volume, worth flagging if this were ever processing high
email throughput in real time.

**Per-class breakdown** — both models are near-perfect on Insurance Claims,
Investment Advisory and Loan Processing (F1 0.92–1.00), which have distinct,
consistent vocabulary ("claim", "premium" vs "portfolio", "fund" vs "loan",
"repayment"). Both are noticeably weaker on **Other**:

| Model | "Other" precision | "Other" recall | "Other" F1 |
|---|---|---|---|
| Baseline | 1.00 | 0.33 | 0.50 |
| Embedding | 0.80 | 0.67 | 0.73 |

This is expected — "Other" is a catch-all with no coherent internal theme,
so both a bag-of-words model and a semantic-similarity model struggle to
build a meaningful class "centroid" or set of discriminating tokens for it.
The embedding classifier's better recall here is a real finding, not noise:
it's genuinely capturing semantic distance from the other four (more
coherent) classes, which a token-overlap model can't.

### Calibration finding

This is the more interesting result. The **baseline is under-confident**:
every single out-of-fold prediction landed below 0.7 confidence (the
five-way softmax spreads probability mass thin with regularized logistic
regression over ~9 examples/class), yet the [0, 0.5) confidence bucket was
still ~89% accurate. The model is right far more often than its own
confidence score would suggest — not dangerous by itself, but it means the
raw confidence number is close to useless as a routing signal (a human
reviewing "send anything under 70% confidence to a person" would end up
reviewing almost everything).

The **embedding classifier's confidence is well-calibrated and monotonic**:

| Confidence bucket | n | Empirical accuracy |
|---|---|---|
| [0.50, 0.70) | 7 | 57% |
| [0.70, 0.85) | 3 | 100% |
| [0.85, 0.95) | 7 | 100% |
| [0.95, 1.00] | 27 | 100% |

Accuracy rises with confidence, and the lowest bucket is exactly where the
model's mistakes concentrate. That makes it a usable signal in production:
"route anything under, say, 0.7 confidence to a human" would be a
defensible policy for this model, and would not be for the baseline. This
is the strongest practical argument in this report for the embedding
approach over the classical one — not just that it's more accurate, but
that its confidence score means something.

## Outcomes and results

The embedding-based classifier (`results/predictions.csv`,
`results/predictions_embedding_centroid.csv`) is the primary submission:
5-fold CV accuracy 0.932 / macro-F1 0.922 on the labeled data, with
confidence scores that are empirically well-calibrated (see table above).
The TF-IDF baseline (`results/predictions_baseline_tfidf_logreg.csv`) is
included for comparison and scored 0.909 / 0.864 — a reasonable model in
its own right, but with confidence scores that don't track its actual
accuracy, which would make it a weaker choice for a production routing
policy that leans on the confidence score to decide what needs human review.
Both classifiers' weakest class is "Other," for the structural reason
described above, not a bug.

## How I'd measure success in production, and what would improve accuracy

**Measuring success**, given the regulatory stakes of misrouting:
- Track precision *per class*, not just overall accuracy — a false positive
  that routes a claim to "Other" is a worse failure than confusing two
  adjacent advisory categories, so the cost of an error should weight the
  metric, not just its existence.
- Track calibration over time (expected calibration error / a reliability
  diagram), not just accuracy — a routing system should be able to say "I'm
  not sure, send this to a human" rather than confidently misfiling something.
  A model that's honest about its uncertainty is safer to automate around
  than one that's occasionally very wrong with high confidence.
- Maintain a **human-in-the-loop review queue** for low-confidence
  predictions (informed directly by the calibration table), and audit a
  sample of *high-confidence* predictions periodically to catch silent
  drift — confidently wrong is the failure mode that costs the most in a
  regulated environment.
- Log every prediction with its input, category, confidence, and (if
  reviewed) the human-corrected label, both for retraining and because
  financial email handling is subject to recordkeeping obligations (SEC
  Rule 17a-4 requires broker-dealers to retain business communications,
  with immediate accessibility for 2 years and 6-year total retention; 2026
  FINRA guidance extends this scrutiny explicitly to AI-assisted tooling —
  see sources below). An automated classifier making routing decisions on
  regulated correspondence should itself produce an auditable decision
  trail, not just a label.

**What would improve accuracy / compliance value most, with more data:**
- More labeled examples per class — 44 total is enough to demonstrate the
  approach, not to fully validate it; the confidence intervals on 5-fold CV
  accuracy with ~9 examples per class are wide.
- Sender domain and historical sender→category patterns (a repeat sender
  previously routed to Loan Processing is informative prior information the
  current model discards entirely).
- Attachment presence/type and structured metadata already present in some
  financial email systems (claim numbers, account numbers, policy references)
  as explicit features rather than relying on the model to infer them from
  free text.
- Thread/reply-chain context — a reply in an existing thread should inherit
  strong priors from that thread's category.
- An explicit "escalate to human" category or confidence floor, since in
  this domain a wrong high-confidence prediction is worse than an honest
  "uncertain."

Sources: [FINRA Email Retention Requirements (2026 Guide) — Smarsh](https://www.smarsh.com/compliance-glossary/finra-email-retention-requirements/), [Understanding Model Calibration — arXiv:2501.19047](https://arxiv.org/pdf/2501.19047)

## Assumptions and trade-offs (consolidated)

1. Used the internal `data-field="email_id"` as the output identifier, not
   the on-disk filename, because filenames collide between `train/` and
   `test/` while the internal field is globally unique — see Lab Book.
2. Substituted a local sentence-embedding classifier for the originally
   planned LLM-API-based classifier, because no API key was available in
   the run environment and the spec requires the solution to run end-to-end
   on open-source tooling alone. Documented above under "Choosing the
   architecture."
3. Used 5-fold stratified cross-validation rather than a single train/val
   split, because 44 examples across 5 classes makes a single split noisy.
4. Subject line is included twice in the text fed to both classifiers
   (`Email.text` in `ingestion.py`) to weight it slightly higher than the
   body for the sparse TF-IDF representation; this is a no-op for the dense
   embedding model but kept consistent across both for a fair comparison.

## Future work

1. **Add a true zero-shot LLM classifier as a third comparison arm** once an
   API key (or a locally-hosted open model via Ollama) is available — this
   was the original plan's primary approach and would likely handle
   ambiguous/boundary-case emails better than similarity-to-centroid,
   particularly ones that don't closely resemble any of the 44 training
   examples.
2. **Calibrate the confidence scores explicitly** (temperature scaling
   fit on a held-out set, or isotonic regression) rather than relying on
   the raw softmax — the baseline's under-confidence finding suggests
   naive softmax output isn't a reliable probability without this step.
3. **Active learning loop**: route low-confidence predictions to a human
   reviewer, feed the correction back into the training set, and re-fit
   periodically — turns the "human-in-the-loop" production recommendation
   above into an actual mechanism for the model to keep improving instead of
   only ever being retrained manually.
4. **Multi-label support**: the current design assumes one category per
   email, but a real email ("please also update my beneficiary and file a
   claim") could legitimately belong to more than one department.
