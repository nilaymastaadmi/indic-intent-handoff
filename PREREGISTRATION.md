# Pre-registration: multilingual intent + unknown-intent hand-off with LoRA on 5 Indian languages

Written 2026-09-24 13:40 IST and committed **before any model code exists** (`git log --diff-filter=A
-- PREREGISTRATION.md iih/` proves the order). Author: Nilay Toshniwal. Built with AI assistance
(Claude Code). Inputs: `RESEARCH.md` (phase 1, verdict GO). Budget: 12 h (Nilay, phase 1).

## Question

A support assistant must (1) route a request to the right intent in the customer's language and
(2) hand off to a person when the request is something it was never trained on. With a small
multilingual encoder (multilingual-e5-small, 117.7M parameters) and LoRA training 0.143% of them:
how much does LoRA add over frozen embeddings, how much accuracy is lost when training only on English
and testing on Hindi, Tamil, Telugu, Kannada and Malayalam, how much of that loss joint multilingual
training recovers, and can the model's own scores tell known intents from unseen ones in each language?

## Data (fixed)

- MASSIVE 1.1 (CC-BY-4.0), tarball 40,251,390 bytes, SHA-256 prefix 4cba5faa, locales en-US, hi-IN,
  ta-IN, te-IN, kn-IN, ml-IN; 11,514 / 2,033 / 2,974 train / dev / test each, **parallel by id**
  (0 of 16,521 ids disagree on split or intent; RESEARCH.md section 1).
- **Held-out intents (unknowns), drawn 2026-09-24 before any model code, `random.Random(20260924)
  .sample(eligible, 6)` over the 46 intents with at least 20 en-US test examples:** `alarm_query`,
  `email_query`, `qa_stock`, `recommendation_events`, `takeaway_order`, `transport_query`.
  Their test counts are 34, 119, 26, 43, 22, 51: **295 unknown test utterances per locale**. They are
  removed from train and dev. **54 known intents** remain (including `cooking_query`, 4 train
  examples, none in test). Known test: 2,679 utterances per locale.
- Several held-out intents share a scenario with a known one (`alarm_query` / `alarm_set`,
  `email_query` / `email_sendemail`): near-domain unknowns, harder than the out-of-domain queries
  in the Mahalanobis literature. That is the realistic case for support routing.
- **The test split is sealed in code**: `data.test()` raises unless `data.unseal(reason)` was called,
  which appends a line to `results/UNSEAL.log`. One pass, after all training and tuning is finished.
  Tuning uses dev only, and zero-shot model selection uses English dev only (IndicXTREME's rule).

## Arms (the same 54-way task everywhere)

| Arm | Model | Tuned on dev | Seeds |
|---|---|---|---|
| A0 | Majority known intent in train | nothing | 1 |
| A1 | Character n-gram TF-IDF (char_wb, 2 to 5) + logistic regression | C in {0.3, 1, 3} | 1 (deterministic) |
| A2 | Frozen e5-small, "query: " prefix, mean-pooled, L2-normalised, + logistic regression | C in {0.3, 1, 3, 10} | 1 (deterministic) |
| A3 | e5-small + LoRA r = 8, alpha = 16, dropout 0.1 on query and value in all 12 layers, + trained linear head (168,246 trainable parameters) | learning rate in {1e-4, 3e-4, 1e-3}, English dev, seed 0, 5 epochs | 3 (0, 1, 2) |

A3 training: batch 32, max length 64 tokens, AdamW (weight decay 0.01), linear decay with 6% warm-up,
final-epoch checkpoint (no early stopping). Regimes, both for A1 to A3:
- **EN-only**: train on en-US known train (5 epochs for A3), test on all 6 locales (zero-shot to 5).
- **JOINT**: train on the known train of all 6 locales (2 epochs for A3), test on all 6.

Deterministic arms are run once; their seed spread is 0 by construction and is reported as such.

## Hand-off scores

- **MSP**: 1 minus the top softmax probability (Hendrycks and Gimpel 2017), for A1, A2 and A3.
- **Mahalanobis**: distance to the nearest known-intent mean under one shared covariance, both fitted
  on the arm's own pooled train embeddings of the known intents (Lee et al. 2018; Podolskiy et al. 2021),
  for A2 and A3.
- **Deployed threshold**: set on **English dev** to keep 95% of known dev utterances, then applied
  unchanged to every locale's test set (the threshold-transfer check from Insights 2022).

## Metrics

- Known-intent accuracy and macro-F1 per locale, 95% bootstrap CIs (2,000 resamples of test ids;
  the same resampled ids for every arm and locale, since the data are parallel), paired differences.
- Zero-shot gap per Indic locale = EN-only English accuracy minus EN-only locale accuracy.
- Gap recovered by JOINT = (JOINT locale acc - EN-only locale acc) / (zero-shot gap), mean over 5.
- Hand-off: AUROC for unknown vs known (unknown = positive), and **unknown acceptance at 95% known
  retention** (the share of unknowns let through when the threshold keeps 95% of known utterances,
  the FPR@95%TPR of the OOD literature), per locale. At the deployed threshold: known retention and
  unknowns caught, per locale.
- A3 numbers are the mean over 3 seeds with the min to max range printed beside every difference.

## Void preconditions (printed beside the headline by the analysis script)

- **V1 pipeline check**: frozen e5-small + logistic regression under MTEB's protocol (all 60 intents,
  8 training examples per intent, 10 draws, max_iter 100) must land within 2.0 points of the model
  card's 70.3 on en-US. Otherwise every A2 result is void (the frozen pipeline is broken).
- **V2**: an arm scoring below 70.3% English known-intent accuracy in EN-only is void.
- **V3**: an A3 seed with English dev accuracy below 50% is reported as a failed seed, not rerun.
- **V4**: hand-off metrics need at least 100 unknown test utterances per locale (295, fixed above).

## Hypotheses and predictions (written before any code; several expected to miss)

| # | Prediction | Pass if | Else |
|---|---|---|---|
| P1 | LoRA is worth its training | A3 EN-only beats A2 EN-only by at least 5.0 points in English accuracy | "PEFT adds nothing measurable here", reported as prominently |
| P2 | Zero-shot costs a lot, least for Hindi | A3 EN-only gap at least 10 points on at least 4 of 5 Indic locales, AND hi-IN has the smallest gap | report each half separately |
| P3 | Joint training recovers most of it | A3 JOINT recovers at least 50% of the gap, mean over 5 locales | |
| P4 | Max-softmax is a weak hand-off signal | A3 EN-only MSP AUROC in English below 0.80 | |
| P5 | Mahalanobis beats MSP | A3 EN-only Mahalanobis AUROC at least 0.03 above MSP in English | may fail: near-domain unknowns |
| P6 | The English threshold does not transfer | at the English-dev threshold, A3 EN-only known retention below 90% on at least 3 of 5 Indic locales | |
| P7 | Where e5-small lands against published models | A3 EN-only 5-locale zero-shot mean between 55 and 70 (published: e5-small frozen 8-shot 59.7, XLM-R base 67.7 to 68.9, over 60 intents; ours is 54, so this is a rough bracket) | |

**Honest-risk line.** If LoRA ties frozen + logistic regression, the PEFT claim reduces to "I ran it",
and that is what gets reported. MASSIVE is voice-assistant commands, not support tickets; the study
measures multilingual routing and hand-off mechanics, not AppleCare's domain.

## Kill switch

**2026-09-25 14:00 IST.** If A3 is not complete by then, `RESULTS.md` is generated from whatever
exists, missing arms and seeds marked NOT RUN, and no resume line is written until the full set exists.

## Build plan (phase 3, after "build it")

Repo layout: `iih/data.py` (load, parallel check, hold-out, sealed test), `iih/arms.py` (A0 to A3),
`iih/handoff.py` (MSP, Mahalanobis, thresholds), `iih/metrics.py` (accuracy, macro-F1, AUROC,
acceptance at 95% retention, paired bootstrap), `iih/run.py` (resume-safe runs, one JSONL per arm,
regime and seed), `iih/analyse.py` (writes `results/RESULTS.md`; nothing typed by hand).

Tests, each shown failing on a planted defect before it counts: no held-out intent in train or dev;
parallel ids across locales; the sealed test raises; accuracy and macro-F1 against scikit-learn;
AUROC and acceptance-at-95%-retention on hand-computed cases; Mahalanobis on synthetic Gaussians with a
known answer; LoRA trainable-parameter count equals 168,246; bootstrap interval contains the point
estimate. Runs hold the machine awake, record CPU load before starting, and time steps on a monotonic clock.

Resume line the study could produce (every letter from `RESULTS.md`): "Fine-tuned a 118M-parameter
multilingual encoder with LoRA (0.14% of weights) on 54 intents in English and 5 Indian languages:
X% English accuracy, Y% mean zero-shot, Z% of the gap recovered by joint training; hand-off to a
person on unseen intents at AUROC W."

## Fixed in advance

Compute is the phase-1 budget (about 5.8 h measured, 7.5 h if 30% slower). One run of the table above;
no reruns to improve numbers. Any change after this commit goes in `AMENDMENTS.md` with its reason
before the results it could affect exist.
