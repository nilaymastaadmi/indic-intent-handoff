# Phase 1 research: multilingual intent + unknown-intent hand-off on MASSIVE's Indian languages

2026-09-24, Nilay Toshniwal. Built with AI assistance (Claude Code). Every number below is either
printed by a script in `phase1/` (named beside it) or transcribed from a cited table whose anchor
`phase1/verify_sources.py` re-fetched and found (19 of 19 sources, HTTP 200; a planted false anchor
was correctly reported missing). Full URL / status / anchor list: `phase1/SOURCES.md`.

**Verdict: GO**, with multilingual-e5-small, on MASSIVE 1.1, inside the 12 h budget (section 5).
The exact study is not published (section 3), but its zero-shot half is: IndicXTREME already fine-tunes
XLM-R, MuRIL and IndicBERT on English MASSIVE and tests on Indic. So the claim this study can own is
the parameter budget and the hand-off signal per language, not the zero-shot accuracy itself.

## 1. Dataset: MASSIVE 1.1 (choice in 3 lines)

1. It is the only public intent set found with 5 Indian languages, and it is parallel by id, so every
   cross-language comparison is paired: same utterance, different language.
2. CC-BY-4.0, with published per-locale baselines at three model scales (section 2) to set void thresholds.
3. The customer-support alternatives cover at most one Indian language: Multi3NLU++ (banking + hotels,
   CC-BY-4.0, 62 intents, 3,080 utterances per language, Marathi only, multi-label) or have no corpus
   licence found (MultiATIS++ Hindi). BANKING77 is English only.

Measured on the download (`phase1/inspect_massive.out`): tarball 40,251,390 bytes, SHA-256
`4cba5faa…c0491577`, 52 locale files. Chosen locales en-US, hi-IN, ta-IN, te-IN, kn-IN, ml-IN: 16,521 rows
each, 11,514 / 2,033 / 2,974 train/dev/test. **Parallel: all 5 Indic id sets equal en-US's, 0 of 16,521
ids disagree on split or intent.** Facts that shape the design:

- Train has 60 intents (4 to 810 examples each); dev and test have 59. `cooking_query` (4 train
  examples) never appears in test. Test examples per intent: median 35, minimum 1 (`general_greet`).
- 21 of 2,974 en-US test utterances also appear verbatim in train (0.7%).
- Translation noise is real: kn-IN id 1 carries a translator's note appended to the command
  (roughly "asking your personal assistant about setting an alarm").
- First rows (id 0, test, `alarm_set`): en "wake me up at five am this week"; hi "मुझे इस सप्ताह सुबह
  पांच बजे जगा दो"; ta "இந்த வாரம் காலை ஐந்து மணிக்கு என்னை எழுப்பு". All 18 first rows: `phase1/inspect_massive.out`.

## 2. Published baselines (test intent accuracy, %)

| Locale | XLM-R full data | mT5 enc full | XLM-R zero-shot | mT5 enc zero-shot | XLM-R zero (IX) | MuRIL zero (IX) | IndicBERT v2 zero (IX) | e5-small frozen, 8/intent (MTEB) |
|---|---|---|---|---|---|---|---|---|
| en-US | 88.3 | 89.0 | 85.2* | 76.0* | | | | 70.3 |
| hi-IN | 85.8 | 86.2 | 77.1 | 66.4 | 74.9 | 82.4 | 82.7 | 64.1 |
| ta-IN | 83.5 | 85.2 | 65.6 | 49.0 | 65.7 | 75.9 | 76.1 | 57.3 |
| te-IN | 84.5 | 85.5 | 66.0 | 52.1 | 66.6 | 74.7 | 77.9 | 57.8 |
| kn-IN | 84.0 | 84.9 | 68.2 | 52.2 | 61.7 | 77.5 | 78.2 | 57.5 |
| ml-IN | 85.1 | 86.5 | 67.6 | 62.0 | 69.5 | 77.4 | 80.4 | 61.9 |
| 5-locale mean | 84.6 | | 68.9 | | 67.7 | 77.6 | 79.1 | 59.7 |

Columns 2 to 5: MASSIVE paper (FitzGerald et al., ACL 2023, arXiv 2204.08582) **Table 8, p21**; per-locale 95%
intervals of ±1.1 to ±1.8 are printed in Table 8; averages in **Table 3, p8**. *en-US in the zero-shot columns is
the English test score of the English-only run. Base-size models (270M XLM-R, 258M mT5 encoder), 128
hyperparameter trials each. IX = IndicXTREME (Doddapaneni et al., ACL 2023, arXiv 2212.05409) **Table 19,
p21**, protocol on **p6**: "fine-tuning the model on English and testing on the available Indic test
sets", model chosen on the English dev set. Last column: the e5-small model card's MTEB scores, where
MTEB fits a logistic regression (max_iter 100) on **8 training examples per intent, averaged over 10
draws** (mteb `AbsTaskClassification`, default unchanged since v1.1.0). It is a few-shot probe, so
full-data frozen + LR should beat it. Means and gaps: `phase1/baselines.out`.

What this says before we run anything: XLM-R's zero-shot drop from its own English run is 8.1 (hi), 19.6
(ta), 19.2 (te), 17.0 (kn), 17.6 (ml) points, and **Hindi has the smallest drop for all 3 models**.
Indic-trained encoders (MuRIL, IndicBERT v2) recover about 10 of those points.

## 3. Methods and prior work

- **LoRA** (Hu et al. 2021, arXiv 2106.09685): freeze W0, learn W0 + ΔW = W0 + BA with B (d×r) and
  A (r×k), B initialised to zero, ΔW scaled by α/r (**p4**); applied to Wq and Wv (**p6**). Here, r = 8 on
  query and value in all 12 layers plus a 54-way head: **168,246 trainable parameters, 0.143% of
  117,653,760** (measured, `phase1/time_encoder.out`).
- **Max-softmax** (Hendrycks and Gimpel, ICLR 2017, arXiv 1610.02136, **p1**): the confidence of the
  top class; low confidence means hand off.
- **Stronger score: Mahalanobis distance** to the nearest known-intent centre, with one shared
  covariance fitted on training embeddings (Lee et al. 2018; for intents, Podolskiy et al., AAAI 2021,
  arXiv 2101.03778, **p4**: "On all datasets, RoBERTa equipped with the Mahalanobis distance outperforms
  baselines ... including RoBERTa with the MSP score", on CLINC150, ROSTD and SNIPS). Why this one: it
  is a whiteboard formula, needs no unknown examples, and reuses the same pooled embeddings. Caveat: their
  unknowns are out-of-domain queries; ours are held-out intents from the same 18 scenarios, which is harder.
- **Threshold transfer** (Insights 2022, aclanthology.org/2022.insights-1.3): distance scores that win
  on AUROC "do not generalize well to the test-set" once the threshold is chosen on validation data. So
  the hand-off rate must also be reported at a threshold chosen on English dev, not only the oracle one.
- **Prior work, and where I looked.** Found: IndicXTREME (zero-shot, full fine-tuning, no unknowns) and
  the MTEB frozen probe (above). **Not found:** LoRA on MASSIVE's Indic locales, or unknown-intent
  detection on MASSIVE in any language. Searched: MMNLU-22 proceedings (10 papers, via your Perplexity
  run), 2 Exa searches, the MASSIVE and IndicXTREME papers in full. The arXiv and Semantic Scholar
  keyword APIs returned HTTP 429 and were **not run**; that is a gap in the negative, not a clean miss.

## 4. Encoder (measured on this machine unless labelled ESTIMATE)

| Encoder | Params (non-embedding) | Licence | Published 5-locale zero-shot | Train step, bs 32 |
|---|---|---|---|---|
| multilingual-e5-small (default) | 117.7M (21.4M) | MIT | none; 59.7 frozen 8-shot | **0.65 to 0.75 s, measured** |
| XLM-R base | 278.9M (85.1M) | MIT | 67.7 (IX), 68.9 (T8) | about 2.6 to 3.0 s, ESTIMATE |
| MuRIL base | 237.0M (85.1M) | Apache-2.0 | 77.6 (IX) | about 2.6 to 3.0 s, ESTIMATE |

The estimate scales the measured step by non-embedding parameters (85.1 / 21.4 = 4.0x). **82% of
e5-small's parameters are its 250,037-token embedding table**, a lookup that costs almost nothing;
the transformer layers that do the arithmetic hold only 21.4M, which is why a step is cheap. MuRIL's published zero-shot mean (77.6) is 9.9 points above XLM-R base's (67.7) on the same
benchmark, but at 4x the step cost the joint regime alone would take about 11 h: it does not fit.

Measured (`phase1/time_encoder.out`; i7-1360P, 12 cores / 16 threads, torch 2.14.0+cpu using 12 threads;
2 repeats, `time.perf_counter`, 3 warm-up steps excluded): **50 LoRA steps, batch 32: 35.1 s and
32.6 s (English), 32.8 s and 37.4 s (6-locale mix); worst single step 1.37 s. Inference: 6.1 to 7.2 ms per
utterance (1,000 utterances, batch 64).** Background CPU load before the run was not recorded (the probe
command failed); the phase-3 runs must record it and hold the machine awake on power.

Tokenizer (XLM-R SentencePiece; mean tokens per train utterance including the 5-token "query: " prefix
and specials): en 13.1, hi 14.8, ta 15.6, te 16.1, kn 15.9, ml 16.1. Tokens per word: en 1.18, hi 1.29,
ta 2.01, te 1.93, kn 2.04, ml 2.12 (the Dravidian four are agglutinative, so fewer, longer words).
Unknown tokens: 29 in te-IN's 11,514 utterances, 0 elsewhere. The 6-locale mix step was not measurably
slower than English.

## 5. Compute budget against 12 hours (`phase1/baselines.out`)

| Item | Runs | Min each | Min |
|---|---|---|---|
| Frozen embeddings, 6 locales, all splits | 1 | 11.9 | 12 |
| TF-IDF + LR and frozen + LR, both regimes (ESTIMATE) | 1 | 15 | 15 |
| Pipeline check, MTEB protocol (ESTIMATE) | 1 | 2 | 2 |
| LoRA learning-rate tuning on English dev, seed 0, 3 rates x 5 epochs | 3 | 23.7 | 71 |
| LoRA English-only, seeds 1 and 2 | 2 | 23.7 | 47 |
| LoRA joint 6-locale, 3 seeds x 2 epochs | 3 | 56.8 | 170 |
| Sealed test pass, 7 models x 6 locales | 7 | 2.1 | 15 |
| Smoke run, bootstrap, RESULTS.md (ESTIMATE) | 1 | 15 | 15 |
| **Compute** | | | **347 (5.8 h; 7.5 h if 30% slower)** |

Plus your time: about 30 min across the two stops and about 2 h studying EXPLAINER.md. **Total 8.3 to
10.0 h of the 12 h.** Tonight is feasible only if the compute starts by about 14:00 IST (it then ends
19:50 to 21:30); building and testing the code first takes about 1.5 h (ESTIMATE), so phase 3 has to
start by about 12:30. Phase 4 runs after the compute.

## 6. What phase 1 changes (inputs to phase 2, not yet decided)

- **Prediction (b) as written is near-certain**: published XLM-R already shows a drop of 10 points or more
  on 4 of 5 locales. Candidate sharper add-on: "hi-IN has the smallest zero-shot gap" (true for all 3
  published models), or a higher bar.
- **Void thresholds now have numbers**: (1) frozen e5-small + LR under the MTEB protocol must reproduce the
  card's 70.3 on en-US; (2) no trained arm may score below that 70.3 in English.
- **The unknown-intent draw** must come from the 59 intents present in test (`cooking_query` has no test
  examples), and it needs a minimum-count void rule, because a random 6 can land on intents with 1 to 6
  test examples.
- **Zero-shot model selection uses English dev only** (IndicXTREME's rule). MASSIVE's paper validated on
  all locales, which quietly uses target-language data.
- **Deterministic arms**: TF-IDF + LR and frozen + LR give the same answer for every seed. Three seeds
  there would report a spread of 0; that is honest but costs nothing, so the phase-2 choice is cosmetic.

Reproduce: `.venv/Scripts/python.exe phase1/inspect_massive.py`, `phase1/time_encoder.py`,
`phase1/verify_sources.py`; `py -3 phase1/baselines.py`.
