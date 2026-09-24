# Amendments to PREREGISTRATION.md

Each entry: what changed, why, when, and which result it could affect. Entries made before any model
was trained or any test prediction existed say so.

## A1, 2026-09-24 13:40 IST: implementation details the pre-registration left open (before any training)

- A1 (TF-IDF): `min_df=2`, `max_features=300,000` to bound memory across 6 scripts; logistic
  regression `max_iter=2000`. A2 (frozen): logistic regression `max_iter=2000`.
- C for A1 and A2 and the learning rate for A3 are all chosen on **English dev** accuracy, in both
  regimes (the pre-registration's "zero-shot model selection uses English dev only", applied uniformly).
- Mahalanobis uses the pseudo-inverse of the shared covariance (numerically safe for 384 dimensions).
- A3 English-only seed 0 is the learning-rate tuning run at the chosen rate, as the phase-1 budget planned.
- V1 (the MTEB pipeline check) needs en-US test, so it runs inside the single sealed pass, after all
  tuning and training; nothing is selected from it.
- Macro-F1 averages over the known intents present in each test set (`cooking_query` has no test examples).
Could affect: absolute accuracy of A1 and A2 slightly (feature cap, iteration cap). No hypothesis threshold changes.

## A2, 2026-09-24 20:20 IST: after the sealed pass had written the EN-only predictions

- **Correction of recorded times.** `PREREGISTRATION.md`'s header and A1 above say 13:40 IST. The git
  commit times are the record: pre-registration 13:08:45, code and A1 13:28:18, training started
  13:28:30 (`results/run.log`), analysis script 13:34:15, test split opened 20:00:14
  (`results/UNSEAL.log`). The order the pre-registration promises holds; only the typed times were wrong.
  `PREREGISTRATION.md` is left unedited.
- **Analysis addition.** The pre-registration's Metrics section lists paired differences between arms,
  but the committed `analyse.py` only computed one (P1, English). Added after seeing the EN-only test
  numbers: the paired A3 minus A2 difference per locale and regime, with the bootstrap interval and the
  A3 seed range. No prediction or threshold depends on it; it is reported, not tested.
Could affect: nothing in the scorecard.

## A3, 2026-09-24 20:35 IST: the sealed pass was interrupted and resumed (before any JOINT LoRA prediction existed)

- The test-pass process ran inside the Claude Code session that launched it and was killed when that
  session ended (about 20:25), after 9 of 16 prediction files (all EN-only arms, A0 to A2 JOINT).
- A first relaunch at 20:28 failed on import before reaching `unseal()` (the runner sat in `scratch/`,
  off the package path). The second relaunch, 20:31:51, runs `python -m iih.run test` unchanged as a
  detached process. `stage_test` logged its unseal again, so `results/UNSEAL.log` has **2 lines for
  one pass**: 20:00:14 (start) and 20:32:38 (resume).
- Nothing was retrained, re-tuned or re-selected between the two lines. `write_preds` skips any file
  that already exists, so the 9 files written before the interruption were kept byte for byte; the
  resume only adds the 3 JOINT LoRA seeds and V1. (It does refit the deterministic A1 and A2 JOINT
  models before skipping their existing files; their output is unused.)
Could affect: nothing in the scorecard.

## A4, 2026-09-24 21:20 IST: V1 failed; the frozen arm A2 is void (after all test predictions existed)

- **V1 result: 64.4% against the model card's 70.3 +/- 2.0 (`results/v1.json`, 10 draws, 63.1% to 67.6%).
  Under the pre-registration every A2 result is void.** A2's numbers stay in `RESULTS.md`, labelled VOID,
  and nothing is concluded from them. P1 (LoRA beats frozen by 5 points) compares against A2, so P1 is
  VOID, not FAIL; it is not re-scored against any other arm.
- **The analysis script did not propagate the void** (it printed P1 as FAIL and A2 rows unlabelled).
  Changed after the results existed: `analyse.py` now labels A2 rows "(VOID)" and prints P1 as VOID when
  V1 fails. Checked both ways on synthetic predictions (V1 forced to 50%: VOID labels and P1 VOID; V1 as
  shipped: output unchanged, P1 PASS).
- **Exploratory diagnosis, not a re-scoring** (`iih/explore_v1.py`, English DEV split only, test not
  reopened): the same MTEB protocol on dev gives 65.6% with the study's L2-normalised embeddings and 71.5%
  with raw ones (mean raw norm 4.74). The check fed the study's normalised features to MTEB's fixed C = 1
  classifier, so it tested a preprocessing choice rather than whether the frozen pipeline was broken. That
  explains the failure; it does not reverse it, because the rule was fixed in advance. What the check
  should have been, and when (on dev, before the seal), is recorded as a lesson in the README.
Could affect: P1 (now VOID). No other prediction uses A2.

## A5, 2026-09-24 21:30 IST: two accidental re-runs of the test stage by pytest collection (after all results existed)

- The relaunch helper from A3 was saved as `scratch/resume_test.py`. Its name matches pytest's `*_test.py`
  pattern, and its top level called `runpy.run_module("iih.run")` with the `test` stage, so `pytest` run from
  the repo root **imported it and re-ran `stage_test`**: once in a full test run (21:19:50) and once in a
  collect-only run made to find the cause (21:26:18). Each appended an unseal line, so `results/UNSEAL.log`
  has 4 lines for 1 pass. The same import opened the seal inside the pytest process, which is why
  `test_test_split_is_sealed` failed in that run; run as `pytest tests/` the suite passes 8 of 8.
- Neither re-run changed anything: `write_preds` and `stage_v1` skip existing outputs. All 12 prediction files
  keep their original modification times (last 21:03:23), `v1.json` keeps 21:03:25, and the 9 files hashed
  before A3 still match. SHA-1 of all 12 plus `v1.json` recorded in `results/preds.sha1`.
- The helper is deleted. Nothing was trained, tuned or selected; every number in `RESULTS.md` is unchanged.
Could affect: nothing in the scorecard. The unseal count printed in `RESULTS.md` is now 4.
