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
