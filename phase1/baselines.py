"""Published intent-accuracy baselines, transcribed from the cited tables, and every derived number
RESEARCH.md quotes (means, gaps, compute budget). Transcription sources are verified by verify_sources.py."""
import math
L5 = ["hi", "ta", "te", "kn", "ml"]
# MASSIVE paper (arXiv 2204.08582) Table 8, p21, test intent accuracy %, columns:
# mT5 T2T full, mT5 Enc full, XLM-R full, mT5 T2T zero, mT5 Enc zero, XLM-R zero ("zero" = trained on en-US only)
T8 = {"en": (87.9, 89.0, 88.3, 77.1, 76.0, 85.2), "hi": (85.6, 86.2, 85.8, 68.0, 66.4, 77.1),
      "ta": (84.4, 85.2, 83.5, 47.2, 49.0, 65.6), "te": (84.9, 85.5, 84.5, 56.5, 52.1, 66.0),
      "kn": (84.9, 84.9, 84.0, 55.0, 52.2, 68.2), "ml": (86.0, 86.5, 85.1, 62.9, 62.0, 67.6)}
# IndicXTREME (arXiv 2212.05409) Table 19, p21: zero-shot (fine-tuned on English), accuracy %
T19 = {"XLM-R base": {"hi": 74.9, "kn": 61.7, "ml": 69.5, "ta": 65.7, "te": 66.6},
       "MuRIL base": {"hi": 82.4, "kn": 77.5, "ml": 77.4, "ta": 75.9, "te": 74.7},
       "IndicBERT v2": {"hi": 82.7, "kn": 78.2, "ml": 80.4, "ta": 76.1, "te": 77.9}}
# multilingual-e5-small model card (HF API cardData, MTEB MassiveIntentClassification test accuracy),
# MTEB protocol: logistic regression max_iter 100 on 8 train examples per intent, mean of 10 draws
MTEB = {"en": 70.29589778076665, "hi": 64.06523201075991, "ta": 57.2595830531271, "te": 57.82784129119032,
        "kn": 57.5084061869536, "ml": 61.90316072629456}
mean = lambda xs: sum(xs) / len(xs)
print("XLM-R zero-shot gap (its own en-US run 85.2 minus locale), Table 8:")
g = {l: T8["en"][5] - T8[l][5] for l in L5}
print("  " + ", ".join(f"{l} {v:.1f}" for l, v in g.items()), f"| >=10 on {sum(v >= 10 for v in g.values())} of 5, >=15 on {sum(v >= 15 for v in g.values())} of 5")
print("  locale with the smallest zero-shot drop, per model (T2T, Enc, XLM-R):",
      [max(L5, key=lambda l: T8[l][c]) for c in (3, 4, 5)])
print("XLM-R full-data minus zero-shot, Table 8: " + ", ".join(f"{l} {T8[l][2]-T8[l][5]:.1f}" for l in L5))
print(f"5-locale means: XLM-R full {mean([T8[l][2] for l in L5]):.1f} | XLM-R zero (T8) {mean([T8[l][5] for l in L5]):.1f} | "
      + " | ".join(f"{m} zero (T19) {mean([T19[m][l] for l in L5]):.1f}" for m in T19)
      + f" | e5-small frozen 8-shot (MTEB) {mean([MTEB[l] for l in L5]):.1f}")
print(f"e5-small MTEB en {MTEB['en']:.1f}; en minus 5-locale mean {MTEB['en']-mean([MTEB[l] for l in L5]):.1f}")

# compute budget from measured per-step and per-utterance times (phase1/time_encoder.out)
STEP = 0.748   # s per LoRA train step, bs 32: the largest of 4 measured means (0.703, 0.657, 0.652, 0.748)
UTT = 0.00718  # s per utterance inference: the largest of 4 measured values
TR, DEV, TEST, N_LOC = 11514, 2033, 2974, 6
en_epoch = math.ceil(TR / 32) * STEP / 60
joint_epoch = math.ceil(TR * N_LOC / 32) * STEP / 60
dev_en, dev_all, test_all = DEV * UTT / 60, DEV * N_LOC * UTT / 60, TEST * N_LOC * UTT / 60
embed_all = (TR + DEV + TEST) * N_LOC * UTT / 60
en_run = 5 * en_epoch + 5 * dev_en
joint_run = 2 * joint_epoch + 2 * dev_all
rows = [("Frozen embeddings, 6 locales x all splits (measured rate)", 1, embed_all),
        ("TF-IDF + LR and frozen + LR fits, both regimes (ESTIMATE, unmeasured)", 1, 15.0),
        ("Pipeline check: MTEB protocol, 8 per intent x 10 draws (ESTIMATE)", 1, 2.0),
        ("LoRA lr tuning, English dev, seed 0, 3 learning rates x 5 epochs", 3, en_run),
        ("LoRA English-only, seeds 1 and 2 (seed 0 reused from tuning)", 2, en_run),
        ("LoRA joint 6-locale, 3 seeds x 2 epochs", 3, joint_run),
        ("Sealed test pass: 7 encoders x 6 locales, both OOD scores", 7, test_all),
        ("Smoke run + bootstrap + RESULTS.md (ESTIMATE)", 1, 15.0)]
print(f"per epoch: English {en_epoch:.1f} min, joint {joint_epoch:.1f} min; English run {en_run:.1f} min, joint run {joint_run:.1f} min")
tot = 0
for name, n, m in rows:
    tot += n * m; print(f"| {name} | {n} | {m:.1f} | {n*m:.0f} |")
print(f"TOTAL compute {tot:.0f} min = {tot/60:.1f} h at measured speed; {1.3*tot/60:.1f} h if 30% slower (largest single step seen: 1.367 s)")
