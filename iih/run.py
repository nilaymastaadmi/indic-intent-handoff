"""Resume-safe stages: embed -> tune -> train -> test (the single sealed pass).

    python -m iih.run embed | tune | train | test
"""
import hashlib
import json
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from iih import arms, data
from iih.data import HELD_OUT, LOCALES, RESULTS, labels
from iih.handoff import Mahalanobis, msp

LRS, C_TFIDF, C_FROZEN = (1e-4, 3e-4, 1e-3), (0.3, 1, 3), (0.3, 1, 3, 10)
EPOCHS, SEEDS, REGIMES = {"en": 5, "joint": 2}, (0, 1, 2), ("en", "joint")
LAB = {name: i for i, name in enumerate(labels())}
TUNE, PREDS = RESULTS / "tune.jsonl", RESULTS / "preds"


def ty(rows):
    return [r["utt"] for r in rows], [LAB[r["intent"]] for r in rows]


def train_rows(regime):
    return data.known("en-US", "train") if regime == "en" else [r for l in LOCALES for r in data.known(l, "train")]


def emb_all(locale, partition):
    """Frozen embeddings of every row of a train or dev partition, known and held-out, in file order."""
    rows = [r for r in data.rows(locale) if r["partition"] == partition]
    return rows, arms.frozen_embeddings([r["utt"] for r in rows], f"{locale}_{partition}_all")


def emb_known(locale, partition):
    rows, emb = emb_all(locale, partition)
    keep = np.array([r["intent"] not in HELD_OUT for r in rows])
    return emb[keep]


def train_emb(regime):
    return emb_known("en-US", "train") if regime == "en" else np.concatenate([emb_known(l, "train") for l in LOCALES])


def done(path, **key):
    return path.exists() and any(all(json.loads(line).get(k) == v for k, v in key.items())
                                 for line in path.open(encoding="utf-8"))


def append(path, rec):
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def en_dev():
    return ty(data.known("en-US", "dev"))


def stage_embed():
    for loc in LOCALES:
        for part in ("train", "dev"):
            t = time.perf_counter()
            rows, emb = emb_all(loc, part)
            print(f"{loc} {part}: {emb.shape} {time.perf_counter() - t:.0f}s", flush=True)


def stage_tune():
    dev_x, dev_y = en_dev()
    dev_emb = emb_known("en-US", "dev")
    for regime in REGIMES:
        x, y = ty(train_rows(regime))
        for C in C_TFIDF:
            if not done(TUNE, arm="A1", regime=regime, C=C):
                acc = float((arms.tfidf_lr(x, y, C).predict(dev_x) == np.array(dev_y)).mean())
                append(TUNE, {"arm": "A1", "regime": regime, "C": C, "dev_acc_en": acc})
                print("A1", regime, C, acc, flush=True)
        emb = train_emb(regime)
        for C in C_FROZEN:
            if not done(TUNE, arm="A2", regime=regime, C=C):
                acc = float((arms.frozen_lr(emb, y, C).predict(dev_emb) == np.array(dev_y)).mean())
                append(TUNE, {"arm": "A2", "regime": regime, "C": C, "dev_acc_en": acc})
                print("A2", regime, C, acc, flush=True)
    x, y = ty(train_rows("en"))
    for lr in LRS:
        lora_run("en", lr, 0, x, y, dev_x, dev_y, tuning=True)


def lora_run(regime, lr, seed, x, y, dev_x, dev_y, tuning=False):
    path = arms.MODELS / f"a3_{regime}_lr{lr:g}_s{seed}.pt"
    if path.exists():
        return
    t = time.perf_counter()
    clf = arms.train_lora(x, y, lr, EPOCHS[regime], seed, RESULTS / f"trainlog_{path.stem}.jsonl")
    acc = float((arms.infer(clf, dev_x)[0].argmax(1) == np.array(dev_y)).mean())
    arms.save_trainable(clf, path)
    append(TUNE if tuning else RESULTS / "train.jsonl",
           {"arm": "A3", "regime": regime, "lr": lr, "seed": seed, "dev_acc_en": acc,
            "trainable": arms.trainable(clf), "minutes": round((time.perf_counter() - t) / 60, 1)})
    print("A3", regime, lr, seed, acc, flush=True)


def best(arm, regime, key):
    rows = [json.loads(line) for line in TUNE.open(encoding="utf-8")]
    rows = [r for r in rows if r["arm"] == arm and r["regime"] == regime]
    return max(rows, key=lambda r: r["dev_acc_en"])[key]


def stage_train():
    lr = best("A3", "en", "lr")
    dev_x, dev_y = en_dev()
    for regime in REGIMES:
        x, y = ty(train_rows(regime))
        for seed in SEEDS:
            if regime == "en" and seed == 0:
                continue                                    # the tuning run at the chosen lr is seed 0
            lora_run(regime, lr, seed, x, y, dev_x, dev_y)


def write_preds(name, score_fn, fixed_pred=None, train_vecs=None, train_y=None):
    """score_fn(texts, locale, split) -> (probs, vecs). Writes English dev + every locale's test."""
    out = PREDS / f"{name}.jsonl"
    if out.exists():
        return
    maha = Mahalanobis().fit(train_vecs, train_y) if train_vecs is not None else None
    recs = []
    sets = [("dev_en", "en-US", data.known("en-US", "dev"), [])]
    sets += [("test", loc, *data.test(loc)) for loc in LOCALES]
    for split, loc, known_rows, unknown_rows in sets:
        rows = known_rows + unknown_rows
        probs, vecs = score_fn([r["utt"] for r in rows], loc, split)
        m = maha.score(vecs) if maha is not None else [None] * len(rows)
        s = msp(probs) if probs is not None else [None] * len(rows)
        for i, r in enumerate(rows):
            recs.append({"split": split, "locale": loc, "id": r["id"], "unknown": r["intent"] in HELD_OUT,
                         "gold": LAB.get(r["intent"], -1),
                         "pred": int(probs[i].argmax()) if probs is not None else int(fixed_pred),
                         "msp": None if s[i] is None else float(s[i]),
                         "maha": None if m[i] is None else float(m[i])})
    PREDS.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    print("preds", name, len(recs), flush=True)


def frozen_test(texts, loc, split):
    key = f"{loc}_{split}_" + hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    return None, arms.frozen_embeddings(texts, key)


def stage_test():
    data.unseal("stage test: all tuning and training finished; single sealed pass")
    for regime in REGIMES:
        x, y = ty(train_rows(regime))
        write_preds(f"A0_{regime}", lambda t, l, s: (None, None), fixed_pred=arms.majority(y))
        m1 = arms.tfidf_lr(x, y, best("A1", regime, "C"))
        write_preds(f"A1_{regime}", lambda t, l, s: (m1.predict_proba(t), None))
        emb = train_emb(regime)
        m2 = arms.frozen_lr(emb, y, best("A2", regime, "C"))

        def a2(t, l, s):
            v = frozen_test(t, l, s)[1]
            return m2.predict_proba(v), v
        write_preds(f"A2_{regime}", a2, train_vecs=emb, train_y=y)
        lr = best("A3", "en", "lr")
        for seed in SEEDS:
            name = f"A3_{regime}_s{seed}"
            if (PREDS / f"{name}.jsonl").exists():
                continue
            clf = arms.load_lora(arms.MODELS / f"a3_{regime}_lr{lr:g}_s{seed}.pt")
            train_vecs = arms.infer(clf, x)[1]
            write_preds(name, lambda t, l, s: arms.infer(clf, t), train_vecs=train_vecs, train_y=y)
    stage_v1()


def stage_v1():
    """MTEB protocol pipeline check: 60 intents, 8 per intent, 10 draws, LogisticRegression(max_iter=100)."""
    out = RESULTS / "v1.json"
    if out.exists():
        return
    tr_rows, tr_emb = emb_all("en-US", "train")
    kn, un = data.test("en-US")
    te_rows = kn + un
    te_emb = frozen_test([r["utt"] for r in te_rows], "en-US", "test")[1]
    tr_y, te_y = np.array([r["intent"] for r in tr_rows]), np.array([r["intent"] for r in te_rows])
    accs = []
    for draw in range(10):
        rng = np.random.default_rng(draw)
        idx = np.concatenate([rng.choice(np.flatnonzero(tr_y == c), min(8, (tr_y == c).sum()), replace=False)
                              for c in np.unique(tr_y)])
        clf = LogisticRegression(max_iter=100).fit(tr_emb[idx], tr_y[idx])
        accs.append(float((clf.predict(te_emb) == te_y).mean()))
    out.write_text(json.dumps({"mean_acc": float(np.mean(accs)), "draws": accs, "reference": 0.703,
                               "tolerance": 0.02}, indent=1), encoding="utf-8")
    print("V1", np.mean(accs), flush=True)


if __name__ == "__main__":
    {"embed": stage_embed, "tune": stage_tune, "train": stage_train, "test": stage_test}[sys.argv[1]]()
