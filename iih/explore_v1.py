"""Exploratory, not pre-registered (added after V1 failed; English DEV split only, so the test split is not
opened again, and nothing here changes the V1 verdict): does L2-normalising the frozen embeddings explain
why the MTEB-protocol check scored 64.4% against the model card's 70.3%?

    python -m iih.explore_v1 > results/explore_v1.out
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from iih import arms, data


def mteb_protocol(tr_emb, tr_y, te_emb, te_y):
    accs = []
    for draw in range(10):
        rng = np.random.default_rng(draw)
        idx = np.concatenate([rng.choice(np.flatnonzero(tr_y == c), min(8, (tr_y == c).sum()), replace=False)
                              for c in np.unique(tr_y)])
        clf = LogisticRegression(max_iter=100).fit(tr_emb[idx], tr_y[idx])
        accs.append(float((clf.predict(te_emb) == te_y).mean()))
    return float(np.mean(accs))


def main():
    tr = [r for r in data.rows("en-US") if r["partition"] == "train"]
    dv = [r for r in data.rows("en-US") if r["partition"] == "dev"]
    tr_y, dv_y = np.array([r["intent"] for r in tr]), np.array([r["intent"] for r in dv])
    tr_n, dv_n = np.load(arms.CACHE / "en-US_train_all.npy"), np.load(arms.CACHE / "en-US_dev_all.npy")
    clf = arms.build(False, 0)
    tr_raw = arms.infer(clf, [r["utt"] for r in tr])[1]
    dv_raw = arms.infer(clf, [r["utt"] for r in dv])[1]
    assert np.allclose(tr_raw / np.linalg.norm(tr_raw, axis=1, keepdims=True), tr_n, atol=1e-4)
    print(f"en-US dev, {len(dv)} utterances, {len(set(dv_y))} intents; MTEB protocol as in iih.run.stage_v1")
    print(f"raw embedding norm: mean {np.linalg.norm(dv_raw, axis=1).mean():.2f}")
    print(f"L2-normalised (as run): {mteb_protocol(tr_n, tr_y, dv_n, dv_y):.1%}")
    print(f"not normalised:         {mteb_protocol(tr_raw, tr_y, dv_raw, dv_y):.1%}")


if __name__ == "__main__":
    main()
