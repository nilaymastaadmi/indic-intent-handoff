"""Exploratory, not pre-registered (added after the sealed pass; English DEV split only, so the test split
is not touched again and nothing is selected): why Mahalanobis on frozen e5 vectors scores near chance.
Dev contains the held-out intents' dev utterances, which no model was trained on.

    python -m iih.explore_maha > results/explore_maha.out
"""
import numpy as np

from iih import arms, data
from iih.data import HELD_OUT, labels
from iih.handoff import Mahalanobis, msp
from iih.metrics import auroc


def main():
    lab = {n: i for i, n in enumerate(labels())}
    tr_rows = [r for r in data.rows("en-US") if r["partition"] == "train"]
    tr_all = np.load(arms.CACHE / "en-US_train_all.npy")
    keep = np.array([r["intent"] not in HELD_OUT for r in tr_rows])
    X, y = tr_all[keep], np.array([lab[r["intent"]] for r in tr_rows if r["intent"] not in HELD_OUT])
    dv_rows = [r for r in data.rows("en-US") if r["partition"] == "dev"]
    D = np.load(arms.CACHE / "en-US_dev_all.npy")
    unk = np.array([r["intent"] in HELD_OUT for r in dv_rows])
    print(f"en-US dev: {(~unk).sum()} known, {unk.sum()} unknown; frozen e5-small, L2-normalised")

    def report(name, s):
        print(f"{name}: AUROC {auroc(s[~unk], s[unk]):.3f}")

    report("MSP, logistic regression C = 10 (A2's tuned C)", msp(arms.frozen_lr(X, y, 10).predict_proba(D)))
    report("Mahalanobis, shared covariance, all 384 directions (A2 as run)", Mahalanobis().fit(X, y).score(D))
    means = np.stack([X[y == c].mean(0) for c in np.unique(y)])
    report("Euclidean distance to the nearest intent centre", ((D[:, None] - means[None]) ** 2).sum(-1).min(1))
    centred = X - means[np.searchsorted(np.unique(y), y)]
    evals, evecs = np.linalg.eigh(centred.T @ centred / len(X))
    evals, evecs = evals[::-1], evecs[:, ::-1]
    print(f"within-intent covariance eigenvalues: largest {evals[0]:.2e}, 50th {evals[49]:.2e}, "
          f"smallest {evals[-1]:.2e} (ratio {evals[0] / evals[-1]:.0f}x)")
    print(f"eigenvalues below 1e-8 x largest: {(evals < 1e-8 * evals[0]).sum()} "
          f"(numpy pinv's default cut-off is 1e-15 x largest, so these are inverted, not dropped)")
    for k in (10, 50, 100, 200, 383):
        P = evecs[:, :k]
        report(f"Mahalanobis on the top {k} variance directions", Mahalanobis().fit(X @ P, y).score(D @ P))
    report("Mahalanobis on the bottom 10 variance directions",
           Mahalanobis().fit(X @ evecs[:, -10:], y).score(D @ evecs[:, -10:]))


if __name__ == "__main__":
    main()
