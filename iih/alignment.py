"""Exploratory, not pre-registered (added after training, dev split only, no model trained or selected
from it): how close a sentence and its translation sit in frozen e5-small space. Explains why zero-shot
transfer can work at all.

    python -m iih.alignment > results/alignment.out
"""
import numpy as np

from iih import data
from iih.arms import CACHE
from iih.data import LOCALES


def main():
    en = [r["id"] for r in data.rows("en-US") if r["partition"] == "dev"]
    E = np.load(CACHE / "en-US_dev_all.npy")
    rng = np.random.default_rng(0)
    print(f"en-US dev sentences: {len(en)}; embeddings L2-normalised, so dot product = cosine")
    for loc in LOCALES[1:]:
        pos = {r["id"]: i for i, r in enumerate(r for r in data.rows(loc) if r["partition"] == "dev")}
        X = np.load(CACHE / f"{loc}_dev_all.npy")[[pos[i] for i in en]]
        same = (E * X).sum(1).mean()
        rand = (E * X[rng.permutation(len(en))]).sum(1).mean()
        top1 = (np.argmax(X @ E.T, 1) == np.arange(len(en))).mean()
        print(f"{loc}: translation cosine {same:.3f}, random-pair cosine {rand:.3f}, "
              f"translation is the nearest of {len(en)} English sentences for {top1:.1%}")


if __name__ == "__main__":
    main()
