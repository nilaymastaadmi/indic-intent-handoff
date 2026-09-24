"""Unknown-intent scores: max-softmax (MSP) and Mahalanobis distance with one shared covariance."""
import numpy as np


def msp(probs):
    """1 - top class probability: higher = more likely an unknown intent."""
    return 1.0 - np.asarray(probs).max(axis=1)


class Mahalanobis:
    """Class means plus one shared covariance, fitted on the known intents' training embeddings."""

    def fit(self, emb, y):
        emb, y = np.asarray(emb, dtype=np.float64), np.asarray(y)
        self.classes = np.unique(y)
        self.means = np.stack([emb[y == c].mean(0) for c in self.classes])
        centred = emb - self.means[np.searchsorted(self.classes, y)]
        self.prec = np.linalg.pinv(centred.T @ centred / len(emb))
        return self

    def score(self, emb):
        """Squared distance to the nearest class mean: higher = more likely unknown."""
        x = np.asarray(emb, dtype=np.float64)
        xp = x @ self.prec
        d = (xp * x).sum(1)[:, None] - 2 * xp @ self.means.T + ((self.means @ self.prec) * self.means).sum(1)[None]
        return d.min(1)
