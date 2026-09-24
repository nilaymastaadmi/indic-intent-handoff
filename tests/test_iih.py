import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from iih import data
from iih.handoff import Mahalanobis, msp
from iih.metrics import accuracy, acceptance_at_retention, auroc, bootstrap, macro_f1


def test_no_held_out_intent_in_train_or_dev():
    for loc in data.LOCALES:
        for part in ("train", "dev"):
            assert not {r["intent"] for r in data.known(loc, part)} & data.HELD_OUT
    assert len(data.labels()) == 54 and not set(data.labels()) & data.HELD_OUT


def test_locales_are_parallel():
    assert data.check_parallel() == 16521


def test_test_split_is_sealed():
    with pytest.raises(PermissionError):
        data.test("en-US")
    with pytest.raises(AssertionError):
        data.known("en-US", "test")


def test_accuracy_and_macro_f1_match_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 7, 500), rng.integers(0, 7, 500)
    assert accuracy(y, p) == pytest.approx(accuracy_score(y, p))
    assert macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro"))


def test_handoff_metrics_on_hand_computed_cases():
    known = np.arange(20.0)                                  # 95% retention threshold = 18.05
    assert acceptance_at_retention(known, [100.0]) == 0.0    # far above the threshold: caught
    assert acceptance_at_retention(known, [10.0, 100.0]) == 0.5
    assert auroc([0.0, 1.0], [2.0]) == 1.0 and auroc([2.0], [0.0, 1.0]) == 0.0
    assert msp(np.array([[0.9, 0.1], [0.5, 0.5]])).tolist() == pytest.approx([0.1, 0.5])


def test_mahalanobis_uses_the_shared_covariance():
    rng = np.random.default_rng(0)
    # one class, wide in x (sd 10) and narrow in y (sd 1): Euclidean ranks (10, 0) as farther than (0, 3),
    # Mahalanobis must rank it closer (1 standard deviation against 3)
    emb = rng.normal(0, 1, (20000, 2)) * [10.0, 1.0]
    d = Mahalanobis().fit(emb, np.zeros(20000)).score(np.array([[10.0, 0.0], [0.0, 3.0]]))
    assert d[0] == pytest.approx(1.0, abs=0.05) and d[1] == pytest.approx(9.0, abs=0.3)


def test_lora_trainable_parameter_count():
    from iih.arms import build, trainable
    assert trainable(build(True, 0)) == 168246


def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 3, 400), rng.integers(0, 3, 400)
    lo, hi = bootstrap(lambda i: accuracy(y[i], p[i]), len(y))
    assert lo <= accuracy(y, p) <= hi and hi - lo < 0.2
