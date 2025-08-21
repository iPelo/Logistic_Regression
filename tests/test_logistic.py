import numpy as np
import pytest

from src.logisctic_regression import LogisticRegressionModel
from src import utils


def test_sigmoid_properties():
    model = LogisticRegressionModel()
    z = np.array([-1000, 0, 1000], dtype=float)
    s = model._sigmoid(z)
    assert np.all(s >= 0) and np.all(s <= 1)
    assert pytest.approx(s[1], rel=1e-6) == 0.5
    assert s[0] < s[1] < s[2]


def test_fit_and_predict_toy_dataset():
    # simple linearly separable data
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionModel(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)

    preds = model.predict(X)
    acc = utils.accuracy_score(y, preds)

    assert acc >= 0.95


def test_confusion_and_metrics_consistency():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])

    cm = utils.confusion_matrix(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    assert tn == 1
    assert fp == 1
    assert fn == 1
    assert tp == 1

    acc = utils.accuracy_score(y_true, y_pred)
    prec = utils.precision_score(y_true, y_pred)
    rec = utils.recall_score(y_true, y_pred)
    f1 = utils.f1_score(y_true, y_pred)

    # sanity checks
    assert np.isclose(acc, (tn + tp) / 4.0)
    assert 0 <= prec <= 1
    assert 0 <= rec <= 1
    assert 0 <= f1 <= 1


def test_threshold_changes_predictions():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionModel(learning_rate=0.1, n_iters=1000)
    model.fit(X, y)

    proba = model.predict_proba(X)
    preds_05 = model.predict(X, threshold=0.5)
    preds_09 = model.predict(X, threshold=0.9)

    # with higher threshold, fewer positives
    assert np.sum(preds_09) <= np.sum(preds_05)


def test_auc_on_toy_dataset():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    fpr, tpr, _ = utils.roc_curve(y_true, y_score)
    auc_val = utils.auc(fpr, tpr)

    assert 0 <= auc_val <= 1
    assert auc_val > 0.5  # better than random