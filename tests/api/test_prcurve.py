import os
import numpy as np
import pytest
import sklearn.metrics as skm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mmu
from mmu.commons._testing import PRCU_skm, ROCCU_skm, generate_test_labels, greater_equal_tol


def test_PR_MU_reference():

    fpath = os.path.abspath(os.path.dirname(__file__))
    conf_mats = np.load(os.path.join(fpath, "pr_curve_reference_conf_mats.npy"))
    ref_scores = np.load(os.path.join(fpath, "pr_curve_reference.npy"))

    curve_err = mmu.PRCU().from_confusion_matrices(conf_mats)
    assert np.allclose(curve_err.chi2_scores, ref_scores)


@pytest.mark.parametrize("curve_class,skm_func", [(mmu.PRCU, PRCU_skm), (mmu.ROCCU, ROCCU_skm)])
def test_MU_from_scores(curve_class, skm_func):
    """Test BaseCurveUncertainty.from_scores"""
    np.random.seed(412)
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    proba, _, y = generate_test_labels(N=500)
    yhat = greater_equal_tol(proba, thresholds[100])

    sk_conf_mat = skm.confusion_matrix(y, yhat)
    err = curve_class.from_scores(y=y, scores=proba, thresholds=thresholds)
    assert err.conf_mats is not None
    assert err.conf_mats.dtype == np.dtype(np.int64)

    y, x = skm_func(y, yhat)

    assert err.chi2_scores.shape == (err.x_grid.size, err.x_grid.size)
    assert np.isclose(err.y[100], y[1])
    assert np.isclose(err.x[100], x[1])
    assert np.array_equal(err.conf_mats[100], sk_conf_mat.flatten())


@pytest.mark.parametrize("curve_class,skm_func", [(mmu.PRCU, PRCU_skm), (mmu.ROCCU, ROCCU_skm)])
def test_MU_from_confusion_matrices(curve_class, skm_func):
    """Test BaseCurveUncertainty.from_confusion_matrices"""
    np.random.seed(412)
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    proba, _, y = generate_test_labels(N=500)
    yhat = greater_equal_tol(proba, thresholds[100])

    sk_conf_mat = skm.confusion_matrix(y, yhat)
    conf_mats = mmu.confusion_matrices_thresholds(y, proba, thresholds)
    err = curve_class.from_confusion_matrices(conf_mats=conf_mats)
    assert err.conf_mats is not None
    assert err.conf_mats.dtype == np.dtype(np.int64)

    y, x = skm_func(y, yhat)

    assert err.chi2_scores.shape == (err.x_grid.size, err.x_grid.size)
    assert np.isclose(err.y[100], y[1])
    assert np.isclose(err.x[100], x[1])
    assert np.array_equal(err.conf_mats[100], sk_conf_mat.flatten())


@pytest.mark.parametrize("curve_class,skm_func", [(mmu.PRCU, PRCU_skm), (mmu.ROCCU, ROCCU_skm)])
def test_MU_from_classifier(curve_class, skm_func):
    """Test BaseCurveUncertainty.from_classifier"""
    # generate seeds to be used by sklearn
    # do not use this in real scenarios,
    # it's a convenience only used in the tutorial notebooks
    seeds = mmu.commons.utils.SeedGenerator(234)

    # generate 2 class dataset
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=seeds())

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seeds())
    # fit a model
    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_scores = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    yhat = greater_equal_tol(y_scores, thresholds[100])
    sk_conf_mat = skm.confusion_matrix(y_test, yhat)
    err = curve_class.from_classifier(clf=model, X=X_test, y=y_test, thresholds=thresholds)
    assert err.conf_mats is not None
    assert err.conf_mats.dtype == np.dtype(np.int64)

    y, x = skm_func(y_test, yhat)

    assert err.chi2_scores.shape == (err.x_grid.size, err.x_grid.size)
    assert np.isclose(err.y[100], y[1])
    assert np.isclose(err.x[100], x[1])
    assert np.array_equal(err.conf_mats[100], sk_conf_mat.flatten())
