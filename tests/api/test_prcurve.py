import os
import numpy as np
import sklearn.metrics as skm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import mmu
from mmu.commons._testing import generate_test_labels
from mmu.commons._testing import greater_equal_tol


def test_PRCMU_from_scores():
    """Test PRMU.from_scores"""
    np.random.seed(412)
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    proba, _, y = generate_test_labels(N=500)
    yhat = greater_equal_tol(proba, thresholds[100])

    sk_conf_mat = skm.confusion_matrix(y, yhat)
    pr_err = mmu.PRCU.from_scores(y=y, scores=proba, thresholds=thresholds)
    assert pr_err.conf_mats is not None
    assert pr_err.conf_mats.dtype == np.dtype(np.int64)

    prec, rec, _, _ = skm.precision_recall_fscore_support(
        y, yhat, zero_division=0.0  # type: ignore
    )

    assert pr_err.chi2_scores.shape == (
        pr_err.rec_grid.size,
        pr_err.rec_grid.size
    )
    assert np.isclose(pr_err.precision[100], prec[1])
    assert np.isclose(pr_err.recall[100], rec[1])
    assert np.array_equal(pr_err.conf_mats[100], sk_conf_mat.flatten())


def test_PRCMU_from_confusion_matrices():
    """Test PRMU.from_scores"""
    np.random.seed(412)
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    proba, _, y = generate_test_labels(N=500)
    yhat = greater_equal_tol(proba, thresholds[100])

    sk_conf_mat = skm.confusion_matrix(y, yhat)
    conf_mats = mmu.confusion_matrices_thresholds(y, proba, thresholds)
    pr_err = mmu.PRCU.from_confusion_matrices(conf_mats=conf_mats)
    assert pr_err.conf_mats is not None
    assert pr_err.conf_mats.dtype == np.dtype(np.int64)

    prec, rec, _, _ = skm.precision_recall_fscore_support(
        y, yhat, zero_division=0.0  # type: ignore
    )

    assert pr_err.chi2_scores.shape == (
        pr_err.rec_grid.size,
        pr_err.rec_grid.size
    )
    assert np.isclose(pr_err.precision[100], prec[1])
    assert np.isclose(pr_err.recall[100], rec[1])
    assert np.array_equal(pr_err.conf_mats[100], sk_conf_mat.flatten())


def test_PRCMU_from_classifier():
    """Test PREU.from_classifier"""
    # generate seeds to be used by sklearn
    # do not use this in real scenarios,
    # it's a convenience only used in the tutorial notebooks
    seeds = mmu.commons.utils.SeedGenerator(234)

    # generate 2 class dataset
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=seeds()
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seeds()
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_scores = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    yhat = greater_equal_tol(y_scores, thresholds[100])
    sk_conf_mat = skm.confusion_matrix(y_test, yhat)
    pr_err = mmu.PRCU.from_classifier(
        clf=model, X=X_test, y=y_test, thresholds=thresholds
    )
    assert pr_err.conf_mats is not None
    assert pr_err.conf_mats.dtype == np.dtype(np.int64)

    prec, rec, _, _ = skm.precision_recall_fscore_support(
        y_test, yhat, zero_division=0.0  # type: ignore
    )

    assert pr_err.chi2_scores.shape == (
        pr_err.rec_grid.size,
        pr_err.rec_grid.size
    )
    assert np.isclose(pr_err.precision[100], prec[1])
    assert np.isclose(pr_err.recall[100], rec[1])
    assert np.array_equal(pr_err.conf_mats[100], sk_conf_mat.flatten())


def test_PRCEU_from_scores():
    """Test PRMU.from_scores"""
    np.random.seed(412)
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    proba, _, y = generate_test_labels(N=500)
    yhat = greater_equal_tol(proba, thresholds[100])

    sk_conf_mat = skm.confusion_matrix(y, yhat)
    pr_err = mmu.PRCU.from_scores(
        y=y, scores=proba, thresholds=thresholds, method='bvn'
    )
    assert pr_err.conf_mats is not None
    assert pr_err.conf_mats.dtype == np.dtype(np.int64)

    prec, rec, _, _ = skm.precision_recall_fscore_support(
        y, yhat, zero_division=0.0  # type: ignore
    )

    assert pr_err.chi2_scores.shape == (
        pr_err.rec_grid.size,
        pr_err.rec_grid.size
    )
    assert np.isclose(pr_err.precision[100], prec[1])
    assert np.isclose(pr_err.recall[100], rec[1])
    assert np.array_equal(pr_err.conf_mats[100], sk_conf_mat.flatten())


def test_PRCEU_from_confusion_matrices():
    """Test PRMU.from_scores"""
    np.random.seed(412)
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    proba, _, y = generate_test_labels(N=500)
    yhat = greater_equal_tol(proba, thresholds[100])

    sk_conf_mat = skm.confusion_matrix(y, yhat)
    conf_mats = mmu.confusion_matrices_thresholds(y, proba, thresholds)
    pr_err = mmu.PRCU.from_confusion_matrices(conf_mats=conf_mats, method='bvn')
    assert pr_err.conf_mats is not None
    assert pr_err.conf_mats.dtype == np.dtype(np.int64)

    prec, rec, _, _ = skm.precision_recall_fscore_support(
        y, yhat, zero_division=0.0  # type: ignore
    )

    assert pr_err.chi2_scores.shape == (
        pr_err.rec_grid.size,
        pr_err.rec_grid.size
    )
    assert np.isclose(pr_err.precision[100], prec[1])
    assert np.isclose(pr_err.recall[100], rec[1])
    assert np.array_equal(pr_err.conf_mats[100], sk_conf_mat.flatten())


def test_PRCEU_from_classifier():
    """Test PREU.from_classifier"""
    # generate seeds to be used by sklearn
    # do not use this in real scenarios,
    # it's a convenience only used in the tutorial notebooks
    seeds = mmu.commons.utils.SeedGenerator(234)

    # generate 2 class dataset
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=seeds()
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=seeds()
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_scores = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    yhat = greater_equal_tol(y_scores, thresholds[100])
    sk_conf_mat = skm.confusion_matrix(y_test, yhat)
    pr_err = mmu.PRCU.from_classifier(
        clf=model, X=X_test, y=y_test, thresholds=thresholds, method='bvn'
    )
    assert pr_err.conf_mats is not None
    assert pr_err.conf_mats.dtype == np.dtype(np.int64)

    prec, rec, _, _ = skm.precision_recall_fscore_support(
        y_test, yhat, zero_division=0.0  # type: ignore
    )

    assert pr_err.chi2_scores.shape == (
        pr_err.rec_grid.size,
        pr_err.rec_grid.size
    )
    assert np.isclose(pr_err.precision[100], prec[1])
    assert np.isclose(pr_err.recall[100], rec[1])
    assert np.array_equal(pr_err.conf_mats[100], sk_conf_mat.flatten())


def test_PRU_from_scores_with_train():
    """Test PREU.from_classifier"""
    ref_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'train_reference_sets.npz'
    )
    ll = np.load(ref_path)
    y_test = ll.get('y_test')
    y_score = ll.get('y_score')
    scores_bs = ll.get('scores_bs')

    pr_err = mmu.PRCU.from_scores_with_train(
        y_test,
        scores=y_score,
        scores_bs=scores_bs,
        obs_axis=0
    )
    exp_shape = pr_err.thresholds.size, scores_bs.shape[1]
    assert pr_err.train_precisions.shape == exp_shape
    assert pr_err.train_recalls.shape == exp_shape
    assert pr_err.train_conf_mats.shape == (
        pr_err.thresholds.size, scores_bs.shape[1], 4
    )
    exp_cov_mat_shape = pr_err.thresholds.size, 4
    assert pr_err.train_cov_mats.shape == exp_cov_mat_shape
    assert pr_err.train_cov_mats.shape == exp_cov_mat_shape
    assert pr_err.total_cov_mats.shape == exp_cov_mat_shape

    # reference sets are computed with threshold 0.4999443530277085 which
    # corresponds to idx: 418
    ref_set_test = [0.00310889, 0.00112212, 0.00112212, 0.00258438]
    ref_set_train = [0.00142558, -0.00041831, -0.00041831,  0.00189029]

    assert np.allclose(pr_err.cov_mats[418].flatten(), ref_set_test, rtol=5e3)
    assert np.allclose(pr_err.train_cov_mats[418].flatten(), ref_set_train, rtol=5e3)
