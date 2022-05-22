import itertools
import numpy as np
import pytest
import sklearn.metrics as skm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


import mmu
from mmu.commons._testing import generate_test_labels
from mmu.commons._testing import greater_equal_tol

Y_DTYPES = [
    bool,
    np.bool_,
    int,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
]

YHAT_DTYPES = [
    bool,
    np.bool_,
    int,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
]

PROBA_DTYPES = [
    float,
    np.float32,
    np.float64,
]


def test_PRMU_from_scores():
    """Test PRMU.from_scores"""
    np.random.seed(412)
    thresholds = np.random.uniform(1e-6, 1-1e-6, 10)
    for y_dtype, proba_dtype, threshold in itertools.product(
        Y_DTYPES, PROBA_DTYPES, thresholds
    ):
        proba, _, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            proba_dtype=proba_dtype
        )
        yhat = greater_equal_tol(proba, threshold)
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        pr_err = mmu.PRU.from_scores(y=y, scores=proba, threshold=threshold)
        assert pr_err.conf_mat is not None
        assert pr_err.conf_mat.dtype == np.dtype(np.int64)

        prec, rec, _, _ = skm.precision_recall_fscore_support(
            y, yhat, zero_division=0.0  # type: ignore
        )

        assert pr_err.chi2_scores.shape == (pr_err.n_bins, pr_err.n_bins)
        assert np.isclose(pr_err.precision, prec[1])
        assert np.isclose(pr_err.recall, rec[1])
        assert np.array_equal(pr_err.conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {proba_dtype}"
            f" and threshold: {threshold}"
        )


def test_PRMU_from_predictions():
    """Test PRMU.from_predictions"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        pr_err = mmu.PRU.from_predictions(y=y, yhat=yhat)
        assert pr_err.conf_mat is not None
        assert pr_err.conf_mat.dtype == np.dtype(np.int64)

        prec, rec, _, _ = skm.precision_recall_fscore_support(
            y, yhat, zero_division=0.0  # type: ignore
        )

        assert pr_err.chi2_scores.shape == (pr_err.n_bins, pr_err.n_bins)
        assert np.isclose(pr_err.precision, prec[1])
        assert np.isclose(pr_err.recall, rec[1])
        assert np.array_equal(pr_err.conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


def test_PRMU_from_confusion_matrix():
    """Test PRMU.from_confusion_matrix"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        pr_err = mmu.PRU.from_confusion_matrix(sk_conf_mat)
        prec, rec, _, _ = skm.precision_recall_fscore_support(
            y, yhat, zero_division=0.0  # type: ignore
        )

        assert pr_err.chi2_scores.shape == (pr_err.n_bins, pr_err.n_bins)
        assert np.isclose(pr_err.precision, prec[1])
        assert np.isclose(pr_err.recall, rec[1])
        assert np.array_equal(pr_err.conf_mat, sk_conf_mat.flatten()), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


def test_PRMU_from_classifier():
    """Test PRMU.from_classifier"""
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

    np.random.seed(2345)
    thresholds = np.random.uniform(1e-6, 1-1e-6, 10)
    for threshold in thresholds:
        yhat = greater_equal_tol(y_scores, threshold)
        sk_conf_mat = skm.confusion_matrix(y_test, yhat)

        pr_err = mmu.PRU.from_classifier(
            model, X_test, y_test, threshold=threshold
        )
        prec, rec, _, _ = skm.precision_recall_fscore_support(
            y_test, yhat, zero_division=0.0  # type: ignore
        )

        assert pr_err.chi2_scores.shape == (pr_err.n_bins, pr_err.n_bins)
        assert np.isclose(pr_err.precision, prec[1])
        assert np.isclose(pr_err.recall, rec[1])
        assert np.array_equal(pr_err.conf_mat, sk_conf_mat), (
            f"test failed for threshold: {threshold}"
        )


def test_PRMU_exceptions():
    """Test PRMU exceptions"""
    proba, _, y = generate_test_labels(N=1000,)
    yhat = greater_equal_tol(proba, 0.5)

    pr_err = mmu.PRU.from_scores(
        y=y, scores=proba, threshold=0.5, n_bins=40
    )
    assert pr_err.n_bins == 40
    assert pr_err.chi2_scores.shape == (40, 40)

    # n_bins >= 1
    with pytest.raises(ValueError):
        pr_err = mmu.PRU.from_scores(
            y=y, scores=proba, threshold=0.5, n_bins=-20
        )

    # n_bins >= 1
    with pytest.raises(ValueError):
        pr_err = mmu.PRU.from_scores(
            y=y, scores=proba, threshold=0.5, n_bins=0
        )

    # n_bins must be an int
    with pytest.raises(TypeError):
        pr_err = mmu.PRU.from_scores(
            y=y, scores=proba, threshold=0.5, n_bins=20.
        )

    pr_err = mmu.PRU.from_scores(
        y=y, scores=proba, threshold=0.5, n_sigmas=1.
    )

    with pytest.raises(TypeError):
        pr_err = mmu.PRU.from_scores(
            y=y, scores=proba, threshold=0.5, n_sigmas=[1., ]
        )

def test_PREU_from_scores():
    """Test PREU.from_scores"""
    np.random.seed(43)
    thresholds = np.random.uniform(1e-6, 1-1e-6, 10)
    for y_dtype, proba_dtype, threshold in itertools.product(
        Y_DTYPES, PROBA_DTYPES, thresholds
    ):
        proba, _, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            proba_dtype=proba_dtype
        )
        yhat = greater_equal_tol(proba, threshold)
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        pr_err = mmu.PRU.from_scores(
            y=y, scores=proba, threshold=threshold, method='bvn'
        )
        assert pr_err.conf_mat is not None
        assert pr_err.conf_mat.dtype == np.dtype(np.int64)
        assert pr_err.cov_mat.shape == (2, 2)
        prec, rec, _, _ = skm.precision_recall_fscore_support(
            y, yhat, zero_division=0.0  # type: ignore
        )

        assert np.isclose(pr_err.precision, prec[1])
        assert np.isclose(pr_err.recall, rec[1])
        assert np.array_equal(pr_err.conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {proba_dtype}"
            f" and threshold: {threshold}"
        )


def test_PREU_from_predictions():
    """Test PREU.from_predictions"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        pr_err = mmu.PRU.from_predictions(y=y, yhat=yhat, method='bvn')
        assert pr_err.conf_mat is not None
        assert pr_err.conf_mat.dtype == np.dtype(np.int64)

        prec, rec, _, _ = skm.precision_recall_fscore_support(
            y, yhat, zero_division=0.0  # type: ignore
        )

        assert pr_err.cov_mat.shape == (2, 2)
        assert np.isclose(pr_err.precision, prec[1])
        assert np.isclose(pr_err.recall, rec[1])
        assert np.array_equal(pr_err.conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


def test_PREU_from_confusion_matrix():
    """Test PREU.from_confusion_matrix"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        pr_err = mmu.PRU.from_confusion_matrix(sk_conf_mat, method='bvn')
        prec, rec, _, _ = skm.precision_recall_fscore_support(
            y, yhat, zero_division=0.0  # type: ignore
        )

        assert np.isclose(pr_err.precision, prec[1])
        assert np.isclose(pr_err.recall, rec[1])
        assert np.array_equal(pr_err.conf_mat, sk_conf_mat.flatten()), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


def test_PREU_from_classifier():
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

    thresholds = np.random.uniform(0, 1, 10)
    for threshold in thresholds:
        yhat = greater_equal_tol(y_scores, threshold)
        sk_conf_mat = skm.confusion_matrix(y_test, yhat)

        pr_err = mmu.PRU.from_classifier(
            model, X_test, y_test, threshold=threshold, method='bvn'
        )
        prec, rec, _, _ = skm.precision_recall_fscore_support(
            y_test, yhat, zero_division=0.0  # type: ignore
        )

        assert np.isclose(pr_err.precision, prec[1])
        assert np.isclose(pr_err.recall, rec[1])
        assert np.array_equal(pr_err.conf_mat, sk_conf_mat), (
            f"test failed for threshold: {threshold}"
        )
