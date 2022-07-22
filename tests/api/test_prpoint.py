import os
import itertools
import numpy as np
import pytest
import scipy.stats as sts
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


def test_PRMU_ref_chi2():
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=1949933174
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=3437779408
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_score = model.predict_proba(X_test)[:, 1]

    pr_err = mmu.PRU.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
    )

    ref_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'multn_chi2_scores.npy'
    )
    ref_chi2_scores = np.load(ref_path)
    assert np.allclose(pr_err.chi2_scores, ref_chi2_scores)


def test_PRMU_compute_score_for():
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=1949933174
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=3437779408
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_score = model.predict_proba(X_test)[:, 1]

    pr_err = mmu.PRU.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
    )
    # check the the profile loglikelihood with itself is zero
    assert (
        pr_err.compute_score_for(pr_err.precision, pr_err.recall)
        < 1e-12
    )

    ref_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'multn_chi2_scores.npy'
    )
    ref_chi2_scores = np.load(ref_path)
    ref_prec = 0.7852763945527024
    ref_rec = 0.8305165343173831
    ref_score = pr_err.compute_score_for(ref_prec, ref_rec)
    assert np.isclose(ref_score, ref_chi2_scores.min())

    prec = np.linspace(0, 1, 100)
    rec = prec[::-1].copy()
    scores = pr_err.compute_score_for(prec, rec)
    assert scores.size == prec.size
    assert np.isnan(scores).sum() == 0


def test_PRMU_compute_pvalue_for():
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=1949933174
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=3437779408
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_score = model.predict_proba(X_test)[:, 1]

    pr_err = mmu.PRU.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
    )
    # check the the profile loglikelihood with itself is zero
    assert (
        abs(pr_err.compute_pvalue_for(pr_err.precision, pr_err.recall) - 1)
        < 1e-12
    )

    ref_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'multn_chi2_scores.npy'
    )
    ref_chi2_scores = np.load(ref_path)
    ref_prec = 0.7852763945527024
    ref_rec = 0.8305165343173831
    ref_score = pr_err.compute_pvalue_for(ref_prec, ref_rec)
    assert np.isclose(ref_score, sts.chi2.sf(ref_chi2_scores.min(), 2))

    prec = np.linspace(0, 1, 100)
    rec = prec[::-1].copy()
    scores = pr_err.compute_score_for(prec, rec)
    assert scores.size == prec.size
    assert np.isnan(scores).sum() == 0


def test_PREU_ref_cov():
    """Test PREU.from_scores"""
    # generate 2 class dataset
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=1949933174
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=3437779408
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_score = model.predict_proba(X_test)[:, 1]

    pr_err = mmu.PRU.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
        method='bvn'
    )

    ref_cov_mat = np.asarray([0.00064625, 0.00011629, 0.00011629, 0.00057463])
    assert np.isclose(ref_cov_mat, pr_err.cov_mat.flatten(), rtol=7e-3).all()


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


def test_PREU_compute_score_for():
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=1949933174
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=3437779408
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_score = model.predict_proba(X_test)[:, 1]

    pr_err = mmu.PRU.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
        method='bvn'
    )
    # check the the profile loglikelihood with itself is zero
    assert (
        pr_err.compute_score_for(pr_err.precision, pr_err.recall)
        < 1e-12
    )

    prec = np.linspace(0, 1, 100)
    rec = prec[::-1].copy()
    scores = pr_err.compute_score_for(prec, rec)
    assert scores.size == prec.size
    assert np.isnan(scores).sum() == 0


def test_PREU_compute_pvalue_for():
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=1949933174
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=3437779408
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_score = model.predict_proba(X_test)[:, 1]

    pr_err = mmu.PRU.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
        method='bvn'
    )
    # check the the profile loglikelihood with itself is zero
    assert (
        abs(pr_err.compute_pvalue_for(pr_err.precision, pr_err.recall) - 1)
        < 1e-12
    )

    prec = np.linspace(0, 1, 100)
    rec = prec[::-1].copy()
    scores = pr_err.compute_score_for(prec, rec)
    assert scores.size == prec.size
    assert np.isnan(scores).sum() == 0


def test_PREU_plot_integration():
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=1949933174
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=3437779408
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_score = model.predict_proba(X_test)[:, 1]

    mmu.PRU.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
        method='bvn'
    ).plot()


def test_PRMU_plot_integration():
    X, y = make_classification(
        n_samples=1000, n_classes=2, random_state=1949933174
    )

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=3437779408
    )
    # fit a model
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_score = model.predict_proba(X_test)[:, 1]

    mmu.PRU.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
    ).plot()
