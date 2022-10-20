import os
import pytest
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
from mmu.commons._testing import PRCU_skm
from mmu.commons._testing import ROCCU_skm

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


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PRMU_from_scores(curve_class, skm_func):
    """Test BaseUncertainty.from_scores"""
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

        err = curve_class.from_scores(y=y, scores=proba, threshold=threshold)
        assert err.conf_mat is not None
        assert err.conf_mat.dtype == np.dtype(np.int64)

        y, x = skm_func(y, yhat)

        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)
        assert np.isclose(err.y, y[1])
        assert np.isclose(err.x, x[1])
        assert np.array_equal(err.conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {proba_dtype}"
            f" and threshold: {threshold}"
        )


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PRMU_from_predictions(curve_class, skm_func):
    """Test PRMU.from_predictions"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        err = curve_class.from_predictions(y=y, yhat=yhat)
        assert err.conf_mat is not None
        assert err.conf_mat.dtype == np.dtype(np.int64)

        y, x = skm_func(y, yhat)

        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)
        assert np.isclose(err.y, y[1])
        assert np.isclose(err.x, x[1])
        assert np.array_equal(err.conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PRMU_from_confusion_matrix(curve_class, skm_func):
    """Test PRMU.from_confusion_matrix"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        err = curve_class.from_confusion_matrix(sk_conf_mat)
        y, x = skm_func(y, yhat)

        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)
        assert np.isclose(err.y, y[1])
        assert np.isclose(err.x, x[1])
        assert np.array_equal(err.conf_mat, sk_conf_mat.flatten()), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PRMU_from_classifier(curve_class, skm_func):
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

        err = curve_class.from_classifier(
            model, X_test, y_test, threshold=threshold
        )
        y, x = skm_func(y_test, yhat)

        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)
        assert np.isclose(err.y, y[1])
        assert np.isclose(err.x, x[1])
        assert np.array_equal(err.conf_mat, sk_conf_mat), (
            f"test failed for threshold: {threshold}"
        )


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PRMU_exceptions(curve_class, skm_func):
    """Test PRMU exceptions"""
    proba, _, y = generate_test_labels(N=1000,)
    yhat = greater_equal_tol(proba, 0.5)

    err = curve_class.from_scores(
        y=y, scores=proba, threshold=0.5, n_bins=40
    )
    assert err.n_bins == 40
    assert err.chi2_scores.shape == (40, 40)

    # n_bins >= 1
    with pytest.raises(ValueError):
        err = curve_class.from_scores(
            y=y, scores=proba, threshold=0.5, n_bins=-20
        )

    # n_bins >= 1
    with pytest.raises(ValueError):
        err = curve_class.from_scores(
            y=y, scores=proba, threshold=0.5, n_bins=0
        )

    # n_bins must be an int
    with pytest.raises(TypeError):
        err = curve_class.from_scores(
            y=y, scores=proba, threshold=0.5, n_bins=20.
        )

    err = curve_class.from_scores(
        y=y, scores=proba, threshold=0.5, n_sigmas=1.
    )

    with pytest.raises(TypeError):
        err = curve_class.from_scores(
            y=y, scores=proba, threshold=0.5, n_sigmas=[1., ]
        )


@pytest.mark.parametrize("curve_class,multn_chi2_scores_filename", [
    (mmu.PRU, 'pr_multn_chi2_scores.npy'),
    (mmu.ROCU, 'roc_multn_chi2_scores.npy'),
])
def test_PRMU_ref_chi2(curve_class, multn_chi2_scores_filename):
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

    err = curve_class.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
    )

    ref_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        multn_chi2_scores_filename
    )
    ref_chi2_scores = np.load(ref_path)
    assert np.allclose(err.chi2_scores, ref_chi2_scores)


@pytest.mark.parametrize("curve_class,multn_chi2_scores_filename,ref_yx", [
    (mmu.PRU, 'pr_multn_chi2_scores.npy', (0.7852763945527024, 0.8305165343173831)),
    (mmu.ROCU, 'roc_multn_chi2_scores.npy', (0.8305165343173831, 0.2163994844947491)),
])
def test_PRMU_compute_score_for(curve_class, multn_chi2_scores_filename, ref_yx):
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

    err = curve_class.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
    )
    # check the the profile loglikelihood with itself is zero
    assert (
        err.compute_score_for(err.y, err.x)
        < 1e-12
    )

    ref_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        multn_chi2_scores_filename
    )
    ref_chi2_scores = np.load(ref_path)
    ref_score = err.compute_score_for(*ref_yx)
    # np.unravel_index(np.argmin(ref_chi2_scores, axis=None), ref_chi2_scores.shape)
    # np.unravel_index(np.argmin(err.chi2_scores, axis=None), err.chi2_scores.shape)
    # For PR minimum is reach at ref_chi2_scores[49,49]
    # ref_yx = (np.linspace(*err.y_bounds, err.n_bins)[49], np.linspace(*err.x_bounds, err.n_bins)[49])
    # For ROC minimum is reach at ref_chi2_scores[49,50]
    # ref_yx = (np.linspace(*err.y_bounds, err.n_bins)[49], np.linspace(*err.x_bounds, err.n_bins)[50])
    assert np.isclose(ref_score, ref_chi2_scores.min())

    y = np.linspace(0, 1, 100)
    x = y[::-1].copy()
    scores = err.compute_score_for(y, x)
    assert scores.size == y.size
    assert np.isnan(scores).sum() == 0


@pytest.mark.parametrize("curve_class,multn_chi2_scores_filename,ref_yx", [
    (mmu.PRU, 'pr_multn_chi2_scores.npy', (0.7852763945527024, 0.8305165343173831)),
    (mmu.ROCU, 'roc_multn_chi2_scores.npy', (0.8305165343173831, 0.2163994844947491)),
])
def test_PRMU_compute_pvalue_for(curve_class, multn_chi2_scores_filename, ref_yx):
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

    err = curve_class.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
    )
    # check the the profile loglikelihood with itself is zero
    assert (
        abs(err.compute_pvalue_for(err.y, err.x) - 1)
        < 1e-12
    )

    ref_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        multn_chi2_scores_filename
    )
    ref_chi2_scores = np.load(ref_path)
    ref_score = err.compute_pvalue_for(*ref_yx)
    # assert np.isclose(ref_score, sts.chi2.sf(ref_chi2_scores.min(), 2))

    y = np.linspace(0, 1, 100)
    x = y[::-1].copy()
    scores = err.compute_score_for(y, x)
    assert scores.size == y.size
    assert np.isnan(scores).sum() == 0


@pytest.mark.parametrize("curve_class,ref_cov_mat", [
    (mmu.PRU, np.asarray([0.00064625, 0.00011629, 0.00011629, 0.00057463])),
    (mmu.ROCU, np.asarray([0.00057294, 0.        , 0.        , 0.00065893])),  # noqa: E203, E202
])
def test_PREU_ref_cov(curve_class, ref_cov_mat):
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

    err = curve_class.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
        method='bvn'
    )

    assert np.isclose(ref_cov_mat, err.cov_mat.flatten(), rtol=7e-3).all()


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PREU_from_scores(curve_class, skm_func):
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

        err = curve_class.from_scores(
            y=y, scores=proba, threshold=threshold, method='bvn'
        )
        assert err.conf_mat is not None
        assert err.conf_mat.dtype == np.dtype(np.int64)
        assert err.cov_mat.shape == (2, 2)
        y, x = skm_func(y, yhat)

        assert np.isclose(err.y, y[1])
        assert np.isclose(err.x, x[1])
        assert np.array_equal(err.conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {proba_dtype}"
            f" and threshold: {threshold}"
        )


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PREU_from_predictions(curve_class, skm_func):
    """Test PREU.from_predictions"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        err = curve_class.from_predictions(y=y, yhat=yhat, method='bvn')
        assert err.conf_mat is not None
        assert err.conf_mat.dtype == np.dtype(np.int64)

        y, x = skm_func(y, yhat)

        assert err.cov_mat.shape == (2, 2)
        assert np.isclose(err.y, y[1])
        assert np.isclose(err.x, x[1])
        assert np.array_equal(err.conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PREU_from_confusion_matrix(curve_class, skm_func):
    """Test PREU.from_confusion_matrix"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        err = curve_class.from_confusion_matrix(sk_conf_mat, method='bvn')
        y, x = skm_func(y, yhat)

        assert np.isclose(err.y, y[1])
        assert np.isclose(err.x, x[1])
        assert np.array_equal(err.conf_mat, sk_conf_mat.flatten()), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


@pytest.mark.parametrize("curve_class,skm_func", [
    (mmu.PRU, PRCU_skm),
    (mmu.ROCU, ROCCU_skm),
])
def test_PREU_from_classifier(curve_class, skm_func):
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

        err = curve_class.from_classifier(
            model, X_test, y_test, threshold=threshold, method='bvn'
        )
        y, x = skm_func(y_test, yhat)

        assert np.isclose(err.y, y[1])
        assert np.isclose(err.x, x[1])
        assert np.array_equal(err.conf_mat, sk_conf_mat), (
            f"test failed for threshold: {threshold}"
        )


@pytest.mark.parametrize("curve_class", [
    (mmu.PRU),
    (mmu.ROCU),
])
def test_PREU_compute_score_for(curve_class):
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

    err = curve_class.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
        method='bvn'
    )
    # check the the profile loglikelihood with itself is zero
    assert (
        err.compute_score_for(err.y, err.x)
        < 1e-12
    )

    y = np.linspace(0, 1, 100)
    x = y[::-1].copy()
    scores = err.compute_score_for(y, x)
    assert scores.size == y.size
    assert np.isnan(scores).sum() == 0


@pytest.mark.parametrize("curve_class", [
    (mmu.PRU),
    (mmu.ROCU),
])
def test_PREU_compute_pvalue_for(curve_class):
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

    err = curve_class.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
        method='bvn'
    )
    # check the the profile loglikelihood with itself is zero
    assert (
        abs(err.compute_pvalue_for(err.y, err.x) - 1)
        < 1e-12
    )

    y = np.linspace(0, 1, 100)
    x = y[::-1].copy()
    scores = err.compute_score_for(y, x)
    assert scores.size == y.size
    assert np.isnan(scores).sum() == 0


@pytest.mark.parametrize("curve_class", [
    (mmu.PRU),
    (mmu.ROCU),
])
def test_PREU_plot_integration(curve_class):
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

    curve_class.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
        method='bvn'
    ).plot()


@pytest.mark.parametrize("curve_class", [
    (mmu.PRU),
    (mmu.ROCU),
])
def test_PRMU_plot_integration(curve_class):
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

    curve_class.from_scores(
        y_test,
        scores=y_score,
        threshold=0.5,
    ).plot()
