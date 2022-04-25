import itertools
import pytest
import sklearn as sk
import sklearn.metrics as skm
import numpy as np
import mmu

from mmu.commons._testing import generate_test_labels

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


def test_confusion_matrix():
    """Check that supported dtypes are handled."""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        conf_mat = mmu.confusion_matrix(y, yhat)
        sk_conf_mat = skm.confusion_matrix(y, yhat)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )


def test_confusion_matrix_shapes():
    """Check if different shapes are handled correctly."""
    _, yhat, y = generate_test_labels(1000)
    y_shapes = [y, y[None, :], y[:, None]]
    yhat_shapes = [yhat, yhat[None, :], yhat[:, None]]

    sk_conf_mat = skm.confusion_matrix(y, yhat)

    for y_, yhat_ in itertools.product(y_shapes, yhat_shapes):
        conf_mat = mmu.confusion_matrix(y_, yhat_)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for shape: {y_.shape}, {yhat_.shape}"
        )

    # unequal length
    with pytest.raises(ValueError):
        mmu.confusion_matrix(y, yhat[:100])
    with pytest.raises(ValueError):
        mmu.confusion_matrix(y[:100], yhat)

    # 2d with more than one row/column for the second dimension or 3d
    y_shapes = [
        np.tile(y[:, None], 2),
        np.tile(y[None, :], (2, 1)),
        np.tile(y[None, :], (2, 2, 1)),
    ]

    yhat_shapes = [
        np.tile(yhat[:, None], 2),
        np.tile(yhat[None, :], (2, 1)),
        np.tile(yhat[None, :], (2, 2, 1)),
    ]
    for y_, yhat_ in itertools.product(y_shapes, yhat_shapes):
        with pytest.raises(ValueError):
            mmu.confusion_matrix(y_, yhat_)


def test_confusion_matrix_order():
    """Check that different orders and shapes are handled correctly."""
    _, yhat, y = generate_test_labels(1000)
    y_orders = [
        y.copy(order='C'),
        y.copy(order='F'),
        y[None, :].copy(order='C'),
        y[:, None].copy(order='C'),
        y[None, :].copy(order='F'),
        y[:, None].copy(order='F'),
    ]

    yhat_orders = [
        yhat.copy(order='C'),
        yhat.copy(order='F'),
        yhat[None, :].copy(order='C'),
        yhat[:, None].copy(order='C'),
        yhat[None, :].copy(order='F'),
        yhat[:, None].copy(order='F'),
    ]

    sk_conf_mat = skm.confusion_matrix(y, yhat)

    for y_, yhat_ in itertools.product(y_orders, yhat_orders):
        conf_mat = mmu.confusion_matrix(y_, yhat_)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for shape: {y_.shape}, {yhat_.shape}"
        )


def test_confusion_matrix_proba():
    """Check that supported dtypes are handled correctly."""
    thresholds = np.random.uniform(0, 1, 10)
    for y_dtype, proba_dtype, threshold in itertools.product(
        Y_DTYPES, PROBA_DTYPES, thresholds
    ):
        proba, _, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            proba_dtype=proba_dtype
        )
        yhat = proba >= threshold
        sk_conf_mat = skm.confusion_matrix(y, yhat)
        conf_mat = mmu.confusion_matrix(
            y, score=proba, threshold=threshold
        )
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {proba_dtype}"
            f" and threshold: {threshold}"
        )

def test_confusion_matrix_proba_shapes():
    """Check if different shapes are handled correctly."""
    proba, _, y = generate_test_labels(1000)
    y_shapes = [y, y[None, :], y[:, None]]
    proba_shapes = [proba, proba[None, :], proba[:, None]]

    yhat = proba >= 0.5
    sk_conf_mat = skm.confusion_matrix(y, yhat)

    for y_, proba_ in itertools.product(y_shapes, proba_shapes):
        conf_mat = mmu.confusion_matrix(
            y, score=proba, threshold=0.5
        )
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for shape: {y_.shape}, {proba_.shape}"
        )

    # unequal length
    with pytest.raises(ValueError):
        mmu.confusion_matrix(y, score=proba[:100])
    with pytest.raises(ValueError):
        mmu.confusion_matrix(y[:100], score=proba)

    # 2d with more than one row/column for the second dimension or 3d
    y_shapes = [
        np.tile(y[:, None], 2),
        np.tile(y[None, :], (2, 1)),
        np.tile(y[None, :], (2, 2, 1)),
    ]

    proba_shapes = [
        np.tile(proba[:, None], 2),
        np.tile(proba[None, :], (2, 1)),
        proba[None, None, :],
        np.tile(proba[None, :], (2, 2, 1)),
    ]
    for y_, proba_ in itertools.product(y_shapes, proba_shapes):
        with pytest.raises(ValueError):
            mmu.confusion_matrix(y_, score=proba_)


def test_confusion_matrix_proba_order():
    """Check that different orders and shapes are handled correctly."""
    proba, _, y = generate_test_labels(1000)
    y_orders = [
        y.copy(order='C'),
        y.copy(order='F'),
        y[None, :].copy(order='C'),
        y[:, None].copy(order='C'),
        y[None, :].copy(order='F'),
        y[:, None].copy(order='F'),
    ]

    proba_orders = [
        proba.copy(order='C'),
        proba.copy(order='F'),
        proba[None, :].copy(order='C'),
        proba[:, None].copy(order='C'),
        proba[None, :].copy(order='F'),
        proba[:, None].copy(order='F'),
    ]

    yhat = proba >= 0.5
    sk_conf_mat = skm.confusion_matrix(y, yhat)

    for y_, proba_ in itertools.product(y_orders, proba_orders):
        conf_mat = mmu.confusion_matrix(y_, score=proba_)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for shape: {y_.shape}, {proba.shape}"
        )
