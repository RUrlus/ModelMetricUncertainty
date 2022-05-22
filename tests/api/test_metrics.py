import itertools
import pytest
import numpy as np
import mmu

from mmu.commons._testing import generate_test_labels
from mmu.commons._testing import compute_reference_metrics

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


def test_binary_metrics_yhat():
    """Test confusion_matrix int64"""
    for y_dtype, yhat_dtype in itertools.product(Y_DTYPES, YHAT_DTYPES):
        _, yhat, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            yhat_dtype=yhat_dtype
        )
        sk_conf_mat, sk_metrics = compute_reference_metrics(y, yhat=yhat)

        conf_mat, metrics = mmu.binary_metrics(y, yhat)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )
        assert np.allclose(metrics, sk_metrics), (
            f"test failed for dtypes: {y_dtype}, {yhat_dtype}"
        )

def test_binary_metrics_yhat_shapes():
    """Check if different shapes are handled correctly."""
    _, yhat, y = generate_test_labels(1000)
    sk_conf_mat, sk_metrics = compute_reference_metrics(y, yhat=yhat)

    y_shapes = [y, y[None, :], y[:, None]]
    yhat_shapes = [yhat, yhat[None, :], yhat[:, None]]

    for y_, yhat_ in itertools.product(y_shapes, yhat_shapes):
        conf_mat, metrics = mmu.binary_metrics(y_, yhat_)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_.shape}, {yhat_.shape}"
        )
        assert np.allclose(metrics, sk_metrics), (
            f"test failed for dtypes: {y_.shape}, {yhat_.shape}"
        )

    # unequal length
    with pytest.raises(ValueError):
        mmu.binary_metrics(y, yhat[:100])
    with pytest.raises(ValueError):
        mmu.binary_metrics(y[:100], yhat)

    # 2d with more than one row/column for the second dimension or 3d
    y_shapes = [
        np.tile(y[:, None], 2),
        np.tile(y[None, :], (2, 1)),
    ]

    yhat_shapes = [
        np.tile(yhat[:, None], 2),
        np.tile(yhat[None, :], (2, 1)),
    ]
    for y_, yhat_ in itertools.product(y_shapes, yhat_shapes):
        with pytest.raises(ValueError):
            mmu.binary_metrics(y_, yhat_)


def test_binary_metrics_order():
    """Check that different orders and shapes are handled correctly."""
    _, yhat, y = generate_test_labels(1000)
    sk_conf_mat, sk_metrics = compute_reference_metrics(y, yhat=yhat)

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

    for y_, yhat_ in itertools.product(y_orders, yhat_orders):
        conf_mat, metrics = mmu.binary_metrics(y_, yhat_)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_.shape}, {yhat_.shape}"
        )
        assert np.allclose(metrics, sk_metrics), (
            f"test failed for dtypes: {y_.shape}, {yhat_.shape}"
        )


def test_binary_metrics_proba():
    """Test confusion_matrix int64"""
    thresholds = np.random.uniform(0, 1, 10)
    for y_dtype, proba_dtype, threshold in itertools.product(
        Y_DTYPES, PROBA_DTYPES, thresholds
    ):
        proba, _, y = generate_test_labels(
            N=1000,
            y_dtype=y_dtype,
            proba_dtype=proba_dtype
        )
        sk_conf_mat, sk_metrics = compute_reference_metrics(
            y, proba=proba, threshold=threshold
        )

        conf_mat, metrics = mmu.binary_metrics(y, scores=proba, threshold=threshold)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for dtypes: {y_dtype}, {proba_dtype}"
            f" and threshold: {threshold}"
        )
        assert np.allclose(metrics, sk_metrics), (
            f"test failed for dtypes: {y_dtype}, {proba_dtype}"
            f" and threshold: {threshold}"
        )

    # test fill settings
    proba, _, y = generate_test_labels(N=1000,)
    thresholds = [1e7, 1. - 1e7]
    fills = [0.0, 1.0]
    threshold = 1e-7

    for threshold, fill in itertools.product(thresholds, fills):
        conf_mat, metrics = mmu.binary_metrics(
            y, scores=proba, threshold=threshold, fill=fill
        )
        sk_conf_mat, sk_metrics = compute_reference_metrics(
            y, proba=proba, threshold=threshold, fill=fill
        )
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for threshold: {threshold}, fill: {fill}"
        )
        assert np.allclose(metrics, sk_metrics), (
            f"test failed for threshold: {threshold}, fill: {fill}"
        )

def test_binary_metrics_proba_shapes():
    """Check if different shapes are handled correctly."""
    proba, _, y = generate_test_labels(1000)
    y_shapes = [y, y[None, :], y[:, None]]
    proba_shapes = [proba, proba[None, :], proba[:, None]]

    sk_conf_mat, sk_metrics = compute_reference_metrics(
        y, proba=proba, threshold=0.5
    )

    for y_, proba_ in itertools.product(y_shapes, proba_shapes):
        conf_mat, metrics = mmu.binary_metrics(y_, scores=proba_, threshold=0.5)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for shapes: {y_.shape}, {proba_.shape}"
        )
        assert np.allclose(metrics, sk_metrics), (
            f"test failed for shapes: {y_.shape}, {proba_.shape}"
        )

    # unequal length
    with pytest.raises(ValueError):
        mmu.binary_metrics(y, scores=proba[:100], threshold=0.5)
    with pytest.raises(ValueError):
        mmu.binary_metrics(y[:100], scores=proba, threshold=0.5)

    # 2d with more than one row/column for the second dimension or 3d
    y_shapes = [
        np.tile(y[:, None], 2),
        np.tile(y[None, :], (2, 1)),
    ]

    proba_shapes = [
        np.tile(proba[:, None], 2),
        np.tile(proba[None, :], (2, 1)),
    ]
    for y_, proba_ in itertools.product(y_shapes, proba_shapes):
        with pytest.raises(ValueError):
            mmu.binary_metrics(y_, scores=proba_, threshold=0.5)


def test_binary_metrics_proba_order():
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

    sk_conf_mat, sk_metrics = compute_reference_metrics(
        y, proba=proba, threshold=0.5
    )

    for y_, proba_ in itertools.product(y_orders, proba_orders):
        conf_mat, metrics = mmu.binary_metrics(y_, scores=proba_, threshold=0.5)
        assert np.array_equal(conf_mat, sk_conf_mat), (
            f"test failed for shapes: {y_.shape}, {proba_.shape}"
        )
        assert np.allclose(metrics, sk_metrics), (
            f"test failed for shapes: {y_.shape}, {proba_.shape}"
        )
