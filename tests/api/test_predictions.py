import numpy as np
import pytest

import mmu

ATOL = 1e-8
RTOL = 1e-5


def _reference(scores, threshold):
    tol = ATOL + RTOL * abs(threshold)
    return (scores >= threshold - tol).astype(np.int64)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("threshold", [-2.0, -0.5, 0.0, 0.5, 1.0, 100.0])
def test_predictions_scalar_threshold(dtype, threshold):
    rng = np.random.default_rng(42)
    scores = (rng.random(1000) * 4.0 - 2.0).astype(dtype)
    scores[:3] = threshold

    yhat = mmu.predictions(scores, threshold)

    assert yhat.shape == scores.shape
    assert yhat.dtype == np.int64
    assert np.array_equal(yhat, _reference(scores, dtype(threshold)))
    assert np.all(yhat[:3] == 1)


@pytest.mark.parametrize("threshold", [-5.0, 0.5, 5.0])
def test_predictions_within_tolerance_is_positive(threshold):
    tol = ATOL + RTOL * abs(threshold)
    scores = np.array([threshold - 0.5 * tol, threshold - 2.0 * tol])
    assert np.array_equal(mmu.predictions(scores, threshold), [1, 0])


def test_predictions_consistent_with_confusion_matrix():
    rng = np.random.default_rng(7)
    scores = rng.random(2000)
    y = rng.integers(0, 2, 2000)
    thresholds = np.linspace(0, 1, 37)

    yhat = mmu.predictions(scores, thresholds)
    expected = mmu.confusion_matrices_thresholds(y, scores, thresholds)
    for i in range(thresholds.size):
        assert np.array_equal(mmu.confusion_matrix(y, yhat=yhat[:, i]).ravel(), expected[i])


def test_predictions_1d_scores_array_thresholds():
    rng = np.random.default_rng(1)
    scores = rng.random(500)
    thresholds = np.array([0.1, 0.5, 0.9])

    yhat = mmu.predictions(scores, thresholds)

    assert yhat.shape == (500, 3)
    for t, thr in enumerate(thresholds):
        assert np.array_equal(yhat[:, t], _reference(scores, thr))


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_predictions_2d_scores_broadcasting(order, dtype):
    rng = np.random.default_rng(3)
    scores = np.asarray(rng.random((200, 5)), order=order, dtype=dtype)
    thresholds = np.linspace(0.05, 0.95, 11)

    yhat = mmu.predictions(scores, thresholds)

    assert yhat.shape == (200, 5, 11)
    for k in range(scores.shape[1]):
        for t, thr in enumerate(thresholds):
            assert np.array_equal(yhat[:, k, t], _reference(scores[:, k], dtype(thr)))

    yhat_scalar = mmu.predictions(scores, 0.5)
    assert yhat_scalar.shape == scores.shape
    assert np.array_equal(yhat_scalar, _reference(scores, dtype(0.5)))


def test_predictions_non_contiguous_scores():
    rng = np.random.default_rng(5)
    scores = rng.random((100, 6))[::2, ::3]
    thresholds = [0.25, 0.75]

    yhat = mmu.predictions(scores, thresholds)

    assert yhat.shape == (50, 2, 2)
    for t, thr in enumerate(thresholds):
        assert np.array_equal(yhat[..., t], _reference(scores, thr))


def test_predictions_single_element_thresholds_keeps_axis():
    scores = np.array([0.2, 0.7])
    assert mmu.predictions(scores, np.array([0.5])).shape == (2, 1)
    assert mmu.predictions(scores, np.float32(0.5)).shape == (2,)


def test_predictions_invalid_input():
    scores = np.array([0.2, 0.7])
    with pytest.raises(ValueError, match="Non-finite"):
        mmu.predictions(np.array([0.1, np.nan]), 0.5)
    with pytest.raises(ValueError, match="finite"):
        mmu.predictions(scores, np.inf)
    with pytest.raises(ValueError, match="Non-finite"):
        mmu.predictions(scores, np.array([0.5, np.nan]))
    with pytest.raises(TypeError):
        mmu.predictions(scores, "0.5")
    with pytest.raises(TypeError):
        mmu.predictions(scores, np.ones((2, 2)))
