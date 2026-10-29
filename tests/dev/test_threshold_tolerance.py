"""Regression tests for the classification threshold comparison (C-2).

The comparison tolerance was computed as ``atol + rtol * threshold`` using the
*signed* threshold. For any negative threshold that makes the tolerance
negative, which turns the documented inclusive ``score >= threshold`` into a
strict ``score > threshold + |tol|``. Every observation sitting on a negative
threshold was silently misclassified, corrupting all downstream metrics for
``decision_function`` output, logits, log-odds and any non-probability score.
"""

import numpy as np
import pytest

import mmu

ATOL = 1e-8
RTOL = 1e-5

# thresholds spanning both signs and several orders of magnitude
THRESHOLDS = [-1e4, -100.0, -5.0, -2.0, -1.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 1.0, 2.0, 5.0, 100.0, 1e4]


def _reference(y, scores, threshold):
    """Oracle implementing the documented inclusive-within-tolerance rule."""
    tol = ATOL + RTOL * abs(threshold)
    return np.bincount(y * 2 + (scores >= threshold - tol).astype(int), minlength=4)


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_score_exactly_on_threshold_is_positive(threshold):
    """The threshold is documented as inclusive, for either sign.

    This is the precise invariant the signed tolerance broke.
    """
    y = np.array([1, 1, 0, 0], dtype=np.int64)
    far_below = threshold - 10.0 * (abs(threshold) + 1.0)
    scores = np.array([threshold, far_below, threshold, far_below])

    conf_mat = mmu.confusion_matrix(y, scores=scores, threshold=float(threshold)).ravel()

    assert conf_mat[3] == 1, f"score == threshold ({threshold}) was not classified positive"
    assert conf_mat[1] == 1, f"score == threshold ({threshold}) was not classified positive"


@pytest.mark.parametrize("threshold", THRESHOLDS)
def test_confusion_matrix_matches_reference(threshold):
    """Confusion matrix must match the tolerance-aware reference."""
    rng = np.random.default_rng(1234)
    scores = rng.random(2000) * 4.0 - 2.0
    y = (rng.random(2000) > 0.6).astype(np.int64)

    conf_mat = mmu.confusion_matrix(y, scores=scores, threshold=float(threshold)).ravel()

    assert np.array_equal(conf_mat, _reference(y, scores, threshold))


@pytest.mark.parametrize("shift", [0.0, -0.5, -5.0, -10.0, 100.0])
def test_thresholds_sweep_with_shifted_scores(shift):
    """The vectorised threshold sweep must agree with the single-threshold path.

    Covers ``confusion_matrices_thresholds``, which duplicated the tolerance
    formula inline rather than sharing it.
    """
    rng = np.random.default_rng(99)
    scores = rng.random(1500) * 2.0 + shift
    y = (rng.random(1500) > 0.5).astype(np.int64)
    thresholds = np.linspace(scores.min(), scores.max(), 25)

    swept = mmu.confusion_matrices_thresholds(y, scores, thresholds)
    expected = np.stack([_reference(y, scores, float(t)) for t in thresholds])

    assert np.array_equal(swept, expected)


def test_negative_threshold_reported_case():
    """The exact case from the C-2 report.

    Previously returned ``[2 0 1 1]`` instead of ``[1 1 0 2]`` -- two of four
    observations in the wrong cell.
    """
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([-3.0, -2.0, -2.0, 3.0])

    conf_mat = mmu.confusion_matrix(y, scores=scores, threshold=-2.0).ravel()

    assert np.array_equal(conf_mat, np.array([1, 1, 0, 2]))


def test_tolerance_band_is_symmetric_in_sign():
    """The tolerance width must depend on |threshold|, not its sign.

    A threshold and its negation should admit the same band width, so mirrored
    data produces mirrored counts.
    """
    t = 5.0
    tol = ATOL + RTOL * t
    y = np.array([1, 1], dtype=np.int64)

    # just inside the band below the threshold, for +t and -t
    pos = mmu.confusion_matrix(y, scores=np.array([t - tol * 0.5, t - tol * 0.5]), threshold=t).ravel()
    neg = mmu.confusion_matrix(y, scores=np.array([-t - tol * 0.5, -t - tol * 0.5]), threshold=-t).ravel()

    assert np.array_equal(pos, neg), f"sign asymmetry: +t gave {pos}, -t gave {neg}"
    assert pos[3] == 2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_negative_threshold_for_both_score_dtypes(dtype):
    """float32 scores must behave like float64 (the tolerance was narrowed)."""
    y = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([-3.0, -2.0, -2.0, 3.0], dtype=dtype)

    conf_mat = mmu.confusion_matrix(y, scores=scores, threshold=dtype(-2.0).item()).ravel()

    assert np.array_equal(conf_mat, np.array([1, 1, 0, 2]))

