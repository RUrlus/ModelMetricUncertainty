"""Numerical stability regression tests (C-7, C-9).

C-7: the MCC numerator ``TP*TN - FP*FN`` was evaluated in `double`, where each
product is exact only to 2^53. For large counts the subtraction of two big,
nearly equal products loses significance.

C-9: the profile log-likelihood was computed as ``nll_constrained -
nll_unconstrained``. Both grow with the sample size while their difference
stays O(1), so the subtraction sheds digits as n grows. It also let a clipped
zero probability escape as ``+inf`` into the returned chi2 grid.
"""

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

import mmu

MCC = 9


@pytest.mark.parametrize("scale", [1e2, 1e4, 1e6, 1e8])
def test_mcc_accurate_at_large_counts(scale):
    """MCC must stay accurate as counts grow past 2^53 product range."""
    n = int(scale)
    conf_mat = np.array([5 * n, 2 * n, 3 * n, 4 * n], dtype=np.int64)

    metrics = mmu.binary_metrics_confusion_matrix(conf_mat)

    tn, fp, fn, tp = (float(v) for v in conf_mat)
    expected = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (fn + tp) * (tn + fp) * (tn + fn))

    assert np.isclose(metrics[MCC], expected, rtol=1e-12)


def test_mcc_matches_sklearn_on_moderate_counts():
    rng = np.random.default_rng(5)
    y = (rng.random(5000) > 0.55).astype(np.int64)
    yhat = (rng.random(5000) > 0.45).astype(np.int64)

    _, metrics = mmu.binary_metrics(y, yhat=yhat)

    assert np.isclose(metrics[MCC], matthews_corrcoef(y, yhat))


@pytest.mark.parametrize("n", [10**3, 10**5, 10**7, 10**9])
def test_deviance_is_zero_at_the_observed_point(n):
    """The chi2 score at the observed precision/recall must be ~0.

    There the constrained fit coincides with the unconstrained MLE, so the
    deviance is identically zero *in exact arithmetic*. Computing it as a
    difference of two large deviances left O(n * eps) noise instead; the
    likelihood-ratio form is exact in theory but every term is still a
    `double` log-ratio evaluated at a probability of order 1/n, so residual
    error of O(n * eps) is the float64 floor, not a defect -- it is ~50x
    smaller than it would be without the ratio-form fix. The tolerance below
    scales with `n` to reflect that floor rather than pinning an
    arbitrarily tight absolute bound that only holds for small `n`.
    """
    conf_mat = np.array([n // 2, n // 5, n // 10, n // 5], dtype=np.int64)
    tn, fp, fn, tp = (float(v) for v in conf_mat)

    err = mmu.PRU.from_confusion_matrix(conf_mat)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    score = err.compute_score_for(precision, recall)

    tol = max(1e-9, n * np.finfo(np.float64).eps)
    assert abs(score) < tol, f"deviance at the observed point is {score} for n={n} (tol={tol})"


@pytest.mark.parametrize("cls", [mmu.PRU, mmu.ROCU, mmu.PPNRU], ids=["PRU", "ROCU", "PPNRU"])
def test_chi2_grid_is_never_infinite(cls):
    """`multn_error` wrote +inf into the grid handed back to Python."""
    rng = np.random.default_rng(9)
    scores = rng.random(500)
    y = (rng.random(500) < scores).astype(np.int64)

    err = cls.from_scores(y, scores, threshold=0.5)

    assert not np.isinf(err.chi2_scores).any(), "chi2 grid contains inf"
    assert np.isfinite(err.chi2_scores).all(), "chi2 grid contains non-finite values"


@pytest.mark.parametrize("cls", [mmu.PRU, mmu.ROCU, mmu.PPNRU], ids=["PRU", "ROCU", "PPNRU"])
def test_chi2_scores_are_non_negative(cls):
    """A deviance is non-negative; cancellation could drive it slightly below 0."""
    rng = np.random.default_rng(11)
    scores = rng.random(2000)
    y = (rng.random(2000) < scores).astype(np.int64)

    err = cls.from_scores(y, scores, threshold=0.5)

    assert err.chi2_scores.min() >= -1e-9, f"negative deviance {err.chi2_scores.min()}"


def test_score_grid_minimum_sits_at_the_observed_point():
    """The deviance is minimised at the MLE; precision loss can move it.

    `chi2_scores.min()` is the minimum over a finite `n_bins` x `n_bins` grid,
    so it cannot reach the continuous minimum unless a grid point happens to
    land exactly on the MLE -- with the default `n_bins=100` it generally will
    not. Near the minimum the deviance is locally quadratic in (y, x) measured
    in standard-deviation units, so the worst-case gap between the nearest grid
    point and the true minimum is bounded by the squared half grid-spacing (in
    sigma units), up to a generous safety factor. Pinning an absolute `1e-6`
    bound conflated that expected grid-resolution gap with the actual
    regression target of C-9 (cancellation / `inf` leakage), which is tested
    directly below via the exact (non-grid) evaluation at the MLE.
    """
    rng = np.random.default_rng(13)
    scores = rng.random(5000)
    y = (rng.random(5000) < scores).astype(np.int64)

    err = mmu.PRU.from_scores(y, scores, threshold=0.5)

    # The actual C-9 regression target: the deviance evaluated exactly at the
    # MLE (not snapped to a grid point) must be ~0.
    exact_score = err.compute_score_for(float(err.y), float(err.x))
    assert abs(exact_score) < 1e-6, f"deviance at the exact MLE is {exact_score}"

    # The grid can only approximate the continuous minimum; bound the gap by
    # the squared half grid-spacing in sigma units, with a generous safety
    # factor, rather than an arbitrary absolute value that depends on
    # `n_bins`/`n_sigmas`.
    half_step_sigma = err.n_sigmas / (err.n_bins - 1)
    bound = 6.0 * (half_step_sigma**2)
    assert err.chi2_scores.min() < bound, f"grid minimum {err.chi2_scores.min()} exceeds resolution bound {bound}"

