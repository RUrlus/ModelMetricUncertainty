"""Regression tests for the optional `thresholds` argument on curve functions (C-12).

`precision_recall_curve` and `ROC_curve` both declare `thresholds=None` in
their signature and document it as optional, but passed the value straight to
`check_array`, which coerces `None` to `array(nan)` and raises
"Input contains NaN". The documented default therefore could never be used --
both public functions failed outright when called as documented.

The fix falls back to `auto_thresholds(scores)`, matching what the curve
uncertainty classes in `methods/curvebase.py` already do when `thresholds` is
not supplied.
"""

import numpy as np
import pytest

import mmu
from mmu.metrics.utils import auto_thresholds

CURVE_FUNCS = [mmu.precision_recall_curve, mmu.ROC_curve]
CURVE_IDS = ["precision_recall_curve", "ROC_curve"]


def _scores(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.random(n)
    y = (rng.random(n) < scores).astype(np.int64)
    return y, scores


@pytest.mark.parametrize("func", CURVE_FUNCS, ids=CURVE_IDS)
def test_thresholds_none_does_not_raise(func):
    """The documented default previously raised ValueError('Input contains NaN')."""
    y, scores = _scores()

    first, second = func(y, scores)

    assert first.ndim == 1
    assert second.ndim == 1
    assert first.size == second.size
    assert first.size > 0


@pytest.mark.parametrize("func", CURVE_FUNCS, ids=CURVE_IDS)
def test_thresholds_none_matches_explicit_auto_thresholds(func):
    """Omitting `thresholds` must equal passing `auto_thresholds(scores)`."""
    y, scores = _scores()

    implicit_a, implicit_b = func(y, scores)
    explicit_a, explicit_b = func(y, scores, thresholds=auto_thresholds(scores))

    np.testing.assert_allclose(implicit_a, explicit_a)
    np.testing.assert_allclose(implicit_b, explicit_b)


@pytest.mark.parametrize("func", CURVE_FUNCS, ids=CURVE_IDS)
def test_thresholds_none_results_are_finite(func):
    y, scores = _scores()

    first, second = func(y, scores)

    assert np.isfinite(first).all()
    assert np.isfinite(second).all()


@pytest.mark.parametrize("func", CURVE_FUNCS, ids=CURVE_IDS)
def test_explicit_thresholds_still_honoured(func):
    """The fix must not change behaviour when `thresholds` is supplied."""
    y, scores = _scores()
    thresholds = np.linspace(0.01, 0.99, 37)

    first, second = func(y, scores, thresholds=thresholds)

    assert first.size == thresholds.size
    assert second.size == thresholds.size


@pytest.mark.parametrize("func", CURVE_FUNCS, ids=CURVE_IDS)
def test_thresholds_none_with_return_df(func):
    """`return_df=True` also indexes `thresholds`; it must see a real array."""
    y, scores = _scores()

    df = func(y, scores, return_df=True)

    assert "thresholds" in df.columns
    assert len(df) > 0
    assert np.isfinite(df["thresholds"].to_numpy()).all()


@pytest.mark.parametrize("func", CURVE_FUNCS, ids=CURVE_IDS)
def test_float32_scores(func):
    y, scores = _scores()

    first, second = func(y, scores.astype(np.float32))

    assert first.size > 0
    assert np.isfinite(first).all()
    assert np.isfinite(second).all()


# ---------------------------------------------------------------------------
# Single-element threshold arrays (C-8 lower-bound gap)
# ---------------------------------------------------------------------------
# `_check_dims` counted only axes with `size > 1`, so `np.array([0.5])` scored
# zero non-trivial dimensions and was rejected as "must have at least 1
# dimensions". A single-threshold sweep was therefore impossible, even though
# trivial axes are exactly what the *upper* bound is designed to forgive.


def test_single_element_thresholds_accepted():
    """`confusion_matrices_thresholds` with one threshold must work."""
    y, scores = _scores()

    cm = mmu.confusion_matrices_thresholds(y, scores, np.array([0.5]))

    assert cm.shape == (1, 4)
    assert cm.sum() == y.size


def test_single_element_matches_scalar_threshold_path():
    """One threshold via the sweep must equal the single-matrix entry point."""
    y, scores = _scores()

    swept = mmu.confusion_matrices_thresholds(y, scores, np.array([0.5]))
    single = mmu.confusion_matrix(y, scores=scores, threshold=0.5)

    np.testing.assert_array_equal(swept.ravel(), single.ravel())


@pytest.mark.parametrize("func", CURVE_FUNCS, ids=CURVE_IDS)
def test_single_element_thresholds_on_curves(func):
    y, scores = _scores()

    first, second = func(y, scores, thresholds=np.array([0.5]))

    assert first.size == 1
    assert second.size == 1


def test_zero_dim_scalar_still_rejected():
    """Relaxing the lower bound must not let a true 0-d scalar through."""
    y, scores = _scores()

    with pytest.raises((ValueError, TypeError)):
        mmu.confusion_matrices_thresholds(y, scores, np.float64(0.5))


