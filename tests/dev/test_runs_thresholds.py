"""Regression tests for `confusion_matrices_runs_thresholds` (C-5).

Two defects:

1. ``fill`` was documented as "value to fill when a metric is not defined" but
   was never referenced in the body. The function returns counts only and
   computes no metric, so the parameter could not have had any effect. Callers
   passing it got a silent no-op. It has been **removed**.
2. ``y.ndim`` was read *before* ``check_array`` coerced ``y``, so a list input
   raised ``AttributeError`` instead of being accepted, and ``y.shape[1]`` was
   indexed without first confirming ``y`` was 2-D.
"""

import inspect

import numpy as np
import pytest

import mmu


@pytest.fixture
def runs_data():
    rng = np.random.default_rng(3)
    n_obs, n_runs = 200, 4
    scores = rng.random((n_obs, n_runs))
    y = (rng.random((n_obs, n_runs)) > 0.6).astype(np.int64)
    thresholds = np.linspace(0.1, 0.9, 6)
    return y, scores, thresholds


def test_fill_parameter_is_gone():
    """`fill` must no longer exist in the signature."""
    params = inspect.signature(mmu.confusion_matrices_runs_thresholds).parameters

    assert "fill" not in params
    assert list(params) == ["y", "scores", "thresholds", "n_obs", "obs_axis"]


def test_passing_fill_raises(runs_data):
    """Passing the removed parameter must fail loudly, not be ignored."""
    y, scores, thresholds = runs_data

    with pytest.raises(TypeError, match="fill"):
        mmu.confusion_matrices_runs_thresholds(y, scores, thresholds, fill=0.0)


def test_obs_axis_still_reachable_positionally(runs_data):
    """`obs_axis` moved up one position now that `fill` is gone."""
    y, scores, thresholds = runs_data

    positional = mmu.confusion_matrices_runs_thresholds(y, scores, thresholds, None, 0)
    keyword = mmu.confusion_matrices_runs_thresholds(y, scores, thresholds, obs_axis=0)

    assert np.array_equal(positional, keyword)


def test_accepts_list_for_y(runs_data):
    """A list `y` must be coerced, not raise AttributeError on `.ndim`."""
    _, scores, thresholds = runs_data
    n_obs = scores.shape[0]
    y_list = [int(i % 2) for i in range(n_obs)]

    result = mmu.confusion_matrices_runs_thresholds(y_list, scores, thresholds)

    expected = mmu.confusion_matrices_runs_thresholds(np.asarray(y_list), scores, thresholds)
    assert np.array_equal(result, expected)


def test_one_dimensional_y_is_tiled(runs_data):
    """A shared 1-D `y` must be broadcast across runs."""
    _, scores, thresholds = runs_data
    n_obs, n_runs = scores.shape
    y_1d = (np.arange(n_obs) % 3 == 0).astype(np.int64)

    from_1d = mmu.confusion_matrices_runs_thresholds(y_1d, scores, thresholds)
    from_2d = mmu.confusion_matrices_runs_thresholds(np.tile(y_1d[:, None], n_runs), scores, thresholds)

    assert np.array_equal(from_1d, from_2d)


def test_column_vector_y_is_tiled(runs_data):
    """An (n, 1) `y` must be broadcast across runs."""
    _, scores, thresholds = runs_data
    n_obs, n_runs = scores.shape
    y_col = (np.arange(n_obs) % 3 == 0).astype(np.int64)[:, None]

    from_col = mmu.confusion_matrices_runs_thresholds(y_col, scores, thresholds)
    from_full = mmu.confusion_matrices_runs_thresholds(np.tile(y_col, n_runs), scores, thresholds)

    assert np.array_equal(from_col, from_full)


def test_counts_match_per_run_reference(runs_data):
    """Counts must match a straightforward per-run, per-threshold reference."""
    y, scores, thresholds = runs_data
    n_obs, n_runs = scores.shape

    result = mmu.confusion_matrices_runs_thresholds(y, scores, thresholds)

    for ti, t in enumerate(thresholds):
        tol = 1e-8 + 1e-5 * abs(t)
        for r in range(n_runs):
            expected = np.bincount(
                y[:, r] * 2 + (scores[:, r] >= t - tol).astype(int), minlength=4
            )
            assert np.array_equal(result[ti, :, r], expected), f"threshold {ti}, run {r}"


