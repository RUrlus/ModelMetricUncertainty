"""Equivalence tests for the sort-based threshold sweep (P-1).

`confusion_matrices_thresholds` dispatches between two algorithms:

* the direct sweep, O(n_thresholds * n_obs), which rescans every observation
  for every threshold;
* the sort-based sweep, O((n_obs + n_thresholds) * log n_obs), which sorts the
  scores once and answers each threshold with a binary search.

`core::prefer_sorted_sweep` picks between them on size alone, so the two must
be observationally identical -- a user must never be able to tell which ran.
These tests pin that equivalence, including the tolerance semantics around the
threshold and the degenerate label distributions.

The dispatch is internal and deliberately not exposed, so the tests drive it
by varying `n_thresholds` across the crossover rather than by forcing a path.
"""

import numpy as np
import pytest

import mmu

# Comfortably either side of `prefer_sorted_sweep` for the n_obs used here.
SMALL_SWEEP = 8
LARGE_SWEEP = 4000


def _data(n_obs, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    scores = rng.random(n_obs).astype(dtype)
    y = (rng.random(n_obs) < scores).astype(np.int64)
    return y, scores


def _reference(y, scores, thresholds):
    """Tolerance-aware oracle, computed independently of the extension.

    Mirrors `core::scaled_tolerance`: an observation is predicted positive
    when `score >= threshold - (atol + rtol * |threshold|)`.
    """
    out = np.empty((thresholds.size, 4), dtype=np.int64)
    y_bool = y.astype(bool)
    for i, t in enumerate(thresholds):
        tol = 1e-8 + 1e-5 * abs(float(t))
        pred = scores >= (t - tol)
        out[i, 0] = np.sum(~y_bool & ~pred)  # TN
        out[i, 1] = np.sum(~y_bool & pred)  # FP
        out[i, 2] = np.sum(y_bool & ~pred)  # FN
        out[i, 3] = np.sum(y_bool & pred)  # TP
    return out


@pytest.mark.parametrize("n_obs", [5_000, 50_000])
def test_large_sweep_matches_reference(n_obs):
    """The sorted path must agree with an independent oracle."""
    y, scores = _data(n_obs)
    thresholds = np.linspace(0.001, 0.999, LARGE_SWEEP)

    got = mmu.confusion_matrices_thresholds(y, scores, thresholds)

    np.testing.assert_array_equal(got, _reference(y, scores, thresholds))


def test_both_paths_agree_on_shared_thresholds():
    """Same thresholds, two dispatch decisions, identical answers.

    The small sweep takes the direct path and the large sweep the sorted one;
    the rows for the thresholds they have in common must match exactly.
    """
    y, scores = _data(20_000)
    shared = np.linspace(0.05, 0.95, SMALL_SWEEP)
    padded = np.concatenate([shared, np.linspace(0.001, 0.999, LARGE_SWEEP)])

    direct = mmu.confusion_matrices_thresholds(y, scores, shared)
    sorted_path = mmu.confusion_matrices_thresholds(y, scores, padded)

    np.testing.assert_array_equal(direct, sorted_path[: shared.size])


def test_rows_always_sum_to_n_obs():
    """Every observation lands in exactly one cell, on either path."""
    n_obs = 20_000
    y, scores = _data(n_obs)

    for n_t in (SMALL_SWEEP, LARGE_SWEEP):
        cm = mmu.confusion_matrices_thresholds(y, scores, np.linspace(0.001, 0.999, n_t))
        assert np.all(cm.sum(axis=1) == n_obs), f"cells do not sum to n_obs for n_t={n_t}"
        assert np.all(cm >= 0), f"negative cell count for n_t={n_t}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_dtypes_agree(dtype):
    y, scores = _data(20_000, dtype=dtype)
    thresholds = np.linspace(0.001, 0.999, LARGE_SWEEP).astype(dtype)

    cm = mmu.confusion_matrices_thresholds(y, scores, thresholds)

    assert np.all(cm.sum(axis=1) == y.size)


@pytest.mark.parametrize(
    ("label", "make"),
    [
        ("all_positive", lambda n: np.ones(n, dtype=np.int64)),
        ("all_negative", lambda n: np.zeros(n, dtype=np.int64)),
        ("single_positive", lambda n: np.eye(1, n, dtype=np.int64).ravel()),
    ],
)
def test_degenerate_label_distributions(label, make):
    """Prefix-sum bookkeeping is easiest to get wrong at the extremes."""
    n_obs = 10_000
    rng = np.random.default_rng(3)
    scores = rng.random(n_obs)
    y = make(n_obs)
    thresholds = np.linspace(0.001, 0.999, LARGE_SWEEP)

    got = mmu.confusion_matrices_thresholds(y, scores, thresholds)

    np.testing.assert_array_equal(got, _reference(y, scores, thresholds), err_msg=label)


def test_thresholds_outside_score_range():
    """Binary search must clamp correctly past both ends of the sorted array."""
    y, scores = _data(20_000)
    # scores are in [0, 1); these bracket it on both sides
    thresholds = np.concatenate(
        [
            np.full(LARGE_SWEEP // 2, -5.0),
            np.full(LARGE_SWEEP // 2, 5.0),
        ]
    )

    cm = mmu.confusion_matrices_thresholds(y, scores, thresholds)

    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    # threshold below every score -> everything predicted positive
    assert np.all(cm[: LARGE_SWEEP // 2, 3] == n_pos)
    assert np.all(cm[: LARGE_SWEEP // 2, 1] == n_neg)
    # threshold above every score -> nothing predicted positive
    assert np.all(cm[LARGE_SWEEP // 2 :, 3] == 0)
    assert np.all(cm[LARGE_SWEEP // 2 :, 0] == n_neg)


def test_duplicate_scores():
    """Ties must all fall on the same side of the cut."""
    n_obs = 20_000
    rng = np.random.default_rng(5)
    # only 5 distinct score values -> massive tie groups
    scores = rng.choice(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), size=n_obs)
    y = (rng.random(n_obs) < scores).astype(np.int64)
    thresholds = np.linspace(0.001, 0.999, LARGE_SWEEP)

    got = mmu.confusion_matrices_thresholds(y, scores, thresholds)

    np.testing.assert_array_equal(got, _reference(y, scores, thresholds))


def test_negative_scores_and_thresholds():
    """C-2's tolerance sign fix must hold on the sorted path too."""
    n_obs = 20_000
    rng = np.random.default_rng(7)
    scores = rng.normal(0.0, 3.0, n_obs)
    y = (rng.random(n_obs) < 0.5).astype(np.int64)
    thresholds = np.linspace(-8.0, 8.0, LARGE_SWEEP)

    got = mmu.confusion_matrices_thresholds(y, scores, thresholds)

    np.testing.assert_array_equal(got, _reference(y, scores, thresholds))

