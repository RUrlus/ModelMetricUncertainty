"""Equivalence tests for the tiled curve-uncertainty grid (P-3).

`multn_grid_curve_error_mt` builds the minimum-chi2 surface over a curve of
confusion matrices. It was parallelised by splitting the *confusion matrices*
across threads, which meant two threads could target the same output cell --
the regions different matrices cover overlap. That forced a private
`n_y_bins * n_x_bins` scratch block per thread (64 MB for a 1000x1000 grid on
8 threads), which had to be allocated, filled, and reduced back down.

It is now parallelised by tiling the *output grid* into disjoint bands of
rows, so threads write straight into the result: no scratch buffer, no fill
over it, and no reduction pass.

The serial routine `multn_grid_curve_error` was not changed -- it already
wrote directly -- which makes it an independent oracle. These tests pin the
multithreaded path to it exactly. A tiling bug would show up as a seam at a
tile boundary, so the parametrisation deliberately includes grid sizes that
do not divide evenly by the tile count.
"""

import numpy as np
import pytest

import mmu

CURVE_CLASSES = [mmu.PRCU, mmu.ROCCU, mmu.PPNRCU]
CURVE_IDS = ["PRCU", "ROCCU", "PPNRCU"]


def _data(n_obs, seed=0):
    rng = np.random.default_rng(seed)
    scores = rng.random(n_obs)
    y = (rng.random(n_obs) < scores).astype(np.int64)
    return y, scores


@pytest.mark.parametrize("cls", CURVE_CLASSES, ids=CURVE_IDS)
@pytest.mark.parametrize(
    "n_bins",
    # 97 and 257 are prime-ish: they cannot divide evenly into tiles, so any
    # off-by-one in the row-band arithmetic leaves an unwritten seam.
    [97, 100, 257],
)
def test_tiled_matches_serial(cls, n_bins):
    y, scores = _data(5_000)
    thresholds = np.linspace(0.01, 0.99, 150)

    serial = cls.from_scores(y, scores, thresholds=thresholds, n_bins=n_bins, n_threads=1)
    tiled = cls.from_scores(y, scores, thresholds=thresholds, n_bins=n_bins, n_threads=8)

    np.testing.assert_array_equal(serial.chi2_scores, tiled.chi2_scores)


@pytest.mark.parametrize("n_threads", [1, 2, 3, 8])
def test_result_is_independent_of_thread_count(n_threads):
    """Tiling must not make the answer depend on how many threads ran."""
    y, scores = _data(8_000, seed=1)
    thresholds = np.linspace(0.01, 0.99, 200)

    reference = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=128, n_threads=1)
    got = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=128, n_threads=n_threads)

    np.testing.assert_array_equal(reference.chi2_scores, got.chi2_scores)


@pytest.mark.parametrize("cls", CURVE_CLASSES, ids=CURVE_IDS)
def test_every_cell_is_written(cls):
    """No tile may leave a gap.

    An unwritten cell would retain whatever the allocation happened to hold.
    Every cell must be either a real deviance or the fill sentinel, and the
    grid must contain at least one genuinely computed value.
    """
    y, scores = _data(5_000, seed=2)
    thresholds = np.linspace(0.01, 0.99, 150)

    err = cls.from_scores(y, scores, thresholds=thresholds, n_bins=101, n_threads=8)
    grid = err.chi2_scores

    assert np.isfinite(grid).all(), "grid contains non-finite values"
    assert (grid >= 0.0).all(), "deviance cannot be negative"
    # MULT_DEFAULT_CHI2_SCORE is 65.0; untouched cells keep it.
    assert (grid <= 65.0).all(), "grid exceeds the fill sentinel"
    assert grid.min() < 1.0, "no cell near the curve was actually evaluated"


def test_single_confusion_matrix():
    """A one-matrix curve exercises the tile/region intersection edge."""
    y, scores = _data(2_000, seed=3)
    thresholds = np.array([0.5])

    serial = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=100, n_threads=1)
    tiled = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=100, n_threads=8)

    np.testing.assert_array_equal(serial.chi2_scores, tiled.chi2_scores)


def test_more_tiles_than_rows():
    """`n_bins` below the tile count leaves some tiles empty; they must be skipped."""
    y, scores = _data(2_000, seed=4)
    thresholds = np.linspace(0.05, 0.95, 40)

    serial = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=4, n_threads=1)
    tiled = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=4, n_threads=8)

    np.testing.assert_array_equal(serial.chi2_scores, tiled.chi2_scores)


def test_asymmetric_grid():
    """Separate y/x bin counts must not transpose inside the tiling."""
    y, scores = _data(5_000, seed=5)
    thresholds = np.linspace(0.01, 0.99, 120)

    serial = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=[64, 96], n_threads=1)
    tiled = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=[64, 96], n_threads=8)

    assert serial.chi2_scores.shape == (64, 96)
    np.testing.assert_array_equal(serial.chi2_scores, tiled.chi2_scores)


def test_large_grid_has_no_scratch_blowup():
    """Regression guard for the allocation this change removed.

    The old design allocated `n_threads * n_y_bins * n_x_bins` doubles --
    64 MB here. The tiled version writes directly into the result, so peak
    scratch is O(n_conf_mats). This asserts the call completes and is exact;
    it is the size that used to be painful, not the arithmetic.
    """
    y, scores = _data(10_000, seed=6)
    thresholds = np.linspace(0.01, 0.99, 100)

    serial = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=1000, n_threads=1)
    tiled = mmu.PRCU.from_scores(y, scores, thresholds=thresholds, n_bins=1000, n_threads=8)

    assert tiled.chi2_scores.shape == (1000, 1000)
    np.testing.assert_array_equal(serial.chi2_scores, tiled.chi2_scores)

