"""Regression tests for degenerate confusion matrices (C-1).

`Profile::bvn_sigma` previously divided by zero whenever a metric's denominator
was zero, producing NaN grid bounds and a wholly non-finite chi2 grid -- with no
exception and no warning. These tests pin the contract that every representable
confusion matrix either yields a finite, well-ordered grid or raises.
"""

import itertools

import numpy as np
import pytest

import mmu

POINT_CLASSES = [mmu.PRU, mmu.ROCU, mmu.PPNRU]
POINT_IDS = ["PRU", "ROCU", "PPNRU"]

# TN, FP, FN, TP
DEGENERATE = [
    (10, 0, 5, 0),  # no positive predictions -> precision undefined
    (10, 5, 0, 0),  # no actual positives -> recall undefined
    (0, 0, 0, 10),  # only true positives
    (10, 0, 0, 0),  # only true negatives
    (0, 10, 0, 0),  # only false positives
    (0, 0, 10, 0),  # only false negatives
    (1, 0, 0, 0),  # single observation
    (0, 0, 0, 1),  # single observation
]


@pytest.mark.parametrize("cls", POINT_CLASSES, ids=POINT_IDS)
@pytest.mark.parametrize("cm", DEGENERATE, ids=[str(c) for c in DEGENERATE])
def test_degenerate_conf_mat_yields_finite_grid(cls, cm):
    """A degenerate confusion matrix must not produce NaN bounds or scores."""
    err = cls.from_confusion_matrix(np.array(cm, dtype=np.int64))

    assert np.isfinite(err.y_bounds).all(), f"non-finite y_bounds {err.y_bounds}"
    assert np.isfinite(err.x_bounds).all(), f"non-finite x_bounds {err.x_bounds}"
    assert np.isfinite(err.chi2_scores).any(), "chi2 grid is wholly non-finite"


@pytest.mark.parametrize("cls", POINT_CLASSES, ids=POINT_IDS)
@pytest.mark.parametrize("cm", DEGENERATE, ids=[str(c) for c in DEGENERATE])
def test_degenerate_conf_mat_bounds_are_ordered(cls, cm):
    """Bounds must be non-empty and correctly ordered, never inverted.

    A zero sigma used to collapse the range to a single point or invert it,
    which silently produced an empty or descending grid axis.
    """
    err = cls.from_confusion_matrix(np.array(cm, dtype=np.int64))

    assert err.y_bounds[0] <= err.y_bounds[1], f"inverted y_bounds {err.y_bounds}"
    assert err.x_bounds[0] <= err.x_bounds[1], f"inverted x_bounds {err.x_bounds}"


@pytest.mark.parametrize("cls", POINT_CLASSES, ids=POINT_IDS)
@pytest.mark.parametrize("cm", list(itertools.product([0, 3], repeat=4)), ids=lambda c: str(c))
def test_all_zero_nonzero_patterns(cls, cm):
    """Exhaustive sweep of every zero/non-zero pattern.

    The all-zero matrix carries no information and must raise; every other
    pattern must produce a finite, ordered grid.
    """
    conf_mat = np.array(cm, dtype=np.int64)

    if conf_mat.sum() == 0:
        with pytest.raises(ValueError, match="no observations"):
            cls.from_confusion_matrix(conf_mat)
        return

    err = cls.from_confusion_matrix(conf_mat)
    assert np.isfinite(err.y_bounds).all()
    assert np.isfinite(err.x_bounds).all()
    assert err.y_bounds[0] <= err.y_bounds[1]
    assert err.x_bounds[0] <= err.x_bounds[1]
    assert np.isfinite(err.chi2_scores).any()


@pytest.mark.parametrize("cls", POINT_CLASSES, ids=POINT_IDS)
def test_empty_conf_mat_raises(cls):
    """An all-zero confusion matrix must raise rather than return a NaN grid."""
    with pytest.raises(ValueError, match="no observations"):
        cls.from_confusion_matrix(np.zeros(4, dtype=np.int64))


@pytest.mark.parametrize("cls", POINT_CLASSES, ids=POINT_IDS)
def test_no_positive_predictions_reported_case(cls):
    """The exact matrix from the C-1 report.

    ``[TN=10, FP=0, FN=5, TP=0]`` previously gave ``y_bounds = [nan, nan]`` and
    a chi2 grid that was 0% finite.
    """
    err = cls.from_confusion_matrix(np.array([10, 0, 5, 0], dtype=np.int64))

    assert not np.isnan(err.y_bounds).any()
    assert not np.isnan(err.x_bounds).any()
    assert np.isfinite(err.chi2_scores).mean() > 0.5

