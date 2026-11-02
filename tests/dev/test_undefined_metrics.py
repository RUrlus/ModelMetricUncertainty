"""Regression tests for undefined-metric handling (C-3).

``accuracy`` was the only metric in ``binary_metrics`` whose division was not
guarded, so an empty confusion matrix produced ``0.0 / 0.0`` and returned NaN
to the caller, silently ignoring ``fill``.
"""

import itertools

import numpy as np
import pytest

import mmu

ACC = 8
FILLS = [0.0, 0.5, 1.0]


@pytest.mark.parametrize("fill", FILLS)
def test_accuracy_on_empty_conf_mat_uses_fill(fill):
    """An empty confusion matrix must yield `fill`, not NaN."""
    metrics = mmu.binary_metrics_confusion_matrix(np.zeros(4, dtype=np.int64), fill=fill)

    assert not np.isnan(metrics[ACC]), "accuracy is NaN for an empty confusion matrix"
    assert metrics[ACC] == fill


@pytest.mark.parametrize("fill", FILLS)
@pytest.mark.parametrize("pattern", list(itertools.product([0, 4], repeat=4)), ids=lambda p: str(p))
def test_no_metric_is_nan(pattern, fill):
    """No metric may be NaN for any zero/non-zero confusion matrix pattern."""
    metrics = mmu.binary_metrics_confusion_matrix(np.array(pattern, dtype=np.int64), fill=fill)

    nan_idx = np.where(np.isnan(metrics))[0]
    assert nan_idx.size == 0, f"NaN at metric indices {nan_idx.tolist()} for cm={pattern}"


@pytest.mark.parametrize("pattern", list(itertools.product([0, 4], repeat=4)), ids=lambda p: str(p))
def test_all_metrics_finite(pattern):
    """Metrics must be finite -- never inf -- for any pattern."""
    metrics = mmu.binary_metrics_confusion_matrix(np.array(pattern, dtype=np.int64), fill=0.0)

    assert np.isfinite(metrics).all(), f"non-finite metrics for cm={pattern}: {metrics}"


def test_accuracy_matches_sklearn():
    """Guarding the division must not perturb the normal path."""
    from sklearn.metrics import accuracy_score

    rng = np.random.default_rng(7)
    y = (rng.random(3000) > 0.7).astype(np.int64)
    yhat = (rng.random(3000) > 0.6).astype(np.int64)

    _, metrics = mmu.binary_metrics(y, yhat=yhat)

    assert np.isclose(metrics[ACC], accuracy_score(y, yhat))


@pytest.mark.parametrize(
    "pattern,expected",
    [
        ((10, 0, 0, 10), 1.0),  # perfect
        ((0, 10, 10, 0), 0.0),  # all wrong
        ((5, 5, 5, 5), 0.5),  # half
    ],
)
def test_accuracy_known_values(pattern, expected):
    """Spot-check accuracy against hand-computed values."""
    metrics = mmu.binary_metrics_confusion_matrix(np.array(pattern, dtype=np.int64))

    assert np.isclose(metrics[ACC], expected)


@pytest.mark.parametrize("fill", FILLS)
def test_f1_undefined_uses_fill(fill):
    """F1 is undefined when its denominator is zero; it must use `fill`.

    Restructuring the F1 guard must preserve the original semantics.
    """
    # TN=0, FP=0, FN=0, TP=n -> negative F1 denominator (2*TN + FP + FN) == 0
    metrics = mmu.binary_metrics_confusion_matrix(np.array([0, 0, 0, 8], dtype=np.int64), fill=fill)
    assert metrics[4] == fill

    # TN=n, FP=0, FN=0, TP=0 -> positive F1 denominator (2*TP + FP + FN) == 0
    metrics = mmu.binary_metrics_confusion_matrix(np.array([8, 0, 0, 0], dtype=np.int64), fill=fill)
    assert metrics[5] == fill


def test_f1_matches_sklearn():
    """The restructured F1 guard must not change defined values."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(11)
    y = (rng.random(3000) > 0.7).astype(np.int64)
    yhat = (rng.random(3000) > 0.6).astype(np.int64)

    _, metrics = mmu.binary_metrics(y, yhat=yhat)

    assert np.isclose(metrics[5], f1_score(y, yhat))
    assert np.isclose(metrics[4], f1_score(y, yhat, pos_label=0))

