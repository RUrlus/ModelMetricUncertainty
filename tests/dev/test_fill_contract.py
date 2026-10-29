"""Regression tests for the `fill` contract on undefined metrics (C-4).

TNR/FPR and TPR/FNR are complements, but the undefined branch filled one member
of each pair with `fill` and hard-coded the other to ``1.0``. With the default
``fill=1.0`` that produced ``recall == 1.0`` *and* ``FNR == 1.0`` -- a pair
summing to 2.0.

The contract is now: an undefined rate is ``fill`` and its complement is
``1 - fill``, so the identity holds for every value of `fill`. This matches the
package's own reference oracle in ``mmu.commons._testing.utils``, which defines
``metrics[6] = 1.0 - metrics[2]`` and ``metrics[7] = 1.0 - metrics[3]``.
"""

import itertools

import numpy as np
import pytest

import mmu

TNR, TPR, FPR, FNR, MCC = 2, 3, 6, 7, 9
FILLS = [0.0, 0.25, 0.5, 1.0]

# TN, FP, FN, TP
NO_POSITIVES = (10, 5, 0, 0)  # P == FN + TP == 0  -> TPR/FNR undefined
NO_NEGATIVES = (0, 0, 5, 10)  # N == TN + FP == 0  -> TNR/FPR undefined


@pytest.mark.parametrize("fill", FILLS)
@pytest.mark.parametrize("pattern", list(itertools.product([0, 4], repeat=4)), ids=lambda p: str(p))
def test_rate_pairs_are_complements(pattern, fill):
    """TNR + FPR == 1 and TPR + FNR == 1, for every pattern and every fill."""
    m = mmu.binary_metrics_confusion_matrix(np.array(pattern, dtype=np.int64), fill=fill)

    assert np.isclose(m[TNR] + m[FPR], 1.0), f"TNR+FPR={m[TNR] + m[FPR]} for cm={pattern}, fill={fill}"
    assert np.isclose(m[TPR] + m[FNR], 1.0), f"TPR+FNR={m[TPR] + m[FNR]} for cm={pattern}, fill={fill}"


@pytest.mark.parametrize("fill", FILLS)
def test_undefined_tpr_uses_fill_and_complement(fill):
    """No actual positives -> TPR == fill, FNR == 1 - fill."""
    m = mmu.binary_metrics_confusion_matrix(np.array(NO_POSITIVES, dtype=np.int64), fill=fill)

    assert m[TPR] == fill
    assert np.isclose(m[FNR], 1.0 - fill)


@pytest.mark.parametrize("fill", FILLS)
def test_undefined_tnr_uses_fill_and_complement(fill):
    """No actual negatives -> TNR == fill, FPR == 1 - fill."""
    m = mmu.binary_metrics_confusion_matrix(np.array(NO_NEGATIVES, dtype=np.int64), fill=fill)

    assert m[TNR] == fill
    assert np.isclose(m[FPR], 1.0 - fill)


def test_default_fill_no_longer_sums_to_two():
    """The exact case from the C-4 report: recall and FNR both returned 1.0."""
    m = mmu.binary_metrics_confusion_matrix(np.array(NO_POSITIVES, dtype=np.int64), fill=1.0)

    assert np.isclose(m[TPR] + m[FNR], 1.0), "recall + FNR must not sum to 2.0"


@pytest.mark.parametrize("fill", FILLS)
@pytest.mark.parametrize("pattern", list(itertools.product([0, 4], repeat=4)), ids=lambda p: str(p))
def test_mcc_is_zero_when_undefined_regardless_of_fill(pattern, fill):
    """MCC is the deliberate exception: 0.0, never `fill`.

    Its denominator vanishes whenever a row or column is empty -- common for a
    single-class predictor -- and scikit-learn returns 0.0 there. Filling with
    the default 1.0 would claim perfect correlation for a degenerate model.
    """
    cm = np.array(pattern, dtype=np.int64)
    tn, fp, fn, tp = (float(v) for v in pattern)
    denominator = (tp + fp) * (fn + tp) * (tn + fp) * (tn + fn)

    m = mmu.binary_metrics_confusion_matrix(cm, fill=fill)

    if denominator == 0.0:
        assert m[MCC] == 0.0, f"MCC should be 0.0 when undefined, got {m[MCC]}"


def test_mcc_matches_sklearn_on_single_class_prediction():
    """A model predicting one class has undefined MCC; must agree with sklearn."""
    from sklearn.metrics import matthews_corrcoef

    y = np.array([0, 0, 1, 1], dtype=np.int64)
    yhat = np.zeros(4, dtype=np.int64)  # predicts all-negative

    _, m = mmu.binary_metrics(y, yhat=yhat, fill=1.0)

    assert np.isclose(m[MCC], matthews_corrcoef(y, yhat))
    assert m[MCC] == 0.0


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_matches_package_reference_oracle(fill):
    """Defined values must still match the package's sklearn-based oracle."""
    from mmu.commons._testing import compute_reference_metrics, generate_test_labels

    scores, yhat, y = generate_test_labels(2000)
    _, metrics = mmu.binary_metrics(y, yhat=yhat, fill=fill)
    _, ref = compute_reference_metrics(y, yhat=yhat, fill=fill)

    assert np.allclose(metrics, ref)

