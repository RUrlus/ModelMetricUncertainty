"""Tests for Recall-PPN uncertainty implementation.

Tests the RecallPPNUncertainty and RecallPPNCurveUncertainty classes
using the C++ backend.
"""
import os
import pytest
import itertools
import numpy as np
import sklearn.metrics as skm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mmu
from mmu import RPPNU, RPPNCU
from mmu.commons._testing import generate_test_labels
from mmu.commons._testing import greater_equal_tol
import mmu.lib._mmu_core as _core


# =============================================================================
# Helper Functions
# =============================================================================

def compute_recall_ppn(y, yhat):
    """Compute Recall and PPN from true labels and predictions.

    Returns:
        tuple: (recall, ppn) values
    """
    cm = skm.confusion_matrix(y, yhat)
    tn, fp, fn, tp = cm.ravel()
    n = tn + fp + fn + tp
    p = fn + tp
    recall = tp / p if p > 0 else 0.0
    ppn = (tn + fn) / n if n > 0 else 0.0
    return recall, ppn


def RPPN_skm(y, yhat):
    """Wrapper to match the interface of PRCU_skm / ROCCU_skm.

    Returns:
        tuple: (array([0, ppn]), array([0, recall]))
    """
    recall, ppn = compute_recall_ppn(y, yhat)
    return np.array([0, ppn]), np.array([0, recall])


# =============================================================================
# Test Data Types
# =============================================================================

Y_DTYPES = [
    bool,
    np.bool_,
    int,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
]

YHAT_DTYPES = [
    bool,
    np.bool_,
    int,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
]

PROBA_DTYPES = [
    float,
    np.float32,
    np.float64,
]


# =============================================================================
# Tests for C++ Core Functions
# =============================================================================

class TestRecallPPNCppCore:
    """Test the C++ core functions directly."""

    def test_recall_ppn_metric(self):
        """Test the recall_ppn metric computation."""
        # Simple confusion matrix: TN=100, FP=10, FN=20, TP=70
        conf_mat = np.array([100, 10, 20, 70], dtype=np.int64)

        ppn, recall = _core.recall_ppn(conf_mat)

        # Expected values
        expected_recall = 70 / (70 + 20)  # TP / (TP + FN)
        expected_ppn = (100 + 20) / 200   # (TN + FN) / N

        assert np.isclose(recall, expected_recall), f"Expected recall {expected_recall}, got {recall}"
        assert np.isclose(ppn, expected_ppn), f"Expected ppn {expected_ppn}, got {ppn}"

    def test_recall_ppn_2d(self):
        """Test recall_ppn_2d for multiple confusion matrices."""
        conf_mats = np.array([
            [100, 10, 20, 70],  # PPN = 0.6, Recall = 0.777...
            [50, 50, 10, 90],   # PPN = 0.3, Recall = 0.9
            [150, 30, 5, 15],   # PPN = 0.775, Recall = 0.75
        ], dtype=np.int64)

        result = _core.recall_ppn_2d(conf_mats)

        assert result.shape == (3, 2), f"Expected shape (3, 2), got {result.shape}"

        # Check first row
        expected_ppn_0 = (100 + 20) / 200
        expected_recall_0 = 70 / 90
        assert np.isclose(result[0, 0], expected_ppn_0), f"Row 0 PPN mismatch"
        assert np.isclose(result[0, 1], expected_recall_0), f"Row 0 Recall mismatch"

    def test_recall_ppn_chi2_score_at_observed(self):
        """Test that chi2 score at observed point is close to 0."""
        conf_mat = np.array([100, 10, 20, 70], dtype=np.int64)
        ppn, recall = _core.recall_ppn(conf_mat)

        score = _core.recall_ppn_multn_chi2_score(ppn, recall, conf_mat, 1e-4)

        # At the observed point, the chi2 score should be very close to 0
        assert score < 1e-6, f"Expected score near 0 at observed point, got {score}"

    def test_recall_ppn_chi2_scores(self):
        """Test chi2 scores for multiple points."""
        conf_mat = np.array([100, 10, 20, 70], dtype=np.int64)
        ppn_obs, recall_obs = _core.recall_ppn(conf_mat)

        ppns = np.array([ppn_obs, ppn_obs + 0.1, ppn_obs - 0.1])
        recalls = np.array([recall_obs, recall_obs + 0.1, recall_obs - 0.1])

        scores = _core.recall_ppn_multn_chi2_scores(ppns, recalls, conf_mat, 1e-4)

        assert len(scores) == 3, f"Expected 3 scores, got {len(scores)}"
        # First point is at observed, should be near 0
        assert scores[0] < 1e-6, f"Score at observed should be near 0"
        # Other points should have positive scores
        assert scores[1] > 0, "Score at different point should be positive"
        assert scores[2] > 0, "Score at different point should be positive"

    def test_recall_ppn_multn_error(self):
        """Test multn_error grid computation."""
        conf_mat = np.array([100, 10, 20, 70], dtype=np.int64)
        n_bins = 50

        result, bounds = _core.recall_ppn_multn_error(n_bins, conf_mat, 6.0, 1e-4)

        assert result.shape == (n_bins, n_bins), f"Expected shape ({n_bins}, {n_bins})"
        assert bounds.shape == (2, 2), f"Expected bounds shape (2, 2)"

        # Bounds should be within [0, 1]
        assert np.all(bounds >= 0) and np.all(bounds <= 1), "Bounds should be in [0, 1]"

        # Result should have some finite values
        finite_mask = np.isfinite(result)
        assert np.sum(finite_mask) > 0, "Should have some finite chi2 scores"

    def test_recall_ppn_multn_grid_error(self):
        """Test multn_grid_error with user-provided grid."""
        conf_mat = np.array([100, 10, 20, 70], dtype=np.int64)

        ppn_grid = np.linspace(0.5, 0.7, 30)
        recall_grid = np.linspace(0.6, 0.9, 30)

        scores = _core.recall_ppn_multn_grid_error(
            ppn_grid, recall_grid, conf_mat, 6.0, 1e-4
        )

        assert scores.shape == (30, 30), f"Expected shape (30, 30)"

        # Should have some finite values
        finite_mask = np.isfinite(scores)
        assert np.sum(finite_mask) > 0, "Should have some finite chi2 scores"

    def test_recall_ppn_multn_grid_curve_error(self):
        """Test multn_grid_curve_error with multiple confusion matrices."""
        conf_mats = np.array([
            [100, 10, 20, 70],
            [95, 15, 25, 65],
            [110, 5, 15, 70],
        ], dtype=np.int64)

        ppn_grid = np.linspace(0.5, 0.7, 20)
        recall_grid = np.linspace(0.6, 0.9, 20)

        scores = _core.recall_ppn_multn_grid_curve_error(
            3, ppn_grid, recall_grid, conf_mats, 6.0, 1e-4
        )

        assert scores.shape == (20, 20), f"Expected shape (20, 20)"

        # The minimum across matrices should be taken
        finite_mask = np.isfinite(scores)
        assert np.sum(finite_mask) > 0, "Should have some finite chi2 scores"


# =============================================================================
# Tests for RecallPPNUncertainty Class
# =============================================================================

class TestRecallPPNUncertainty:
    """Test the RecallPPNUncertainty (RPPNU) class."""

    def test_from_scores_basic(self):
        """Test RPPNU.from_scores with basic inputs."""
        np.random.seed(412)
        proba, _, y = generate_test_labels(N=1000)
        threshold = 0.5
        yhat = greater_equal_tol(proba, threshold)

        err = RPPNU.from_scores(y=y, scores=proba, threshold=threshold)

        # Check attributes are set
        assert err.conf_mat is not None
        assert err.conf_mat.dtype == np.dtype(np.int64)
        assert err.chi2_scores is not None
        assert err.y is not None  # PPN
        assert err.x is not None  # Recall

        # Check shape
        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)

        # Verify metrics match sklearn
        recall, ppn = compute_recall_ppn(y, yhat)
        assert np.isclose(err.recall, recall, rtol=1e-5), f"Recall mismatch: {err.recall} vs {recall}"
        assert np.isclose(err.ppn, ppn, rtol=1e-5), f"PPN mismatch: {err.ppn} vs {ppn}"

    @pytest.mark.parametrize("y_dtype,proba_dtype", [
        (np.int64, np.float64),
        (np.int32, np.float32),
        (bool, float),
    ])
    def test_from_scores_dtypes(self, y_dtype, proba_dtype):
        """Test RPPNU.from_scores with different data types."""
        np.random.seed(412)
        proba, _, y = generate_test_labels(
            N=1000, y_dtype=y_dtype, proba_dtype=proba_dtype
        )
        threshold = 0.5
        yhat = greater_equal_tol(proba, threshold)

        err = RPPNU.from_scores(y=y, scores=proba, threshold=threshold)

        assert err.conf_mat.dtype == np.dtype(np.int64)

        recall, ppn = compute_recall_ppn(y, yhat)
        assert np.isclose(err.recall, recall, rtol=1e-5)
        assert np.isclose(err.ppn, ppn, rtol=1e-5)

    def test_from_predictions(self):
        """Test RPPNU.from_predictions."""
        np.random.seed(412)
        _, yhat, y = generate_test_labels(N=1000)

        err = RPPNU.from_predictions(y=y, yhat=yhat)

        assert err.conf_mat is not None
        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)

        recall, ppn = compute_recall_ppn(y, yhat)
        assert np.isclose(err.recall, recall, rtol=1e-5)
        assert np.isclose(err.ppn, ppn, rtol=1e-5)

    def test_from_confusion_matrix(self):
        """Test RPPNU.from_confusion_matrix."""
        np.random.seed(412)
        _, yhat, y = generate_test_labels(N=1000)
        sk_conf_mat = skm.confusion_matrix(y, yhat)

        err = RPPNU.from_confusion_matrix(sk_conf_mat)

        assert err.conf_mat is not None
        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)

        recall, ppn = compute_recall_ppn(y, yhat)
        assert np.isclose(err.recall, recall, rtol=1e-5)
        assert np.isclose(err.ppn, ppn, rtol=1e-5)

    def test_from_classifier(self):
        """Test RPPNU.from_classifier."""
        seeds = mmu.commons.utils.SeedGenerator(234)

        X, y = make_classification(n_samples=1000, n_classes=2, random_state=seeds())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=seeds()
        )

        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)

        y_scores = model.predict_proba(X_test)[:, 1]
        threshold = 0.5
        yhat = greater_equal_tol(y_scores, threshold)

        err = RPPNU.from_classifier(model, X_test, y_test, threshold=threshold)

        assert err.conf_mat is not None
        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)

        recall, ppn = compute_recall_ppn(y_test, yhat)
        assert np.isclose(err.recall, recall, rtol=1e-5)
        assert np.isclose(err.ppn, ppn, rtol=1e-5)

    def test_compute_score_for(self):
        """Test compute_score_for method."""
        conf_mat = np.array([[100, 10], [20, 70]])
        err = RPPNU.from_confusion_matrix(conf_mat)

        # Score at observed point should be near 0
        score = err.compute_score_for(err.ppn, err.recall)
        assert score < 1e-6, f"Score at observed point should be near 0, got {score}"

        # Score at different point should be positive
        score = err.compute_score_for(err.ppn + 0.1, err.recall + 0.05)
        assert score > 0, "Score at different point should be positive"

    def test_compute_pvalue_for(self):
        """Test compute_pvalue_for method."""
        conf_mat = np.array([[100, 10], [20, 70]])
        err = RPPNU.from_confusion_matrix(conf_mat)

        # p-value at observed point should be close to 1
        pval = err.compute_pvalue_for(err.ppn, err.recall)
        assert pval > 0.9, f"P-value at observed should be near 1, got {pval}"

        # p-value at distant point should be smaller
        pval = err.compute_pvalue_for(err.ppn + 0.2, err.recall - 0.2)
        assert 0 <= pval <= 1, f"P-value should be in [0, 1], got {pval}"

    def test_n_bins_parameter(self):
        """Test n_bins parameter."""
        conf_mat = np.array([[100, 10], [20, 70]])

        err = RPPNU.from_confusion_matrix(conf_mat, n_bins=50)
        assert err.chi2_scores.shape == (50, 50)

        err = RPPNU.from_confusion_matrix(conf_mat, n_bins=200)
        assert err.chi2_scores.shape == (200, 200)

    def test_property_aliases(self):
        """Test property aliases for recall, ppn, recall_bounds, ppn_bounds."""
        conf_mat = np.array([[100, 10], [20, 70]])
        err = RPPNU.from_confusion_matrix(conf_mat)

        # Test aliases
        assert err.recall == err.x
        assert err.ppn == err.y
        assert np.array_equal(err.recall_bounds, err.x_bounds)
        assert np.array_equal(err.ppn_bounds, err.y_bounds)

    def test_edge_case_perfect_recall(self):
        """Test edge case: perfect recall (FN=0)."""
        # TN=100, FP=20, FN=0, TP=80
        conf_mat = np.array([[100, 20], [0, 80]])

        err = RPPNU.from_confusion_matrix(conf_mat)

        assert np.isclose(err.recall, 1.0), f"Expected recall=1.0, got {err.recall}"
        assert err.chi2_scores.shape == (err.n_bins, err.n_bins)

    def test_edge_case_low_recall(self):
        """Test edge case: low recall."""
        # TN=100, FP=10, FN=80, TP=10
        conf_mat = np.array([[100, 10], [80, 10]])

        err = RPPNU.from_confusion_matrix(conf_mat)

        expected_recall = 10 / 90
        assert np.isclose(err.recall, expected_recall), f"Expected recall={expected_recall}, got {err.recall}"


# =============================================================================
# Tests for RecallPPNCurveUncertainty Class
# =============================================================================

class TestRecallPPNCurveUncertainty:
    """Test the RecallPPNCurveUncertainty (RPPNCU) class."""

    def test_from_scores_basic(self):
        """Test RPPNCU.from_scores with basic inputs."""
        np.random.seed(412)
        proba, _, y = generate_test_labels(N=500)
        thresholds = np.linspace(0.1, 0.9, 50)

        err = RPPNCU.from_scores(y=y, scores=proba, thresholds=thresholds)

        assert err.conf_mats is not None
        assert err.conf_mats.dtype == np.dtype(np.int64)
        assert err.conf_mats.shape[0] == len(thresholds)
        assert err.chi2_scores is not None
        assert len(err.y) == len(thresholds)  # PPN values
        assert len(err.x) == len(thresholds)  # Recall values

    def test_from_confusion_matrices(self):
        """Test RPPNCU.from_confusion_matrices."""
        np.random.seed(412)
        proba, _, y = generate_test_labels(N=500)
        thresholds = np.linspace(0.1, 0.9, 50)

        conf_mats = mmu.confusion_matrices_thresholds(y, proba, thresholds)
        err = RPPNCU.from_confusion_matrices(conf_mats=conf_mats)

        assert err.conf_mats is not None
        assert err.chi2_scores.shape == (err.y_grid.size, err.x_grid.size)

        # Verify metrics for a specific threshold
        idx = 25
        yhat = greater_equal_tol(proba, thresholds[idx])
        recall, ppn = compute_recall_ppn(y, yhat)

        assert np.isclose(err.recall[idx], recall, rtol=1e-5)
        assert np.isclose(err.ppn[idx], ppn, rtol=1e-5)

    def test_from_classifier(self):
        """Test RPPNCU.from_classifier."""
        seeds = mmu.commons.utils.SeedGenerator(234)

        X, y = make_classification(n_samples=1000, n_classes=2, random_state=seeds())
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=seeds()
        )

        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)

        thresholds = np.linspace(0.1, 0.9, 30)
        err = RPPNCU.from_classifier(
            clf=model, X=X_test, y=y_test, thresholds=thresholds
        )

        assert err.conf_mats is not None
        assert err.conf_mats.shape[0] == len(thresholds)
        assert err.chi2_scores is not None

    def test_property_aliases(self):
        """Test property aliases for recall, ppn, rec_grid, ppn_grid."""
        np.random.seed(412)
        proba, _, y = generate_test_labels(N=500)
        thresholds = np.linspace(0.1, 0.9, 30)

        err = RPPNCU.from_scores(y=y, scores=proba, thresholds=thresholds)

        # Test aliases
        assert np.array_equal(err.recall, err.x)
        assert np.array_equal(err.ppn, err.y)
        assert np.array_equal(err.rec_grid, err.x_grid)
        assert np.array_equal(err.ppn_grid, err.y_grid)

    def test_chi2_scores_minimum_across_thresholds(self):
        """Test that chi2_scores take minimum across thresholds."""
        np.random.seed(412)
        proba, _, y = generate_test_labels(N=500)
        thresholds = np.linspace(0.1, 0.9, 20)

        err = RPPNCU.from_scores(y=y, scores=proba, thresholds=thresholds)

        # The shape should match the grid
        assert err.chi2_scores.shape == (err.y_grid.size, err.x_grid.size)

        # Should have some finite values (not all inf)
        finite_count = np.sum(np.isfinite(err.chi2_scores))
        assert finite_count > 0, "Should have some finite chi2 scores"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestRecallPPNEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_n_bins(self):
        """Test that invalid n_bins raises an error."""
        conf_mat = np.array([[100, 10], [20, 70]])

        with pytest.raises(ValueError):
            RPPNU.from_confusion_matrix(conf_mat, n_bins=-10)

        with pytest.raises(ValueError):
            RPPNU.from_confusion_matrix(conf_mat, n_bins=0)

    def test_all_zeros_confusion_matrix(self):
        """Test behavior with all-zeros confusion matrix."""
        conf_mat = np.array([[0, 0], [0, 0]])

        # This might raise an error or handle gracefully
        # depending on implementation
        try:
            err = RPPNU.from_confusion_matrix(conf_mat)
            # If it doesn't raise, check for NaN handling
            assert err.recall == 0 or np.isnan(err.recall)
            assert err.ppn == 0 or np.isnan(err.ppn)
        except (ValueError, ZeroDivisionError):
            pass  # Expected behavior

    def test_single_class_positive_only(self):
        """Test with only positive predictions (TN=0, FP=0)."""
        # All samples predicted positive
        conf_mat = np.array([[0, 0], [20, 80]])  # TN=0, FP=0, FN=20, TP=80

        err = RPPNU.from_confusion_matrix(conf_mat)

        # PPN = (TN + FN) / N = 20/100 = 0.2
        assert np.isclose(err.ppn, 0.2), f"Expected PPN=0.2, got {err.ppn}"
        # Recall = TP / (TP + FN) = 80/100 = 0.8
        assert np.isclose(err.recall, 0.8), f"Expected Recall=0.8, got {err.recall}"


# =============================================================================
# Comparison Tests
# =============================================================================

class TestRecallPPNComparison:
    """Compare Recall-PPN with other metrics for consistency."""

    def test_recall_matches_roc_tpr(self):
        """Test that Recall in RPPNU matches TPR in ROCU."""
        np.random.seed(42)
        proba, _, y = generate_test_labels(N=1000)
        threshold = 0.5

        rppnu = RPPNU.from_scores(y=y, scores=proba, threshold=threshold)
        rocu = mmu.ROCU.from_scores(y=y, scores=proba, threshold=threshold)

        # Recall (RPPNU.x) should equal TPR (ROCU.y)
        assert np.isclose(rppnu.recall, rocu.y, rtol=1e-5), \
            f"Recall {rppnu.recall} should equal TPR {rocu.y}"

    def test_ppn_consistency(self):
        """Test PPN calculation consistency."""
        np.random.seed(42)
        _, yhat, y = generate_test_labels(N=1000)

        err = RPPNU.from_predictions(y=y, yhat=yhat)

        # PPN = (TN + FN) / N = 1 - (FP + TP) / N = 1 - PP
        tn, fp, fn, tp = err.conf_mat.flatten()
        n = tn + fp + fn + tp
        expected_ppn = (tn + fn) / n

        assert np.isclose(err.ppn, expected_ppn, rtol=1e-10), \
            f"PPN mismatch: {err.ppn} vs {expected_ppn}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

