"""Module containing the API for Recall vs Proportion Predicted Negative uncertainty.

This module provides analytical profile likelihood solutions for the joint
uncertainty of Recall (TPR) and Proportion Predicted Negative (PPN) metrics.
"""

import numpy as np

from mmu.methods.pointbase import BaseUncertainty
from mmu.methods.curvebase import BaseCurveUncertainty

import mmu.lib._mmu_core as _core
from mmu.lib import _MMU_MT_SUPPORT


class RecallPPNUncertainty(BaseUncertainty):
    """Compute joint uncertainty on Recall and Proportion Predicted Negative.

    The joint statistical uncertainty is computed using profile log-likelihoods
    between the observed and most conservative confusion matrix for that
    (Recall, PPN) pair.

    Metrics:
        Recall = TP / (TP + FN)
        PPN = (TN + FN) / N  (Proportion Predicted Negative)

    Attributes
    ----------
    conf_mat : np.ndarray[int64]
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
        A DataFrame can be obtained by calling `get_conf_mat`.
    y : float
        PPN (Proportion Predicted Negative) - vertical axis
    x : float
        Recall (True Positive Rate) - horizontal axis
    threshold : float, optional
        the inclusive threshold used to determine the confusion matrix.
        Is None when the class is instantiated with `from_predictions` or
        `from_confusion_matrix`.
    chi2_scores : np.ndarray[float64]
        the chi2 scores for the grid with shape (`n_bins`, `n_bins`) and
        bounds y_bounds on the y-axis (PPN), x_bounds on the x-axis (Recall)
    y_bounds : np.ndarray[float64]
        the lower and upper bound for which PPN was evaluated, equal
        to ppn +- `n_sigmas` * sigma(ppn)
    x_bounds : np.ndarray[float64]
        the lower and upper bound for which Recall was evaluated, equal
        to recall +- `n_sigmas` * sigma(recall)
    n_sigmas : int, float
        the number of marginal standard deviations used to determine the
        bounds of the grid which is evaluated.
    epsilon : float
        the value used to prevent the bounds from reaching recall/ppn
        1.0/0.0 which would result in NaNs.

    """

    def __init__(self):
        BaseUncertainty.__init__(self)

        # Set metric and error functions (C++ implementations)
        self.metric_func = _core.recall_ppn
        self.multn_error_func = _core.recall_ppn_multn_error
        self.multn_chi2_score_func = _core.recall_ppn_multn_chi2_score
        self.multn_chi2_scores_func = _core.recall_ppn_multn_chi2_scores

        if _MMU_MT_SUPPORT:
            self.multn_error_mt_func = _core.recall_ppn_multn_error_mt
            self.multn_chi2_scores_mt_func = _core.recall_ppn_multn_chi2_scores_mt

        self.y_label = "PPN"
        self.x_label = "Recall"


    @property
    def recall(self):
        """Alias of the x coordinate (horizontal axis).

        :type: float
        """
        return self.x

    @property
    def ppn(self):
        """Alias of the y coordinate (vertical axis).

        :type: float
        """
        return self.y

    @property
    def recall_bounds(self):
        """Alias of the x_bounds (horizontal axis bounds).

        :type: np.ndarray[float64]
        """
        return self.x_bounds

    @property
    def ppn_bounds(self):
        """Alias of the y_bounds (vertical axis bounds).

        :type: np.ndarray[float64]
        """
        return self.y_bounds


RPPNU = RecallPPNUncertainty


# =============================================================================
# RecallPPNCurveUncertainty - Curve Uncertainty Class
# =============================================================================


class RecallPPNCurveUncertainty(BaseCurveUncertainty):
    """Compute joint uncertainty for Recall-PPN curve over multiple thresholds.

    The joint statistical uncertainty is computed using profile log-likelihoods
    between the observed and most conservative confusion matrix for each
    (Recall, PPN) pair.

    Metrics:
        Recall = TP / (TP + FN)
        PPN = (TN + FN) / N  (Proportion Predicted Negative)

    Attributes
    ----------
    conf_mats : np.ndarray[int64]
        the confusion_matrices over the thresholds with columns [TN, FP, FN, TP].
        A DataFrame can be obtained by calling `get_conf_mats`.
    y : np.ndarray[float64]
        PPN (Proportion Predicted Negative) for each threshold - vertical axis
    x : np.ndarray[float64]
        Recall (True Positive Rate) for each threshold - horizontal axis
    chi2_scores : np.ndarray[float64]
        the chi2 scores for the grid with shape (`n_bins_y`, `n_bins_x`).
        Each grid point contains the minimum chi2 score across all thresholds.
    thresholds : np.ndarray[float64], Optional
        the inclusive classification/discrimination thresholds used to compute
        the confusion matrices. Is None when the class is instantiated with
        `from_confusion_matrices`.
    y_grid : np.ndarray[float64]
        the PPN values that were evaluated.
    x_grid : np.ndarray[float64]
        the Recall values that were evaluated.
    n_sigmas : int, float
        the number of marginal standard deviations used to determine the
        bounds of the grid which is evaluated.
    epsilon : float
        the value used to prevent the bounds from reaching recall/ppn
        1.0/0.0 which would result in NaNs.

    """

    def __init__(self):
        BaseCurveUncertainty.__init__(self)

        # Set metric and error functions (C++ implementations)
        self.multn_grid_curve_error_func = _core.recall_ppn_multn_grid_curve_error
        self.metric_2d_func = _core.recall_ppn_2d

        if _MMU_MT_SUPPORT:
            self.multn_grid_curve_error_mt_func = _core.recall_ppn_multn_grid_curve_error_mt

        self.y_label = "PPN"
        self.x_label = "Recall"


    @property
    def recall(self):
        """Alias of the x coordinate of the curve (horizontal axis).

        :type: np.ndarray[float64]
        """
        return self.x

    @property
    def ppn(self):
        """Alias of the y coordinate of the curve (vertical axis).

        :type: np.ndarray[float64]
        """
        return self.y

    @property
    def rec_grid(self):
        """Alias of the x values that were evaluated.

        :type: np.ndarray[float64]
        """
        return self.x_grid

    @property
    def ppn_grid(self):
        """Alias of the y values that were evaluated.

        :type: np.ndarray[float64]
        """
        return self.y_grid


# Alias for convenience
RPPNCU = RecallPPNCurveUncertainty
