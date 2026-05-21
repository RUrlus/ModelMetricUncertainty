"""Module containing the API for Recall vs Proportion Predicted Negative uncertainty.

This module provides analytical profile likelihood solutions for the joint
uncertainty of Recall (TPR) and Proportion Predicted Negative (PPN) metrics.
"""

import numpy as np

from mmu.methods.pointbase import BaseUncertainty
from mmu.methods.curvebase import BaseCurveUncertainty


def _safe_xlogp(x, p):
    """Compute x * log(p) safely."""
    if p <= 0:
        return 0.0 if x == 0 else -np.inf
    return x * np.log(p)


def _loglike_counts(cm, p_tn, p_fp, p_fn, p_tp):
    """Multinomial log-likelihood."""
    return (
        _safe_xlogp(cm[0], p_tn)
        + _safe_xlogp(cm[1], p_fp)
        + _safe_xlogp(cm[2], p_fn)
        + _safe_xlogp(cm[3], p_tp)
    )


def _profile_t_hat(conf_mat, recall, ppn, eps=1e-12):
    """Profile likelihood over t = p_TP for fixed (recall, ppn).

    Returns (t_hat, ll_prof). If infeasible: (None, -inf).
    """
    n_total = conf_mat.sum()
    tn, fp, fn, tp = conf_mat.ravel()
    n_pos = fn + tp

    if not (eps < recall < 1.0 - eps and eps < ppn < 1.0 - eps):
        return None, -np.inf

    fn_tp_ratio = (1.0 - recall) / recall
    pp = 1.0 - ppn

    if fn_tp_ratio <= 0:
        return None, -np.inf

    t_max = min(pp, ppn / fn_tp_ratio)
    if t_max <= eps:
        return None, -np.inf

    linear_coef = n_pos * (pp * fn_tp_ratio + ppn) + fp * ppn + fn_tp_ratio * tn * pp
    discriminant = linear_coef**2 - 4.0 * fn_tp_ratio * n_total * n_pos * ppn * pp

    if discriminant < 0:
        if discriminant > -1e-12:
            discriminant = 0.0
        else:
            return None, -np.inf

    sqrt_d = np.sqrt(discriminant)
    denom = 2.0 * fn_tp_ratio * n_total
    roots = [(linear_coef - sqrt_d) / denom, (linear_coef + sqrt_d) / denom]

    best_ll, best_t = -np.inf, None
    for t in roots:
        if not (eps < t < t_max - eps):
            continue
        p_tp, p_fn = t, fn_tp_ratio * t
        p_tn, p_fp = ppn - p_fn, pp - t
        if p_tn < 0 or p_fp < 0:
            continue
        ll = _loglike_counts(conf_mat.ravel(), p_tn, p_fp, p_fn, p_tp)
        if ll > best_ll:
            best_ll, best_t = ll, t

    return best_t, best_ll


def _q_stat(conf_mat, recall, ppn, eps=1e-12):
    """Profile likelihood ratio statistic q(recall, ppn)."""
    n = conf_mat.sum()
    ll_hat = _loglike_counts(conf_mat.ravel(), *(conf_mat / n).ravel())
    _, ll_prof = _profile_t_hat(conf_mat, recall, ppn, eps=eps)
    if not np.isfinite(ll_prof):
        return np.inf
    return -2.0 * (ll_prof - ll_hat)


def _recall_ppn_metric(conf_mat):
    tn, fp, fn, tp = conf_mat.ravel()
    n = tn + fp + fn + tp
    p = tp + fn
    recall = tp / p if p > 0 else np.nan
    ppn = (tn + fn) / n if n > 0 else np.nan
    return ppn, recall


def _recall_ppn_metric_2d(conf_mats):
    """Compute (y, x) = (PPN, Recall) for multiple confusion matrices."""
    tn = conf_mats[:, 0]
    fp = conf_mats[:, 1]
    fn = conf_mats[:, 2]
    tp = conf_mats[:, 3]
    n = tn + fp + fn + tp
    p = tp + fn
    recall = np.where(p > 0, tp / p, np.nan)
    ppn = np.where(n > 0, (tn + fn) / n, np.nan)
    return np.column_stack([ppn, recall])  # columns: [y, x] = [ppn, recall]


def _recall_ppn_multn_error(n_bins, conf_mat, n_sigmas, epsilon, n_threads=None):
    """Compute chi2 scores on a grid. Returns (scores, bounds).

    Interface compatible with BaseUncertainty.multn_error_func.
    Note: n_threads is ignored as this is a pure Python implementation.
    """
    tn, fp, fn, tp = conf_mat.ravel()
    n = conf_mat.sum()
    p = tp + fn
    pn = tn + fn
    pp = tp + fp

    recall = tp / p if p > 0 else np.nan
    ppn = (tn + fn) / n if n > 0 else np.nan

    # Marginal variances
    if p > 0 and tp > 0 and fn > 0:
        var_r = (tp * fn) / (p**3)
    else:
        var_r = 0.01
    if n > 0 and pn > 0 and pp > 0:
        var_ppn = (pn * pp) / (n**3)
    else:
        var_ppn = 0.01

    sigma_r = np.sqrt(var_r) if np.isfinite(var_r) else 0.1
    sigma_ppn = np.sqrt(var_ppn) if np.isfinite(var_ppn) else 0.1

    r_lo = max(epsilon, recall - n_sigmas * sigma_r)
    r_hi = min(1 - epsilon, recall + n_sigmas * sigma_r)
    ppn_lo = max(epsilon, ppn - n_sigmas * sigma_ppn)
    ppn_hi = min(1 - epsilon, ppn + n_sigmas * sigma_ppn)

    bounds = np.array([[ppn_lo, ppn_hi], [r_lo, r_hi]])
    ppn_grid = np.linspace(ppn_lo, ppn_hi, n_bins)
    r_grid = np.linspace(r_lo, r_hi, n_bins)

    scores = np.full((n_bins, n_bins), 65.0, dtype=np.float64)
    for i, pv in enumerate(ppn_grid):
        for j, rv in enumerate(r_grid):
            scores[i, j] = _q_stat(conf_mat, rv, pv, eps=epsilon)

    return scores, bounds


def _recall_ppn_multn_chi2_score(y, x, conf_mat, epsilon):
    """Compute chi2 score for single point (y=ppn, x=recall)."""
    return _q_stat(conf_mat, x, y, eps=epsilon)


def _recall_ppn_multn_chi2_scores(y_arr, x_arr, conf_mat, epsilon):
    """Compute chi2 scores for arrays.

    Interface compatible with BaseUncertainty.multn_chi2_scores_func.
    """
    scores = np.empty(len(y_arr), dtype=np.float64)
    for i, (pv, rv) in enumerate(zip(y_arr, x_arr)):
        scores[i] = _q_stat(conf_mat, rv, pv, eps=epsilon)
    return scores


def _recall_ppn_multn_grid_curve_error(
    n_conf_mats, y_grid, x_grid, conf_mat, n_sigmas, epsilon, n_threads=None
):
    """Compute curve chi2 scores by taking min across confusion matrices.

    Interface compatible with BaseCurveUncertainty.multn_grid_curve_error_func.
    Note: n_threads is ignored as this is a pure Python implementation.
    """
    scores = np.full((len(y_grid), len(x_grid)), 65.0, dtype=np.float64)
    for cm_idx in range(n_conf_mats):
        for i, pv in enumerate(y_grid):
            for j, rv in enumerate(x_grid):
                q = _q_stat(conf_mat[cm_idx], rv, pv, eps=epsilon)
                if q < scores[i, j]:
                    scores[i, j] = q
    return scores


class RecallPPNUncertainty(BaseUncertainty):
    """Compute joint uncertainty on Recall and Proportion Predicted Negative.

    The joint statistical uncertainty is computed using profile log-likelihoods
    between the observed and most conservative confusion matrix for that
    (Recall, PPN) pair.

    Metrics:
        Recall = TP / (TP + FN)
        PPN = (TN + FN) / N  (Proportion Predicted Negative)

    Note: For this metric pair, only the 'multinomial' method is supported
    since the bivariate normal approximation requires C++ core functions
    that are not yet implemented for Recall-PPN.

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

        # Set metric and error functions (Python implementations)
        self.metric_func = _recall_ppn_metric
        self.multn_error_func = _recall_ppn_multn_error

        # Multinomial-only for Recall-PPN (no C++ BVN support yet)
        self.multn_chi2_score_func = _recall_ppn_multn_chi2_score
        self.multn_chi2_scores_func = _recall_ppn_multn_chi2_scores

        # Use same functions for MT versions (Python is single-threaded here)
        self.multn_error_mt_func = _recall_ppn_multn_error
        self.multn_chi2_scores_mt_func = _recall_ppn_multn_chi2_scores

        # BVN is not supported for Recall-PPN, set to None
        self.bvn_cov_func = None
        self.bvn_chi2_score_func = None
        self.bvn_chi2_scores_func = None
        self.bvn_chi2_scores_mt_func = None

        self.y_label = "PPN"
        self.x_label = "Recall"

        # Override moptions to only support multinomial
        self._moptions = {
            "mult": {"mult", "multinomial"},
            "bvn": set(),  # BVN not supported
        }

    def _parse_method(self, method):
        """Override to only allow multinomial method for Recall-PPN."""
        if method in self._moptions["mult"]:
            self.method = method
            self._compute_scores = self._compute_multn_scores
        else:
            msg = (
                "Only 'multinomial' or 'mult' method is supported for RecallPPNUncertainty. "
                + "The bivariate-normal/elliptical method is not available for this metric pair."
            )
            raise ValueError(msg)

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

    Note: For this metric pair, only the 'multinomial' method is supported
    since the bivariate normal approximation requires C++ core functions
    that are not yet implemented for Recall-PPN.

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

        # Set metric and error functions (Python implementations)
        self.multn_grid_curve_error_func = _recall_ppn_multn_grid_curve_error
        self.metric_2d_func = _recall_ppn_metric_2d

        # Use same function for MT version (Python is single-threaded here)
        self.multn_grid_curve_error_mt_func = _recall_ppn_multn_grid_curve_error

        # BVN is not supported for Recall-PPN, set to None
        self.bvn_grid_curve_error_func = None
        self.bvn_grid_curve_error_mt_func = None

        self.y_label = "PPN"
        self.x_label = "Recall"

        # Override moptions to only support multinomial
        self._moptions = {
            "mult": {"mult", "multinomial"},
            "bvn": set(),  # BVN not supported
        }

    def _parse_method(self, method):
        """Override to only allow multinomial method for Recall-PPN."""
        if method in self._moptions["mult"]:
            self.method = method
            self._compute_scores = self._compute_multn_scores
        else:
            raise ValueError(
                "Only 'multinomial' or 'mult' method is supported for RecallPPNCurveUncertainty. "
                "The bivariate-normal/elliptical method is not available for this metric pair."
            )

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
