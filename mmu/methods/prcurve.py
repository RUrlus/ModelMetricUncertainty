"""Module containing the API for the precision-recall with Multinomial uncertainty."""
import warnings
from typing import Union, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as sts

from mmu.commons import check_array, _convert_to_float, _convert_to_int
from mmu.commons.checks import _check_n_threads
from mmu.metrics.utils import auto_thresholds
from mmu.metrics.confmat import confusion_matrices_thresholds
from mmu.metrics.confmat import confusion_matrices_to_dataframe
from mmu.viz.contours import _plot_pr_curve_contours
from mmu.lib import MMU_MT_SUPPORT as _MMU_MT_SUPPORT

import mmu.lib._mmu_core as _core
from mmu.lib._mmu_core import (
    multinomial_uncertainty_over_grid_thresholds as mult_error_grid_thresh,
    multinomial_uncertainty_over_grid_thresholds_mt as mult_error_grid_thresh_mt
)

class _PrecisionRecallCurveBase:
    def __init__(self):
        pass

    def _parse_thresholds(self, thresholds, scores, max_steps, seed):
        if thresholds is None:
            thresholds = auto_thresholds(scores, max_steps=max_steps, seed=seed)
        else:
            thresholds = check_array(
                thresholds,
                max_dim=1,
                dtype_check=_convert_to_float
            )
        if thresholds.min() <= 0.0:
            raise ValueError("`thresholds` should be in (0., 1.0)")
        if thresholds.max() >= 1.0:
            raise ValueError("`thresholds` should be in (0., 1.0)")
        self.thresholds = thresholds

    def _parse_nbins(self, n_bins):
        # -- validate n_bins arg
        if n_bins is None:
            self.prec_grid = self.rec_grid = np.linspace(1-12, 1-1e-12, 1000)
        elif isinstance(n_bins, int):
            if n_bins < 1:
                raise ValueError("`n_bins` must be bigger than 0")
            self.prec_grid = self.rec_grid = np.linspace(1-12, 1-1e-12, n_bins)
        elif isinstance(n_bins, np.ndarray):
            if not np.issubdtype(n_bins.dtype, np.integer):
                raise TypeError("`n_bins` must be an int or list-like ints")
            self.prec_grid = np.linspace(1-12, 1-1e-12, n_bins[0])
            self.rec_grid = np.linspace(1-12, 1-1e-12, n_bins[1])

        elif isinstance(n_bins, (list, tuple)) and len(n_bins) == 2:
            if (
                (not isinstance(n_bins[0], int))
                or (not isinstance(n_bins[1], int))
            ):
                raise TypeError("`n_bins` must be an int or list-like ints")
            self.prec_grid = np.linspace(1-12, 1-1e-12, n_bins[0])
            self.rec_grid = np.linspace(1-12, 1-1e-12, n_bins[1])
        else:
            raise TypeError("`n_bins` must be an int or list-like ints")


class PrecisionRecallCurveMultinomialUncertainty(_PrecisionRecallCurveBase):
    """Precision-Recall curve uncertainty modelled as a Multinomial.

    Model's the uncertainty using profile log-likelihoods between
    the observed and most conservative confusion matrix for that
    precision recall. Unlike the PrecisionRecallEllipticalUncertainty
    this approach is valid for relatively low statistic samples and
    at the edges of the curve. However, it does not allow one to
    add the training sample uncertainty to it.

    Attributes
    ----------
    conf_mat : np.ndarray[int64]
        the confusion_matrices over the thresholds with columns
        [0] = TN, [1] = FP, [2] = FN, [3] = TP
        A DataFrame can be obtained by calling `get_conf_mat`.
    precision : np.ndarray[float64]
        the Positive Predictive Values aka positive precisions
    recall : np.ndarray[float64]
        True Positive Rate aka Sensitivity aka positive recall
    chi2_scores : np.ndarray[float64]
        the chi2 scores for the grid with shape (`n_bins`, `n_bins`) and
        bounds precision_bounds on the y-axis, recall_bounds on the x-axis

    Methods
    -------
    from_scores
        compute the uncertainty based on classifier scores and true labels
    from_confusion_matrix
        compute the uncertainty based on a confusion matrix
    from_classifier
        compute the uncertainty based on classifier scores from a trained
        classifier and true labels
    plot
        plot the joint uncertainty of precision and recall
    get_conf_mats
        get the confusion matrices
    get_scores
        get the profile loglikelihood scores

    """
    def __init__(self):
        self.n_conf_mats = None
        self.conf_mats = None
        self.precision = None
        self.recall = None
        self.chi2_scores = None
        self.prec_grid = None
        self.rec_grid = None
        self.n_sigmas = None
        self.epsilon = None

    def _compute_loglike_scores(self, n_sigmas, epsilon, n_threads):
        n_threads = _check_n_threads(n_threads)

        # -- validate n_sigmas arg
        if not isinstance(n_sigmas, (int, float)):
            raise TypeError("`n_sigmas` must be an int or float.")
        elif n_sigmas < 1.0:
            raise ValueError("`n_sigmas` must be greater than 1.")
        self.n_sigmas = n_sigmas

        # -- validate epsilon arg
        if not isinstance(epsilon, float):
            raise TypeError("`epsilon` must be a float")
        elif not (1e-15 <= epsilon <= 0.1):
            raise ValueError("`epsilon` must be  in [1e-15, 0.1]")
        self.epsilon = epsilon
        # compute precision and recall
        self.precision, self.recall = _core.precision_recall_2d(self.conf_mats)
        # compute scores
        if _MMU_MT_SUPPORT and n_threads > 1:
            self.chi2_scores = mult_error_grid_thresh_mt(
                n_conf_mats=self.n_conf_mats,
                precs_grid=self.prec_grid,
                recs_grid=self.rec_grid,
                conf_mat=self.conf_mats,
                n_sigmas=self.n_sigmas,
                epsilon=self.epsilon,
                n_threads=n_threads,
            )
        elif (n_threads > 1):
            warnings.warn(
                "mmu was not compiled with multi-threading enabled,"
                " ignoring `n_threads`"
            )
        else:
            self.chi2_scores = mult_error_grid_thresh(
                n_conf_mats=self.n_conf_mats,
                precs_grid=self.prec_grid,
                recs_grid=self.rec_grid,
                conf_mat=self.conf_mats,
                n_sigmas=self.n_sigmas,
                epsilon=self.epsilon,
            )

    @classmethod
    def from_scores(cls,
        y : np.ndarray,
        score : np.ndarray,
        thresholds : Optional[np.ndarray] = None,
        n_bins : Union[int, Tuple[int], List[int], np.ndarray, None] = 1000,
        n_sigmas : Union[int, float] = 6.0,
        epsilon : float = 1e-12,
        auto_max_steps : Optional[int] = None,
        auto_seed : Optional[int] = None,
    ):
        """Compute Multinomial uncertainty on precision and recall.

        Model's the uncertainty using profile log-likelihoods between
        the observed and most conservative confusion matrix for that
        precision recall.

        Parameters
        ----------
        y : np.ndarray
            true labels for observations, supported dtypes are [bool, int32,
            int64, float32, float64]
        scores : np.ndarray, default=None
            the classifier score to be evaluated against the `thresholds`, i.e.
            `yhat` = `score` >= `threshold`.
            Supported dtypes are float32 and float64.
        thresholds : np.ndarray, default=None
            the inclusive classification threshold against which the classifier
            score is evaluated. If None the classification thresholds are
            determined such that each thresholds results in a different
            confusion matrix. Note that the maximum number of thresholds can
            be set using `max_steps`.
        n_bins : int, array-like[int], default=1000
            the number of bins in the precision/recall grid for which the
            uncertainty is computed. If an int the `chi2_scores` will be a
            `n_bins` by `n_bins` array. If list-like it must be of length two
            where the first values determines the number of bins for
            precision/y-axis and the second the recall/x-axis
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching precision/recall
            1.0/0.0 which would result in NaNs.
        auto_max_steps : int, default=None
            the maximum number of thresholds for `auto_thresholds`, is ignored
            if `thresholds` is not None.
        auto_seed : int, default=None
            the seed/random_state used by `auto_thresholds` when `max_steps` is
            not None. Ignored when `thresholds` is not None.

        """
        self = cls()
        self._parse_thresholds(thresholds, score, auto_max_steps, auto_seed)
        self.conf_mats = confusion_matrices_thresholds(
            y=y, score=score, thresholds=thresholds
        )
        self.n_conf_mats = self.conf_mats.shape[0]
        self._compute_loglike_scores(n_bins, n_sigmas, epsilon)
        return self

    @classmethod
    def from_confusion_matrix(cls,
        conf_mats : np.ndarray,
        n_bins : Union[int, Tuple[int], List[int], np.ndarray, None] = 1000,
        n_sigmas : Union[int, float] = 6.0,
        epsilon : float = 1e-12,
        obs_axis : int = 0,
    ):
        """Compute Multinomial uncertainty on precision and recall.

        Model's the uncertainty using profile log-likelihoods between
        the observed and most conservative confusion matrix for that
        precision recall.

        Parameters
        ----------
        conf_mat : np.ndarray,
            confusion matrix as returned by mmu.confusion_matrix, i.e.
            with layout [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP or
            the flattened equivalent. Supported dtypes are int32, int64
        n_bins : int, array-like[int], default=1000
            the number of bins in the precision/recall grid for which the
            uncertainty is computed. If an int the `chi2_scores` will be a
            `n_bins` by `n_bins` array. If list-like it must be of length two
            where the first values determines the number of bins for
            precision/y-axis and the second the recall/x-axis
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching precision/recall
            1.0/0.0 which would result in NaNs.

        """
        self = cls()
        self.conf_mats = check_array(
            conf_mats,
            min_dim=2,
            max_dim=2,
            axis=obs_axis,
            target_axis=obs_axis,
            target_order=0,
            dtype_check=_convert_to_int
        )
        self.n_conf_mats = self.conf_mats.shape[obs_axis]
        self._compute_loglike_scores(n_bins, n_sigmas, epsilon)
        return self

    @classmethod
    def from_classifier(cls,
        clf,
        X : np.ndarray,
        y : np.ndarray,
        thresholds : Optional[np.ndarray] = None,
        n_bins : Union[int, Tuple[int], List[int], np.ndarray, None] = 1000,
        n_sigmas : Union[int, float] = 6.0,
        epsilon : float = 1e-12,
        auto_max_steps : Optional[int] = None,
        auto_seed : Optional[int] = None,
    ):
        """Compute Multinomial uncertainty on precision and recall.

        Model's the uncertainty using profile log-likelihoods between
        the observed and most conservative confusion matrix for that
        precision recall.

        Parameters
        ----------
        clf : sklearn.Predictor
            a trained model with method `predict_proba`, used to compute
            the classifier scores
        X : np.ndarray
            the feature array to be used to compute the classifier scores
        y : np.ndarray
            true labels for observations, supported dtypes are [bool, int32,
            int64, float32, float64]
        threshold : float, default=0.5
            the classification threshold to which the classifier score is evaluated,
            is inclusive.
        n_bins : int, array-like[int], default=1000
            the number of bins in the precision/recall grid for which the
            uncertainty is computed. If an int the `chi2_scores` will be a
            `n_bins` by `n_bins` array. If list-like it must be of length two
            where the first values determines the number of bins for
            precision/y-axis and the second the recall/x-axis
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching precision/recall
            1.0/0.0 which would result in NaNs.
        auto_max_steps : int, default=None
            the maximum number of thresholds for `auto_thresholds`, is ignored
            if `thresholds` is not None.
        auto_seed : int, default=None
            the seed/random_state used by `auto_thresholds` when `max_steps` is
            not None. Ignored when `thresholds` is not None.

        """
        self = cls()
        if not hasattr(clf, 'predict_proba'):
            raise TypeError("`clf` must have a method `predict_proba`")
        score = clf.predict_proba(X)[:, 1]
        self._parse_thresholds(thresholds, score, auto_max_steps, auto_seed)
        self.conf_mats = confusion_matrices_thresholds(
            y=y, score=score, thresholds=thresholds
        )
        self.n_conf_mats = self.conf_mats.shape[0]
        self._compute_loglike_scores(n_bins, n_sigmas, epsilon)
        return self

    def get_conf_mat(self) -> pd.DataFrame:
        """Obtain confusion matrix as a DataFrame.

        Returns
        -------
        pd.DataFrame
            the confusion matrix of the test set
        """
        return confusion_matrices_to_dataframe(self.conf_mats)

    def _get_critical_values_std(self, n_std):
        """Compute the critical values for a chi2 with 2df using the continuity
        correction."""
        alphas = 2. * (sts.norm.cdf(n_std) - 0.5)
        # confidence limits in two dimensions
        return sts.chi2.ppf(alphas, 2)

    def _get_critical_values_alpha(self, alphas):
        """Compute the critical values for a chi2 with 2df."""
        # confidence limits in two dimensions
        return sts.chi2.ppf(alphas, 2)

    def plot(
        self,
        levels : Union[int, float, np.ndarray, None] = None,
        ax=None,
        cmap_name : str = 'Blues',
        equal_aspect : bool = False,
        limit_axis : bool = True,
    ):
        """Plot confidence interval(s) for precision and recall

        Parameters
        ----------
        levels : int, float np.ndarray, default=np.array((1, 2, 3,))
            if int(s) levels is treated as the number of standard deviations
            for the confidence interval.
            If float(s) it is taken to be the density to be contained in the
            confidence interval
            By default we plot 1, 2 and 3 std deviations
        ax : matplotlib.axes.Axes, default=None
            Pre-existing axes for the plot
        cmap_name : str, default='Blues'
            matplotlib cmap name to use for CIs
        equal_aspect : bool, default=False
            enforce square axis
        limit_axis : bool, default=True
            allow ax to be limited for optimal CI plot

        Returns
        -------
        ax : matplotlib.axes.Axes
            the axis with the ellipse added to it

        """
        if self.chi2_scores is None:
            raise RuntimeError("the class needs to be initialised with from_*")

        # quick catch for list and tuples
        if isinstance(levels, (list, tuple)):
            levels = np.asarray(levels)

        # transform levels into scaling factors for the ellipse
        if levels is None:
            levels = self._get_critical_values_std(np.array((1, 2, 3)))
            labels = [r'$1\sigma$ CI', r'$2\sigma$ CI', r'$3\sigma$ CI']
        elif isinstance(levels, int):
            labels = [f'{levels}' + r'$\sigma$ CI']
            levels = self._get_critical_values_std(np.array((levels,)))
        elif (
            isinstance(levels, np.ndarray)
            and np.issubdtype(levels.dtype, np.integer)
        ):
            levels = np.sort(levels)
            labels = [f'{l}' + r'$\sigma$ CI' for l in levels]
            levels = self._get_critical_values_std(levels)
        elif isinstance(levels, float):
            labels = [f'{round(levels * 100, 3)}% CI']
            levels = self._get_critical_values_alpha(np.array((levels,)))
        elif (
            isinstance(levels, np.ndarray)
            and np.issubdtype(levels.dtype, np.floating)
        ):
            levels = np.sort(levels)
            labels = [f'{round(l * 100, 3)}% CI' for l in levels]
            levels = self._get_critical_values_alpha(levels)
        else:
            raise TypeError(
                "`levels` must be a int, float, array-like or None"
            )

        return _plot_pr_curve_contours(
            self.precision,
            self.recall,
            self.chi2_scores,
            self.prec_grid,
            self.rec_grid,
            levels,
            labels,
            cmap_name=cmap_name,
            ax=ax,
            equal_aspect=equal_aspect,
            limit_axis=limit_axis,
        )
