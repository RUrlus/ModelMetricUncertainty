"""Module containing the API for the precision-recall with Multinomial uncertainty."""
import warnings
from itertools import zip_longest
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import scipy.stats as sts

from mmu.commons import (
    check_array,
    _convert_to_float,
    _convert_to_int,
)
from mmu.commons.checks import _check_n_threads
from mmu.metrics.utils import auto_thresholds
from mmu.metrics.confmat import confusion_matrices_thresholds
from mmu.metrics.confmat import confusion_matrices_to_dataframe
from mmu.viz.contours import _plot_pr_curve_contours
from mmu.lib import _MMU_MT_SUPPORT
from mmu.methods.prpoint import PrecisionRecallUncertainty

import mmu.lib._mmu_core as _core
from mmu.lib._mmu_core import (
    pr_multn_grid_curve_error,
    pr_bvn_grid_curve_error
)

if _MMU_MT_SUPPORT:
    from mmu.lib._mmu_core import (
        pr_multn_grid_curve_error_mt,
        pr_bvn_grid_curve_error_mt,
    )


class PrecisionRecallCurveUncertainty:
    """Compute joint uncertainty Precision-Recall curve.

    The joint statistical uncertainty can be computed using:

    Multinomial method:

    Model's the uncertainty using profile log-likelihoods between
    the observed and most conservative confusion matrix for that
    precision recall. Unlike the Bivariate-Normal/Elliptical approach,
    this approach is valid for relatively low statistic samples and
    at the edges of the curve. However, it does not allow one to
    add the training sample uncertainty to it.

    Bivariate Normal / Elliptical method:

    Model's the linearly propogated errors of the confusion matrix as a
    bivariate Normal distribution. Note that this method is not valid
    for low statistic sets or for precision/recall close to 1.0/0.0.
    In these scenarios the Multinomial method should be used.

    Attributes
    ----------
    conf_mat : np.ndarray[int64]
        the confusion_matrices over the thresholds with columns [TN, FP, FN, TP].
        A DataFrame can be obtained by calling `get_conf_mat`.
    precision : np.ndarray[float64]
        the Positive Predictive Values aka positive precisions
    recall : np.ndarray[float64]
        True Positive Rate aka Sensitivity aka positive recall
    chi2_scores : np.ndarray[float64]
        the sum of squared z scores which follow a chi2 distribution with
        two degrees of freedom. Has shape (`n_bins`, `n_bins`) with bounds
        `precision_bounds` on the y-axis and `recall_bounds` on the x-axis.
    thresholds : np.ndarray[float64], Optional
        the inclusive classification/discrimination thresholds used to compute
        the confusion matrices. Is None when the class is instantiated with
        `from_confusion_matrices`.
    rec_grid : np.ndarray[float64]
        the recall values that where evaluated.
    prec_grid : np.ndarray[float64]
        the precision values that where evaluated.
    n_sigmas : int, float
        the number of marginal standard deviations used to determine the
        bounds of the grid which is evaluated for each observed precision and
        recall.
    epsilon : float
        the value used to prevent the bounds from reaching precision/recall
        1.0/0.0 which would result in NaNs.
    cov_mats : np.ndarray[float64], optional
        flattened covariance matrices for each threshold.
        **Only set when method is bivariate/elliptical.**
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
        self.thresholds = None
        self.cov_mats = None
        self.total_cov_mats = None
        self._has_cov = False
        self._moptions = {
            "mult": {"mult", "multinomial"},
            "bvn": {"bvn", "bivariate", "elliptical"},
        }

    def _compute_bvn_scores(self, n_sigmas, epsilon, n_threads):
        n_threads = _check_n_threads(n_threads)

        # -- validate n_sigmas arg
        self._parse_n_sigmas(n_sigmas)

        # -- validate epsilon arg
        self._parse_epsilon(epsilon)

        # compute scores
        if _MMU_MT_SUPPORT and n_threads > 1:
            prec_rec, self.chi2_scores = pr_bvn_grid_curve_error_mt(
                n_conf_mats=self.n_conf_mats,
                precs_grid=self.prec_grid,
                recs_grid=self.rec_grid,
                conf_mat=self.conf_mats,
                n_sigmas=self.n_sigmas,
                epsilon=self.epsilon,
                n_threads=n_threads,
            )
        else:
            if n_threads > 1:
                warnings.warn(
                    "mmu was not compiled with multi-threading enabled,"
                    " ignoring `n_threads`"
                )
            prec_rec, self.chi2_scores = pr_bvn_grid_curve_error(
                n_conf_mats=self.n_conf_mats,
                precs_grid=self.prec_grid,
                recs_grid=self.rec_grid,
                conf_mat=self.conf_mats,
                n_sigmas=self.n_sigmas,
                epsilon=self.epsilon,
            )

        # compute precision and recall
        self.precision = prec_rec[:, 0]
        self.recall = prec_rec[:, 1]
        self.cov_mats = prec_rec[:, 2:]

    def _compute_multn_scores(self, n_sigmas, epsilon, n_threads):
        n_threads = _check_n_threads(n_threads)

        # -- validate n_sigmas arg
        self._parse_n_sigmas(n_sigmas)

        # -- validate epsilon arg
        self._parse_epsilon(epsilon)

        # compute precision and recall
        mtr = _core.precision_recall_2d(self.conf_mats)
        self.precision = mtr[:, 0].copy()
        self.recall = mtr[:, 1].copy()
        # compute scores
        if _MMU_MT_SUPPORT and n_threads > 1:
            self.chi2_scores = pr_multn_grid_curve_error_mt(
                n_conf_mats=self.n_conf_mats,
                precs_grid=self.prec_grid,
                recs_grid=self.rec_grid,
                conf_mat=self.conf_mats,
                n_sigmas=self.n_sigmas,
                epsilon=self.epsilon,
                n_threads=n_threads,
            )
        else:
            if n_threads > 1:
                warnings.warn(
                    "mmu was not compiled with multi-threading enabled,"
                    " ignoring `n_threads`"
                )
            self.chi2_scores = pr_multn_grid_curve_error(
                n_conf_mats=self.n_conf_mats,
                precs_grid=self.prec_grid,
                recs_grid=self.rec_grid,
                conf_mat=self.conf_mats,
                n_sigmas=self.n_sigmas,
                epsilon=self.epsilon,
            )

    def _parse_method(self, method):
        if method in self._moptions["mult"]:
            self.method = method
            self._compute_scores = self._compute_multn_scores
        elif method in self._moptions["bvn"]:
            self._has_cov = True
            self.method = method
            self._compute_scores = self._compute_bvn_scores
        else:
            raise ValueError(
                "``method`` must be one of 'multinomial', 'mult', 'elliptical'"
                ", 'bivariate', 'bvn'"
            )

    def _parse_thresholds(self, thresholds, scores, max_steps, seed):
        if thresholds is None:
            thresholds = auto_thresholds(scores, max_steps=max_steps, seed=seed)
        else:
            thresholds = check_array(
                thresholds, max_dim=1, dtype_check=_convert_to_float
            )
        if thresholds.min() <= 0.0:
            raise ValueError("`thresholds` should be in (0., 1.0)")
        if thresholds.max() >= 1.0:
            raise ValueError("`thresholds` should be in (0., 1.0)")
        self.thresholds = thresholds

    def _parse_nbins(self, n_bins):
        if self.thresholds is not None:
            lb = min(self.thresholds[0], 1e-12)
            ub = max(self.thresholds[-1], 1 - 1e-12)
        else:
            lb = 1e-12
            ub = 1 - lb
        # -- validate n_bins arg
        if n_bins is None:
            self.prec_grid = self.rec_grid = np.linspace(lb, ub, 1000)
        elif isinstance(n_bins, int):
            if n_bins < 1:
                raise ValueError("`n_bins` must be bigger than 0")
            self.prec_grid = self.rec_grid = np.linspace(lb, ub, n_bins)
        elif isinstance(n_bins, np.ndarray):
            if not np.issubdtype(n_bins.dtype, np.integer):
                raise TypeError("`n_bins` must be an int or list-like ints")
            self.prec_grid = np.linspace(lb, ub, n_bins[0])
            self.rec_grid = np.linspace(lb, ub, n_bins[1])

        elif isinstance(n_bins, (list, tuple)) and len(n_bins) == 2:
            if (not isinstance(n_bins[0], int)) or (not isinstance(n_bins[1], int)):
                raise TypeError("`n_bins` must be an int or list-like ints")
            self.prec_grid = np.linspace(lb, ub, n_bins[0])
            self.rec_grid = np.linspace(lb, ub, n_bins[1])
        else:
            raise TypeError("`n_bins` must be an int or list-like ints")

    def _parse_n_sigmas(self, n_sigmas):
        # -- validate n_sigmas arg
        if not isinstance(n_sigmas, (int, float)):
            raise TypeError("`n_sigmas` must be an int or float.")
        elif n_sigmas < 1.0:
            raise ValueError("`n_sigmas` must be greater than 1.")
        self.n_sigmas = n_sigmas

    def _parse_epsilon(self, epsilon):
        # -- validate epsilon arg
        if not isinstance(epsilon, float):
            raise TypeError("`epsilon` must be a float")
        elif not (1e-15 <= epsilon <= 0.1):
            raise ValueError("`epsilon` must be  in [1e-15, 0.1]")
        self.epsilon = epsilon

    @classmethod
    def from_scores(
        cls,
        y: np.ndarray,
        scores: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        method: str = "multinomial",
        n_bins: Union[int, Tuple[int], List[int], np.ndarray, None] = 1000,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        auto_max_steps: Optional[int] = None,
        auto_seed: Optional[int] = None,
        n_threads: Optional[int] = None,
    ):
        """Compute Precision-Recall curve uncertainty from classifier scores.

        Parameters
        ----------
        y : np.ndarray[bool, int32, int64, float32, float64]
            true labels for the observations
        scores : np.ndarray[float32, float64], default=None
            the classifier score to be evaluated against the `thresholds`, i.e.
            `yhat` = `score` >= `threshold`.
        thresholds : np.ndarray[float64], default=None
            the inclusive classification threshold against which the classifier
            score is evaluated. If None the classification thresholds are
            determined such that each thresholds results in a different
            confusion matrix. Note that the maximum number of thresholds can
            be set using `max_steps`.
        method : str, default='multinomial',
            which method to use, options are the Multinomial approach
            {'multinomial', 'mult'} or the bivariate-normal/elliptical approach
            {'bvn', 'bivariate', 'elliptical'}. Default is 'multinomial'.
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
        n_threads : int, default=None
            the number of threads to use when computing the scores. By default
            we use 4 threads if OpenMP was found, otherwise the computation
            is single threaded. As is common, -1 indicates that all threads but
            one should be used.

        """
        self = cls()
        self._parse_method(method)
        self._parse_thresholds(thresholds, scores, auto_max_steps, auto_seed)
        self._parse_nbins(n_bins)
        self.conf_mats = confusion_matrices_thresholds(
            y=y, scores=scores, thresholds=self.thresholds
        )
        self.n_conf_mats = self.conf_mats.shape[0]
        self._compute_scores(n_sigmas, epsilon, n_threads)
        return self

    @classmethod
    def from_confusion_matrices(
        cls,
        conf_mats: np.ndarray,
        method: str = "multinomial",
        n_bins: Union[int, Tuple[int], List[int], np.ndarray, None] = 1000,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        obs_axis: int = 0,
        n_threads: Optional[int] = None,
    ):
        """Compute Precision-Recall curve uncertainty from confusion matrices.

        Parameters
        ----------
        conf_mat : np.ndarray[int64],
            confusion matrix as returned by mmu.confusion_matrix, i.e.
            with layout [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP or
            the flattened equivalent.
        method : str, default='multinomial',
            which method to use, options are the Multinomial approach
            {'multinomial', 'mult'} or the bivariate-normal/elliptical approach
            {'bvn', 'bivariate', 'elliptical'}. Default is 'multinomial'.
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
        n_threads : int, default=None
            the number of threads to use when computing the scores. By default
            we use 4 threads if OpenMP was found, otherwise the computation
            is single threaded. As is common, -1 indicates that all threads but
            one should be used.

        """
        self = cls()
        self._parse_method(method)
        self.conf_mats = check_array(
            conf_mats,
            min_dim=2,
            max_dim=2,
            axis=obs_axis,
            target_axis=obs_axis,
            target_order=0,
            dtype_check=_convert_to_int,
        )
        self.n_conf_mats = self.conf_mats.shape[obs_axis]
        self._parse_nbins(n_bins)
        self._compute_scores(n_sigmas, epsilon, n_threads)
        return self

    @classmethod
    def from_classifier(
        cls,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
        method: str = "multinomial",
        n_bins: Union[int, Tuple[int], List[int], np.ndarray, None] = 1000,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        auto_max_steps: Optional[int] = None,
        auto_seed: Optional[int] = None,
        n_threads: Optional[int] = None,
    ):
        """Compute Precision-Recall curve uncertainty from a trained classifier.

        Parameters
        ----------
        clf : sklearn.Predictor
            a trained model with method `predict_proba`, used to compute
            the classifier scores
        X : np.ndarray
            the feature array to be used to compute the classifier scores
        y : np.ndarray[bool, int32, int64, float32, float64]
            true labels for observations, supported dtypes are
        threshold : float, default=0.5
            the classification threshold to which the classifier score is evaluated,
            is inclusive.
        method : str, default='multinomial',
            which method to use, options are the Multinomial approach
            {'multinomial', 'mult'} or the bivariate-normal/elliptical approach
            {'bvn', 'bivariate', 'elliptical'}. Default is 'multinomial'.
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
        n_threads : int, default=None
            the number of threads to use when computing the scores. By default
            we use 4 threads if OpenMP was found, otherwise the computation
            is single threaded. As is common, -1 indicates that all threads but
            one should be used.

        """
        self = cls()
        self._parse_method(method)
        if not hasattr(clf, "predict_proba"):
            raise TypeError("`clf` must have a method `predict_proba`")
        score = clf.predict_proba(X)[:, 1]
        self._parse_thresholds(thresholds, score, auto_max_steps, auto_seed)
        self._parse_nbins(n_bins)
        self.conf_mats = confusion_matrices_thresholds(
            y=y, scores=score, thresholds=self.thresholds
        )
        self.n_conf_mats = self.conf_mats.shape[0]
        self._compute_scores(n_sigmas, epsilon, n_threads)
        return self

    def get_conf_mats(self) -> pd.DataFrame:
        """Obtain confusion matrix as a DataFrame.

        Returns
        -------
        pd.DataFrame
            the confusion matrix of the test set
        """
        return confusion_matrices_to_dataframe(self.conf_mats)

    def get_cov_mats(self):
        """Get the covariance matrices over the thresholds.

        Returns
        -------
        cov_df = pd.DataFrame
            the flattened covariance matrix and the thresholds

        Raises
        ------
        NotImplementedError
            when method is not Bivariate-Normal/Elliptical

        """
        if not self._has_cov:
            raise NotImplementedError(
                "`cov_mats` are not computed when method is not"
                " Bivariate-Normal/Elliptical."
            )
        cov_df = pd.DataFrame(
            self.cov_mats, columns=["var_prec", "cov", "cov", "var_rec"]
        )
        cov_df["thresholds"] = self.thresholds
        return cov_df

    def _get_critical_values_std(self, n_std):
        """Compute the critical values for a chi2 with 2df using the continuity
        correction."""
        alphas = 2.0 * (sts.norm.cdf(n_std) - 0.5)
        # confidence limits in two dimensions
        return sts.chi2.ppf(alphas, 2)

    def _get_critical_values_alpha(self, alphas):
        """Compute the critical values for a chi2 with 2df."""
        # confidence limits in two dimensions
        return sts.chi2.ppf(alphas, 2)

    def _add_point_to_plot(self, point, point_kwargs):
        if not isinstance(point, PrecisionRecallUncertainty):
            raise TypeError("``point`` must be PrecisionRecallUncertainty isinstance.")
        if isinstance(point_kwargs, dict):
            if "cmap" not in point_kwargs:
                point_kwargs["cmap"] = "Reds"
            if "ax" in point_kwargs:
                point_kwargs.pop("ax")
            if "equal_aspect" not in point_kwargs:
                point_kwargs["equal_aspect"] = False
        elif point_kwargs is None:
            point_kwargs = {"cmap": "Reds"}
        else:
            raise TypeError("`point_kwargs` must be a Dict or None")
        self._ax = point.plot(ax=self._ax, **point_kwargs)
        self._handles = self._handles + point._handles
        self._ax.legend(handles=self._handles, loc="lower center", fontsize=12)  # type: ignore

    def _add_points_to_plot(self, point, point_kwargs):
        if isinstance(point, PrecisionRecallUncertainty):
            self._add_point_to_plot(point, point_kwargs)
        elif isinstance(point, (list, tuple)):
            if point_kwargs is None:
                point_kwargs = {}
            elif isinstance(point_kwargs, dict):
                point_kwargs = [
                    point_kwargs,
                ] * len(point)
            for p, k in zip_longest(point, point_kwargs):
                self._add_point_to_plot(p, k)
        else:
            raise TypeError(
                "``point_uncertainty`` must be a PrecisionRecallUncertainty"
                " isinstance or a list of PrecisionRecallUncertainty's."
            )

    def plot(
        self,
        levels: Union[int, float, np.ndarray, None] = None,
        ax=None,
        cmap: str = "Blues",
        equal_aspect: bool = False,
        limit_axis: bool = True,
        legend_loc: Optional[str] = None,
        alpha: float = 0.8,
        point_uncertainty: Union[
            PrecisionRecallUncertainty, List[PrecisionRecallUncertainty], None
        ] = None,
        point_kwargs: Union[Dict, List[Dict], None] = None,
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
        cmap : str, default='Blues'
            matplotlib cmap name to use for CIs
        equal_aspect : bool, default=False
            enforce square axis
        limit_axis : bool, default=True
            allow ax to be limited for optimal CI plot
        legend_loc : str, default=None
            location of the legend, default is `lower center`
        alpha : float, defualt=0.8
            opacity value of the contours
        point_uncertainty : PrecisionRecallUncertainty, List, default=None
            Add a point uncertainty(ies) plot to the curve plot, by default the
            `Reds` cmap is used for the point plot(s).
        point_kwargs : dict, list[dict], default=None
            Keyword arguments passed to `point_uncertainty.plot()`, ignored if
            point_uncertainty is None. If `point_uncertainty` is a list and
            `point_kwargs` is a dict the kwargs are used for all point
            uncertainties.

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
            labels = [r"$1\sigma$ CI", r"$2\sigma$ CI", r"$3\sigma$ CI"]
        elif isinstance(levels, int):
            labels = [f"{levels}" + r"$\sigma$ CI"]
            levels = self._get_critical_values_std(np.array((levels,)))
        elif isinstance(levels, np.ndarray) and np.issubdtype(levels.dtype, np.integer):
            levels = np.sort(np.unique(levels))
            labels = [f"{l}" + r"$\sigma$ CI" for l in levels]
            levels = self._get_critical_values_std(levels)
        elif isinstance(levels, float):
            labels = [f"{round(levels * 100, 3)}% CI"]
            levels = self._get_critical_values_alpha(np.array((levels,)))
        elif isinstance(levels, np.ndarray) and np.issubdtype(
            levels.dtype, np.floating
        ):
            levels = np.sort(np.unique(levels))
            labels = [f"{round(l * 100, 3)}% CI" for l in levels]
            levels = self._get_critical_values_alpha(levels)
        else:
            raise TypeError("`levels` must be a int, float, array-like or None")

        self.critical_values_plot = levels

        self._ax, self._handles = _plot_pr_curve_contours(
            precision=self.precision,
            recall=self.recall,
            scores=self.chi2_scores,
            prec_grid=self.prec_grid,
            rec_grid=self.rec_grid,
            levels=levels,
            labels=labels,
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            equal_aspect=equal_aspect,
            limit_axis=limit_axis,
            legend_loc=legend_loc,
        )

        if point_uncertainty is not None:
            self._add_points_to_plot(point_uncertainty, point_kwargs)

        return self._ax
