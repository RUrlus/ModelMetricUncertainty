"""Module containing the API for the uncertainty modelled through profile log likelihoods."""
import warnings
from itertools import zip_longest
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import scipy.stats as sts

from mmu.commons import check_array
from mmu.commons import _convert_to_int, _convert_to_float
from mmu.commons.checks import _check_n_threads
from mmu.metrics.confmat import confusion_matrix
from mmu.metrics.confmat import confusion_matrix_to_dataframe
from mmu.viz.ellipse import _plot_ellipse
from mmu.viz.contours import _plot_contours

from mmu.lib import _MMU_MT_SUPPORT


class BaseUncertainty:
    """Compute joint uncertainty for a point like Precision-Recall or TPR-FPR.

    The joint statistical uncertainty can be computed using:

    Multinomial method:

    Model's the uncertainty using profile log-likelihoods between
    the observed and most conservative confusion matrix for that point.
    Unlike the Bivariate-Normal/Elliptical approach,
    this approach is valid for relatively low statistic samples and
    at the edges of the curve. However, it does not allow one to
    add the training sample uncertainty to it.

    Bivariate Normal / Elliptical method:

    Model's the linearly propogated errors of the confusion matrix as a
    bivariate Normal distribution. Note that this method is not valid
    for low statistic sets or for points close to (y=1.0, x=0.0).
    In these scenarios the Multinomial method should be used.

    Attributes
    ----------
    conf_mat : np.ndarray[int64]
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
        A DataFrame can be obtained by calling `get_conf_mat`.
    y : float
        - Precision in Precision-Recall
        - TPR       in ROC
    x : float
        - Recall in Precision-Recall
        - FPR    in ROC
    threshold : float, optional
        the inclusive threshold used to determine the confusion matrix.
        Is None when the class is instantiated with `from_predictions` or
        `from_confusion_matrix`.
    cov_mat : np.ndarray[float64], optional
        the covariance matrix of with layout
        [0, 0] = V[y], [0, 1] = COV[y, x], [1, 0] = COV[y, x], [1, 1] = V[x]
        For example for Precision-Recall:
        [0, 0] = V[P], [0, 1] = COV[P, R], [1, 0] = COV[P, R], [1, 1] = V[R]
        A DataFrame can be obtained by calling `get_cov_mat`.
        **Only set when Bivariate/Elliptical method is used.**
    chi2_scores : np.ndarray[float64], optional
        the chi2 scores for the grid with shape (`n_bins`, `n_bins`) and
        bounds y_bounds on the y-axis, x_bounds on the x-axis
        **Only set when Multinomial method is used.**
    y_bounds : np.ndarray[float64], optional
        the lower and upper bound for which y was evaluated, equal
        to y +- `n_sigmas` * sigma(y)
        **Only set when Multinomial method is used.**
    x_bounds : np.ndarray[float64], optional
        the lower and upper bound for which x was evaluated, equal
        to x +- `n_sigmas` * sigma(x)
        **Only set when Multinomial method is used.**
    n_sigmas : int, float, optional
        the number of marginal standard deviations used to determine the
        bounds of the grid which is evaluated for each observed y and x.
        **Only set when Multinomial method is used.**
    epsilon : float, optional
        the value used to prevent the bounds from reaching
        the point (y=1.0, x=0.0) which would result in NaNs.
        **Only set when Multinomial method is used.**
    y_label : str
        the label of the y-avis.
    x_label : str
        the label of the x-avis.
    """

    def __init__(self):
        self.conf_mat = None
        self.y = None
        self.x = None
        self.cov_mat = None
        self._bounds = None
        self.y_bounds = None
        self.x_bounds = None
        self.chi2_scores = None
        self.n_bins = None
        self.n_sigmas = None
        self.epsilon = None
        self._has_cov = False
        self._moptions = {
            "mult": {"mult", "multinomial"},
            "bvn": {"bvn", "bivariate", "elliptical"},
        }
        self.metric_func = None
        self.multn_error_func = None
        self.bvn_cov_func = None
        self.bvn_chi2_score_func = None
        self.multn_chi2_score_func = None
        self.bvn_chi2_scores_func = None
        self.multn_chi2_scores_func = None
        self.multn_error_mt_func = None
        self.bvn_chi2_scores_mt_func = None
        self.multn_chi2_scores_mt_func = None
        self.y_label = None
        self.x_label = None

    def _parse_threshold(self, threshold):
        if not isinstance(threshold, float) or not (0.0 < threshold < 1.0):
            raise TypeError("`threshold` must be a float in [0, 1]")
        self.threshold = threshold

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

    def _compute_multn_scores(self, n_bins, n_sigmas, epsilon, n_threads):
        # -- validate n_bins arg
        if n_bins is None:
            self.n_bins = 100
        elif isinstance(n_bins, int):
            if n_bins < 1:
                raise ValueError("`n_bins` must be bigger than 0")
            self.n_bins = n_bins
        else:
            raise TypeError("`n_bins` must be an int")

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

        self.y, self.x = self.metric_func(self.conf_mat)

        # compute scores
        n_threads = _check_n_threads(n_threads)
        if _MMU_MT_SUPPORT and n_threads > 1:
            self.chi2_scores, bounds = self.multn_error_mt_func(
                n_bins=self.n_bins,
                conf_mat=self.conf_mat,
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
            self.chi2_scores, bounds = self.multn_error_func(
                n_bins=self.n_bins,
                conf_mat=self.conf_mat,
                n_sigmas=self.n_sigmas,
                epsilon=self.epsilon,
            )
        self.y_bounds = bounds[0, :].copy()
        self.x_bounds = bounds[1, :].copy()
        self._bounds = bounds.flatten()

    def _compute_bvn_scores(self, *args, **kwargs):
        out = self.bvn_cov_func(self.conf_mat)
        self.y = out[0]
        self.x = out[1]
        self.cov_mat = out[2:].reshape(2, 2)

        if self.y < 1e-12:
            warnings.warn(f"y `{self.y_label}` is close to zero, COV[P, R] is not valid")
        elif (1 - self.y) < 1e-12:
            warnings.warn(f"y `{self.y_label}` is close to one, COV[P, R] is not valid")
        if self.x < 1e-12:
            warnings.warn(f"x `{self.x_label}` is close to zero, COV[P, R] is not valid")
        elif (1 - self.x) < 1e-12:
            warnings.warn(f"x `{self.x_label}` is close to one, COV[P, R] is not valid")

        # check if we have enough power for binomial approximation
        # n * p * (1 - p) > 10
        fcmat = self.conf_mat.flatten()  # type: ignore
        # p = TP / P + N
        # TODO
        p = fcmat[3] / fcmat.sum()
        n = min(fcmat[1] + fcmat[3], fcmat[2] + fcmat[3])
        min_val_score = n * p * (1 - p)
        if min_val_score <= 10.0:
            warnings.warn(
                "Low statistics, Normal approximation to Binomial may not be"
                f" robust for the observed confusion matrix probabilities"
                " and counts"
            )

    @classmethod
    def from_scores(
        cls,
        y: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.5,
        method: str = "multinomial",
        n_bins: int = 100,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        n_threads: Optional[int] = None,
    ):
        """Compute joint-uncertainty for a point.

        Parameters
        ----------
        y : np.ndarray
            true labels for observations, supported dtypes are [bool, int32,
            int64, float32, float64]
        scores : np.ndarray, default=None
            the classifier scores to be evaluated against the `threshold`, i.e.
            `yhat` = `scores` >= `threshold`.
            Supported dtypes are float32 and float64.
        threshold : float, default=0.5
            the classification threshold to which the classifier scores are evaluated,
            is inclusive.
        method : str, default='multinomial',
            which method to use, options are the Multinomial approach
            {'multinomial', 'mult'} or the bivariate-normal/elliptical approach
            {'bvn', 'bivariate', 'elliptical'}. Default is 'multinomial'.
        n_bins : int, default=100
            the number of bins in the y/x grid for which the
            uncertainty is computed. `scores` will be a `n_bins` by `n_bins`
            array.
            Ignored when method is not the Multinomial approach.
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
            Ignored when method is not the Multinomial approach.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.
        n_threads : int, default=None
            number of threads to use in the computation of multinomial.
            If mmu installed from a wheel it won't have multithreading support.
            If it was compiled with OpenMP support the default is 4, otherwise 1.

        """
        self = cls()
        self._parse_method(method)
        self._parse_threshold(threshold)
        self.conf_mat = confusion_matrix(y=y, scores=scores, threshold=threshold)
        self._compute_scores(n_bins, n_sigmas, epsilon, n_threads)
        return self

    @classmethod
    def from_predictions(
        cls,
        y: np.ndarray,
        yhat: np.ndarray,
        method: str = "multinomial",
        n_bins: int = 100,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        n_threads: Optional[int] = None,
    ):
        """Compute joint-uncertainty for a point.

        Parameters
        ----------
        y : np.ndarray[bool, int32, int64, float32, float64]
            true labels for observations, supported dtypes are
        yhat : yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
            the predicted labels, the same dtypes are supported as y.
        method : str, default='multinomial',
            which method to use, options are the Multinomial approach
            {'multinomial', 'mult'} or the bivariate-normal/elliptical approach
            {'bvn', 'bivariate', 'elliptical'}. Default is 'multinomial'.
        n_bins : int, default=100
            the number of bins in the y/x grid for which the
            uncertainty is computed. `scores` will be a `n_bins` by `n_bins`
            array.
            Ignored when method is not the Multinomial approach.
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
            Ignored when method is not the Multinomial approach.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.
        n_threads : int, default=None
            number of threads to use in the computation of multinomial.
            If mmu installed from a wheel it won't have multithreading support.
            If it was compiled with OpenMP support the default is 4, otherwise 1.

        """
        self = cls()
        self._parse_method(method)
        self.conf_mat = confusion_matrix(y=y, yhat=yhat)
        self._compute_scores(n_bins, n_sigmas, epsilon, n_threads)
        return self

    @classmethod
    def from_confusion_matrix(
        cls,
        conf_mat: np.ndarray,
        method: str = "multinomial",
        n_bins: int = 100,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        n_threads: Optional[int] = None,
    ):
        """Compute joint-uncertainty for a point.

        Parameters
        ----------
        conf_mat : np.ndarray[int64],
            confusion matrix as returned by mmu.confusion_matrix, i.e.
            with layout [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP or
            the flattened equivalent. Supported dtypes are int32, int64
        method : str, default='multinomial',
            which method to use, options are the Multinomial approach
            {'multinomial', 'mult'} or the bivariate-normal/elliptical approach
            {'bvn', 'bivariate', 'elliptical'}. Default is 'multinomial'.
        n_bins : int, default=100
            the number of bins in the y/x grid for which the
            uncertainty is computed. `scores` will be a `n_bins` by `n_bins`
            array.
            Ignored when method is not the Multinomial approach.
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
            Ignored when method is not the Multinomial approach.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.
        n_threads : int, default=None
            number of threads to use in the computation of multinomial.
            If mmu installed from a wheel it won't have multithreading support.
            If it was compiled with OpenMP support the default is 4, otherwise 1.

        """
        self = cls()
        self._parse_method(method)
        if conf_mat.shape == (2, 2):
            conf_mat = conf_mat.ravel()
        self.conf_mat = check_array(conf_mat, max_dim=1, dtype_check=_convert_to_int)
        self._compute_scores(n_bins, n_sigmas, epsilon, n_threads)
        return self

    @classmethod
    def from_classifier(
        cls,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
        method: str = "multinomial",
        n_bins: int = 100,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        n_threads: Optional[int] = None,
    ):
        """Compute joint-uncertainty for a point.

        Parameters
        ----------
        clf : sklearn.Predictor
            a trained model with method `predict_proba`, used to compute
            the classifier scores
        X : np.ndarray
            the feature array to be used to compute the classifier scores
        y : np.ndarray[bool, int32, int64, float32, float64]
            true labels for observations
        threshold : float, default=0.5
            the classification threshold to which the classifier score is evaluated,
            is inclusive.
        method : str, default='multinomial',
            which method to use, options are the Multinomial approach
            {'multinomial', 'mult'} or the bivariate-normal/elliptical approach
            {'bvn', 'bivariate', 'elliptical'}. Default is 'multinomial'.
        n_bins : int, default=100
            the number of bins in the y/x grid for which the
            uncertainty is computed. `scores` will be a `n_bins` by `n_bins`
            array.
            Ignored when method is not the Multinomial approach.
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
            Ignored when method is not the Multinomial approach.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.
        n_threads : int, default=None
            number of threads to use in the computation of multinomial.
            If mmu installed from a wheel it won't have multithreading support.
            If it was compiled with OpenMP support the default is 4, otherwise 1.

        """
        self = cls()
        self._parse_method(method)
        self._parse_threshold(threshold)
        if not hasattr(clf, "predict_proba"):
            raise TypeError("`clf` must have a method `predict_proba`")
        scores = clf.predict_proba(X)[:, 1]
        self.conf_mat = confusion_matrix(y=y, scores=scores, threshold=threshold)
        self._compute_scores(n_bins, n_sigmas, epsilon, n_threads)
        return self

    def _get_scaling_factor_alpha(self, alphas):
        """Compute critical value given a number alphas."""
        # Get the scale for 2 degrees of freedom confidence interval
        # We use chi2 because the equation of an ellipse is a sum of squared variable,
        return np.sqrt(sts.chi2.ppf(alphas, 2))

    def _get_scaling_factor_std(self, stds):
        alphas = 2.0 * (sts.norm.cdf(stds) - 0.5)
        return np.sqrt(sts.chi2.ppf(alphas, 2))

    def _get_critical_values_std(self, n_std):
        """Compute the critical values for a chi2 with 2df using the continuity correction"""
        alphas = 2.0 * (sts.norm.cdf(n_std) - 0.5)
        # confidence limits in two dimensions
        return sts.chi2.ppf(alphas, 2)

    def _get_critical_values_alpha(self, alphas):
        """Compute the critical values for a chi2 with 2df."""
        # confidence limits in two dimensions
        return sts.chi2.ppf(alphas, 2)

    def get_cov_mat(self) -> pd.DataFrame:
        """Obtain covariance matrix of the test set.

        Returns
        -------
        pd.DataFrame
            the covariance matrix

        Raises
        ------
        NotImplementedError
            when method is not Bivariate-Normal/Elliptical

        """
        if not self._has_cov:
            raise NotImplementedError(
                "`cov_mat` is not computed when method is not"
                " Bivariate-Normal/Elliptical."
            )
        cov_cols = [self.y_label, self.x_label]
        return pd.DataFrame(self.cov_mat, index=cov_cols, columns=cov_cols)

    def compute_score_for(
        self,
        y: Union[float, np.ndarray],
        x: Union[float, np.ndarray],
        epsilon: float = 1e-12,
    ) -> float:
        """Compute score for a given y(s) and x(s).
        If method is `bvn` the sum of squared Z scores is computed, if method
        is 'mult' the profile loglikelihood is computed. Both follow a chi2
        distribution with 2 degrees of freedom.

        Parameters
        ----------
        y : float, np.ndarray[float64, float32]
            value(s) to evaluate

            - Precision in Precision-Recall
            - TPR       in ROC
        x : float, np.ndarray[float64, float32]
            value(s) to evaluate

            - Recall in Precision-Recall
            - FPR    in ROC
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.

        Returns
        -------
        chi2_score : float, np.ndarray[float64]
            the chi2_score(s) for the given y(s) and x(s).

        """
        if self.conf_mat is None:
            raise RuntimeError("the class needs to be initialised with from_*")
        if not isinstance(epsilon, float):
            raise TypeError("`epsilon` must be a float")
        elif not (1e-15 <= epsilon <= 0.1):
            raise ValueError("`epsilon` must be  in [1e-15, 0.1]")

        if (
            isinstance(y, float)
            and (0.0 <= y <= 1.0)
            and isinstance(x, float)
            and (0.0 <= x <= 1.0)
        ):
            if self._has_cov:
                return self.bvn_chi2_score_func(y, x, self.conf_mat, epsilon)
            return self.multn_chi2_score_func(y, x, self.conf_mat, epsilon)
        elif isinstance(y, np.ndarray) and isinstance(x, np.ndarray):
            y = check_array(y, max_dim=1, dtype_check=_convert_to_float)
            x = check_array(x, max_dim=1, dtype_check=_convert_to_float)
            if self._has_cov:
                if _MMU_MT_SUPPORT:
                    return self.bvn_chi2_scores_mt_func(
                        y, x, self.conf_mat, epsilon
                    )
                return self.bvn_chi2_scores_func(y, x, self.conf_mat, epsilon)
            if _MMU_MT_SUPPORT:
                return self.multn_chi2_scores_mt_func(y, x, self.conf_mat, epsilon)
            return self.multn_chi2_scores_func(y, x, self.conf_mat, epsilon)
        else:
            raise ValueError(
                "``prec`` and ``rec`` must bot be floats or np.ndarray's of"
                " floats in [0, 1]"
            )

    def compute_pvalue_for(
        self,
        y: Union[float, np.ndarray],
        x: Union[float, np.ndarray],
        epsilon: float = 1e-12,
    ) -> Union[float, np.ndarray]:
        """Compute p-value(s) for a given y(s) and x(s).
        If method is `bvn` the sum of squared Z scores is computed, if method
        is 'mult' the profile loglikelihood is computed. Both follow are chi2
        distribution with 2 degrees of freedom.

        Parameters
        ----------
        y : float, np.ndarray[float64, float32]
            value(s) to evaluate

            - Precision in Precision-Recall
            - TPR       in ROC
        x : float, np.ndarray[float64, float32]
            value(s) to evaluate

            - Recall in Precision-Recall
            - FPR    in ROC
        level : int, float
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.

        Returns
        -------
        chi2_score : float, np.ndarray[float64]
            the chi2_score(s) for the given y(s) and x(s).

        """
        chi2_score = self.compute_score_for(y, x, epsilon)
        return sts.chi2.sf(chi2_score, 2)

    def get_conf_mat(self) -> pd.DataFrame:
        """Obtain confusion matrix as a DataFrame.

        Returns
        -------
        pd.DataFrame
            the confusion matrix of the test set
        """
        return confusion_matrix_to_dataframe(self.conf_mat)

    def _add_point_to_plot(self, point, point_kwargs):
        if isinstance(point_kwargs, dict):
            if "cmap" not in point_kwargs:
                point_kwargs["cmap"] = "Reds"
            if "ax" in point_kwargs:
                point_kwargs.pop("ax")
        elif point_kwargs is None:
            point_kwargs = {"cmap": "Reds"}
        else:
            raise TypeError("`point_kwargs` must be a Dict or None")
        self._ax = point.plot(ax=self._ax, **point_kwargs)
        self._handles = self._handles + point._handles
        self._ax.legend(handles=self._handles, loc="lower left", fontsize=12)  # type: ignore

    def _add_other_to_plot(self, other, other_kwargs):
        if isinstance(other, BaseUncertainty) or isinstance(
            other, BaseSimulatedUncertainty
        ):
            self._add_point_to_plot(other, other_kwargs)
        elif isinstance(other, (list, tuple)):
            if other_kwargs is None:
                other_kwargs = {}
            elif isinstance(other_kwargs, dict):
                other_kwargs = [
                    other_kwargs,
                ] * len(other)
            for point, kwargs in zip_longest(other, other_kwargs):
                self._add_point_to_plot(point, kwargs)
        else:
            raise TypeError(
                "`point_uncertainty` must be of type BaseUncertainty"
                " , BaseSimulatedUncertainty or a list of those"
            )

    def _plot_ellipse(
        self,
        levels,
        ax,
        cmap,
        equal_aspect,
        limit_axis,
        legend_loc,
        alpha,
        other,
        other_kwargs,
    ):
        """Plot elliptical confidence interval(s) for the point."""
        if self.cov_mat is None:
            raise RuntimeError("the class needs to be initialised with from_*")

        # quick catch for list and tuples
        if isinstance(levels, (list, tuple)):
            levels = np.asarray(levels)

        # transform levels into scaling factors for the ellipse
        if levels is None:
            scales = self._get_scaling_factor_std(np.array((1, 2, 3)))
            labels = [r"$1\sigma$ CI", r"$2\sigma$ CI", r"$3\sigma$ CI"]
        elif isinstance(levels, int):
            labels = [f"{levels}" + r"$\sigma$ CI"]
            scales = self._get_scaling_factor_std(np.array((levels,)))
        elif isinstance(levels, np.ndarray) and np.issubdtype(levels.dtype, np.integer):
            levels = np.sort(np.unique(levels))
            labels = [f"{l}" + r"$\sigma$ CI" for l in levels]
            scales = self._get_scaling_factor_std(levels)
        elif isinstance(levels, float):
            labels = [f"{round(levels * 100, 3)}% CI"]
            scales = self._get_scaling_factor_alpha(np.array((levels,)))
        elif isinstance(levels, np.ndarray) and np.issubdtype(
            levels.dtype, np.floating
        ):
            levels = np.sort(np.unique(levels))
            labels = [f"{round(l * 100, 3)}% CI" for l in levels]
            scales = self._get_scaling_factor_alpha(levels)
        else:
            raise TypeError("`levels` must be a int, float, array-like or None")

        self.critical_values_plot = levels

        self._ax, self._handles = _plot_ellipse(
            y=self.y,
            x=self.x,
            cov_mat=self.cov_mat,
            scales=scales,
            labels=labels,
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            equal_aspect=equal_aspect,
            limit_axis=limit_axis,
            legend_loc=legend_loc,
            y_label=self.y_label,
            x_label=self.x_label,
        )
        if other is not None:
            self._add_other_to_plot(other, other_kwargs)
        return self._ax

    def _plot_contour(
        self,
        levels,
        ax,
        cmap,
        equal_aspect,
        limit_axis,
        legend_loc,
        alpha,
        other,
        other_kwargs,
    ):
        """Plot confidence interval(s) for the point."""
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

        self._ax, self._handles = _plot_contours(
            n_bins=self.n_bins,
            y=self.y,
            x=self.x,
            scores=self.chi2_scores,
            bounds=self._bounds,
            levels=levels,
            labels=labels,
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            equal_aspect=equal_aspect,
            limit_axis=limit_axis,
            legend_loc=legend_loc,
            y_label=self.y_label,
            x_label=self.x_label,
        )

        if other is not None:
            self._add_other_to_plot(other, other_kwargs)
        return self._ax

    def plot(
        self,
        levels: Union[int, float, np.ndarray, None] = None,
        ax=None,
        cmap: str = "Blues",
        equal_aspect: bool = True,
        limit_axis: bool = True,
        legend_loc: Optional[str] = None,
        alpha: float = 0.8,
        other: Union[
            "BaseUncertainty",
            "BaseSimulatedUncertainty",
            List["BaseUncertainty"],
            List["BaseSimulatedUncertainty"],
            None,
        ] = None,
        other_kwargs: Union[Dict, List[Dict], None] = None,
    ):
        """Plot confidence interval(s).

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
        equal_aspect : bool, default=True
            ensure the same scaling for x and y axis
        limit_axis : bool, default=True
            allow ax to be limited for optimal CI plot
        legend_loc : str, default=None
            location of the legend, default is `lower left`
        alpha : float, defualt=0.8
            opacity value of the contours
        other : BaseUncertainty, BaseSimulatedUncertainty, List, default=None
            Add other point uncertainty(ies) plot to the plot, by default the
            `Reds` cmap is used for the `other` plot(s).
        other_kwargs : dict, list[dict], default=None
            Keyword arguments passed to `other.plot()`, ignored if
            `other` is None. If `other` is a list and
            `other_kwargs` is a dict, the kwargs are used for all point
            others.

        Returns
        -------
        ax : matplotlib.axes.Axes
            the axis with the ellipse added to it

        """
        if self._has_cov is True:
            return self._plot_ellipse(
                levels=levels,
                ax=ax,
                cmap=cmap,
                equal_aspect=equal_aspect,
                limit_axis=limit_axis,
                legend_loc=legend_loc,
                alpha=alpha,
                other=other,
                other_kwargs=other_kwargs,
            )
        else:
            return self._plot_contour(
                levels=levels,
                ax=ax,
                cmap=cmap,
                equal_aspect=equal_aspect,
                limit_axis=limit_axis,
                legend_loc=legend_loc,
                alpha=alpha,
                other=other,
                other_kwargs=other_kwargs,
            )


class BaseSimulatedUncertainty:
    """Compute joint uncertainty through simulation.

    Model's the uncertainty using profile log-likelihoods between
    the observed and most conservative confusion matrix for that point
    and checks how often random multinomial given the observed
    probabilities of the confusion matrix result in lower profile
    log-likelihoods.

    This approach is much slower than the BaseUncertainty with
    Multinomial method, and is likely to give less well-defined contours unless
    the number of simulations is high enough.

    Attributes
    ----------
    conf_mat : np.ndarray[int64]
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
        A DataFrame can be obtained by calling `get_conf_mat`.
    y : float
        - Precision in Precision-Recall
        - TPR       in ROC
    x : float
        - Recall in Precision-Recall
        - FPR    in ROC
    threshold : float, optional
        the inclusive threshold used to determine the confusion matrix.
        Is None when the class is instantiated with `from_predictions` or
        `from_confusion_matrix`.
    coverage : np.ndarray[float64]
        the percentage of simulations with a lower profile loglikelihood
        for the grid with shape (`n_bins`, `n_bins`) and
        bounds y_bounds on the y-axis, x_bounds on the x-axis
    y_bounds : np.ndarray[float64]
        the lower and upper bound for which y was evaluated, equal
        to y +- `n_sigmas` * sigma(y)
    x_bounds : np.ndarray[float64]
        the lower and upper bound for which x was evaluated, equal
        to x +- `n_sigmas` * sigma(x)
    n_sigmas : int, float
        the number of marginal standard deviations used to determine the
        bounds of the grid which is evaluated for each observed point.
    epsilon : float
        the value used to prevent the bounds from reaching
        the point (y=1.0, x=0.0) which would result in NaNs.
    n_simulations : int
        the number of simulations performed per grid point

    """

    def __init__(self):
        self.conf_mat = None
        self.y = None
        self.x = None
        self._bounds = None
        self.y_bounds = None
        self.x_bounds = None
        self.coverage = None
        self.n_bins = None
        self.n_sigmas = None
        self.epsilon = None
        self.metric_func = None
        self.multn_sim_error_mt_func = None

    def _parse_threshold(self, threshold):
        if not isinstance(threshold, float) or not (0.0 < threshold < 1.0):
            raise TypeError("`threshold` must be a float in [0, 1]")
        self.threshold = threshold

    def _simulate_multn_scores(
        self, n_simulations, n_bins, n_sigmas, epsilon, n_threads
    ):
        n_threads = _check_n_threads(n_threads)
        if isinstance(n_simulations, int):
            if n_simulations <= 29:
                raise ValueError(
                    "``n_simulations`` must be at least 30."
                    " The results will be unreliable for small number of simulations."
                )
            self.n_simulations = n_simulations
        else:
            raise TypeError("``n_simulations`` must be an int")

        # -- validate n_bins arg
        if n_bins is None:
            self.n_bins = 100
        elif isinstance(n_bins, int):
            if n_bins < 1:
                raise ValueError("`n_bins` must be bigger than 0")
            self.n_bins = n_bins
        else:
            raise TypeError("`n_bins` must be an int")

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

        self.y, self.x = self.metric_func(self.conf_mat)
        # compute scores
        if _MMU_MT_SUPPORT and n_threads > 1:
            self.coverage, bounds = self.multn_sim_error_mt_func(
                n_sims=n_simulations,
                n_bins=self.n_bins,
                conf_mat=self.conf_mat,
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
            self.coverage, bounds = self.multn_sim_error_mt_func(
                n_sims=n_simulations,
                n_bins=self.n_bins,
                conf_mat=self.conf_mat,
                n_sigmas=self.n_sigmas,
                epsilon=self.epsilon,
            )
        self.y_bounds = bounds[0, :].copy()
        self.x_bounds = bounds[1, :].copy()
        self._bounds = bounds.flatten()

    @classmethod
    def from_scores(
        cls,
        y: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.5,
        n_simulations: int = 10000,
        n_bins: int = 100,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        n_threads: Optional[int] = None,
    ):
        """Compute joint-uncertainty for a point.

        Parameters
        ----------
        y : np.ndarray
            true labels for observations, supported dtypes are [bool, int32,
            int64, float32, float64]
        scores : np.ndarray, default=None
            the classifier scores to be evaluated against the `threshold`, i.e.
            `yhat` = `scores` >= `threshold`.
            Supported dtypes are float32 and float64.
        threshold : float, default=0.5
            the classification threshold to which the classifier scores are evaluated,
            is inclusive.
        n_simulations : int, default=10000
            the number of simulations to perform per grid point, note that the
            total number of simulations is (n_bins ** 2 * n_simulations)
            It is advised ``n_simulations`` >= 10000
        n_bins : int, default=100
            the number of bins in the y/x grid for which the
            uncertainty is computed. `scores` will be a `n_bins` by `n_bins`
            array.
            Ignored when method is not the Multinomial approach.
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
            Ignored when method is not the Multinomial approach.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.
        n_threads : int, default=None
            number of threads to use in the computation. If mmu installed from a
            wheel it won't have multithreading support. If it was compiled
            with OpenMP support the default is 4, otherwise 1.

        """
        self = cls()
        self._parse_threshold(threshold)
        self.conf_mat = confusion_matrix(y=y, scores=scores, threshold=threshold)
        self._simulate_multn_scores(n_simulations, n_bins, n_sigmas, epsilon, n_threads)
        return self

    @classmethod
    def from_predictions(
        cls,
        y: np.ndarray,
        yhat: np.ndarray,
        n_simulations: int = 10000,
        n_bins: int = 100,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        n_threads: Optional[int] = None,
    ):
        """Compute joint-uncertainty for a point.

        Parameters
        ----------
        y : np.ndarray[bool, int32, int64, float32, float64]
            true labels for observations, supported dtypes are
        yhat : yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
            the predicted labels, the same dtypes are supported as y.
        n_simulations : int, default=10000
            the number of simulations to perform per grid point, note that the
            total number of simulations is (n_bins ** 2 * n_simulations)
            It is advised ``n_simulations`` >= 10000
        n_bins : int, default=100
            the number of bins in the y/x grid for which the
            uncertainty is computed. `scores` will be a `n_bins` by `n_bins`
            array.
            Ignored when method is not the Multinomial approach.
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
            Ignored when method is not the Multinomial approach.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.
        n_threads : int, default=None
            number of threads to use in the computation. If mmu installed from a
            wheel it won't have multithreading support. If it was compiled
            with OpenMP support the default is 4, otherwise 1.

        """
        self = cls()
        self.conf_mat = confusion_matrix(y=y, yhat=yhat)
        self._simulate_multn_scores(n_simulations, n_bins, n_sigmas, epsilon, n_threads)
        return self

    @classmethod
    def from_confusion_matrix(
        cls,
        conf_mat: np.ndarray,
        n_simulations: int = 10000,
        n_bins: int = 100,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        n_threads: Optional[int] = None,
    ):
        """Compute joint-uncertainty for a point.

        Parameters
        ----------
        conf_mat : np.ndarray[int64],
            confusion matrix as returned by mmu.confusion_matrix, i.e.
            with layout [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP or
            the flattened equivalent. Supported dtypes are int32, int64
        n_simulations : int, default=10000
            the number of simulations to perform per grid point, note that the
            total number of simulations is (n_bins ** 2 * n_simulations)
            It is advised ``n_simulations`` >= 10000
        n_bins : int, default=100
            the number of bins in the y/x grid for which the
            uncertainty is computed. `scores` will be a `n_bins` by `n_bins`
            array.
            Ignored when method is not the Multinomial approach.
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
            Ignored when method is not the Multinomial approach.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.
        n_threads : int, default=None
            number of threads to use in the computation. If mmu installed from a
            wheel it won't have multithreading support. If it was compiled
            with OpenMP support the default is 4, otherwise 1.

        """
        self = cls()
        if conf_mat.shape == (2, 2):
            conf_mat = conf_mat.ravel()
        self.conf_mat = check_array(conf_mat, max_dim=1, dtype_check=_convert_to_int)
        self._simulate_multn_scores(n_simulations, n_bins, n_sigmas, epsilon, n_threads)
        return self

    @classmethod
    def from_classifier(
        cls,
        clf,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.5,
        n_simulations: int = 10000,
        n_bins: int = 100,
        n_sigmas: Union[int, float] = 6.0,
        epsilon: float = 1e-12,
        n_threads: Optional[int] = None,
    ):
        """Compute joint-uncertainty for a point.

        Parameters
        ----------
        clf : sklearn.Predictor
            a trained model with method `predict_proba`, used to compute
            the classifier scores
        X : np.ndarray
            the feature array to be used to compute the classifier scores
        y : np.ndarray[bool, int32, int64, float32, float64]
            true labels for observations
        threshold : float, default=0.5
            the classification threshold to which the classifier score is evaluated,
            is inclusive.
        n_simulations : int, default=10000
            the number of simulations to perform per grid point, note that the
            total number of simulations is (n_bins ** 2 * n_simulations)
            It is advised ``n_simulations`` >= 10000
        n_bins : int, default=100
            the number of bins in the y/x grid for which the
            uncertainty is computed. `scores` will be a `n_bins` by `n_bins`
            array.
            Ignored when method is not the Multinomial approach.
        n_sigmas : int, float, default=6.0
            the number of marginal standard deviations used to determine the
            bounds of the grid which is evaluated.
            Ignored when method is not the Multinomial approach.
        epsilon : float, default=1e-12
            the value used to prevent the bounds from reaching
            the point (y=1.0, x=0.0) which would result in NaNs.
            Ignored when method is not the Multinomial approach.
        n_threads : int, default=None
            number of threads to use in the computation. If mmu installed from a
            wheel it won't have multithreading support. If it was compiled
            with OpenMP support the default is 4, otherwise 1.

        """
        self = cls()
        self._parse_threshold(threshold)
        if not hasattr(clf, "predict_proba"):
            raise TypeError("`clf` must have a method `predict_proba`")
        scores = clf.predict_proba(X)[:, 1]
        self.conf_mat = confusion_matrix(y=y, scores=scores, threshold=threshold)
        self._simulate_multn_scores(n_simulations, n_bins, n_sigmas, epsilon, n_threads)
        return self

    def _get_cdf_factor_std(self, stds):
        return 2.0 * (sts.norm.cdf(stds) - 0.5)

    def get_conf_mat(self) -> pd.DataFrame:
        """Obtain confusion matrix as a DataFrame.

        Returns
        -------
        pd.DataFrame
            the confusion matrix of the test set
        """
        return confusion_matrix_to_dataframe(self.conf_mat)

    def _add_point_to_plot(self, point, point_kwargs):
        if isinstance(point_kwargs, dict):
            if "cmap" not in point_kwargs:
                point_kwargs["cmap"] = "Reds"
            if "ax" in point_kwargs:
                point_kwargs.pop("ax")
        elif point_kwargs is None:
            point_kwargs = {"cmap": "Reds"}
        else:
            raise TypeError("`point_kwargs` must be a Dict or None")
        self._ax = point.plot(ax=self._ax, **point_kwargs)
        self._handles = self._handles + point._handles
        self._ax.legend(handles=self._handles, loc="lower left", fontsize=12)  # type: ignore

    def _add_other_to_plot(self, other, other_kwargs):
        if isinstance(other, BaseUncertainty) or isinstance(
            other, BaseSimulatedUncertainty
        ):
            self._add_point_to_plot(other, other_kwargs)
        elif isinstance(other, (list, tuple)):
            if other_kwargs is None:
                other_kwargs = {}
            elif isinstance(other_kwargs, dict):
                other_kwargs = [
                    other_kwargs,
                ] * len(other)
            for point, kwargs in zip_longest(other, other_kwargs):
                self._add_point_to_plot(point, kwargs)
        else:
            raise TypeError(
                "`point_uncertainty` must be a subclass of PointUncertainty"
                " or a list of PointUncertainty's"
            )

    def plot(
        self,
        levels: Union[int, float, np.ndarray, None] = None,
        ax=None,
        cmap: str = "Blues",
        equal_aspect: bool = True,
        limit_axis: bool = True,
        legend_loc: Optional[str] = None,
        alpha: float = 0.8,
        other: Union[
            "BaseUncertainty",
            "BaseSimulatedUncertainty",
            List["BaseUncertainty"],
            List["BaseSimulatedUncertainty"],
            None,
        ] = None,
        other_kwargs: Union[Dict, List[Dict], None] = None,
    ):
        """Plot confidence interval(s) a point.

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
            location of the legend, default is `lower left`
        alpha : float, defualt=0.8
            opacity value of the contours
        other : BaseUncertainty, BaseSimulatedUncertainty, List, default=None
            Add other point uncertainty(ies) plot to the plot, by default the
            `Reds` cmap is used for the `other` plot(s).
        other_kwargs : dict, list[dict], default=None
            Keyword arguments passed to `other.plot()`, ignored if
            `other` is None. If `other` is a list and
            `other_kwargs` is a dict, the kwargs are used for all point
            others.

        Returns
        -------
        ax : matplotlib.axes.Axes
            the axis with the ellipse added to it

        """
        if self.coverage is None:
            raise RuntimeError("the class needs to be initialised with from_*")

        # quick catch for list and tuples
        if isinstance(levels, (list, tuple)):
            levels = np.asarray(levels)

        # transform levels into scaling factors for the ellipse
        if levels is None:
            levels = self._get_cdf_factor_std(np.array((1, 2, 3)))
            labels = [r"$1\sigma$ CI", r"$2\sigma$ CI", r"$3\sigma$ CI"]
        elif isinstance(levels, int):
            labels = [f"{levels}" + r"$\sigma$ CI"]
            levels = self._get_cdf_factor_std(np.array((levels,)))
        elif isinstance(levels, np.ndarray) and np.issubdtype(levels.dtype, np.integer):
            levels = np.sort(np.unique(levels))
            labels = [f"{l}" + r"$\sigma$ CI" for l in levels]
        elif isinstance(levels, float):
            labels = [f"{round(levels * 100, 3)}% CI"]
            levels = np.array((levels,))
        elif isinstance(levels, np.ndarray) and np.issubdtype(
            levels.dtype, np.floating
        ):
            levels = np.sort(np.unique(levels))
            labels = [f"{round(l * 100, 3)}% CI" for l in levels]
        else:
            raise TypeError("`levels` must be a int, float, array-like or None")

        self.critical_values_plot = levels

        self._ax, self._handles = _plot_contours(
            n_bins=self.n_bins,
            y=self.y,
            x=self.x,
            scores=self.coverage,
            bounds=self._bounds,
            levels=levels,
            labels=labels,
            cmap=cmap,
            ax=ax,
            alpha=alpha,
            equal_aspect=equal_aspect,
            limit_axis=limit_axis,
            legend_loc=legend_loc,
            y_label=self.y_label,
            x_label=self.x_label,
        )

        if other is not None:
            self._add_other_to_plot(other, other_kwargs)
        return self._ax
