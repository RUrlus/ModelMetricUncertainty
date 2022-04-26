"""Module containing the API for the precision-recall uncertainty modelled as
a Multivariate Normal.
"""
import numpy as np
import pandas as pd

from mmu.commons import check_array
from mmu.metrics.confmat import confusion_matrix_to_dataframe
from mmu.metrics.confmat import confusion_matrices_to_dataframe

import mmu.lib._mmu_core as _core
from mmu.lib._mmu_core import pr_mvn_error
from mmu.lib._mmu_core import pr_curve_mvn_error
from mmu.metrics.confmat import confusion_matrix
from mmu.metrics.confmat import _CONF_MAT_SUPPORTED_DTYPES


def precision_recall_uncertainty(
    y, yhat=None, score=None, threshold=None, alpha=0.95, return_df=False
):
    """Compute Precision, Recall and their joint uncertainty.

    The uncertainty on the precision and recall are computed
    using a Multivariate Normal over the linearly propogated errors of the
    confusion matrix.

    Parameters
    ----------
    y : np.ndarray
        true labels for observations, supported dtypes are [bool, int32,
        int64, float32, float64]
    yhat : np.ndarray, default=None
        the predicted labels, the same dtypes are supported as y. Can be `None`
        if `score` is not `None`, if both are provided, `score` is ignored.
    score : np.ndarray, default=None
        the classifier score to be evaluated against the `threshold`, i.e.
        `yhat` = `score` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
        Supported dtypes are float32 and float64.
    threshold : float, default=0.5
        the classification threshold to which the classifier score is evaluated,
        is inclusive.
    alpha : float, default=0.95
        the density in the confidence interval
    return_df : bool, default=False
        return confusion matrix and uncertainties as a pd.DataFrame

    Returns
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrix
    metrics : np.ndarray, pd.DataFrame
        the precision, it's confidence interval and recall and it's confidence
        interval
    cov : np.ndarray, pd.DataFrame
        covariance matrix of precision and recall

    """
    conf_mat = confusion_matrix(y, yhat, score, threshold, return_df=False)
    mtr = pr_mvn_error(conf_mat, alpha)
    cov = mtr[-4:].reshape(2, 2)
    metrics = mtr[:-4]

    if return_df:
        cov_cols = ['precision', 'recall']
        cov_df = pd.DataFrame(cov, index=cov_cols, columns=cov_cols)
        metrics_df = pd.DataFrame(
            index=['precision', 'recall'],
            columns=['metric', 'lb_ci', 'ub_ci']
        )
        metrics_df.loc['precision', :] = metrics[:3]
        metrics_df.loc['recall', :] = metrics[3:6]

        return (confusion_matrix_to_dataframe(conf_mat), metrics_df, cov_df)
    return conf_mat, metrics, cov


class PrecisionRecallCurveUncertainty():
    """Estimate the uncertainty over the Precision and Recall as Multivariate
    Normal.

    Parameters
    ----------
    thresholds : np.ndarray[float32/64], default=None
        the thresholds for the curve. If None a range of thresholds of
        length `n_thresholds` spanning (0.0, 1.0) will be created
    n_thresholds : int, default=1000
        number of points used to compute the curve for.
    alpha : float, default=0.95
        the density in the confidence interval
    """
    def __init__(self, thresholds=None, n_thresholds=1000, alpha=0.95):
        """Initialise the class.

        Parameters
        ----------
        thresholds : np.ndarray[float32/64], default=None
            the thresholds for the curve. If None a range of thresholds of
            length `n_thresholds` spanning (0.0, 1.0) will be created
        n_thresholds : int, default=1000
            number of points used to compute the curve for.
        alpha : float, default=0.95
            the density in the confidence interval

        """
        if (
            thresholds is None
            or (
                isinstance(thresholds, np.ndarray)
                and np.issubdtype(thresholds.dtype, np.floating)
            )
        ):
            self.thresholds = thresholds
        else:
            raise TypeError("`thresholds` must be None or an ndarray of floats.")

        if isinstance(n_thresholds, int):
            self.n_thresholds = n_thresholds
        else:
            raise TypeError("`n_thresholds` must be an int")
        if isinstance(alpha, float) and (0.0 > alpha < 1.0):
            self.alpha = alpha
        else:
            raise TypeError("`alpha` must be a float in (0.0, 1.0)")

    def _fit(self, alpha):
        mtr = pr_curve_mvn_error(self.conf_mat, alpha)
        self.precision_ = mtr[:, 0]
        self.recall_ = mtr[:, 3]
        self.precision_ci_ = mtr[:, [1, 2]]
        self.recall_ci_ = mtr[:, [4, 5]]
        self.cov_ = mtr[:, -4:]

    def fit(self, X, y=None, alpha=None):
        """Compute the PR curve with uncertainties.

        Parameters
        ----------
        X : np.ndarray[int32/64]
            2D array containing the confusion matrix with shape (N, 4) with
            column order TN, FP, FN, TP
        y : None
            ignored and only present to comply with scikit api
        alpha : float, default=None
            the density in the confidence interval

        """
        self.conf_mat = check_array(
            X,
            target_order=0,
            axis=0,
            min_dim=2,
            max_dim=2,
            dtype=['int32', 'int64']
        )
        self._fit(alpha or self.alpha)

    def fit_from_scores(self, X, y, thresholds=None, n_thresholds=None, alpha=None):
        """Compute the PR curve with uncertainties from scores and labels.

        Parameters
        ----------
        X : np.ndarray[float32/64]
            1D array containing the classifier scores.
        y : np.ndarray[bool, int, float]
            1D array containing the true labels
        thresholds : np.ndarray[float32/64], default=None
            the thresholds for the curve. If None a range of thresholds of
            length `n_thresholds` spanning (0.0, 1.0) will be created
        n_thresholds : int, default=None
            number of points used to compute the curve for.
        alpha : float, default=None
            the density in the confidence interval

        """
        X = check_array(X, max_dim=1, dtype=_CONF_MAT_SUPPORTED_DTYPES['score'])
        y = check_array(y, max_dim=1, dtype=_CONF_MAT_SUPPORTED_DTYPES['y'])

        n_thresholds = n_thresholds or self.n_thresholds

        if (
            thresholds is not None
            and not (
                isinstance(thresholds, np.ndarray)
                and np.issubdtype(thresholds.dtype, np.floating)
            )
        ):
            raise TypeError("`thresholds` must be None or an ndarray of floats.")
        else:
            thresholds = thresholds or self.thresholds or np.linspace(
                start=0., stop=1.0, endpoint=False, num=n_thresholds
            )

        self.conf_mat = _core.confusion_matrix_score_runs(y, X, thresholds)
        self._fit(alpha or self.alpha)
