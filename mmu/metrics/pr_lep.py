"""Module containing functions to compute uncertainties on precision-recall
using Bivariate Normal over the linearly propogated errors of the confusion
matrix."""
import pandas as pd

from mmu.commons import check_array
from mmu.commons import _convert_to_int
from mmu.metrics.confmat import confusion_matrix
from mmu.metrics.confmat import confusion_matrix_to_dataframe
from mmu.metrics.confmat import confusion_matrices_to_dataframe
from mmu.metrics.confmat import confusion_matrices
from mmu.lib._mmu_core import pr_bvn_error
from mmu.lib._mmu_core import pr_bvn_error_runs


def precision_recall_bvn_uncertainty(
    y, yhat=None, scores=None, threshold=None, alpha=0.95, return_df=False
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
    scores : np.ndarray, default=None
        the classifier scores to be evaluated against the `threshold`, i.e.
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
    conf_mat = confusion_matrix(y, yhat, scores, threshold, return_df=False)
    mtr = pr_bvn_error(conf_mat, alpha)
    cov = mtr[-4:].reshape(2, 2)
    metrics = mtr[:-4]

    if return_df:
        cov_cols = ["precision", "recall"]
        cov_df = pd.DataFrame(cov, index=cov_cols, columns=cov_cols)
        metrics_df = pd.DataFrame(
            index=["precision", "recall"], columns=["metric", "lb_ci", "ub_ci"]
        )
        metrics_df.loc["precision", :] = metrics[:3]
        metrics_df.loc["recall", :] = metrics[3:6]

        return (confusion_matrix_to_dataframe(conf_mat), metrics_df, cov_df)
    return conf_mat, metrics, cov


def precision_recall_bvn_uncertainty_confusion_matrix(
    conf_mat, alpha=0.95, return_df=False
):
    """Compute Precision, Recall and their joint uncertainty.

    The uncertainty on the precision and recall are computed
    using a Multivariate Normal over the linearly propogated errors of the
    confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray[int64]
        the confusion_matrix whith flattened entries: TN, FP, FN, TP
    alpha : float, default=0.95
        the density in the confidence interval
    return_df : bool, default=False
        return confusion matrix and uncertainties as a pd.DataFrame

    confusion_matrix : np.ndarray, pd.DataFrame

    Returns
    -------
    metrics : np.ndarray, pd.DataFrame
        the precision, it's confidence interval and recall and it's confidence
        interval
    cov : np.ndarray, pd.DataFrame
        covariance matrix of precision and recall

    """
    if conf_mat.shape == (2, 2):
        conf_mat = conf_mat.ravel()
    conf_mat = check_array(conf_mat, max_dim=1, dtype_check=_convert_to_int)
    mtr = pr_bvn_error(conf_mat, alpha)
    cov = mtr[-4:].reshape(2, 2)
    metrics = mtr[:-4]

    if return_df:
        cov_cols = ["precision", "recall"]
        cov_df = pd.DataFrame(cov, index=cov_cols, columns=cov_cols)
        metrics_df = pd.DataFrame(
            index=["precision", "recall"], columns=["metric", "lb_ci", "ub_ci"]
        )
        metrics_df.loc["precision", :] = metrics[:3]
        metrics_df.loc["recall", :] = metrics[3:6]

        return (confusion_matrix_to_dataframe(conf_mat), metrics_df, cov_df)
    return conf_mat, metrics, cov


def precision_recall_bvn_uncertainty_confusion_matrices(
    conf_mat, alpha=0.95, return_df=False
):
    """Compute Precision, Recall and their joint uncertainty.

    The uncertainty on the precision and recall are computed
    using a Multivariate Normal over the linearly propogated errors of the
    confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray[int64]
        the confusion_matrices whith columns: TN, FP, FN, TP
    alpha : float, default=0.95
        the density in the confidence interval
    return_df : bool, default=False
        return confusion matrix and uncertainties as a pd.DataFrame

    confusion_matrix : np.ndarray, pd.DataFrame

    Returns
    -------
    metrics : np.ndarray, pd.DataFrame
        the precision, it's confidence interval and recall and it's confidence
        interval
    cov : np.ndarray, pd.DataFrame
        covariance matrix of precision and recall

    """
    if conf_mat.shape == (2, 2):
        conf_mat = conf_mat.ravel()

    conf_mat = check_array(
        conf_mat,
        target_axis=0,
        target_order=0,
        max_dim=2,
        dtype_check=_convert_to_int,
    )
    mtr = pr_bvn_error_runs(conf_mat, alpha)
    cov = mtr[:, -4:]
    metrics = mtr[:, :-4]

    if return_df:
        cov_cols = ["precision", "recall"]
        cov_df = pd.DataFrame(cov, index=cov_cols, columns=cov_cols)
        metrics_df = pd.DataFrame(
            data=metrics,
            columns=["precision", "lb_ci", "ub_ci", "recall", "lb_ci", "ub_ci"],
        )
        return metrics_df, cov_df
    return metrics, cov


def precision_recall_bvn_uncertainty_runs(
    y, yhat=None, scores=None, threshold=None, alpha=0.95, return_df=False
):
    """Compute Precision, Recall and their joint uncertainty over multiple runs.

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
        if `scores` is not `None`, if both are provided, `scores` is ignored.
    scores : np.ndarray, default=None
        the classifier scores to be evaluated against the `threshold`, i.e.
        `yhat` = `score` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
        Supported dtypes are float32 and float64.
    threshold : float, default=0.5
        the classification threshold to which the classifier scores are evaluated,
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
    conf_mat = confusion_matrices(y, yhat, scores, threshold, return_df=False)
    mtr = pr_bvn_error_runs(conf_mat, alpha)
    cov = mtr[:, -4:]
    metrics = mtr[:, :-4]

    if return_df:
        cov_cols = ["precision", "recall"]
        cov_df = pd.DataFrame(cov, index=cov_cols, columns=cov_cols)
        metrics_df = pd.DataFrame(
            data=metrics,
            columns=["precision", "lb_ci", "ub_ci", "recall", "lb_ci", "ub_ci"],
        )
        return (confusion_matrices_to_dataframe(conf_mat), metrics_df, cov_df)
    return conf_mat, metrics, cov
