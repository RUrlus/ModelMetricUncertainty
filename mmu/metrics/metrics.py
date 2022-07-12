import numpy as np
import pandas as pd

import mmu.lib._mmu_core as _core
from mmu.commons import check_array
from mmu.metrics.confmat import confusion_matrix_to_dataframe
from mmu.metrics.confmat import confusion_matrices_to_dataframe

from mmu.commons import _convert_to_ext_types
from mmu.commons import _convert_to_int
from mmu.commons import _convert_to_float

col_index = {
    "neg.precision": 0,
    "neg.prec": 0,
    "npv": 0,
    "pos.precision": 1,
    "pos.prec": 1,
    "ppv": 1,
    "neg.recall": 2,
    "neg.rec": 2,
    "tnr": 2,
    "specificity": 2,
    "pos.recall": 3,
    "pos.rec": 3,
    "tpr": 3,
    "sensitivity": 3,
    "neg.f1": 4,
    "neg.f1_score": 4,
    "pos.f1": 5,
    "pos.f1_score": 5,
    "fpr": 6,
    "fnr": 7,
    "accuracy": 8,
    "acc": 8,
    "mcc": 9,
}

col_names = [
    "neg.precision",
    "pos.precision",
    "neg.recall",
    "pos.recall",
    "neg.f1",
    "pos.f1",
    "fpr",
    "fnr",
    "acc",
    "mcc",
]


def metrics_to_dataframe(metrics, metric_names=None):
    """Return DataFrame with metrics.

    Parameters
    ----------
    metrics : np.ndarray
        metrics where the rows are the metrics for various runs or
        classification thresholds and the columns are the metrics.
    metric_names : str, list[str], default=None
        if you computed a subset of the metrics you should set the column
        names here

    Returns
    -------
    pd.DataFrame
        the metrics as a DataFrame

    """
    if metric_names is None:
        metric_names = col_names
    elif isinstance(metric_names, str):
        metric_names = [metric_names]
    elif isinstance(metric_names, (tuple, list, np.ndarray)):
        if not isinstance(metric_names[0], str):
            raise TypeError("``metrics_names`` should contain strings.")
    else:
        raise TypeError("``metrics_names`` has an unsupported type.")
    if metrics.ndim == 1:
        return pd.DataFrame(metrics[None, :], columns=metric_names)
    return pd.DataFrame(metrics, columns=metric_names)


def binary_metrics(
    y, yhat=None, scores=None, threshold=None, fill=1.0, return_df=False
):
    r"""Compute binary classification metrics.

    `bmetrics` is an alias for this function.

    Computes the following metrics where [i] indicates the i'th value in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations
    yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
        the predicted labels, the same dtypes are supported as y. Can be `None`
        if `scores` is not `None`, if both are provided, `scores` is ignored.
    scores : np.ndarray[float32, float64], default=None
        the classifier scores to be evaluated against the `threshold`, i.e.
        `yhat` = `scores` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
    threshold : float, default=0.5
        the classification threshold to which the classifier scores is evaluated,
        is inclusive.
    fill : float, default=1.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return confusion matrix as pd.DataFrame

    Returns
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
    metrics : np.ndarray, pd.DataFrame
        the computed metrics

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")

    y = check_array(y, max_dim=1, dtype_check=_convert_to_ext_types)

    if scores is not None:
        scores = check_array(
            scores,
            max_dim=1,
            dtype_check=_convert_to_float,
        )
        if not isinstance(threshold, float):
            raise TypeError("`threshold` must be a float if scores is not None")
        if scores.size != y.size:
            raise ValueError("`scores` and `y` must have equal length.")
        conf_mat = _core.confusion_matrix_score(y, scores, threshold)

    elif yhat is not None:
        yhat = check_array(
            yhat,
            max_dim=1,
            dtype_check=_convert_to_ext_types,
        )
        if yhat.size != y.size:
            raise ValueError("`yhat` and `y` must have equal length.")
        conf_mat = _core.confusion_matrix(y, yhat)
    else:
        raise TypeError("`yhat` must not be None if `scores` is None")

    metrics = _core.binary_metrics(conf_mat, fill)

    if return_df:
        return (confusion_matrix_to_dataframe(conf_mat), metrics_to_dataframe(metrics))
    return conf_mat, metrics


def binary_metrics_confusion_matrix(conf_mat, fill=1.0, return_df=False):
    """Compute binary classification metrics.

    `bmetrics_conf_mat` is an alias for this function.

    Computes the following metrics where [i] indicates the i'th value in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC

    Parameters
    ----------
    conf_mat : np.ndarray[int32, int64],
        confusion_matrix as returned by mmu.confusion_matrix
    fill : float, default=1.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return the metrics confusion matrix and metrics as a DataFrame

    Returns
    -------
    metrics : np.ndarray, pd.DataFrame
        the computed metrics

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")

    if conf_mat.shape == (2, 2):
        conf_mat = conf_mat.flatten()

    conf_mat = check_array(
        conf_mat,
        max_dim=1,
        target_order=0,
        dtype_check=_convert_to_int,
    )
    metrics = _core.binary_metrics(conf_mat, fill)

    if return_df:
        return metrics_to_dataframe(metrics)
    return metrics


def binary_metrics_confusion_matrices(conf_mat, fill=1.0, return_df=False):
    """Compute binary classification metrics.

    `bmetrics_conf_mats` is an alias for this function.

    Computes the following metrics where [i] indicates the i'th value in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC

    Parameters
    ----------
    conf_mat : np.ndarray[int32, int64],
        confusion_matrix as returned by mmu.confusion_matrices, should have
        shape (N, 4) and be C-Contiguous
    fill : float, default=1.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return the metrics confusion matrix and metrics as a DataFrame

    Returns
    -------
    metrics : np.ndarray, pd.DataFrame
        the computed metrics

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")

    conf_mat = check_array(
        conf_mat,
        max_dim=2,
        target_order=0,
        dtype_check=_convert_to_int,
    )
    metrics = _core.binary_metrics_2d(conf_mat, fill)

    if return_df:
        return metrics_to_dataframe(metrics)
    return metrics


def binary_metrics_thresholds(y, scores, thresholds, fill=1.0, return_df=False):
    """Compute binary classification metrics over multiple thresholds.

    `bmetrics_thresh` is an alias for this function.

    Computes the following metrics where [i] indicates the i'th column in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations
    yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
        the predicted labels, the same dtypes are supported as y. Can be `None`
        if `scores` is not `None`, if both are provided, `scores` is ignored.
    scores : np.ndarray[float32, float64], default=None
        the classifier scores to be evaluated against the `threshold`, i.e.
        `yhat` = `scores` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
    thresholds : np.ndarray[float32, float64]
        the classification thresholds for which the classifier scores is evaluated,
        is inclusive.
    fill : float, default=1.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return the metrics confusion matrix and metrics as a DataFrame

    Returns
    -------
    conf_mat : np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for a
        threshold, [i, 0] = TN, [i, 1] = FP, [i, 2] = FN, [i, 3] = TP
    metrics : np.ndarray, pd.DataFrame
        the computed metrics where the rows contain the metrics for a single
        threshold

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")

    y = check_array(
        y,
        max_dim=1,
        dtype_check=_convert_to_ext_types,
    )

    scores = check_array(
        scores,
        max_dim=1,
        dtype_check=_convert_to_float,
    )

    thresholds = check_array(
        thresholds,
        max_dim=1,
        dtype_check=_convert_to_float,
    )

    if scores.size != y.size:
        raise ValueError("`scores` and `y` must have equal length.")
    conf_mat = _core.confusion_matrix_thresholds(y, scores, thresholds)
    metrics = _core.binary_metrics_2d(conf_mat, fill)

    if return_df:
        return (
            confusion_matrices_to_dataframe(conf_mat),
            metrics_to_dataframe(metrics),
        )
    return conf_mat, metrics


def binary_metrics_runs(
    y, yhat=None, scores=None, threshold=None, obs_axis=0, fill=1.0, return_df=False
):
    """Compute binary classification metrics over multiple runs.

    `bmetrics_runs` is an alias for this function.

    Computes the following metrics where [i] indicates the i'th column in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations, should have shape (N, K) for `K` runs
        each consisting of `N` observations if `obs_axis`
    yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
        the predicted labels, the same dtypes are supported as y. Can be `None`
        if `scores` is not `None`, if both are provided, `scores` is ignored.
        `yhat` shape must be compatible with `y`.
    scores : np.ndarray[float32, float64], default=None
        the classifier scores to be evaluated against the `threshold`, i.e.
        `yhat` = `scores` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
        `scores` shape must be compatible with `y`.
    thresholds : np.ndarray[float32, float64]
        the classification thresholds for which the classifier scores is evaluated,
        is inclusive.
    obs_axis : int, default=0
        the axis containing the observations for a single run, e.g. 0 when the
        labels and scoress are stored as columns
    fill : float, default=1.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return the metrics confusion matrix and metrics as a DataFrame

    Returns
    -------
    conf_mat : np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for a
        run, [i, 0] = TN, [i, 1] = FP, [i, 2] = FN, [i, 3] = TP
    metrics : np.ndarray, pd.DataFrame
        the computed metrics where the rows contain the metrics for a single
        run

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")
    if not isinstance(obs_axis, int) or (obs_axis != 0 and obs_axis != 1):
        raise TypeError("`obs_axis` must be either 0 or 1.")

    y = check_array(
        y,
        axis=obs_axis,
        target_axis=obs_axis,
        target_order=1 - obs_axis,
        max_dim=2,
        dtype_check=_convert_to_ext_types,
    )

    if scores is not None:
        scores = check_array(
            scores,
            axis=obs_axis,
            target_axis=obs_axis,
            target_order=1 - obs_axis,
            max_dim=2,
            dtype_check=_convert_to_float,
        )
        if not isinstance(threshold, float):
            raise TypeError("`threshold` must be a float if scores is not None")
        if scores.size != y.size:
            raise ValueError("`scores` and `y` must have equal length.")
        conf_mat = _core.confusion_matrix_score_runs(y, scores, threshold, obs_axis)

    elif yhat is not None:
        yhat = check_array(
            yhat,
            axis=obs_axis,
            target_axis=obs_axis,
            target_order=1 - obs_axis,
            max_dim=2,
            dtype_check=_convert_to_ext_types,
        )
        if yhat.size != y.size:
            raise ValueError("`yhat` and `y` must have equal length.")
        conf_mat = _core.confusion_matrix_runs(y, yhat, obs_axis)
    else:
        raise TypeError("`yhat` must not be None if `scores` is None")

    metrics = _core.binary_metrics_2d(conf_mat, fill)

    if return_df:
        return (
            confusion_matrices_to_dataframe(conf_mat),
            metrics_to_dataframe(metrics),
        )
    return conf_mat, metrics


def binary_metrics_runs_thresholds(
    y, scores, thresholds, n_obs=None, fill=1.0, obs_axis=0
):
    """Compute binary classification metrics over runs and thresholds.

    `bmetrics_runs_thresh` is an alias for this function.

    Computes the following metrics where [i] indicates the i'th column in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        the ground truth labels, if different runs have different number of
        observations the n_obs parameter must be set to avoid computing metrics
        of the filled values. If ``y`` is one dimensional and ``scores`` is not
        the ``y`` values are assumed to be the same for each run.
    scores : np.array[float32, float64]
        the classifier scoress, if different runs have different number of
        observations the n_obs parameter must be set to avoid computing metrics
        of the filled values.
    thresholds : np.array[float32, float64]
        classification thresholds
    n_obs : np.array[int64], default=None
        the number of observations per run, if None the same number of
        observations are assumed exist for each run.
    fill : double, default=1.0
        value to fill when a metric is not defined, e.g. divide by zero.
    obs_axis : {0, 1}, default=0
        0 if the observations for a single run is a column (e.g. from
        pd.DataFrame) and 1 otherwhise

    Returns
    -------
    conf_mat : np.ndarray[int64]
        3D array where the rows contain the counts for a threshold,
        the columns the confusion matrix entries and the slices the counts for
        a run
    metrics : np.ndarray[float64]
        3D array where the first axis is the threshold, the second the metrics
        and the third the run

    """
    thresholds = check_array(
        thresholds,
        max_dim=1,
        dtype_check=_convert_to_float,
    )

    scores = check_array(
        scores,
        axis=obs_axis,
        target_axis=obs_axis,
        target_order=1 - obs_axis,
        max_dim=2,
        dtype_check=_convert_to_float,
    )

    n_runs = scores.shape[1 - obs_axis]
    max_obs = scores.shape[obs_axis]

    if y.ndim == 1:
        y = np.tile(y[:, None], n_runs)
    elif y.shape[1] == 1 and y.shape[0] >= 2:
        y = np.tile(y, n_runs)

    y = check_array(
        y,
        axis=obs_axis,
        target_axis=obs_axis,
        target_order=1 - obs_axis,
        max_dim=2,
        dtype_check=_convert_to_ext_types,
    )

    n_thresholds = thresholds.size
    if n_obs is None:
        n_obs = np.repeat(max_obs, n_runs)

    cm = _core.confusion_matrix_runs_thresholds(y, scores, thresholds, n_obs)
    mtr = _core.binary_metrics_2d(cm, fill)

    # cm and mtr are both flat arrays with order conf_mat, thresholds, runs
    # as this is fastest to create. However, how the cubes will be sliced
    # later doesn't align with this. So we incur a copy such that the cubes
    # have the optimal strides for further processing
    if n_thresholds == 1:
        # create cube from flat array
        cm = cm.reshape(n_runs, 4, order="C")
    else:
        # create cube from flat array
        cm = cm.reshape(n_runs, n_thresholds, 4, order="C")
        # reorder such that with F-order we get from smallest to largest
        # strides: conf_mat, runs, thresholds
        cm = np.swapaxes(np.swapaxes(cm, 0, 2), 1, 2)
        # make values over the confusion matrix and runs contiguous
        cm = np.asarray(cm, order="F")
        # change order s.t. we have thresholds, conf_mat, runs
        cm = np.swapaxes(cm.T, 1, 2)

    # create cube from flat array
    # order is runs, thresholds, metrics
    if n_thresholds == 1:
        # make values over the runs contiguous
        mtr = np.asarray(mtr.reshape(n_runs, 10, order="C"), order="F")
    else:
        mtr = mtr.reshape(n_runs, n_thresholds, 10, order="C")
        # make values over the runs contiguous
        mtr = np.asarray(mtr, order="F")
        # change order s.t. we have thresholds, metrics, runs
        mtr = np.swapaxes(mtr.T, 0, 1)
    return cm, mtr


def precision_recall(
    y, yhat=None, scores=None, threshold=None, fill=1.0, return_df=False
):
    r"""Compute precision and recall.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations
    yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
        the predicted labels, the same dtypes are supported as y. Can be `None`
        if `scores` is not `None`, if both are provided, `scores` is ignored.
    scores : np.ndarray[float32, float64], default=None
        the classifier scores to be evaluated against the `threshold`, i.e.
        `yhat` = `scores` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
    threshold : float, default=0.5
        the classification threshold to which the classifier scores is evaluated,
        is inclusive.
    fill : float, default=1.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return confusion matrix as pd.DataFrame

    Returns
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
    prec_rec : np.ndarray, pd.DataFrame
        precision and recall

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")

    y = check_array(y, max_dim=1, dtype_check=_convert_to_ext_types)

    if scores is not None:
        scores = check_array(
            scores,
            max_dim=1,
            dtype_check=_convert_to_float,
        )
        if not isinstance(threshold, float):
            raise TypeError("`threshold` must be a float if scores is not None")
        if scores.size != y.size:
            raise ValueError("`scores` and `y` must have equal length.")
        conf_mat = _core.confusion_matrix_score(y, scores, threshold)

    elif yhat is not None:
        yhat = check_array(
            yhat,
            max_dim=1,
            dtype_check=_convert_to_ext_types,
        )
        if yhat.size != y.size:
            raise ValueError("`yhat` and `y` must have equal length.")
        conf_mat = _core.confusion_matrix(y, yhat)
    else:
        raise TypeError("`yhat` must not be None if `scores` is None")

    prec_rec = _core.precision_recall(conf_mat, fill)

    if return_df:
        return (
            confusion_matrix_to_dataframe(conf_mat),
            pd.DataFrame(prec_rec, index=["precision", "recall"]).T,
        )
    return conf_mat, prec_rec


def precision_recall_curve(y, scores, thresholds=None, fill=1.0, return_df=False):
    """Compute precision and recall over the thresholds.

    `pr_curve` is an alias for this function.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations
    scores : np.ndarray[float32, float64]
        the classifier scores to be evaluated against the `threshold`, i.e.
        `yhat` = `scores` >= `threshold`
    threshold : np.ndarray[float32, float64]
        the classification thresholds to which the classifier scores is evaluated,
        is inclusive.
    fill : float, default=1.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return confusion matrix as pd.DataFrame

    Returns
    -------
    precision : np.ndarray[float64]
        the precision for each threshold
    recall : np.ndarray[float64]
        the recall for each threshold

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")

    y = check_array(
        y,
        max_dim=1,
        dtype_check=_convert_to_ext_types,
    )

    scores = check_array(
        scores,
        max_dim=1,
        dtype_check=_convert_to_float,
    )

    thresholds = check_array(
        thresholds,
        max_dim=1,
        dtype_check=_convert_to_float,
    )

    if scores.size != y.size:
        raise ValueError("`scores` and `y` must have equal length.")
    conf_mat = _core.confusion_matrix_thresholds(y, scores, thresholds)
    metrics = _core.precision_recall_2d(conf_mat, fill)

    if return_df:
        df = pd.DataFrame(metrics, columns=["precision", "recall"])
        df["thresholds"] = thresholds
        return df
    return metrics[:, 0].copy(), metrics[:, 1].copy()
