import numpy as np
import pandas as pd

import mmu.lib._mmu_core as _core
from mmu.commons import check_array
from mmu.metrics.confmat import confusion_matrix_to_dataframe
from mmu.metrics.confmat import confusion_matrices_to_dataframe

_BINARY_METRICS_SUPPORTED_DTYPES = {
    'y': ['bool', 'int32', 'int64', 'float32', 'float64'],
    'yhat': ['bool', 'int32', 'int64', 'float32', 'float64'],
    'score': ['float32', 'float64']
}

col_index = {
    'neg.precision': 0,
    'neg.prec': 0,
    'npv': 0,
    'pos.precision': 1,
    'pos.prec': 1,
    'ppv': 1,
    'neg.recall': 2,
    'neg.rec': 2,
    'tnr': 2,
    'specificity': 2,
    'pos.recall': 3,
    'pos.rec': 3,
    'tpr': 3,
    'sensitivity': 3,
    'neg.f1': 4,
    'neg.f1_score': 4,
    'pos.f1': 5,
    'pos.f1_score': 5,
    'fpr': 6,
    'fnr': 7,
    'accuracy': 8,
    'acc': 8,
    'mcc': 9,
}

col_names = [
    'neg.precision',
    'pos.precision',
    'neg.recall',
    'pos.recall',
    'neg.f1',
    'pos.f1',
    'fpr',
    'fnr',
    'acc',
    'mcc',
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
            raise TypeError('``metrics_names`` should contain strings.')
    else:
        raise TypeError('``metrics_names`` has an unsupported type.')
    if metrics.ndim == 1:
        return pd.DataFrame(metrics[None, :], columns=metric_names)
    return pd.DataFrame(metrics, columns=metric_names)


def binary_metrics(y, yhat=None, score=None, threshold=None, fill=0.0, return_df=False):
    """Compute binary classification metrics.

    Computes the following metrics:
        0 - neg.precision aka Negative Predictive Value (NPV)
        1 - pos.precision aka Positive Predictive Value (PPV)
        2 - neg.recall aka True Negative Rate (TNR) aka Specificity
        3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
        4 - neg.f1 score
        5 - pos.f1 score
        6 - False Positive Rate (FPR)
        7 - False Negative Rate (FNR)
        8 - Accuracy
        9 - MCC

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
    return_df : bool, default=False
        return confusion matrix as pd.DataFrame

    Returns
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrix
    metrics : np.ndarray, pd.DataFrame
        the computed metrics

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")

    y = check_array(
        y,
        max_dim=1,
        dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['y'],
    )

    if score is not None:
        score = check_array(
            score,
            max_dim=1,
            dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['score'],
        )
        if not isinstance(threshold, float):
            raise TypeError("`threshold` must be a float if score is not None")
        if score.size != y.size:
            raise ValueError('`score` and `y` must have equal length.')

        conf_mat, metrics = _core.binary_metrics_score(y, score, threshold, fill)

    elif yhat is not None:
        yhat = check_array(
            yhat,
            max_dim=1,
            dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['yhat'],
        )
        if yhat.size != y.size:
            raise ValueError('`yhat` and `y` must have equal length.')
        conf_mat, metrics = _core.binary_metrics(y, yhat, fill)
    else:
        raise TypeError("`yhat` must not be None if `score` is None")

    if return_df:
        return (
            confusion_matrix_to_dataframe(conf_mat),
            metrics_to_dataframe(metrics)
        )
    return conf_mat, metrics


def binary_metrics_confusion_matrix(confusion_matrix, fill=0.0, return_df=False):
    """Compute binary classification metrics.

    Computes the following metrics:
        0 - neg.precision aka Negative Predictive Value (NPV)
        1 - pos.precision aka Positive Predictive Value (PPV)
        2 - neg.recall aka True Negative Rate (TNR) aka Specificity
        3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
        4 - neg.f1 score
        5 - pos.f1 score
        6 - False Positive Rate (FPR)
        7 - False Negative Rate (FNR)
        8 - Accuracy
        9 - MCC

    Parameters
    ----------
    confusion_matrix : np.ndarray,
        confusion_matrix as returned by mmu.confusion_matrix
    fill : float, default=0.0
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

    if confusion_matrix.shape == (2, 2):
        confusion_matrix = confusion_matrix.flatten()

    confusion_matrix = check_array(
        confusion_matrix,
        max_dim=1,
        dtypes=['int32', 'int64'],
    )
    metrics = _core.binary_metrics_confusion(confusion_matrix, fill)

    if return_df:
        return metrics_to_dataframe(metrics)
    return metrics


def binary_metrics_thresholds(
    y, score, thresholds, fill=0.0, return_df=False
):
    """Compute binary classification metrics over multiple thresholds.

    Computes the following metrics:
        0 - neg.precision aka Negative Predictive Value (NPV)
        1 - pos.precision aka Positive Predictive Value (PPV)
        2 - neg.recall aka True Negative Rate (TNR) aka Specificity
        3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
        4 - neg.f1 score
        5 - pos.f1 score
        6 - False Positive Rate (FPR)
        7 - False Negative Rate (FNR)
        8 - Accuracy
        9 - MCC

    Parameters
    ----------
    y : np.ndarray
        true labels for observations, supported dtypes are [bool, int32,
        int64, float32, float64]
    score : np.ndarray, default=None
        the classifier score to be evaluated against the `threshold`, i.e.
        `yhat` = `score` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
        Supported dtypes are float32 and float64.
    thresholds : np.ndarray
        the classification thresholds for which the classifier score is evaluated,
        is inclusive.
    fill : float, default=0.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return the metrics confusion matrix and metrics as a DataFrame

    Returns
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for a
        threshold
    metrics : np.ndarray, pd.DataFrame
        the computed metrics where the rows contain the metrics for a single
        threshold

    """
    if not isinstance(fill, float):
        raise TypeError("`fill` must be a float.")

    y = check_array(
        y,
        max_dim=1,
        dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['y'],
    )

    score = check_array(
        score,
        max_dim=1,
        dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['score'],
    )

    thresholds = check_array(
        thresholds,
        max_dim=1,
        dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['score'],
    )

    if score.size != y.size:
        raise ValueError('`score` and `y` must have equal length.')
    conf_mat, metrics = _core.binary_metrics_thresholds(y, score, thresholds, fill)

    if return_df:
        return (
            confusion_matrices_to_dataframe(conf_mat),
            metrics_to_dataframe(metrics)
        )
    return conf_mat, metrics


def binary_metrics_runs(
    y, yhat=None, score=None, threshold=None, obs_axis=0, fill=0.0, return_df=False
):
    """Compute binary classification metrics over multiple runs.

    Computes the following metrics:
        0 - neg.precision aka Negative Predictive Value (NPV)
        1 - pos.precision aka Positive Predictive Value (PPV)
        2 - neg.recall aka True Negative Rate (TNR) aka Specificity
        3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
        4 - neg.f1 score
        5 - pos.f1 score
        6 - False Positive Rate (FPR)
        7 - False Negative Rate (FNR)
        8 - Accuracy
        9 - MCC

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
        the classification threshold for which the classifier score is evaluated,
        is inclusive.
    obs_axis : int, default=0
        the axis containing the observations for a single run, e.g. 0 when the
        labels and scores are stored as columns
    fill : float, default=0.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return the metrics confusion matrix and metrics as a DataFrame

    Returns
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for a single
        run
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
        max_dim=2,
        dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['y'],
    )

    if score is not None:
        score = check_array(
            score,
            axis=obs_axis,
            target_axis=obs_axis,
            max_dim=2,
            dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['score'],
        )
        if not isinstance(threshold, float):
            raise TypeError("`threshold` must be a float if score is not None")
        if score.size != y.size:
            raise ValueError('`score` and `y` must have equal length.')
        conf_mat, metrics = _core.binary_metrics_runs(y, score, threshold, fill, obs_axis)

    elif yhat is not None:
        yhat = check_array(
            yhat,
            axis=obs_axis,
            target_axis=obs_axis,
            max_dim=2,
            dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['yhat'],
        )
        if yhat.size != y.size:
            raise ValueError('`yhat` and `y` must have equal length.')
        conf_mat = _core.confusion_matrix_runs(y, yhat, obs_axis)
        metrics = _core.binary_metrics_confusion(conf_mat, fill)
    else:
        raise TypeError("`yhat` must not be None if `score` is None")

    if return_df:
        return (
            confusion_matrices_to_dataframe(conf_mat),
            metrics_to_dataframe(metrics)
        )
    return conf_mat, metrics



def binary_metrics_runs_thresholds(
    y, scores, thresholds, n_obs=None, fill=0.0, obs_axis=0):
    """Compute binary classification metrics over runs and thresholds.

    Computes the following metrics:
        0 - neg.precision aka Negative Predictive Value (NPV)
        1 - pos.precision aka Positive Predictive Value (PPV)
        2 - neg.recall aka True Negative Rate (TNR) aka Specificity
        3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
        4 - neg.f1 score
        5 - pos.f1 score
        6 - False Positive Rate (FPR)
        7 - False Negative Rate (FNR)
        8 - Accuracy
        9 - MCC

    Parameters
    ----------
    y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
        the ground truth labels, if different runs have different number of
        observations the n_obs parameter must be set to avoid computing metrics
        of the filled values. If ``y`` is one dimensional and ``scores`` is not
        the ``y`` values are assumed to be the same for each run.
    scores : np.array[np.float[32/64]]
        the classifier scores, if different runs have different number of
        observations the n_obs parameter must be set to avoid computing metrics
        of the filled values.
    thresholds : np.array[np.float[32/64]]
        classification thresholds
    n_obs : np.array[np.int64], default=None
        the number of observations per run, if None the same number of
        observations are assumed exist for each run.
    fill : double
        value to fill when a metric is not defined, e.g. divide by zero.
    obs_axis : {0, 1}, default=0
        0 if the observations for a single run is a column (e.g. from
        pd.DataFrame) and 1 otherwhise

    Returns
    -------
    tuple[np.array[np.int64], np.array[np.float64]]
        confusion matrix and metrics array
        the arrays are 3-dimensional where the first axis is the threshold,
        the second the metric and the third the run

    """
    y = check_array(
        y,
        axis=obs_axis,
        target_axis=obs_axis,
        max_dim=2,
        dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['y'],
    )

    scores = check_array(
        scores,
        axis=obs_axis,
        target_axis=obs_axis,
        max_dim=2,
        dtypes=_BINARY_METRICS_SUPPORTED_DTYPES['score'],
    )

    n_runs = scores.shape[1 - obs_axis]
    max_obs = scores.shape[obs_axis]

    if y.shape[1] < 2:
        y = np.tile(y, (y.shape[0], n_runs))

    n_thresholds = thresholds.size
    if n_obs is None:
        n_obs = np.repeat(max_obs, n_runs)

    cm, mtr = _core._binary_metrics_runs_thresholds(
        y, scores, thresholds, n_obs, fill
    )
    # cm and mtr are both flat arrays with order conf_mat, thresholds, runs
    # as this is fastest to create. However, how the cubes will be sliced
    # later doesn't align with this. So we incur a copy such that the cubes
    # have the optimal strides for further processing
    if n_thresholds == 1:
        # create cube from flat array
        cm = cm.reshape(n_runs, 4, order='C')
    else:
        # create cube from flat array
        cm = cm.reshape(n_runs, n_thresholds, 4, order='C')
        # reorder such that with F-order we get from smallest to largest
        # strides: conf_mat, runs, thresholds
        cm = np.swapaxes(np.swapaxes(cm, 0, 2), 1, 2)
        # make values over the confusion matrix and runs contiguous
        cm = np.asarray(cm, order='F')
        # change order s.t. we have thresholds, conf_mat, runs
        cm = np.swapaxes(cm.T, 1, 2)

    # create cube from flat array
    # order is runs, thresholds, metrics
    if n_thresholds == 1:
        mtr = mtr.reshape(n_runs, 10, order='C')
        # make values over the runs contiguous
        mtr = np.asarray(mtr, order='F')
    else:
        mtr = mtr.reshape(n_runs, n_thresholds, 10, order='C')
        # make values over the runs contiguous
        mtr = np.asarray(mtr, order='F')
        # change order s.t. we have thresholds, metrics, runs
        mtr = np.swapaxes(mtr.T, 0, 1)
    return cm, mtr
