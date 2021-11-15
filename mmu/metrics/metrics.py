import numpy as np
import pandas as pd
from sklearn.utils import check_array

import mmu.lib._mmu_core as _core
from mmu.commons import _check_shape_order

col_index = {
    'neg.precision': 0,
    'npv': 0,
    'pos.precision': 1,
    'ppv': 1,
    'neg.recall': 2,
    'tnr': 2,
    'specificity': 2,
    'pos.recall': 3,
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


def metrics_to_dataframe(metrics):
    """Return DataFrame with metrics.

    Parameters
    ----------
    metrics : np.ndarray
        metrics where the rows are the metrics for various runs or
        classification thresholds and the columns are the metrics.

    Returns
    -------
    pd.DataFrame
        the metrics as a DataFrame

    """
    if metrics.ndim == 1:
        return pd.DataFrame(metrics[None, :], columns=col_names)
    return pd.DataFrame(metrics, columns=col_names)


def confusion_matrix_to_dataframe(conf_mat):
    """Create dataframe with confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray
        array containing a single confusion matrix

    Returns
    -------
    pd.DataFrame
        the confusion matrix

    """
    index = (('observed', 'negative'), ('observed', 'positive'))
    cols = (('estimated', 'negative'), ('estimated', 'positive'))
    return pd.DataFrame(conf_mat, index=index, columns=cols)


def binary_metrics_runs_thresholds(y, proba, thresholds, n_obs=None, fill=0.0):
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
        of the filled values. If ``y`` is one dimensional and ``proba`` is not
        the ``y`` values are assumed to be the same for each run.
    proba : np.array[np.float[32/64]]
        the predicted probability, if different runs have different number of
        observations the n_obs parameter must be set to avoid computing metrics
        of the filled values.
    thresholds : np.array[np.float[32/64]]
        classification thresholds
    n_obs : np.array[np.int64], default=None
        the number of observations per run, if None the same number of
        observations are assumed exist for each run.
    fill : double
        value to fill when a metric is not defined, e.g. divide by zero.

    Returns
    -------
    tuple[np.array[np.int64], np.array[np.float64]]
        confusion matrix and metrics array
        the arrays are 3-dimensional where the first axis is the threshold,
        the second the metric and the third the run

    """
    y = check_array(y, accept_large_sparse=False)
    proba = check_array(proba, accept_large_sparse=False)

    y = _check_shape_order(y, 'y')
    proba = _check_shape_order(proba, 'proba')

    if y.shape[1] < proba.shape[1]:
        y = np.tile(y, (y.shape[0], proba.shape[1]))

    n_runs = min(y.shape)
    n_thresholds = thresholds.size
    if n_obs is None:
        max_obs = max(y.shape)
        n_obs = np.repeat(max_obs, n_runs)
    cm, mtr = _core._binary_metrics_run_thresholds(
        y, proba, thresholds, n_obs, fill
    )
    # cm and mtr are both flat arrays with order conf_mat, thresholds, runs
    # as this is fastest to create. However, how the cubes will be sliced
    # later doesn't align with this. So we incur a copy such that the cubes
    # have the optimal strides for further processing

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
    mtr = mtr.reshape(n_runs, n_thresholds, 10, order='C')
    # make values over the runs contiguous
    mtr = np.asarray(mtr, order='F')
    # change order s.t. we have thresholds, metrics, runs
    mtr = np.swapaxes(mtr.T, 0, 1)
    return cm, mtr
