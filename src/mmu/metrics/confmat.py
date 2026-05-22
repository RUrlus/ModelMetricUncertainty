import numpy as np
import pandas as pd
from mmu.lib import _core
from mmu.commons import check_array
from mmu.commons import _convert_to_ext_types
from mmu.commons import _convert_to_float


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
    index = (("observed", "negative"), ("observed", "positive"))
    cols = (("estimated", "negative"), ("estimated", "positive"))
    if conf_mat.size == 4:
        conf_mat = conf_mat.reshape(2, 2)
    return pd.DataFrame(conf_mat, index=index, columns=cols)


def confusion_matrices_to_dataframe(conf_mat):
    """Create dataframe with confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray
        array containing multiple confusion matrices as an (N, 4) array

    Returns
    -------
    pd.DataFrame
        the confusion matrix

    """
    return pd.DataFrame(conf_mat, columns=["TN", "FP", "FN", "TP"])


def confusion_matrix(y, yhat=None, scores=None, threshold=0.5, return_df=False):
    """Compute binary confusion matrix.

    `conf_mat` is alias for this function.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations, supported dtypes are
    yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
        the predicted labels, the same dtypes are supported as y. Can be `None`
        if `scores` is not `None`, if both are provided, `scores` is ignored.
    scores : np.ndarray[float32, float64], default=None
        the classifier scores to be evaluated against the `threshold`, i.e.
        `yhat` = `scores` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
        Supported dtypes are float32 and float64.
    threshold : float, default=0.5
        the classification threshold to which the classifier scores is evaluated,
        is inclusive.
    return_df : bool, default=False
        return confusion matrix as pd.DataFrame

    Raises
    ------
    TypeError
        if both `scores` and `yhat` are None
    TypeError
        if `scores` is not None and `threshold` is not a float

    Returns
    -------
    conf_mat : np.ndarray, pd.DataFrame
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
        Returned as dataframe when `return_df` is True

    """
    # condition checks
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

    if return_df:
        return confusion_matrix_to_dataframe(conf_mat)
    return conf_mat


def confusion_matrices(
    y, yhat=None, scores=None, threshold=0.5, obs_axis=0, return_df=False
):
    """Compute binary confusion matrices over multiple runs.

    `conf_mats` is alias for this function.

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
    obs_axis : int, default=0
        the axis containing the observations for a single run, e.g. 0 when the
        labels and scoress are stored as columns
    return_df : bool, default=False
        return confusion matrix as pd.DataFrame

    Raises
    ------
    TypeError
        if both `scores` and `yhat` are None
    TypeError
        if `scores` is not None and `threshold` is not a float

    Returns
    -------
    confusion_matrices np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for the runs
        [i, 0] = TN, [i, 1] = FP, [i, 2] = FN, [i, 3] = TP

    """
    # condition checks
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
        conf_mat = _core.confusion_matrix_score_runs(
            y, scores, threshold, obs_axis=obs_axis
        )
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

        conf_mat = _core.confusion_matrix_runs(y, yhat, obs_axis=obs_axis)
    else:
        raise TypeError("`yhat` must not be None if `scores` is None")

    if return_df:
        return confusion_matrices_to_dataframe(conf_mat)
    return conf_mat


def confusion_matrices_thresholds(y, scores, thresholds, return_df=False):
    """Compute binary confusion matrix over a range of thresholds.

    `conf_mats_thresh` is an alias for this function.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations
    scores : np.ndarray[float32, float64]
        the classifier scores to be evaluated against the `threshold`, i.e.
        `yhat` = `scores` >= `threshold`.
    thresholds : np.ndarray[float32, float64], default=None
        the classification thresholds for which the classifier scores is evaluated,
        is inclusive.
    return_df : bool, default=False
        return the metrics confusion matrix and metrics as a DataFrame

    Returns
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for a
        threshold, [i, 0] = TN, [i, 1] = FP, [i, 2] = FN, [i, 3] = TP

    """
    # condition checks
    y = check_array(
        y,
        max_dim=2,
        target_order=1,
        dtype_check=_convert_to_ext_types,
    )

    scores = check_array(
        scores,
        max_dim=2,
        target_order=1,
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

    if return_df:
        return confusion_matrices_to_dataframe(conf_mat)
    return conf_mat


def confusion_matrices_runs_thresholds(
    y, scores, thresholds, n_obs=None, fill=0.0, obs_axis=0
):
    """Compute confusion matrices over runs and thresholds.

    `conf_mats_runs_thresh` is an alias for this function.

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
    fill : double
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
    return cm
