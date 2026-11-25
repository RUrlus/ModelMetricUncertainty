import numpy as np
import pandas as pd

from mmu.commons import _convert_to_ext_types, _convert_to_float, check_array
from mmu.commons.checks import _check_threshold
from mmu.lib import _core


def confusion_matrix_to_dataframe(conf_mat):
    """Create dataframe with confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray
        array containing a single confusion matrix

    Returns:
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

    Returns:
    -------
    pd.DataFrame
        the confusion matrix

    """
    return pd.DataFrame(conf_mat, columns=["TN", "FP", "FN", "TP"])


def predictions(scores, thresholds):
    """Compute binary predictions from classifier scores and threshold(s).

    The comparison is identical to the one used by the confusion matrix
    functions, i.e. the threshold is inclusive and floating point noise is
    accounted for: ``yhat = scores >= threshold - (1e-8 + 1e-5 * |threshold|)``.

    Parameters
    ----------
    scores : np.ndarray[float32, float64]
        the classifier scores, e.g. shape (N, ) for a single run or (N, K) for
        K runs stored as columns. Other dtypes are converted to float64.
    thresholds : float, np.ndarray[float32, float64]
        the inclusive classification threshold(s). Thresholds are cast to the
        dtype of `scores`.

    Raises:
    ------
    TypeError
        if `thresholds` is not a real number or a one dimensional array
    ValueError
        if `scores` or `thresholds` contain non-finite values

    Returns:
    -------
    yhat : np.ndarray[int64]
        the predictions in {0, 1}. If `thresholds` is a scalar the shape is
        equal to that of `scores`. If `thresholds` is an array of length T, every
        score is compared to every threshold and the shape is
        ``scores.shape + (T, )``, e.g. (N, K, T) where ``yhat[:, k, t]``
        contains the predictions for column ``k`` and threshold ``t``.

    """
    scores = np.asarray(scores)
    if scores.ndim < 1:
        msg = "`scores` must be at least one dimensional"
        raise ValueError(msg)
    if scores.dtype not in (np.float32, np.float64):
        scores = scores.astype(np.float64)
    if not (scores.flags.c_contiguous or scores.flags.f_contiguous) or not scores.flags.aligned:
        scores = np.ascontiguousarray(scores)
    if not _core.all_finite(scores):
        msg_0 = "Non-finite values encountered in `scores`"
        raise ValueError(msg_0)

    is_scalar = np.ndim(thresholds) == 0
    if is_scalar:
        thresholds = np.asarray([_check_threshold(thresholds)], dtype=scores.dtype)
    else:
        thresholds = np.asarray(thresholds)
        if thresholds.ndim != 1:
            msg_1 = "`thresholds` must be a real number or a one dimensional array"
            raise TypeError(msg_1)
        if not np.issubdtype(thresholds.dtype, np.number) or np.issubdtype(thresholds.dtype, np.complexfloating):
            msg_2 = "`thresholds` must contain real numbers"
            raise TypeError(msg_2)
        thresholds = np.ascontiguousarray(thresholds, dtype=scores.dtype)
        if not _core.all_finite(thresholds):
            msg_3 = "Non-finite values encountered in `thresholds`"
            raise ValueError(msg_3)

    # flat array of shape (n_thresholds, scores.size) where each row follows
    # the memory order of `scores`
    yhat = _core.predictions(scores, thresholds)
    shape = scores.shape
    if scores.flags.c_contiguous:
        yhat = np.moveaxis(yhat.reshape((thresholds.size, *shape)), 0, -1)
    else:
        yhat = yhat.ravel().reshape((*shape, thresholds.size), order="F")

    if is_scalar:
        return yhat[..., 0]
    return yhat


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

    Raises:
    ------
    TypeError
        if both `scores` and `yhat` are None
    TypeError
        if `scores` is not None and `threshold` is not a float

    Returns:
    -------
    conf_mat : np.ndarray, pd.DataFrame
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
        Returned as dataframe when `return_df` is True

    """
    # condition checks
    y = check_array(y, max_dim=1, dtype_check=_convert_to_ext_types)

    if scores is not None:
        scores = check_array(scores, max_dim=1, dtype_check=_convert_to_float)
        if not isinstance(threshold, float):
            msg = "`threshold` must be a float if scores is not None"
            raise TypeError(msg)
        if scores.size != y.size:
            msg_0 = "`scores` and `y` must have equal length."
            raise ValueError(msg_0)
        conf_mat = _core.confusion_matrix_score(y, scores, threshold)
    elif yhat is not None:
        yhat = check_array(yhat, max_dim=1, dtype_check=_convert_to_ext_types)
        if yhat.size != y.size:
            msg_1 = "`yhat` and `y` must have equal length."
            raise ValueError(msg_1)

        conf_mat = _core.confusion_matrix(y, yhat)
    else:
        msg_2 = "`yhat` must not be None if `scores` is None"
        raise TypeError(msg_2)

    if return_df:
        return confusion_matrix_to_dataframe(conf_mat)
    return conf_mat


def confusion_matrices(y, yhat=None, scores=None, threshold=0.5, obs_axis=0, return_df=False):
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

    Raises:
    ------
    TypeError
        if both `scores` and `yhat` are None
    TypeError
        if `scores` is not None and `threshold` is not a float

    Returns:
    -------
    confusion_matrices np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for the runs
        [i, 0] = TN, [i, 1] = FP, [i, 2] = FN, [i, 3] = TP

    """
    # condition checks
    y = check_array(
        y, axis=obs_axis, target_axis=obs_axis, target_order=1 - obs_axis, max_dim=2, dtype_check=_convert_to_ext_types
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
        if scores.size != y.size:
            msg_0 = "`scores` and `y` must have equal length."
            raise ValueError(msg_0)
        conf_mat = _core.confusion_matrix_score_runs(y, scores, _check_threshold(threshold), obs_axis=obs_axis)
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
            msg_1 = "`yhat` and `y` must have equal length."
            raise ValueError(msg_1)

        conf_mat = _core.confusion_matrix_runs(y, yhat, obs_axis=obs_axis)
    else:
        msg_2 = "`yhat` must not be None if `scores` is None"
        raise TypeError(msg_2)

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

    Returns:
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for a
        threshold, [i, 0] = TN, [i, 1] = FP, [i, 2] = FN, [i, 3] = TP

    """
    # condition checks
    y = check_array(y, max_dim=2, target_order=1, dtype_check=_convert_to_ext_types)

    scores = check_array(scores, max_dim=2, target_order=1, dtype_check=_convert_to_float)

    thresholds = check_array(thresholds, max_dim=1, dtype_check=_convert_to_float)

    if scores.size != y.size:
        msg = "`scores` and `y` must have equal length."
        raise ValueError(msg)

    conf_mat = _core.confusion_matrix_thresholds(y, scores, thresholds)

    if return_df:
        return confusion_matrices_to_dataframe(conf_mat)
    return conf_mat


def confusion_matrices_runs_thresholds(y, scores, thresholds, n_obs=None, obs_axis=0):
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
    obs_axis : {0, 1}, default=0
        0 if the observations for a single run is a column (e.g. from
        pd.DataFrame) and 1 otherwhise

    Returns:
    -------
    conf_mat : np.ndarray[int64]
        3D array where the rows contain the counts for a threshold,
        the columns the confusion matrix entries and the slices the counts for
        a run

    """

    thresholds = check_array(thresholds, max_dim=1, dtype_check=_convert_to_float)

    scores = check_array(
        scores, axis=obs_axis, target_axis=obs_axis, target_order=1 - obs_axis, max_dim=2, dtype_check=_convert_to_float
    )

    n_runs = scores.shape[1 - obs_axis]
    max_obs = scores.shape[obs_axis]

    y = np.asarray(y)

    if y.ndim == 1:
        y = np.tile(y[:, None], n_runs)
    elif y.ndim == 2 and y.shape[1] == 1 and y.shape[0] >= 2:
        y = np.tile(y, n_runs)

    y = check_array(
        y, axis=obs_axis, target_axis=obs_axis, target_order=1 - obs_axis, max_dim=2, dtype_check=_convert_to_ext_types
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
