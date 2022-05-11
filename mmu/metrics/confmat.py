import numpy as np
import pandas as pd
from mmu.lib import _core
from mmu.commons import check_array

_CONF_MAT_SUPPORTED_DTYPES = {
    'y': ['bool', 'int32', 'int64', 'float32', 'float64'],
    'yhat': ['bool', 'int32', 'int64', 'float32', 'float64'],
    'score': ['float32', 'float64']
}


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
    return pd.DataFrame(conf_mat, columns=['TN', 'FP', 'FN', 'TP'])


def confusion_matrix(y, yhat=None, score=None, threshold=0.5, return_df=False):
    """Compute binary confusion matrix.

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

    Raises
    ------
    TypeError
        if both `score` and `yhat` are None
    TypeError
        if `score` is not None and `threshold` is not a float

    Returns
    -------
    conf_mat : np.ndarray, pd.DataFrame
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
        Returned as dataframe when `return_df` is True

    """
    # condition checks
    y = check_array(
        y,
        max_dim=1,
        dtypes=_CONF_MAT_SUPPORTED_DTYPES['y'],
    )

    if score is not None:
        score = check_array(
            score,
            max_dim=1,
            dtype=_CONF_MAT_SUPPORTED_DTYPES['score'],
        )
        if not isinstance(threshold, float):
            raise TypeError("`threshold` must be a float if score is not None")
        if score.size != y.size:
            raise ValueError('`score` and `y` must have equal length.')
        conf_mat = _core.confusion_matrix_score(y, score, threshold)
    elif yhat is not None:
        yhat = check_array(
            yhat,
            max_dim=1,
            dtype=_CONF_MAT_SUPPORTED_DTYPES['yhat'],
        )
        if yhat.size != y.size:
            raise ValueError('`yhat` and `y` must have equal length.')

        conf_mat = _core.confusion_matrix(y, yhat)
    else:
        raise TypeError("`yhat` must not be None if `score` is None")

    if return_df:
        return confusion_matrix_to_dataframe(conf_mat)
    return conf_mat


def confusion_matrices(
    y, yhat=None, score=None, threshold=0.5, obs_axis=0, return_df=False
):
    """Compute binary confusion matrix.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations
    yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
        the predicted labels, the same dtypes are supported as y. Can be `None`
        if `score` is not `None`, if both are provided, `score` is ignored.
    score : np.ndarray[float32, float64], default=None
        the classifier score to be evaluated against the `threshold`, i.e.
        `yhat` = `score` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
    threshold : float, default=0.5
        the classification threshold to which the classifier score is evaluated,
        is inclusive.
    obs_axis : int, default=0
        the axis containing the observations for a single run, e.g. 0 when the
        labels and scores are stored as columns
    return_df : bool, default=False
        return confusion matrix as pd.DataFrame

    Raises
    ------
    TypeError
        if both `score` and `yhat` are None
    TypeError
        if `score` is not None and `threshold` is not a float

    Returns
    -------
    confusion_matrix : np.ndarray, pd.DataFrame
        the confusion_matrices where the rows contain the counts for a
        threshold, [i, 0] = TN, [i, 1] = FP, [i, 2] = FN, [i, 3] = TP

    """
    # condition checks
    y = check_array(
        y,
        axis=obs_axis,
        max_dim=2,
        target_order=1,
        dtypes=_CONF_MAT_SUPPORTED_DTYPES['y'],
    )

    if score is not None:
        score = check_array(
            score,
            axis=obs_axis,
            max_dim=2,
            target_order=1,
            dtype=_CONF_MAT_SUPPORTED_DTYPES['score'],
        )
        if not isinstance(threshold, float):
            raise TypeError("`threshold` must be a float if score is not None")
        if score.size != y.size:
            raise ValueError('`score` and `y` must have equal length.')
        conf_mat = _core.confusion_matrix_score_runs(y, score, threshold, obs_axis=obs_axis)
    elif yhat is not None:
        yhat = check_array(
            yhat,
            axis=obs_axis,
            max_dim=2,
            target_order=1,
            dtype=_CONF_MAT_SUPPORTED_DTYPES['yhat'],
        )
        if yhat.size != y.size:
            raise ValueError('`yhat` and `y` must have equal length.')

        conf_mat = _core.confusion_matrix_runs(y, yhat, obs_axis=obs_axis)
    else:
        raise TypeError("`yhat` must not be None if `score` is None")

    if return_df:
        return confusion_matrices_to_dataframe(conf_mat)
    return conf_mat


def confusion_matrices_thresholds(y, yhat=None, score=None, thresholds=None, return_df=False):
    """Compute binary confusion matrix over a range of thresholds.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels for observations
    yhat : np.ndarray[bool, int32, int64, float32, float64], default=None
        the predicted labels, the same dtypes are supported as y. Can be `None`
        if `score` is not `None`, if both are provided, `score` is ignored.
    score : np.ndarray[float32, float64], default=None
        the classifier score to be evaluated against the `threshold`, i.e.
        `yhat` = `score` >= `threshold`. Can be `None` if `yhat` is not `None`,
        if both are provided, this parameter is ignored.
    thresholds : np.ndarray[float32, float64]
        the classification thresholds for which the classifier score is evaluated,
        is inclusive.
    fill : float, default=0.0
        value to fill when a metric is not defined, e.g. divide by zero.
    return_df : bool, default=False
        return the metrics confusion matrix and metrics as a DataFrame

    Raises
    ------
    TypeError
        if both `score` and `yhat` are None
    TypeError
        if `score` is not None and `threshold` is not a float

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
        dtypes=_CONF_MAT_SUPPORTED_DTYPES['y'],
    )

    if score is not None:
        score = check_array(
            score,
            max_dim=2,
            target_order=1,
            dtype=_CONF_MAT_SUPPORTED_DTYPES['score'],
        )
        if not isinstance(thresholds, np.ndarray):
            raise TypeError("`threshold` must be an array of floats if not None")
        if score.size != y.size:
            raise ValueError('`score` and `y` must have equal length.')
        thresholds = check_array(
            thresholds,
            max_dim=1,
            dtype=['float32', 'float64']
        )
        conf_mat = _core.confusion_matrix_thresholds(y, score, thresholds)
    elif yhat is not None:
        yhat = check_array(
            yhat,
            max_dim=2,
            target_order=1,
            dtype=_CONF_MAT_SUPPORTED_DTYPES['yhat'],
        )
        if yhat.size != y.size:
            raise ValueError('`yhat` and `y` must have equal length.')

        conf_mat = _core.confusion_matrix_runs(y, yhat, obs_axis=0)
    else:
        raise TypeError("`yhat` must not be None if `score` is None")

    if return_df:
        return confusion_matrices_to_dataframe(conf_mat)
    return conf_mat


def confusion_matrices_runs_thresholds(
    y, score, thresholds, n_obs=None, fill=0.0, obs_axis=0):
    """Compute confusion matrices over runs and thresholds.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        the ground truth labels, if different runs have different number of
        observations the n_obs parameter must be set to avoid computing metrics
        of the filled values. If ``y`` is one dimensional and ``score`` is not
        the ``y`` values are assumed to be the same for each run.
    score : np.array[float32, float64]
        the classifier scores, if different runs have different number of
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
    y = check_array(
        y,
        axis=obs_axis,
        target_axis=obs_axis,
        max_dim=2,
        dtypes=_CONF_MAT_SUPPORTED_DTYPES['y'],
    )

    score = check_array(
        score,
        axis=obs_axis,
        target_axis=obs_axis,
        max_dim=2,
        dtypes=_CONF_MAT_SUPPORTED_DTYPES['score'],
    )

    n_runs = score.shape[1 - obs_axis]
    max_obs = score.shape[obs_axis]

    if y.shape[1] < 2:
        y = np.tile(y, (y.shape[0], n_runs))

    n_thresholds = thresholds.size
    if n_obs is None:
        n_obs = np.repeat(max_obs, n_runs)

    cm = _core.confusion_matrix_runs_thresholds(
        y, score, thresholds, n_obs
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
    return cm
