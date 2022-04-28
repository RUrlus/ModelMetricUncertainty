"""Public API to confusion functions in mmu/core/.../confusion_matrix.hpp.

"""
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

    Exceptions
    ----------
    TypeError
        if both `score` and `yhat` are None
    TypeError
        if `score` is not None and `threshold` is not a float

    Returns
    -------
    conf_mat : np.ndarray, optional
        confusion matrix with layout:
        ```
                  yhat
                0     1
        y  0    TN    FP
           1    FN    TP
        ```
    conf_mat : pd.DataFrame, optional
        confusion matrix as DataFrame

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


def confusion_matrices(y, yhat=None, score=None, threshold=0.5, return_df=False):
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

    Exceptions
    ----------
    TypeError
        if both `score` and `yhat` are None
    TypeError
        if `score` is not None and `threshold` is not a float

    Returns
    -------
    conf_mat : np.ndarray, optional
        confusion matrix with layout:
        ```
                  yhat
                0     1
        y  0    TN    FP
           1    FN    TP
        ```
    conf_mat : pd.DataFrame, optional
        confusion matrix as DataFrame

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
        if not isinstance(threshold, float):
            raise TypeError("`threshold` must be a float if score is not None")
        if score.size != y.size:
            raise ValueError('`score` and `y` must have equal length.')
        conf_mat = _core.confusion_matrix_score_runs(y, score, threshold, obs_axis=0)
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
