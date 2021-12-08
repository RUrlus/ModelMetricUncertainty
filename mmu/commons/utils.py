import numpy as np
import matplotlib.pyplot as plt


def generate_data(
    n_samples=10000, n_sets=1, proba_dtype=np.float64, yhat_dtype=np.int64,
    y_dtype=np.int64
):
    """Generate data from a logistic process.

    Parameters
    ----------
    n_samples : int, default=10000
        number of samples to generate for each set
    n_sets : int, default=1
        number of sets to generate, each containing ``n_samples``
    proba_dtype : np.dtype, default=np.float64
        dtype for generated probabilities
    yhat_dtype : np.dtype, default=np.int64
        dtype for generated estimated labels
    y_dtype : np.dtype, default=np.int64
        dtype for generated true labels

    Returns
    -------
    proba : np.ndarray
        array of shape (n_samples, n_sets) with generated
        probabilities of dtype ``proba_dtype``
    yhat : np.ndarray
        array of shape (n_samples, n_sets) with generated
        estimated labels of dtype ``yhat_dtype``
    y : np.ndarray[]
        array of shape (n_samples, n_sets) with generated
        true labels of dtype ``y_dtype``

    """
    t_samples = n_samples * n_sets
    proba = np.random.beta(5, 3, size=t_samples).astype(proba_dtype)
    yhat = np.rint(proba).astype(yhat_dtype)
    y = np.random.binomial(1, np.mean(proba), t_samples).astype(y_dtype)
    return (
        proba.reshape(n_samples, n_sets),
        yhat.reshape(n_samples, n_sets),
        y.reshape(n_samples, n_sets),
    )


def _check_shape_order(arr, name, obs_axis=0):
    """Check if order matches shape of the array.

    We expect the observations (rows or columns) to be contiguous in memory.

    Parameters
    ----------
    arr : np.ndarray
        the array to validate
    name : string
        the name of the parameter

    Returns
    -------
    np.ndarray
        the input array or the input array with the correct memory order

    """
    if arr.ndim > 2:
        raise ValueError(f"``{name}`` must be at most two dimensional.")
    if (obs_axis == 0):
        if arr.ndim == 1:
            arr = arr[:, None]
        if not arr.flags['F_CONTIGUOUS']:
            return np.asarray(arr, order='F')
        return arr
    elif (obs_axis == 1):
        if arr.ndim == 1:
            arr = arr[None, :]
        if not arr.flags['C_CONTIGUOUS']:
            return np.asarray(arr, order='C')
        return arr
    else:
        raise ValueError('``obs_axis`` should be 0 or 1.')


def _set_plot_style():
    plt.style.use('ggplot')
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['figure.max_open_warning'] = 0
    return [i['color'] for i in plt.rcParams['axes.prop_cycle']]
