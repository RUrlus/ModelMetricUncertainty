import numpy as np
import matplotlib.pyplot as plt


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
