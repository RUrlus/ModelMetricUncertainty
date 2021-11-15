import numpy as np


def _check_shape_order(arr, name):
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
    rows, cols = arr.shape
    if (rows >= cols):
        if not arr.flags['F_CONTIGUOUS']:
            return arr
        return np.asarray(arr, order='F')
    if (rows < cols):
        if arr.flags['C_CONTIGUOUS']:
            return arr
        return np.asarray(arr, order='C')
