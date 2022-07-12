import multiprocessing
import numpy as np
from sklearn.utils import check_array as sk_check_array

from mmu.commons.types import _convert_to_ext_types
from mmu.commons.types import _is_ext_compat
from mmu.lib import _core
from mmu.lib import _MMU_MT_SUPPORT

_ORDER_SH = {
    "C_CONTIGUOUS": "C",
    0: "C",
    "F_CONTIGUOUS": "F",
    1: "F",
}


def _check_n_threads(n_threads=None):
    maxt_m1 = multiprocessing.cpu_count() - 1
    if n_threads is None:
        if _MMU_MT_SUPPORT:
            n_threads = 4
        else:
            n_threads = 1
    elif not isinstance(n_threads, int):
        raise TypeError("`n_threads` must be an int")
    if n_threads == -1:
        n_threads = maxt_m1
    if (n_threads == 0) or (n_threads < -1):
        raise ValueError("`n_threads` must be strictly positive or -1")
    return n_threads


def _check_array(
    arr,
    dtype_check,
    axis=None,
    target_axis=0,
    target_order=1,
    min_dim=1,
    max_dim=2,
    copy=False,
    check_finite=True,
):
    """Specialisation of check_array for Numpy arrays."""
    convert = copy

    c, dtype = dtype_check(arr)
    convert += c

    ndims = 0
    for s in arr.shape:
        ndims += s > 1
    if ndims > max_dim:
        raise ValueError(f"Array must be at most {max_dim} dimensional.")
    elif ndims < min_dim:
        raise ValueError(f"Array must have at least {min_dim} dimensions.")

    if check_finite and (not _core.all_finite(arr)):
        raise ValueError("Non-finite values encountered")

    # check if array has assumed layout row, column wise
    if axis is None:
        axis = np.argmax(arr.shape)
    if axis != target_axis:
        arr = arr.T

    # check if strides match the layout
    if arr.flags.c_contiguous:
        order = 0
    elif arr.flags.f_contiguous:
        order = 1
    else:
        order = -1

    # the arr is not contiguous or is not aligned
    convert += (
        not arr.flags.aligned
        or (order == -1)
        or ((ndims > 1) and (order != target_order))
    )

    if convert > 0:
        return np.asarray(arr, order=_ORDER_SH[target_order], dtype=dtype)  # type: ignore
    return arr


def check_array(
    arr,
    axis=None,
    target_axis=0,
    target_order=1,
    min_dim=1,
    max_dim=2,
    dtype_check=_convert_to_ext_types,
    copy=False,
    check_finite=True,
    **kwargs,
):
    """Check if array has the appropiate properties.

    There exists a fast path for numpy arrays for Specific dtypes, all other
    types are send to sklearn's check_array.

    Parameters
    ----------
    arr : np.ndarray
        the array to validate
    axis : int, default=None
        the axis containing the observations that are part of a single run,
        e.g. 0 if the values of a single run are stored as a column
    target_axis : int, default=0,
        the axis along which the values should be contiguous
    target_order : int, default=1
        1 for Fortran order/column contiguous and 0 for C order/row contiguous
    min_dim : int, default=1,
        minimal number of dimensions the array should have
    max_dim : int, default=2,
        maximum number of dimensions the array should have
    dtype : str, np.dtype, list[str], default=None
        the supported dtype(s)
    copy : bool, default=False,
        always copy the array
    **kwargs
        keyword arguments passed to sklearn's check_array


    Returns
    -------
    np.ndarray
        the input array or the input array with the correct properties

    """
    # fast path, already a numpy array and of a numeric type we know how to
    # handle
    if isinstance(arr, np.ndarray) and _is_ext_compat(arr):
        return _check_array(
            arr=arr,
            axis=axis,
            target_axis=target_axis,
            target_order=target_order,
            min_dim=min_dim,
            max_dim=max_dim,
            dtype_check=dtype_check,
            copy=copy,
            check_finite=check_finite,
        )
    # slow path handles more types

    # these settings must not change
    kwargs["accept_sparse"] = False
    kwargs["accept_large_sparse"] = False
    kwargs["force_all_finite"] = True
    kwargs["allow_nd"] = False

    if "ensure_2d" not in kwargs:
        kwargs["ensure_2d"] = min_dim == 2

    _, dtype = dtype_check(arr)
    kwargs["dtype"] = dtype

    ndims = 0
    for s in arr.shape:
        ndims += s > 1
    if ndims > max_dim:
        raise ValueError(f"Array must be at most {max_dim} dimensional.")
    elif ndims < min_dim:
        raise ValueError(f"Array must have at least {min_dim} dimensions.")

    # check if array has assumed layout row, column wise
    if axis is None:
        axis = np.argmax(arr.shape)
    if axis != target_axis:
        arr = arr.T

    # check if strides match the layout
    if arr.flags.c_contiguous:
        order = 0
    elif arr.flags.f_contiguous:
        order = 1
    else:
        order = -1
    # the arr is not contiguous or is not aligned, set to target order
    if (
        (not arr.flags.aligned)
        or (order == -1)
        or (ndims > 1 and order != target_order)
    ):
        kwargs["order"] = _ORDER_SH[target_order]
    else:
        kwargs["order"] = None
    return sk_check_array(arr, **kwargs)
