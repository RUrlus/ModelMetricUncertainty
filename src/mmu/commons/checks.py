import multiprocessing

import numpy as np
from sklearn.utils import check_array as sk_check_array

from mmu.commons.types import _convert_to_ext_types, _is_ext_compat
from mmu.lib import _MMU_MT_SUPPORT, _core

_ORDER_SH = {"C_CONTIGUOUS": "C", 0: "C", "F_CONTIGUOUS": "F", 1: "F"}


def _check_n_threads(n_threads=None):
    maxt_m1 = multiprocessing.cpu_count() - 1
    if n_threads is None:
        n_threads = 4 if _MMU_MT_SUPPORT else 1
    elif not isinstance(n_threads, int):
        msg = "`n_threads` must be an int"
        raise TypeError(msg)
    if n_threads == -1:
        n_threads = maxt_m1
    if (n_threads == 0) or (n_threads < -1):
        msg = "`n_threads` must be strictly positive or -1"
        raise ValueError(msg)
    return n_threads


def _check_threshold(threshold, name="threshold"):
    """Validate a classification threshold and return it as a float.

    Accepts any finite real scalar, including numpy scalars.

    Classifier scores are not required to be probabilities -- the output of
    `decision_function`, logits and log-odds are all legitimate -- so no [0, 1]
    range constraint is imposed. Restricting the range here would reject
    exactly the inputs whose handling was fixed in the tolerance computation.

    Parameters
    ----------
    threshold : int, float, np.integer, np.floating
        the classification threshold
    name : str, default="threshold"
        parameter name to use in error messages

    Raises:
    ------
    TypeError
        if `threshold` is not a real scalar
    ValueError
        if `threshold` is not finite

    Returns:
    -------
    float
        the threshold as a Python float

    """
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(
        threshold, (int, float, np.integer, np.floating)
    ):
        msg = f"`{name}` must be a real number, got {type(threshold).__name__}"
        raise TypeError(msg)
    value = float(threshold)
    if not np.isfinite(value):
        msg_0 = f"`{name}` must be finite"
        raise ValueError(msg_0)
    return value


def _check_conf_mat_not_empty(conf_mat):
    """Check that a confusion matrix contains at least one observation.

    An all-zero confusion matrix carries no information, so no uncertainty can
    be estimated from it. Without this check the profile log-likelihood
    degenerates to 0/0 and the whole chi2 grid silently becomes NaN.

    Parameters
    ----------
    conf_mat : np.ndarray
        one or more confusion matrices

    Raises:
    ------
    ValueError
        if the confusion matrix contains no observations

    """
    if int(np.asarray(conf_mat).sum()) < 1:
        msg = (
            "`conf_mat` contains no observations; the uncertainty is undefined "
            "for an empty confusion matrix."
        )
        raise ValueError(msg)


def _check_dims(arr, min_dim, max_dim):
    """Validate the semantic dimensionality of `arr`.

    Counts only non-trivial axes, so an (n, 1) column vector is treated as a
    1-D set of observations. Returns that count.

    Note this is deliberately *not* the same as `arr.ndim`; see `_check_array`,
    which additionally constrains the true ndim.

    --- NOTE ---
    The *lower* bound is tested against `arr.ndim`, not against the
    non-trivial count. Counting only axes with `size > 1` means a
    single-element array such as `np.array([0.5])` scores zero, so it was
    rejected as "must have at least 1 dimensions" -- which made a
    single-threshold sweep, `confusion_matrices_thresholds(y, s, [0.5])`,
    impossible. Trivial axes are exactly what the upper bound is meant to
    forgive, so letting them fail the lower bound was incoherent.

    An array is therefore acceptable when *either* notion of its
    dimensionality reaches `min_dim`; a true 0-d scalar (`np.float64(0.5)`,
    `ndim == 0`) is still rejected.
    --- NOTE ---
    """
    non_trivial = 0
    for s in arr.shape:
        non_trivial += s > 1
    if non_trivial > max_dim:
        msg = f"Array must be at most {max_dim} dimensional."
        raise ValueError(msg)
    if max(non_trivial, arr.ndim) < min_dim:
        msg_0 = f"Array must have at least {min_dim} dimensions."
        raise ValueError(msg_0)
    return non_trivial


def _check_array(
    arr, dtype_check, axis=None, target_axis=0, target_order=1, min_dim=1, max_dim=2, copy=False, check_finite=True
):
    """Specialisation of check_array for Numpy arrays."""
    convert = copy

    c, dtype = dtype_check(arr)
    convert += c

    ndims = _check_dims(arr, min_dim, max_dim)

    if arr.ndim > max_dim:
        arr = np.reshape(arr, tuple(s for s in arr.shape if s > 1) or (arr.size,))
        convert += 1

    if check_finite and (not _core.all_finite(arr)):
        msg_1 = "Non-finite values encountered"
        raise ValueError(msg_1)

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
    convert += not arr.flags.aligned or (order == -1) or ((ndims > 1) and (order != target_order))

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


    Returns:
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
    kwargs["ensure_all_finite"] = True
    kwargs["allow_nd"] = False

    if "ensure_2d" not in kwargs:
        kwargs["ensure_2d"] = min_dim == 2

    _, kwargs["dtype"] = dtype_check(arr)
    kwargs["order"] = _ORDER_SH[target_order]
    arr = sk_check_array(arr, **kwargs)

    return _check_array(
        arr,
        dtype_check=dtype_check,
        axis=axis,
        target_axis=target_axis,
        target_order=target_order,
        min_dim=min_dim,
        max_dim=max_dim,
        copy=False,
        check_finite=False,
    )
