import numpy as np


DTYPES_L = [
    np.dtype("int64"),
    np.dtype("bool"),
    np.dtype("int32"),
    np.dtype("float64"),
    np.dtype("float32"),
]


DTYPES_S = {
    np.dtype("int64"),
    np.dtype("bool"),
    np.dtype("int32"),
    np.dtype("float64"),
    np.dtype("float32"),
}


F_TYPES_L = [np.dtype("float64"), np.dtype("float32")]


F_TYPES_S = {np.dtype("float64"), np.dtype("float32")}


I_TYPES_L = [np.dtype("int64"), np.dtype("int32")]


I_TYPES_S = {np.dtype("int64"), np.dtype("int32")}


IB_TYPES_S = {
    np.dtype("int64"),
    np.dtype("bool"),
    np.dtype("int32"),
}


IB_TYPES_L = [
    np.dtype("int64"),
    np.dtype("bool"),
    np.dtype("int32"),
]


def _convert_to_float(arr):
    if (not hasattr(arr, "dtype")) or (arr.dtype not in F_TYPES_S):
        return True, F_TYPES_L[0]
    return False, None


def _convert_to_int(arr):
    if (not hasattr(arr, "dtype")) or (arr.dtype not in I_TYPES_S):
        return True, I_TYPES_L[0]
    return False, None


def _convert_to_int_bool(arr):
    if (not hasattr(arr, "dtype")) or (arr.dtype not in IB_TYPES_S):
        return True, IB_TYPES_L[0]
    return False, None


def _convert_to_ext_types(arr):
    if (not hasattr(arr, "dtype")) or (arr.dtype not in DTYPES_S):
        return True, DTYPES_L[0]
    return False, None


def _is_ext_compat(arr):
    return hasattr(arr, "dtype") or (arr.dtype in DTYPES_S)
