import numpy as np
import pytest

from mmu_tests import _mmu_core_tests

def test_is_contiguous():
    carr = np.ones((10), order='C', dtype=np.float64)
    farr = np.ones((10), order='F', dtype=np.float64)
    assert _mmu_core_tests.is_contiguous(carr)
    assert _mmu_core_tests.is_contiguous(farr)

    carr = np.ones((10, 4), order='C', dtype=np.float64)
    farr = np.ones((10, 4), order='F', dtype=np.float64)
    assert _mmu_core_tests.is_contiguous(carr)
    assert _mmu_core_tests.is_contiguous(farr)

    assert _mmu_core_tests.is_contiguous(carr[:, [1, 3]])
    assert _mmu_core_tests.is_contiguous(farr[:, [1, 3]])
    assert _mmu_core_tests.is_contiguous(carr[[1, 3], :])
    assert _mmu_core_tests.is_contiguous(farr[[1, 3], :])


def test_is_c_contiguous():
    carr = np.ones((10), order='C', dtype=np.float64)
    farr = np.ones((10), order='F', dtype=np.float64)
    assert _mmu_core_tests.is_c_contiguous(carr)
    assert _mmu_core_tests.is_c_contiguous(farr)

    carr = np.ones((10, 4), order='C', dtype=np.float64)
    farr = np.ones((10, 4), order='F', dtype=np.float64)

    assert _mmu_core_tests.is_c_contiguous(carr)
    assert _mmu_core_tests.is_c_contiguous(farr[[1, 3], :])
    assert _mmu_core_tests.is_c_contiguous(carr[[1, 3], :])

    assert _mmu_core_tests.is_c_contiguous(farr) is False
    assert _mmu_core_tests.is_c_contiguous(carr[:, [1, 3]]) is False
    assert _mmu_core_tests.is_c_contiguous(farr[:, [1, 3]]) is False


def test_is_f_contiguous():
    carr = np.ones((10), order='C', dtype=np.float64)
    farr = np.ones((10), order='F', dtype=np.float64)
    assert _mmu_core_tests.is_f_contiguous(carr)
    assert _mmu_core_tests.is_f_contiguous(farr)

    carr = np.ones((10, 4), order='C', dtype=np.float64)
    farr = np.ones((10, 4), order='F', dtype=np.float64)

    assert _mmu_core_tests.is_f_contiguous(carr) is False
    assert _mmu_core_tests.is_f_contiguous(farr[[1, 3], :]) is False
    assert _mmu_core_tests.is_f_contiguous(carr[[1, 3], :]) is False

    assert _mmu_core_tests.is_f_contiguous(farr)
    assert _mmu_core_tests.is_f_contiguous(carr[:, [1, 3]])
    assert _mmu_core_tests.is_f_contiguous(farr[:, [1, 3]])

def test_check_shape_order():
    """Test check_shape_order.

    check_shape_order checks if the array is contiguous along the obs_axis,
    if not it will return a copy of the array in the correct order, C for
    obs_axis == 1 and F order for obs_axis == 2
    """
    carr_good = np.zeros((2, 10), order='C', dtype=np.float64)
    carr_bad = np.zeros((10, 2), order='C', dtype=np.float64)

    arr = _mmu_core_tests.check_shape_order(carr_good, 'x', 0)
    assert arr.flags['F_CONTIGUOUS']

    arr = _mmu_core_tests.check_shape_order(carr_bad, 'x', 0)
    assert arr.flags['F_CONTIGUOUS']

    arr = _mmu_core_tests.check_shape_order(carr_bad, 'x', 1)
    assert arr.flags['C_CONTIGUOUS']

    arr = _mmu_core_tests.check_shape_order(carr_good, 'x', 1)
    assert arr.flags['C_CONTIGUOUS']

    farr_good = np.zeros((10, 2), order='F', dtype=np.float64)
    farr_bad = np.zeros((2, 10), order='F', dtype=np.float64)

    arr = _mmu_core_tests.check_shape_order(farr_good, 'x', 0)
    assert arr.flags['F_CONTIGUOUS']

    arr = _mmu_core_tests.check_shape_order(farr_bad, 'x', 0)
    assert arr.flags['F_CONTIGUOUS']

    arr = _mmu_core_tests.check_shape_order(farr_bad, 'x', 1)
    assert arr.flags['C_CONTIGUOUS']

    arr = _mmu_core_tests.check_shape_order(farr_good, 'x', 1)
    assert arr.flags['C_CONTIGUOUS']

    # check non-contiguous views
    inp = np.zeros((10, 100), order='C', dtype=np.float64)[:, :90]
    arr = _mmu_core_tests.check_shape_order(inp, 'x', 1)
    assert arr.flags['C_CONTIGUOUS']
    assert arr.shape == (10, 90), arr.shape
    assert np.isclose(arr[2, 10], inp[2, 10])

    inp = np.zeros((10, 100), order='C', dtype=np.float64)[:, :90]
    arr = _mmu_core_tests.check_shape_order(inp, 'x', 0)
    assert arr.flags['F_CONTIGUOUS']
    assert arr.shape == (10, 90), arr.shape
    assert np.isclose(arr[2, 10], inp[2, 10])

    inp = np.zeros((100, 10), order='F', dtype=np.float64)[:90, :]
    arr = _mmu_core_tests.check_shape_order(inp, 'x', 0)
    assert arr.flags['F_CONTIGUOUS']
    assert arr.shape == (90, 10), arr.shape
    assert np.isclose(arr[10, 2], inp[10, 2])

    inp = np.zeros((100, 10), order='F', dtype=np.float64)[:90, :]
    arr = _mmu_core_tests.check_shape_order(inp, 'x', 1)
    assert arr.flags['C_CONTIGUOUS']
    assert arr.shape == (90, 10), arr.shape
    assert np.isclose(arr[10, 2], inp[10, 2])

    with pytest.raises(RuntimeError):
        _mmu_core_tests.check_shape_order(np.zeros((10, 4, 2)), 'arr', 0)

    with pytest.raises(RuntimeError):
        _mmu_core_tests.check_shape_order(np.zeros((10, 2)), 'arr', 2)

def test_assert_shape_order():
    """Test assrt_shape_order.

    assert_shape_order checks if the array is contiguous along the axis of
    length ``expected`` if not it will throw a RuntimeError
    """
    carr_good = np.zeros((10, 4), order='C', dtype=np.float64)
    carr_bad = np.zeros((4, 10), order='C', dtype=np.float64)

    _mmu_core_tests.assert_shape_order(carr_good, 'x', 4)
    with pytest.raises(RuntimeError):
        _mmu_core_tests.assert_shape_order(carr_bad, 'x', 4)

    farr_good = np.zeros((4, 10), order='F', dtype=np.float64)
    farr_bad = np.zeros((10, 4), order='F', dtype=np.float64)

    _mmu_core_tests.assert_shape_order(farr_good, 'x', 4)
    with pytest.raises(RuntimeError):
        _mmu_core_tests.assert_shape_order(farr_bad, 'x', 4)
    with pytest.raises(RuntimeError):
        _mmu_core_tests.assert_shape_order(farr_good[:, [0, 3, 5, 4]], 'x', 4)

    arr_bad = np.zeros((10, 5), dtype=np.float64)
    with pytest.raises(RuntimeError):
        _mmu_core_tests.assert_shape_order(arr_bad, 'x', 4)

    arr_bad = np.zeros((10, 4, 2), dtype=np.float64)
    with pytest.raises(RuntimeError):
        _mmu_core_tests.assert_shape_order(arr_bad, 'x', 4)

    arr_good = np.zeros((4,), dtype=np.float64)
    assert _mmu_core_tests.assert_shape_order(arr_good, 'x', 4) is None
