import numpy as np
import pytest

from mmu_tests import _mmu_core_tests


def test_check_contiguous():
    carr = np.ones((10), order='C', dtype=np.float64)
    farr = np.ones((10), order='F', dtype=np.float64)
    assert _mmu_core_tests.check_contiguous(carr) is None
    assert _mmu_core_tests.check_contiguous(farr) is None

    carr = np.ones((10, 4), order='C', dtype=np.float64)
    farr = np.ones((10, 4), order='F', dtype=np.float64)
    assert _mmu_core_tests.check_contiguous(carr) is None
    assert _mmu_core_tests.check_contiguous(farr) is None

    assert _mmu_core_tests.check_contiguous(carr[:, [1, 3]]) is None
    assert _mmu_core_tests.check_contiguous(farr[:, [1, 3]]) is None
    assert _mmu_core_tests.check_contiguous(carr[[1, 3], :]) is None
    assert _mmu_core_tests.check_contiguous(farr[[1, 3], :]) is None


def test_1d_soft():
    x = np.zeros((100, ))
    assert _mmu_core_tests.check_1d_soft(x) == 0
    x = np.zeros((100, 1))
    assert _mmu_core_tests.check_1d_soft(x) == 0
    assert _mmu_core_tests.check_1d_soft(x.T) == 1
    x = np.zeros((100, 2))
    with pytest.raises(RuntimeError):
        _mmu_core_tests.check_1d_soft(x)
        _mmu_core_tests.check_1d_soft(x.T)
        _mmu_core_tests.check_1d_soft(np.zeros((100, 4, 1)))


def test_check_equal_length():
    x = np.zeros((100, 4))
    y = np.zeros((100, 4))
    assert _mmu_core_tests.check_equal_length(x, y) is None
    assert _mmu_core_tests.check_equal_length(x.T, y.T) is None
    assert _mmu_core_tests.check_equal_length(x[:, [0, 1]], y) is None
    assert _mmu_core_tests.check_equal_length(x, y[:, [0, 1]]) is None
    with pytest.raises(RuntimeError):
        _mmu_core_tests.check_equal_length(x[:10], y)
        _mmu_core_tests.check_equal_length(x, y[:10])
        _mmu_core_tests.check_equal_length(x.T, y)
        _mmu_core_tests.check_equal_length(x, y.T)


def test_check_shape_length():
    x = np.zeros((100, 2))
    y = np.zeros((100, 2))
    assert _mmu_core_tests.check_equal_shape(x, y) is None
    assert _mmu_core_tests.check_equal_shape(x.T, y.T) is None
    with pytest.raises(RuntimeError):
        _mmu_core_tests.check_equal_shape(x[:10], y)
        _mmu_core_tests.check_equal_shape(x, y[:10])
        _mmu_core_tests.check_equal_shape(x.T, y)
        _mmu_core_tests.check_equal_shape(x, y.T)
        _mmu_core_tests.check_equal_shape(x[:, [0, 1]], y)
        _mmu_core_tests.check_equal_shape(x, y[:, [0, 1]])

    x = np.zeros((100, 2))
    y = np.zeros((100, 4, 2))
    with pytest.raises(RuntimeError):
        _mmu_core_tests.check_equal_shape(x, y)
        _mmu_core_tests.check_equal_shape(y, x)
