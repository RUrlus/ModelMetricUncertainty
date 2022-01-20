import numpy as np
import pytest
from mmu_tests.utils import create_unaligned_array

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


def test_is_well_behaved():
    carr = np.ones((10), order='C', dtype=np.float64)
    farr = np.ones((10), order='F', dtype=np.float64)
    assert _mmu_core_tests.is_well_behaved(carr)
    assert _mmu_core_tests.is_well_behaved(farr)

    carr = np.ones((10, 4), order='C', dtype=np.float64)
    farr = np.ones((10, 4), order='F', dtype=np.float64)
    assert _mmu_core_tests.is_well_behaved(carr)
    assert _mmu_core_tests.is_well_behaved(farr)

    assert _mmu_core_tests.is_well_behaved(carr[:, [1, 3]])
    assert _mmu_core_tests.is_well_behaved(farr[:, [1, 3]])
    assert _mmu_core_tests.is_well_behaved(carr[[1, 3], :])
    assert _mmu_core_tests.is_well_behaved(farr[[1, 3], :])

    carr = np.arange(12, dtype=np.float64).reshape((3, 4))
    farr = carr.copy(order='F')
    assert not _mmu_core_tests.is_f_contiguous(carr[:, 1:3])
    assert not _mmu_core_tests.is_c_contiguous(carr[:, 1:3])
    assert not _mmu_core_tests.is_well_behaved(carr[:, 1:3])
    assert not _mmu_core_tests.is_well_behaved(farr[1:3, ])

    # the array being generated is c_contiguous but not aligned
    carr = create_unaligned_array(np.float64)
    assert not _mmu_core_tests.is_well_behaved(carr)

def test_get_data():
    assert _mmu_core_tests.test_get_data(np.zeros((10, 10)))
    assert _mmu_core_tests.test_get_data(np.zeros((10, 10)).view())
    samples = np.asarray(list(range(100)), dtype=np.float64)
    assert _mmu_core_tests.test_get_data(samples)


def test_zero_array():
    dtypes = [np.float64, np.int64]
    for dtype in dtypes:
        og_arr = np.empty((100, 100), dtype=dtype)
        arr = og_arr.copy()
        _mmu_core_tests.zero_array(arr)
        if dtype == np.int64:
            assert arr.sum() == 0
        else:
            assert np.isclose(arr.sum(), 0.0)

        og_arr = np.asarray(
            np.random.uniform(0, 100, size=100),
            dtype=dtype
        )
        arr = og_arr.copy()
        _mmu_core_tests.zero_array(arr)
        if dtype == np.int64:
            assert arr.sum() == 0
        else:
            assert np.isclose(arr.sum(), 0.0)


def test_zero_array_fixed():
    dtypes = [np.float64, np.int64]
    for dtype in dtypes:
        og_arr = np.empty((4, 3), dtype=dtype)
        arr = og_arr.copy()
        _mmu_core_tests.zero_array_fixed(arr)
        if dtype == np.int64:
            assert arr.sum() == 0
        else:
            assert np.isclose(arr.sum(), 0.0)

        og_arr = np.asarray(
            np.random.uniform(0, 100, size=12),
            dtype=dtype
        )
        arr = og_arr.copy()
        _mmu_core_tests.zero_array_fixed(arr)
        if dtype == np.int64:
            assert arr.sum() == 0
        else:
            assert np.isclose(arr.sum(), 0.0)

        # check that only the first 12 elements of the array
        # are zero'd as we specify this in the template
        og_arr = np.ones(13, dtype=dtype) * 10
        arr = og_arr.copy()
        _mmu_core_tests.zero_array_fixed(arr)
        if dtype == np.int64:
            assert not arr.sum() == 0
            assert arr.sum() == 10
        else:
            assert not np.isclose(arr.sum(), 0.0)
            assert np.isclose(arr.sum(), 10.0)


def test_allocate_confusion_matrix():
    cm = _mmu_core_tests.allocate_confusion_matrix()
    assert cm.dtype == np.int64
    assert cm.shape == (2, 2)
    assert cm.flags['C_CONTIGUOUS']
    assert cm.sum() == 0

    cm = _mmu_core_tests.allocate_confusion_matrix(True)
    assert cm.dtype == np.float64
    assert cm.shape == (2, 2)
    assert cm.flags['C_CONTIGUOUS']
    assert np.isclose(cm.sum(), 0.0)


def test_allocate_n_confusion_matrices():
    cm = _mmu_core_tests.allocate_n_confusion_matrices(100)
    assert cm.dtype == np.int64
    assert cm.shape == (100, 4)
    assert cm.flags['C_CONTIGUOUS']
    assert cm.sum() == 0

    cm = _mmu_core_tests.allocate_n_confusion_matrices(1)
    assert cm.shape == (1, 4)
    cm = _mmu_core_tests.allocate_n_confusion_matrices(0)
    assert cm.shape == (0, 4)


    cm = _mmu_core_tests.allocate_n_confusion_matrices(100, True)
    assert cm.dtype == np.float64
    assert cm.shape == (100, 4)
    assert cm.flags['C_CONTIGUOUS']
    assert np.isclose(cm.sum(), 0.0)
