import numpy as np

from mmu.commons._testing import create_unaligned_array
from mmu.lib import _mmu_core
from mmu.lib import _mmu_core_tests

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


def test_all_finite():
    for dtype_ in [int, bool, np.int32, np.int64]:
        # clean
        arr = np.ones(100, order='C', dtype=dtype_)
        assert _mmu_core.all_finite(arr)

    for dtype_ in [float, np.float32, np.float64]:
        # clean
        arr = np.ones(100, order='C', dtype=dtype_)
        assert _mmu_core.all_finite(arr)
        # with NaNs
        arr[np.random.randint(0, 99, size=3)] = np.nan
        assert not _mmu_core.all_finite(arr)

        # with infty
        arr = np.ones(100, order='C', dtype=dtype_)
        arr[np.random.randint(0, 99, size=3)] = np.infty
        assert not _mmu_core.all_finite(arr)

        # with both infty
        arr = np.ones(100, order='C', dtype=dtype_)
        arr[np.random.randint(0, 99, size=3)] = np.infty
        arr[np.random.randint(0, 99, size=3)] = np.nan
        assert not _mmu_core.all_finite(arr)


def test_all_finite_non_contiguous():
    for dtype_ in [int, np.int32, np.int64]:
        # clean
        arr = np.arange(12, dtype=dtype_).reshape((3, 4))[:, 1:3]
        assert _mmu_core.all_finite(arr)

    for dtype_ in [float, np.float32, np.float64]:
        # clean
        arr = np.arange(100, dtype=dtype_).reshape((10, 10))[:, 1:4]
        assert _mmu_core.all_finite(arr)
        # with NaNs
        arr = np.arange(100, dtype=dtype_).reshape((10, 10))[:, 1:4]
        col_idx = np.random.randint(1, 3, size=1)
        row_idxs = np.random.randint(0, 10, size=2)
        arr[row_idxs, col_idx] = np.nan
        assert not _mmu_core.all_finite(arr)

        # with infty
        arr = np.arange(100, dtype=dtype_).reshape((10, 10))[:, 1:4]
        col_idx = np.random.randint(1, 3, size=1)
        row_idxs = np.random.randint(0, 10, size=2)
        arr[row_idxs, col_idx] = np.infty
        assert not _mmu_core.all_finite(arr)

        # with both infty
        arr = np.arange(100, dtype=dtype_).reshape((10, 10))[:, 1:4]
        col_idx = np.random.randint(1, 3, size=1)
        row_idxs = np.random.randint(0, 10, size=2)
        arr[row_idxs, col_idx] = np.infty
        col_idx = np.random.randint(1, 3, size=1)
        row_idxs = np.random.randint(0, 10, size=2)
        arr[row_idxs, col_idx] = np.nan
        assert not _mmu_core.all_finite(arr)


def test_all_finite_non_contiguous_not_aligned():
    arr = create_unaligned_array(np.int64).reshape((10, 10))[:, 1:4]
    assert _mmu_core.all_finite(arr)

    arr = create_unaligned_array(np.float64).reshape((10, 10))[:, 1:4]
    col_idx = np.random.randint(1, 3, size=1)
    row_idxs = np.random.randint(0, 10, size=2)
    arr[row_idxs, col_idx] = np.nan
    assert not _mmu_core.all_finite(arr)


def test_is_well_behaved_finite_int_t():
    for dtype_ in [int, bool, np.int32, np.int64]:
        carr = np.ones((10), order='C', dtype=dtype_)
        farr = np.ones((10), order='F', dtype=dtype_)
        assert _mmu_core.is_well_behaved_finite(carr)
        assert _mmu_core.is_well_behaved_finite(farr)

        carr = np.ones((10, 4), order='C', dtype=dtype_)
        farr = np.ones((10, 4), order='F', dtype=dtype_)
        assert _mmu_core.is_well_behaved_finite(carr)
        assert _mmu_core.is_well_behaved_finite(farr)

        assert _mmu_core.is_well_behaved_finite(carr[:, [1, 3]])
        assert _mmu_core.is_well_behaved_finite(farr[:, [1, 3]])
        assert _mmu_core.is_well_behaved_finite(carr[[1, 3], :])
        assert _mmu_core.is_well_behaved_finite(farr[[1, 3], :])

        if dtype_ != bool:
            carr = np.arange(12, dtype=dtype_).reshape((3, 4))
            farr = carr.copy(order='F')
            assert not _mmu_core.is_well_behaved_finite(carr[:, 1:3])
            assert not _mmu_core.is_well_behaved_finite(farr[1:3, ])

        if dtype_ == np.int64:
            # the array being generated is c_contiguous but not aligned
            carr = create_unaligned_array(dtype_)
            assert not _mmu_core.is_well_behaved_finite(carr)


def test_is_well_behaved_finite_float_t():
    for dtype_ in [float, np.float32, np.float64]:
        carr = np.ones((10), order='C', dtype=dtype_)
        farr = np.ones((10), order='F', dtype=dtype_)
        assert _mmu_core.is_well_behaved_finite(carr)
        assert _mmu_core.is_well_behaved_finite(farr)

        carr = np.ones((10, 4), order='C', dtype=dtype_)
        farr = np.ones((10, 4), order='F', dtype=dtype_)
        assert _mmu_core.is_well_behaved_finite(carr)
        assert _mmu_core.is_well_behaved_finite(farr)

        assert _mmu_core.is_well_behaved_finite(carr[:, [1, 3]])
        assert _mmu_core.is_well_behaved_finite(farr[:, [1, 3]])
        assert _mmu_core.is_well_behaved_finite(carr[[1, 3], :])
        assert _mmu_core.is_well_behaved_finite(farr[[1, 3], :])

        carr = np.arange(12, dtype=dtype_).reshape((3, 4))
        farr = carr.copy(order='F')
        assert not _mmu_core.is_well_behaved_finite(carr[:, 1:3])
        assert not _mmu_core.is_well_behaved_finite(farr[1:3, ])

        if dtype_ == np.float64:
            # the array being generated is c_contiguous but not aligned
            carr = create_unaligned_array(dtype_)
            assert not _mmu_core.is_well_behaved_finite(carr)

        # clean
        arr = np.ones(100, order='C', dtype=dtype_)
        assert _mmu_core.is_well_behaved_finite(arr)
        # with NaNs
        arr[np.random.randint(0, 99, size=3)] = np.nan
        assert not _mmu_core.is_well_behaved_finite(arr)

        # with infty
        arr = np.ones(100, order='C', dtype=dtype_)
        arr[np.random.randint(0, 99, size=3)] = np.infty
        assert not _mmu_core.is_well_behaved_finite(arr)

        # with both infty
        arr = np.ones(100, order='C', dtype=dtype_)
        arr[np.random.randint(0, 99, size=3)] = np.infty
        arr[np.random.randint(0, 99, size=3)] = np.nan
        assert not _mmu_core.is_well_behaved_finite(arr)

        arr = np.arange(12, dtype=dtype_).reshape((3, 4))
        arr[0, 0] = np.infty
        arr[1, 2] = np.nan
        assert not _mmu_core.is_well_behaved_finite(arr[:, 1:3])

        if dtype_ == np.float64:
            # the array being generated is c_contiguous but not aligned
            arr = create_unaligned_array(dtype_)
            arr[np.random.randint(0, 9, size=3), np.random.randint(0, 9, size=3)] = np.infty
            arr[np.random.randint(0, 9, size=3), np.random.randint(0, 9, size=3)] = np.nan
            assert not _mmu_core.is_well_behaved_finite(arr)


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
