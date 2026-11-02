"""Regression tests for the strided `all_finite` fallback (C-6).

`all_finite_strided` handles arrays that are not well-behaved (non-contiguous
and/or unaligned). The previous implementation hand-unrolled the ndim 1/2/3
cases and had four distinct defects:

* the index vector was sized to a hard-coded 3 while ``ndim`` entries were
  sorted, running off the end for ndim > 3 (undefined behaviour), and no branch
  handled ndim > 3 so such arrays were always reported as non-finite;
* the ndim-3 back stride was ``strides[2] - n1 * (n0 * s0)`` instead of
  ``strides[2] - n1 * strides[1]``, which differ whenever the middle axis is
  itself non-contiguous -- so the walk visited the wrong addresses;
* strides were held in ``int``/``size_t``, narrowing 64-bit strides and
  wrapping on the subtraction for negative strides;
* values were read through a ``T*`` cast, which is undefined behaviour for the
  unaligned arrays this path exists to serve.
"""

import numpy as np
import pytest

from mmu.lib import _mmu_core


def _assert_not_well_behaved(arr):
    """These tests are meaningless unless the strided path is actually taken."""
    assert not (arr.flags.aligned and (arr.flags.c_contiguous or arr.flags.f_contiguous))


def test_4d_non_contiguous_is_finite():
    """ndim > 3 previously fell through every branch and returned False."""
    base = np.ones((2, 4, 3, 5))
    view = base[:, ::2, :, :]
    _assert_not_well_behaved(view)

    assert _mmu_core.all_finite(view)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_4d_non_contiguous_detects_non_finite(bad):
    """A non-finite value in a 4-D strided array must be found."""
    base = np.ones((2, 4, 3, 5))
    view = base[:, ::2, :, :]
    view[1, 1, 2, 3] = bad

    assert not _mmu_core.all_finite(view)


def test_5d_non_contiguous_is_finite():
    """Arbitrary dimensionality must work, not just up to 3."""
    base = np.ones((2, 4, 2, 3, 2))
    view = base[:, ::2, :, :, :]
    _assert_not_well_behaved(view)

    assert _mmu_core.all_finite(view)


def test_3d_strided_middle_axis_detects_nan_everywhere():
    """The ndim-3 back-stride bug: middle axis non-contiguous.

    ``strides[1] != n0 * s0`` here, so the old back-stride formula walked the
    wrong addresses and could both miss real NaNs and read outside the view.
    """
    shape = (4, 3, 5)
    for idx in np.ndindex(*shape):
        base = np.ones((4, 6, 5))
        view = base[:, ::2, :]
        assert view.shape == shape
        view[idx] = np.nan

        assert not _mmu_core.all_finite(view), f"missed NaN at {idx}"


def test_3d_strided_middle_axis_is_finite():
    base = np.ones((4, 6, 5))
    view = base[:, ::2, :]
    _assert_not_well_behaved(view)

    assert _mmu_core.all_finite(view)


@pytest.mark.parametrize("shape", [(50,), (10, 5), (4, 3, 5)])
def test_reversed_views(shape):
    """Negative strides wrapped the unsigned back-stride arithmetic."""
    base = np.ones(shape)
    view = base[::-1]
    _assert_not_well_behaved(view)

    assert _mmu_core.all_finite(view)

    view[0] = np.nan
    assert not _mmu_core.all_finite(view)


def test_fully_reversed_2d():
    base = np.ones((6, 7))
    view = base[::-1, ::-1]

    assert _mmu_core.all_finite(view)
    view[3, 4] = np.inf
    assert not _mmu_core.all_finite(view)


def test_transposed_3d_detects_nan_everywhere():
    """A transposed 3-D array is neither C- nor F-contiguous."""
    shape = (3, 4, 5)
    for idx in np.ndindex(*shape):
        base = np.ones((5, 4, 3))
        view = base.transpose(2, 1, 0)
        assert view.shape == shape
        view[idx] = np.nan

        assert not _mmu_core.all_finite(view), f"missed NaN at {idx}"


def test_unaligned_array():
    """Unaligned data was read through a T* cast -- undefined behaviour."""
    from mmu.commons._testing import create_unaligned_array

    view = create_unaligned_array(np.float64)
    assert not view.flags.aligned

    assert _mmu_core.all_finite(view)

    view[5, 5] = np.nan
    assert not _mmu_core.all_finite(view)


def test_unaligned_and_transposed():
    """Unaligned *and* non-contiguous at once."""
    from mmu.commons._testing import create_unaligned_array

    view = create_unaligned_array(np.float64).T
    assert not view.flags.aligned

    assert _mmu_core.all_finite(view)

    view[7, 2] = np.inf
    assert not _mmu_core.all_finite(view)


def test_empty_strided_array():
    base = np.ones((0, 4))
    view = base.T

    assert _mmu_core.all_finite(view)

