import mmu.lib._mmu_core as _core

from mmu.lib._mmu_core import _has_openmp_support
_MMU_MT_SUPPORT = _has_openmp_support

__all__ = ["_core"]
