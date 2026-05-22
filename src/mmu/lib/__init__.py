from mmu.lib import _mmu_core

_core = _mmu_core
_MMU_MT_SUPPORT = _mmu_core._has_openmp_support

__all__ = ["_mmu_core", "_core", "_MMU_MT_SUPPORT"]
