import mmu.lib._mmu_core as _core
try:
    from mmu.lib._mmu_core import multinomial_uncertainty_over_grid_mt
    MMU_MT_SUPPORT = True
except ImportError:
    MMU_MT_SUPPORT = False

__all__ = ["_core"]
