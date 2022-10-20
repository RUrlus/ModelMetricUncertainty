"""Module containing the API for the ROC with Multinomial uncertainty."""
from mmu.methods.curvebase import BaseCurveUncertainty
from mmu.lib import _MMU_MT_SUPPORT

from mmu.lib._mmu_core import (
    roc_multn_grid_curve_error,
    roc_bvn_grid_curve_error
)

if _MMU_MT_SUPPORT:
    from mmu.lib._mmu_core import (
        roc_multn_grid_curve_error_mt,
        roc_bvn_grid_curve_error_mt,
    )

import mmu.lib._mmu_core as _core


class ROCCurveUncertainty(BaseCurveUncertainty):
    __doc__ = BaseCurveUncertainty.__doc__

    def __init__(self):
        BaseCurveUncertainty.__init__(self)

        self.bvn_grid_curve_error_func = roc_bvn_grid_curve_error
        self.multn_grid_curve_error_func = roc_multn_grid_curve_error
        self.metric_2d_func = _core.ROC_2d

        if _MMU_MT_SUPPORT:
            self.bvn_grid_curve_error_mt_func = roc_bvn_grid_curve_error_mt
            self.multn_grid_curve_error_mt_func = roc_multn_grid_curve_error_mt

        self.y_label = 'True Positive Rate (Recall)'
        self.x_label = 'False Positive Rate'

    @property
    def TPR(self):
        """Alias of the y coordinate of the curve

        :type: np.ndarray[float64]
        """
        return self.y

    @property
    def FPR(self):
        """Alias of the x coordinate of the curve

        :type: np.ndarray[float64]
        """
        return self.x

    @property
    def TPR_grid(self):
        """Alias of the y values that where evaluated.

        :type: np.ndarray[float64]
        """
        return self.y_grid

    @property
    def FPR_grid(self):
        """Alias of the x values that where evaluated.

        :type: np.ndarray[float64]
        """
        return self.x_grid
