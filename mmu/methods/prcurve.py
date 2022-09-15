"""Module containing the API for the precision-recall with Multinomial uncertainty."""
from mmu.methods.curvebase import BaseCurveUncertainty
from mmu.lib import _MMU_MT_SUPPORT

from mmu.lib._mmu_core import (
    pr_multn_grid_curve_error,
    pr_bvn_grid_curve_error
)

if _MMU_MT_SUPPORT:
    from mmu.lib._mmu_core import (
        pr_multn_grid_curve_error_mt,
        pr_bvn_grid_curve_error_mt,
    )

import mmu.lib._mmu_core as _core


class PrecisionRecallCurveUncertainty(BaseCurveUncertainty):
    __doc__ = BaseCurveUncertainty.__doc__

    def __init__(self):
        BaseCurveUncertainty.__init__(self)

        self.bvn_grid_curve_error_func = pr_bvn_grid_curve_error
        self.multn_grid_curve_error_func = pr_multn_grid_curve_error
        self.metric_2d_func = _core.precision_recall_2d

        if _MMU_MT_SUPPORT:
            self.bvn_grid_curve_error_mt_func = pr_bvn_grid_curve_error_mt
            self.multn_grid_curve_error_mt_func = pr_multn_grid_curve_error_mt

        self.y_label = 'Precision'
        self.x_label = 'Recall'

    @property
    def precision(self):
        """Alias of the y coordinate of the curve

        :type: np.ndarray[float64]
        """
        return self.y

    @property
    def recall(self):
        """Alias of the x coordinate of the curve

        :type: np.ndarray[float64]
        """
        return self.x

    @property
    def prec_grid(self):
        """Alias of the y values that where evaluated.

        :type: np.ndarray[float64]
        """
        return self.y_grid

    @property
    def rec_grid(self):
        """Alias of the x values that where evaluated.

        :type: np.ndarray[float64]
        """
        return self.x_grid
