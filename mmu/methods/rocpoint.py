"""Module containing the API for the precision-recall uncertainty modelled through profile log likelihoods."""
from mmu.methods.pointbase import BaseUncertainty, BaseSimulatedUncertainty

import mmu.lib._mmu_core as _core
from mmu.lib import _MMU_MT_SUPPORT


class ROCUncertainty(BaseUncertainty):
    __doc__ = BaseUncertainty.__doc__

    def __init__(self):
        BaseUncertainty.__init__(self)

        self.metric_func = _core.ROC
        self.multn_error_func = _core.roc_multn_error
        self.bvn_cov_func = _core.roc_bvn_cov

        self.bvn_chi2_score_func = _core.roc_bvn_chi2_score
        self.multn_chi2_score_func = _core.roc_multn_chi2_score

        self.bvn_chi2_scores_func = _core.roc_bvn_chi2_scores
        self.multn_chi2_scores_func = _core.roc_multn_chi2_scores

        if _MMU_MT_SUPPORT:
            self.multn_error_mt_func = _core.roc_multn_error_mt
            self.bvn_chi2_scores_mt_func = _core.roc_bvn_chi2_scores_mt
            self.multn_chi2_scores_mt_func = _core.roc_multn_chi2_scores_mt

        self.y_label = 'True Positive Rate (Recall)'
        self.x_label = 'False Positive Rate'

    @property
    def TPR(self):
        """Alias of the y coordinate

        :type: np.ndarray[float64]
        """
        return self.y

    @property
    def FPR(self):
        """Alias of the x coordinate

        :type: np.ndarray[float64]
        """
        return self.x


class ROCSimulatedUncertainty(BaseSimulatedUncertainty):
    __doc__ = BaseSimulatedUncertainty.__doc__

    def __init__(self):
        BaseSimulatedUncertainty.__init__(self)

        self.metric_func = _core.ROC
        if _MMU_MT_SUPPORT:
            self.multn_sim_error_mt_func = _core.roc_multn_sim_error_mt

    @property
    def TPR(self):
        """Alias of the y coordinate

        :type: np.ndarray[float64]
        """
        return self.y

    @property
    def FPR(self):
        """Alias of the x coordinate

        :type: np.ndarray[float64]
        """
        return self.x
