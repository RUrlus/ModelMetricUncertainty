"""Module containing the API for the precision-recall uncertainty modelled through profile log likelihoods."""
from mmu.methods.pointbase import BaseUncertainty, BaseSimulatedUncertainty

import mmu.lib._mmu_core as _core
from mmu.lib import _MMU_MT_SUPPORT


class PrecisionRecallUncertainty(BaseUncertainty):
    __doc__ = BaseUncertainty.__doc__

    def __init__(self):
        BaseUncertainty.__init__(self)

        self.metric_func = _core.precision_recall
        self.multn_error_func = _core.pr_multn_error
        self.bvn_cov_func = _core.pr_bvn_cov

        self.bvn_chi2_score_func = _core.pr_bvn_chi2_score
        self.multn_chi2_score_func = _core.pr_multn_chi2_score

        self.bvn_chi2_scores_func = _core.pr_bvn_chi2_scores
        self.multn_chi2_scores_func = _core.pr_multn_chi2_scores

        if _MMU_MT_SUPPORT:
            self.multn_error_mt_func = _core.pr_multn_error_mt
            self.bvn_chi2_scores_mt_func = _core.pr_bvn_chi2_scores_mt
            self.multn_chi2_scores_mt_func = _core.pr_multn_chi2_scores_mt

        self.y_label = 'Precision'
        self.x_label = 'Recall'

    @property
    def precision(self):
        """Alias of the y coordinate

        :type: float
        """
        return self.y

    @property
    def recall(self):
        """Alias of the x coordinate

        :type: float
        """
        return self.x


class PrecisionRecallSimulatedUncertainty(BaseSimulatedUncertainty):
    __doc__ = BaseSimulatedUncertainty.__doc__

    def __init__(self):
        BaseSimulatedUncertainty.__init__(self)

        self.metric_func = _core.precision_recall
        if _MMU_MT_SUPPORT:
            self.multn_sim_error_mt_func = _core.pr_multn_sim_error_mt

    @property
    def precision(self):
        return self.y

    @property
    def recall(self):
        return self.x
