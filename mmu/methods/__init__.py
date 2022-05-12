from mmu.methods.pr_mvn import precision_recall_uncertainty
from mmu.methods.pr_mvn import precision_recall_uncertainty_runs
from mmu.methods.pr_mvn import precision_recall_uncertainty_confusion_matrix
from mmu.methods.pr_mvn import precision_recall_uncertainty_confusion_matrices
from mmu.methods.pr_mvn import PrecisionRecallCurveUncertainty

pr_error = precision_recall_uncertainty
pr_error_runs = precision_recall_uncertainty_runs
pr_error_conf_mat = precision_recall_uncertainty_confusion_matrix
pr_error_conf_mats = precision_recall_uncertainty_confusion_matrices

__all__ = [
    'pr_error',
    'pr_error_runs',
    'pr_error_conf_mat',
    'pr_error_conf_mats',
    'precision_recall_uncertainty',
    'precision_recall_uncertainty_runs',
    'precision_recall_uncertainty_confusion_matrix',
    'precision_recall_uncertainty_confusion_matrices',
    'PrecisionRecallCurveUncertainty',
]
