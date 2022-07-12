import mmu.metrics as metrics
import mmu.methods as methods

from mmu.lib import _MMU_MT_SUPPORT

from mmu.metrics import auto_thresholds
from mmu.metrics import binary_metrics
from mmu.metrics import binary_metrics_runs
from mmu.metrics import binary_metrics_thresholds
from mmu.metrics import binary_metrics_runs_thresholds
from mmu.metrics import binary_metrics_confusion_matrix
from mmu.metrics import binary_metrics_confusion_matrices
from mmu.metrics import confusion_matrix
from mmu.metrics import confusion_matrices
from mmu.metrics import confusion_matrices_thresholds
from mmu.metrics import confusion_matrices_runs_thresholds
from mmu.metrics import confusion_matrix_to_dataframe
from mmu.metrics import confusion_matrices_to_dataframe
from mmu.metrics import metrics_to_dataframe
from mmu.metrics import precision_recall
from mmu.metrics import precision_recall_curve

# aliases
bmetrics = binary_metrics
bmetrics_runs = binary_metrics_runs
bmetrics_thresh = binary_metrics_thresholds
bmetrics_runs_thresh = binary_metrics_runs_thresholds
bmetrics_conf_mat = binary_metrics_confusion_matrix
bmetrics_conf_mats = binary_metrics_confusion_matrices

conf_mat = confusion_matrix
conf_mats = confusion_matrices
conf_mats_thresh = confusion_matrices_thresholds
conf_mats_runs_thresh = confusion_matrices_runs_thresholds

conf_mat_to_df = confusion_matrix_to_dataframe
conf_mats_to_df = confusion_matrices_to_dataframe

pr_curve = precision_recall_curve

from mmu.methods import PRU
from mmu.methods import PRCU
from mmu.methods import PrecisionRecallUncertainty
from mmu.methods import PrecisionRecallSimulatedUncertainty
from mmu.methods import PrecisionRecallCurveUncertainty

from mmu.commons.utils import generate_data

from mmu.version import full_version as __version__

__all__ = [
    "_MMU_MT_SUPPORT",
    # funcs
    "auto_thresholds",
    "binary_metrics",
    "binary_metrics_runs",
    "binary_metrics_thresholds",
    "binary_metrics_runs_thresholds",
    "binary_metrics_confusion_matrix",
    "binary_metrics_confusion_matrices",
    "confusion_matrix",
    "confusion_matrices",
    "confusion_matrices_thresholds",
    "confusion_matrices_runs_thresholds",
    "confusion_matrix_to_dataframe",
    "confusion_matrices_to_dataframe",
    "generate_data",
    "metrics_to_dataframe",
    "precision_recall",
    "pr_curve",
    "precision_recall_curve",
    # classes
    "PRU",
    "PRCU",
    "PrecisionRecallUncertainty",
    "PrecisionRecallSimulatedUncertainty",
    "PrecisionRecallCurveUncertainty",
    # modules
    "metrics",
    "methods",
]
