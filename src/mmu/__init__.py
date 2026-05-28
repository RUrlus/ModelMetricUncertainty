import importlib.metadata
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

from mmu import methods, metrics
from mmu.commons.utils import generate_data
from mmu.lib import _MMU_MT_SUPPORT
from mmu.methods import (
    PPNRCU,
    PPNRU,
    PRCU,
    PRU,
    ROCCU,
    ROCU,
    PPNRecallCurveUncertainty,
    PPNRecallUncertainty,
    PrecisionRecallCurveUncertainty,
    PrecisionRecallUncertainty,
    ROCCurveUncertainty,
    ROCUncertainty,
)
from mmu.metrics import (
    ROC_curve,
    auto_thresholds,
    binary_metrics,
    binary_metrics_confusion_matrices,
    binary_metrics_confusion_matrix,
    binary_metrics_runs,
    binary_metrics_runs_thresholds,
    binary_metrics_thresholds,
    confusion_matrices,
    confusion_matrices_runs_thresholds,
    confusion_matrices_thresholds,
    confusion_matrices_to_dataframe,
    confusion_matrix,
    confusion_matrix_to_dataframe,
    metrics_to_dataframe,
    precision_recall,
    precision_recall_curve,
)

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

__version__ = importlib.metadata.version("mmu")

__all__ = [
    "PPNRCU",
    "PPNRU",
    "PRCU",
    "PRU",
    "ROCCU",
    "ROCU",
    "_MMU_MT_SUPPORT",
    "PPNRecallCurveUncertainty",
    "PPNRecallUncertainty",
    "PrecisionRecallCurveUncertainty",
    "PrecisionRecallUncertainty",
    "ROCCurveUncertainty",
    "ROCUncertainty",
    "ROC_curve",
    "__version__",
    "auto_thresholds",
    "binary_metrics",
    "binary_metrics_confusion_matrices",
    "binary_metrics_confusion_matrix",
    "binary_metrics_runs",
    "binary_metrics_runs_thresholds",
    "binary_metrics_thresholds",
    "confusion_matrices",
    "confusion_matrices_runs_thresholds",
    "confusion_matrices_thresholds",
    "confusion_matrices_to_dataframe",
    "confusion_matrix",
    "confusion_matrix_to_dataframe",
    "generate_data",
    "methods",
    "metrics",
    "metrics_to_dataframe",
    "pr_curve",
    "precision_recall",
    "precision_recall_curve",
]
