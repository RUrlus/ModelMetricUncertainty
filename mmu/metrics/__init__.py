from mmu.metrics.confmat import confusion_matrix
from mmu.metrics.confmat import confusion_matrices
from mmu.metrics.confmat import confusion_matrices_thresholds
from mmu.metrics.confmat import confusion_matrices_runs_thresholds
from mmu.metrics.confmat import confusion_matrix_to_dataframe
from mmu.metrics.confmat import confusion_matrices_to_dataframe
from mmu.metrics.metrics import col_index
from mmu.metrics.metrics import col_names
from mmu.metrics.metrics import metrics_to_dataframe
from mmu.metrics.metrics import binary_metrics
from mmu.metrics.metrics import binary_metrics_runs
from mmu.metrics.metrics import binary_metrics_thresholds
from mmu.metrics.metrics import binary_metrics_runs_thresholds
from mmu.metrics.metrics import binary_metrics_confusion_matrix
from mmu.metrics.metrics import binary_metrics_confusion_matrices
from mmu.metrics.metrics import precision_recall_curve
from mmu.metrics.utils import auto_thresholds

pr_curve = precision_recall_curve
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

__all__ = [
    'auto_thresholds',
    'bmetrics',
    'bmetrics_runs',
    'bmetrics_thresh',
    'bmetrics_runs_thresh',
    'bmetrics_conf_mat',
    'bmetrics_conf_mats',
    'binary_metrics',
    'binary_metrics_runs',
    'binary_metrics_thresholds',
    'binary_metrics_runs_thresholds',
    'binary_metrics_confusion_matrix',
    'binary_metrics_confusion_matrices',
    'col_index',
    'col_names',
    'conf_mat',
    'conf_mats',
    'conf_mats_thresh',
    'conf_mats_runs_thresh',
    'conf_mat_to_df',
    'conf_mats_to_df',
    'confusion_matrix',
    'confusion_matrices',
    'confusion_matrices_thresholds',
    'confusion_matrices_runs_thresholds',
    'confusion_matrix_to_dataframe',
    'confusion_matrices_to_dataframe',
    'metrics_to_dataframe',
    'pr_curve',
    'precision_recall_curve',
]
