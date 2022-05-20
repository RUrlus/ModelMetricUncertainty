import mmu.metrics as metrics
import mmu.methods as methods
import mmu.stats as stats

from mmu.lib import MMU_MT_SUPPORT as _MMU_MT_SUPPORT

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
from mmu.metrics import pr_curve, precision_recall_curve

from mmu.methods import PREU
from mmu.methods import PRMU
from mmu.methods import PRCMU
from mmu.methods import PRCEU
from mmu.methods import PrecisionRecallEllipticalUncertainty
from mmu.methods import PrecisionRecallMultinomialUncertainty
from mmu.methods import PrecisionRecallCurveMultinomialUncertainty
from mmu.methods import PrecisionRecallCurveEllipticalUncertainty

from mmu.commons.utils import generate_data

__all__ = [
    '_MMU_MT_SUPPORT',
    # funcs
    'auto_thresholds',
    'binary_metrics',
    'binary_metrics_runs',
    'binary_metrics_thresholds',
    'binary_metrics_runs_thresholds',
    'binary_metrics_confusion_matrix',
    'binary_metrics_confusion_matrices',
    'confusion_matrix',
    'confusion_matrices',
    'confusion_matrices_thresholds',
    'confusion_matrices_runs_thresholds',
    'confusion_matrix_to_dataframe',
    'confusion_matrices_to_dataframe',
    'generate_data',
    'metrics_to_dataframe',
    'precision_recall',
    'pr_curve',
    'precision_recall_curve',
    # classes
    'PREU',
    'PRMU',
    'PRCMU',
    'PRCEU',
    'PrecisionRecallEllipticalUncertainty',
    'PrecisionRecallMultinomialUncertainty',
    'PrecisionRecallCurveMultinomialUncertainty',
    'PrecisionRecallCurveEllipticalUncertainty',
    # modules
    'metrics',
    'methods',
    'stats',
]
