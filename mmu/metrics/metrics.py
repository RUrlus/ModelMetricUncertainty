import numpy as np
import pandas as pd
from sklearn.utils import check_array

import mmu.lib._mmu_core as _core
from mmu.commons import _check_shape_order

col_index = {
    'neg.precision': 0,
    'npv': 0,
    'pos.precision': 1,
    'ppv': 1,
    'neg.recall': 2,
    'tnr': 2,
    'specificity': 2,
    'pos.recall': 3,
    'tpr': 3,
    'sensitivity': 3,
    'neg.f1': 4,
    'neg.f1_score': 4,
    'pos.f1': 5,
    'pos.f1_score': 5,
    'fpr': 6,
    'fnr': 7,
    'accuracy': 8,
    'acc': 8,
    'mcc': 9,
}

col_names = [
    'neg.precision',
    'pos.precision',
    'neg.recall',
    'pos.recall',
    'neg.f1',
    'pos.f1',
    'fpr',
    'fnr',
    'acc',
    'mcc',
]


def metrics_to_dataframe(metrics):
    """Return DataFrame with metrics.

    Parameters
    ----------
    metrics : np.ndarray
        metrics where the rows are the metrics for various runs or
        classification thresholds and the columns are the metrics.

    Returns
    -------
    pd.DataFrame
        the metrics as a DataFrame

    """
    if metrics.ndim == 1:
        return pd.DataFrame(metrics[None, :], columns=col_names)
    return pd.DataFrame(metrics, columns=col_names)


def confusion_matrix_to_dataframe(conf_mat):
    """Create dataframe with confusion matrix.

    Parameters
    ----------
    conf_mat : np.ndarray
        array containing a single confusion matrix

    Returns
    -------
    pd.DataFrame
        the confusion matrix

    """
    index = (('observed', 'negative'), ('observed', 'positive'))
    cols = (('estimated', 'negative'), ('estimated', 'positive'))
    return pd.DataFrame(conf_mat, index=index, columns=cols)
