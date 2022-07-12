import numpy as np
from mmu.commons import check_array
from mmu.commons import _convert_to_float


def auto_thresholds(scores, max_steps=None, epsilon=None, seed=None):
    """Determine the thresholds s.t. each threshold results in a different
    confusion matrix.

    The thresholds can be subsampled by setting `max_steps`. The points are
    sampled where regions with small difference between score are oversampled.

    Parameters
    ----------
    scores : np.ndarray[float32/float64]
        the classifier scores
    max_steps : int, double, default=None
        limit the number of steps in the threshold.
        Default is None which does not limit the number of thresholds
    epsilon : float, default=None
        the minimum difference between two scores for them to be considered
        different. If None the machine precision is used for the dtype scores
    seed : int, default=None
        seed to use when subsampeling, ignored when `max_steps` is None or
        when the number of unique thresholds is smaller than `max_steps`.

    Returns
    -------
    thresholds : np.ndarray[float64]
        the thresholds that result in different confusion matrices. Length
        of which is at most the number of elements in scores or `max_steps`

    """
    n_elem = scores.size
    scores = scores.ravel()
    scores = check_array(scores, max_dim=1, dtype_check=_convert_to_float)

    if epsilon is None:
        epsilon = np.finfo(scores.dtype).eps
    scores = np.sort(scores, kind="mergesort")
    # start and end of thresholds should start with extreme values of scores
    min_thresh = scores[0]
    max_thresh = scores[-1]

    # compute the first difference
    fdiff = np.diff(scores)
    # select the index where the difference between scores
    # is greater than epsilon (non-zero by default)
    distinct_idx = np.where(fdiff > epsilon)[0]
    threshold_idxs = np.r_[distinct_idx, n_elem - 1]

    if max_steps is not None and distinct_idx.size > max_steps:
        rng = np.random.default_rng(seed)
        weights = np.log(1.0 / fdiff[distinct_idx])
        weights /= weights.sum()
        threshold_idxs = np.sort(
            rng.choice(distinct_idx, replace=False, size=max_steps, p=weights)
        )

    if threshold_idxs[0] != 0 and threshold_idxs[-1] != n_elem - 1:
        thresholds = np.empty(threshold_idxs.size + 2, dtype=np.float64)
        thresholds[1:-1] = scores[threshold_idxs]
        thresholds[0] = min_thresh
        thresholds[-1] = max_thresh
    elif threshold_idxs[0] != 0:
        thresholds = np.empty(threshold_idxs.size + 1, dtype=np.float64)
        thresholds[1:] = scores[threshold_idxs]
        thresholds[0] = min_thresh
    elif threshold_idxs[-1] != n_elem - 1:
        thresholds = np.empty(threshold_idxs.size + 1, dtype=np.float64)
        thresholds[:-1] = scores[threshold_idxs]
        thresholds[-1] = max_thresh
    else:
        thresholds = scores[threshold_idxs]
    return thresholds
