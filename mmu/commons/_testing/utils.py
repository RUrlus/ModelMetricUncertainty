import numpy as np
import sklearn.metrics as skm

DEFAULT_OVERLOAD_DTYPES = [
    bool,
    np.bool_,
    int,
    np.int32,
    np.int64,
    float,
    np.float32,
    np.float64,
]


def create_unaligned_array(dtype):
    if dtype != np.float64 and dtype != np.int64:
        raise ValueError("Only dtypes np.int64 and np.float64 are supported.")
    # Allocate 802 bytes of memory (allocated on boundary)
    a = np.arange(802, dtype=np.uint8)

    # Create an array with boundary offset 4
    z = np.frombuffer(a.data, offset=2, count=100, dtype=dtype)
    z.shape = 10, 10
    z[:] = 0
    return z


def greater_equal_tol(a, b, return_dtype=np.int64):
    """Check if a >= b, taking into account floating point noise.

    Parameters
    ----------
    a : np.ndarray[floating]
        array containing floating points to be checked against b
    b : np.ndarray[floating]
        array containing floating points to be checked against a

    Returns
    -------
    np.ndarray[`return_dtype`]
        the greater equal values with dtype `return_dtype`

    """
    return (np.isclose(a, b) | (a > b)).astype(return_dtype)


def generate_test_labels(
    N,
    y_dtype=np.int64,
    yhat_dtype=np.int64,
    proba_dtype=np.float64,
):
    """Generate test data.

    Parameters
    ----------
    N : int64
        number of data points to generate
    dtype : np.dtype
        dtype for yhat and y

    Returns
    -------
    Tuple[np.array[np.float64], np.array[dtype], np.array[dtype]]
        proba, yhat, y

    """
    rng = np.random.default_rng()

    # class balance of 0.425 with the below beta parameters results in
    # approximately balanced data
    pos_size = np.floor(N * 0.425).astype(np.int64)

    scores = np.empty(N)
    scores[:pos_size] = rng.beta(12, 1.5, size=pos_size)
    scores[pos_size:] = rng.beta(5.5, 20, size=N - pos_size)

    # shuffle scores
    rng.shuffle(scores)

    y = rng.binomial(1, scores, N).astype(y_dtype)  # type: ignore
    yhat = np.rint(scores).astype(yhat_dtype)
    return (scores, yhat, y)


def compute_reference_metrics(y, yhat=None, proba=None, threshold=None, fill=1.0):
    """Compute the set of metrics based on sklearn's implementation.

    Parameters
    ----------
    y : np.ndarray
        the labels
    yhat : np.ndarray, default=None
        the predicted values for yhat, can be None if proba is not None
    proba : np.ndarray, default=None
        the estimated probabilities, can be None if yhat is not None
    threshold : float, default=None
        the classification threshold, can be None if proba is None

    Returns
    -------
    metrics : np.ndarray
        array containing the reference metrics
        * 0 - neg.precision aka Negative Predictive Value
        * 1 - pos.precision aka Positive Predictive Value
        * 2 - neg.recall aka True Negative Rate & Specificity
        * 3 - pos.recall aka True Positive Rate aka Sensitivity
        * 4 - neg.f1 score
        * 5 - pos.f1 score
        * 6 - False Positive Rate
        * 7 - False Negative Rate
        * 8 - Accuracy
        * 9 - MCC
    """
    if yhat is None:
        if proba is None:
            raise ValueError("``proba`` must not be None when yhat is None")
        if threshold is None:
            raise ValueError("``threshold`` must note be None when Proba is not None")
        yhat = greater_equal_tol(proba, threshold)

    metrics = np.zeros(10, dtype=np.float64)
    prec, rec, f1, _ = skm.precision_recall_fscore_support(y, yhat, zero_division=fill)
    metrics[:2] = prec
    metrics[2:4] = rec
    metrics[4:6] = f1
    metrics[6] = 1.0 - metrics[2]
    metrics[7] = 1.0 - metrics[3]
    metrics[8] = skm.accuracy_score(y, yhat)
    metrics[9] = skm.matthews_corrcoef(y, yhat)
    return skm.confusion_matrix(y, yhat), metrics
