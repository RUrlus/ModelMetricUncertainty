import numpy as np


def generate_test_labels(N, dtype=np.int64):
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
    proba = np.random.beta(5, 3, size=N)
    yhat = np.rint(proba).astype(dtype)
    y = np.random.binomial(1, np.mean(proba), N).astype(dtype)
    return (proba, yhat, y)
