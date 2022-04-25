import numpy as np
import pandas as pd
import arviz as az


def compute_hdi(X, prob=0.95, **kwargs):
    """Compute Highest Density Interval over marginal distributions.

    Parameters
    ----------
    X : pd.DataFrame, array-like
        at most two dimensional array where the rows contain the observations
        and the column different distributions.
    prob : float, default=0.95
        the cumulative probability that should be contained in the HDI
    kwargs
        keyword arguments passed to az.hdi

    Returns
    -------
    hdi : pd.DataFrame, optional
        the HDI interval for each column, only returned when ``X`` is a
        DataFrame
    hdi : np.ndarray, optional
        the HID interval for each column in ``X``, only returned if ``X`` is a
        np.ndarray. The first column is the lower bound, the second the upper
        bound

    """
    to_frame = False
    X_cols = []
    if isinstance(X, pd.DataFrame):
        X_cols = X.columns
        X = X.values
        to_frame = True
    elif isinstance(X, np.ndarray):
        if X.ndim > 2:
            raise TypeError('``X`` must be at most two dimensional.')
    elif isinstance(X, (tuple, list, set)):
        X = np.asarray(X)
    else:
        raise TypeError('``X`` must be a DataFrame or array-like')

    if X.ndim > 1:
        n_vars = X.shape[1]
        hdi = np.zeros((n_vars, 2))
        # loop over X as az.hdi has furture warning where the columns will be
        # interpreted as chains rather than distributions
        for i in range(n_vars):
            hdi[i, :] = az.hdi(ary=X[:, i], hdi_prob=prob, **kwargs)
    else:
        hdi = az.hdi(ary=X, hdi_prob=prob, **kwargs)

    if to_frame:
        return pd.DataFrame(hdi, index=X_cols, columns=['lb', 'ub'])
    return hdi
