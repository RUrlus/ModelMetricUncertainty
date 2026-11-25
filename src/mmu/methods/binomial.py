"""Marginal binomial intervals, independent of the joint estimator hierarchy."""

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import beta, norm

from mmu.commons.levels import _parse_levels
from mmu.metrics.metrics import col_index, col_names

CountsLike = npt.ArrayLike
Side = Literal["two-sided", "lower", "upper"]
IntervalMethod = Literal["clopper-pearson", "wilson", "jeffreys"]
_BINOMIAL_INDICES = np.array([0, 1, 2, 3, 6, 7, 8], dtype=np.intp)

__all__ = ["binary_metrics_confidence_intervals", "clopper_pearson", "jeffreys", "wilson"]


def _counts(values):
    """Validate counts before floating-point arithmetic."""
    values = np.asarray(values)
    if values.dtype.kind not in "iuf":
        msg = "Counts must be real numbers"
        raise TypeError(msg)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        msg = "Intervals require finite nonnegative integer counts"
        raise ValueError(msg)
    return values.astype(np.float64)


def _interval_inputs(x, n, levels, side):
    """Broadcast counts and append a separate axis for array-valued levels."""
    if side not in ("two-sided", "lower", "upper"):
        msg = f"Invalid side: {side!r}"
        raise ValueError(msg)
    x, n = np.broadcast_arrays(_counts(x), _counts(n))
    if np.any(x > n):
        msg = "Success counts x must not exceed trial counts n"
        raise ValueError(msg)
    _, probabilities = _parse_levels(levels)
    if probabilities.ndim:
        x, n = x[..., None], n[..., None]
    alpha = (1.0 - probabilities) / (2.0 if side == "two-sided" else 1.0)
    return x, n, alpha


def _interval_result(lower, upper, x, n, side):
    """Enforce boundaries, undefined counts, and scalar return values."""
    lower = np.where(n == 0, np.nan, np.where(x == 0, 0.0, np.clip(lower, 0.0, 1.0)))
    upper = np.where(n == 0, np.nan, np.where(x == n, 1.0, np.clip(upper, 0.0, 1.0)))
    lower = lower.item() if lower.ndim == 0 else lower
    upper = upper.item() if upper.ndim == 0 else upper
    if side == "two-sided":
        return lower, upper
    return lower if side == "lower" else upper


def _conf_mat_counts_last(cm, axis):
    """Return a view of the confusion matrices with the counts on the last axis."""
    if axis is not None:
        if isinstance(axis, bool | np.bool_) or not isinstance(axis, int | np.integer):
            msg = "`axis` must be an int or None"
            raise TypeError(msg)
        if not -cm.ndim <= axis < cm.ndim or cm.shape[axis] != 4:
            msg = f"conf_mat must have 4 counts along axis {axis}"
            raise ValueError(msg)
        return np.moveaxis(cm, axis, -1)
    if cm.ndim >= 2 and cm.shape[-2:] == (2, 2):
        return cm.reshape(*cm.shape[:-2], 4)
    if cm.ndim == 3 and cm.shape[1] == 4:
        if cm.shape[2] == 4:
            msg = (
                "conf_mat with shape (n, 4, 4) is ambiguous: set `axis=1` for the "
                "(n_thresholds, 4, n_runs) layout or `axis=-1` for (..., 4)"
            )
            raise ValueError(msg)
        return np.moveaxis(cm, 1, -1)
    if cm.ndim == 0 or cm.shape[-1] != 4:
        msg = "conf_mat must have shape (4,), (2, 2), (..., 2, 2), (..., 4) or (n_thresholds, 4, n_runs)"
        raise ValueError(msg)
    return cm


def clopper_pearson(x: CountsLike, n: CountsLike, levels=0.95, side: Side = "two-sided"):
    """Compute exact Clopper-Pearson binomial confidence bounds.

    Parameters
    ----------
    x, n : int, float, array-like
        Nonnegative integer successes and trials, with x <= n. Broadcast
        together; integer-valued floats are accepted.
    levels : int, float, array-like, default=0.95
        Integers denote normal standard deviations, floats probability mass
        strictly between zero and one. A one-dimensional array appends a levels
        axis to the broadcast count shape, retaining input order and duplicates.
        For example, counts of shape (3,) and two levels give bounds of shape
        (3, 2). A scalar level does not add an axis. Integers and floats must
        not be mixed.
    side : {"two-sided", "lower", "upper"}, default="two-sided"
        Equal-tailed interval or a single one-sided confidence bound.

    Returns:
    -------
    tuple, float, np.ndarray
        Two-sided calls return (lower, upper); one-sided calls return only the
        requested bound. x=0 has lower=0, x=n has upper=1; n=0 yields NaN.

    Notes:
    -----
    Coverage is at least nominal under the binomial sampling model. These are
    marginal intervals, not a simultaneous confidence region for several metrics.

    """
    x, n, alpha = _interval_inputs(x, n, levels, side)
    lower = beta.ppf(alpha, x, n - x + 1.0)
    upper = beta.isf(alpha, x + 1.0, n - x)
    return _interval_result(lower, upper, x, n, side)


def wilson(x: CountsLike, n: CountsLike, levels=0.95, side: Side = "two-sided", continuity: bool = False):
    """Compute Wilson score bounds, optionally with continuity correction.

    Parameters
    ----------
    x, n : int, float, array-like
        Nonnegative integer successes and trials, with x <= n. Broadcast
        together; integer-valued floats are accepted.
    levels : int, float, array-like, default=0.95
        Integers denote normal standard deviations, floats probability mass,
        see :func:`clopper_pearson`.
    side : {"two-sided", "lower", "upper"}, default="two-sided"
        Equal-tailed interval or a single one-sided confidence bound.
    continuity : bool, default=False
        Invert the continuity-corrected score test.

    Returns:
    -------
    tuple, float, np.ndarray
        Two-sided calls return (lower, upper); one-sided calls return only the
        requested bound. Shapes and boundary handling follow
        :func:`clopper_pearson`.

    Notes:
    -----
    Wilson intervals are approximate: coverage is not guaranteed to be at
    least nominal.

    """
    if not isinstance(continuity, bool | np.bool_):
        msg = "`continuity` must be a bool"
        raise TypeError(msg)
    x, n, alpha = _interval_inputs(x, n, levels, side)
    # z is negative for one-sided levels below 0.5, which selects the root on
    # the other side of the point estimate, matching the inverted score test.
    z = norm.isf(alpha)
    z2 = z * z
    safe_n = np.where(n == 0, 1.0, n)
    correction = 0.5 if continuity else 0.0
    lower_p = np.maximum(x - correction, 0.0) / safe_n
    upper_p = np.minimum(x + correction, n) / safe_n
    z2_over_n = z2 / safe_n
    denominator = 1.0 + z2_over_n
    lower = (
        lower_p + 0.5 * z2_over_n - z * (np.sqrt(lower_p * (1.0 - lower_p) + 0.25 * z2_over_n) / np.sqrt(safe_n))
    ) / denominator
    upper = (
        upper_p + 0.5 * z2_over_n + z * (np.sqrt(upper_p * (1.0 - upper_p) + 0.25 * z2_over_n) / np.sqrt(safe_n))
    ) / denominator
    return _interval_result(lower, upper, x, n, side)


def jeffreys(x: CountsLike, n: CountsLike, levels=0.95, side: Side = "two-sided"):
    """Compute boundary-modified Jeffreys binomial credible bounds.

    Parameters
    ----------
    x, n : int, float, array-like
        Nonnegative integer successes and trials, with x <= n. Broadcast
        together; integer-valued floats are accepted.
    levels : int, float, array-like, default=0.95
        Integers denote normal standard deviations, floats probability mass,
        see :func:`clopper_pearson`.
    side : {"two-sided", "lower", "upper"}, default="two-sided"
        Equal-tailed interval or a single one-sided bound.

    Returns:
    -------
    tuple, float, np.ndarray
        Two-sided calls return (lower, upper); one-sided calls return only the
        requested bound. Shapes follow :func:`clopper_pearson`, including
        lower=0 at x=0, upper=1 at x=n, and NaN at n=0.

    Notes:
    -----
    Uses the Beta(x + 1/2, n - x + 1/2) posterior. The boundary modification
    differs from a strictly equal-tailed posterior interval. Frequentist
    coverage is not guaranteed.

    """
    x, n, alpha = _interval_inputs(x, n, levels, side)
    lower = beta.ppf(alpha, x + 0.5, n - x + 0.5)
    upper = beta.isf(alpha, x + 0.5, n - x + 0.5)
    return _interval_result(lower, upper, x, n, side)


def binary_metrics_confidence_intervals(
    conf_mat: CountsLike,
    *,
    levels=0.95,
    side: Side = "two-sided",
    method: IntervalMethod = "clopper-pearson",
    continuity: bool = False,
    metrics=None,
    axis: int | None = None,
    return_df: bool = False,
):
    """Compute marginal binomial intervals from TN, FP, FN, TP counts.

    Parameters
    ----------
    conf_mat : array-like of integer or floating dtype
        Nonnegative integer counts in TN, FP, FN, TP order with shape (4,),
        (2, 2), (..., 2, 2) or (..., 4). (2, 2) matrices are interpreted in
        row-major order. The (n_thresholds, 4, n_runs) output of
        :func:`mmu.confusion_matrices_runs_thresholds` is also supported.
    levels : int, float, array-like, default=0.95
        Normal sigmas for integers, probability mass for floats, as in
        :func:`clopper_pearson`. Array levels retain order and duplicates.
    side : {"two-sided", "lower", "upper"}, default="two-sided"
        Return both bounds, or just the requested one-sided bound.
    method : {"clopper-pearson", "wilson", "jeffreys"}, default="clopper-pearson"
        Interval construction. Jeffreys is a boundary-modified credible interval.
    continuity : bool, default=False
        Apply continuity correction for Wilson only; ignored with other methods.
    metrics : sequence of str, optional
        Metric names/aliases from ``mmu.metrics.col_index``, in requested order.
        A bare string raises TypeError. None returns all ten standard slots,
        with NaN for F1 and MCC; explicitly requesting F1 or MCC raises ValueError.
    axis : int, default=None
        Axis of `conf_mat` holding the four counts. By default this is
        inferred: a three dimensional array with 4 elements along axis 1 and
        not along the last axis is treated as
        (n_thresholds, 4, n_runs), otherwise the last axis is used. When both
        axis 1 and the last axis have length 4 the layout is ambiguous and
        `axis` must be set explicitly.
    return_df : bool, default=False
        Return a DataFrame with canonical (metric, bound) MultiIndex columns.
        Batch axes are flattened in row-major order. With array levels, a
        (sample, level) row MultiIndex preserves their input order.

    Raises:
    ------
    ValueError
        if `conf_mat` has an unsupported or ambiguous layout

    Returns:
    -------
    np.ndarray, pd.DataFrame
        A scalar level returns (..., n_metrics, 2), lower/upper last, where
        ``...`` are the batch axes of `conf_mat` with the count axis removed,
        e.g. (n_thresholds, n_runs) for the output of
        :func:`mmu.confusion_matrices_runs_thresholds`. Array levels return
        (..., n_levels, n_metrics, 2). One-sided calls omit the final bounds
        axis. Zero-denominator metrics have NaN bounds.

    Notes:
    -----
    Recall, specificity, FPR and FNR condition on observed class counts.
    PPV (precision) and NPV condition on predicted class counts, which are
    themselves random. Their intervals describe conditional uncertainty only;
    they do not capture all joint precision-recall uncertainty and can understate
    it if interpreted unconditionally. Use
    :class:`mmu.methods.PrecisionRecallUncertainty` (PRU) for joint uncertainty.
    Marginal intervals do not provide simultaneous coverage.

    """
    estimators = {"clopper-pearson": clopper_pearson, "wilson": wilson, "jeffreys": jeffreys}
    if method not in estimators:
        msg = f"Invalid method: {method!r}"
        raise ValueError(msg)
    if not isinstance(continuity, bool | np.bool_):
        msg = "`continuity` must be a bool"
        raise TypeError(msg)
    cm = _conf_mat_counts_last(_counts(conf_mat), axis)
    if metrics is None:
        selected = np.arange(len(col_names), dtype=np.intp)
    else:
        if isinstance(metrics, str):
            msg = "metrics must be a sequence of names, not a string"
            raise TypeError(msg)
        try:
            selected = np.asarray([col_index[name] for name in metrics], dtype=np.intp)
        except KeyError as exc:
            msg = f"Unknown metric: {exc.args[0]!r}"
            raise ValueError(msg) from exc
        if not np.isin(selected, _BINOMIAL_INDICES).all():
            msg = "Binomial intervals do not apply to F1 or MCC"
            raise ValueError(msg)

    tn, fp, fn, tp = np.moveaxis(cm, -1, 0)
    with np.errstate(over="ignore"):
        tn_fp = tn + fp
        tp_fn = tp + fn
        total = tn_fp + tp_fn

        x = np.stack(
            (
                tn,  # NPV
                tp,  # Precision (PPV)
                tn,  # Specificity (TNR)
                tp,  # Recall (TPR)
                fp,  # FPR
                fn,  # FNR
                tn + tp,  # Accuracy
            ),
            axis=-1,
        )
        n = np.stack(
            (
                tn + fn,  # NPV
                tp + fp,  # Precision (PPV)
                tn_fp,  # Specificity (TNR)
                tp_fn,  # Recall (TPR)
                tn_fp,  # FPR
                tp_fn,  # FNR
                total,  # Accuracy
            ),
            axis=-1,
        )
    if not np.isfinite(n).all():
        msg = "Confusion-matrix totals overflowed"
        raise ValueError(msg)
    kwargs = {"continuity": continuity} if method == "wilson" else {}
    bounds = estimators[method](x, n, levels=levels, side=side, **kwargs)
    level_values, _ = _parse_levels(levels)
    if side == "two-sided":
        lower, upper = bounds
        values = np.stack((lower, upper), axis=-1)
        if level_values.ndim:
            values = np.swapaxes(values, -3, -2)
        result = np.full((*values.shape[:-2], len(col_names), 2), np.nan)
        result[..., _BINOMIAL_INDICES, :] = values
        result = result[..., selected, :]
    else:
        values = np.swapaxes(bounds, -2, -1) if level_values.ndim else bounds
        result = np.full((*values.shape[:-1], len(col_names)), np.nan)
        result[..., _BINOMIAL_INDICES] = values
        result = result[..., selected]
    if not return_df:
        return result

    names = [col_names[index] for index in selected]
    bound_names = ["lower", "upper"] if side == "two-sided" else [side]
    columns = pd.MultiIndex.from_product([names, bound_names], names=["metric", "bound"])
    n_samples = int(np.prod(cm.shape[:-1], dtype=np.intp))
    if level_values.ndim:
        index = pd.MultiIndex.from_product([range(n_samples), level_values], names=["sample", "level"])
    else:
        index = pd.RangeIndex(n_samples, name="sample")
    return pd.DataFrame(result.reshape(len(index), len(columns)), index=index, columns=columns)
