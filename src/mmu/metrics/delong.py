"""DeLong's test and confidence intervals for (correlated) ROC AUCs.

Implements DeLong, DeLong & Clarke-Pearson (1988) using the O(N log N)
algorithm of Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm for
Comparing the Areas Under Correlated Receiver Operating Characteristic Curves",
IEEE Signal Processing Letters 21(11).

The C++ extension computes the AUCs and structural components; contrasts,
one-sided p-values and confidence intervals are computed here as they cost
O(c * N) after ranking.
"""

import math
import warnings
from typing import NamedTuple

import numpy as np
import scipy.stats as sts

import mmu.lib._mmu_core as _core
from mmu.commons import _convert_to_ext_types, _convert_to_float, check_array
from mmu.commons.checks import _check_n_threads
from mmu.lib import _MMU_MT_SUPPORT

_ALTERNATIVES = ("two-sided", "greater", "less")
_CI_METHODS = ("wald", "logit")

# Contrast estimates (on the AUC scale) in a zero-variance direction smaller
# than this are treated as zero.
_DEGENERATE_ATOL = 1e-12


class DeLongTestResult(NamedTuple):
    """Result of :func:`delong_test`."""

    auc1: float
    auc2: float
    difference: float
    variance: float
    z: float
    p_value: float


class DeLongContrastResult(NamedTuple):
    """Result of :func:`delong_contrast_test`.

    ``statistic`` is the signed z-score when the contrast has a single row
    (``df == 1``) and the chi-squared statistic otherwise.
    """

    aucs: np.ndarray
    estimate: np.ndarray
    covariance: np.ndarray
    statistic: float
    df: int
    p_value: float


class AUCConfidenceInterval(NamedTuple):
    """Result of :func:`auc_confidence_interval`."""

    auc: float | np.ndarray
    lower: float | np.ndarray
    upper: float | np.ndarray
    std_err: float | np.ndarray


def _delong_n_threads(n_threads):
    n_threads = _check_n_threads(n_threads)
    if n_threads > 1 and not _MMU_MT_SUPPORT:
        warnings.warn("mmu was not compiled with multi-threading enabled, ignoring `n_threads`", stacklevel=3)
        return 1
    return n_threads


def _check_labels(y):
    y = check_array(y, max_dim=1, dtype_check=_convert_to_ext_types)
    if not np.all((y == 0) | (y == 1)):
        msg = "`y` should contain only binary labels {0, 1}."
        raise ValueError(msg)
    return np.asarray(y, dtype=np.int64, order="C")


def _check_scores_2d(scores, obs_axis, n_obs):
    if isinstance(obs_axis, bool) or not isinstance(obs_axis, (int, np.integer)) or (obs_axis not in {0, 1}):
        msg = "`obs_axis` must be either 0 or 1."
        raise TypeError(msg)
    scores = check_array(
        scores, axis=int(obs_axis), target_axis=1, target_order=0, min_dim=2, max_dim=2, dtype_check=_convert_to_float
    )
    if scores.shape[1] != n_obs:
        msg = "`scores` and `y` are incompatible: expected shape (n_models, n_obs)."
        raise ValueError(msg)
    return np.asarray(scores, dtype=np.float64, order="C")


def _check_alternative(alternative):
    if alternative not in _ALTERNATIVES:
        msg = f"`alternative` must be one of {_ALTERNATIVES}."
        raise ValueError(msg)


def _p_value_from_z(z, alternative):
    if alternative == "two-sided":
        return math.erfc(abs(z) / math.sqrt(2.0))
    if alternative == "greater":
        return 0.5 * math.erfc(z / math.sqrt(2.0))
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _warn_degenerate(stacklevel=3):
    warnings.warn(
        "DeLong variance estimate is zero while the AUCs differ; the test is undefined, returning NaN.",
        RuntimeWarning,
        stacklevel=stacklevel,
    )


def _centred(components):
    """Centre rows; shifting by the first column first keeps constant rows exactly zero."""
    out = components - components[:, :1]
    out -= out.mean(axis=1, keepdims=True)
    return out


def _component_covariance(w10, w01):
    """DeLong covariance S10 / m + S01 / n of (contrasted) structural components."""
    m = w10.shape[1]
    n = w01.shape[1]
    c10 = _centred(w10)
    c01 = _centred(w01)
    cov = (c10 @ c10.T) / (m * (m - 1.0)) + (c01 @ c01.T) / (n * (n - 1.0))
    return 0.5 * (cov + cov.T)


def delong_covariance(y, scores, obs_axis=0, n_threads=None):
    """Compute DeLong AUC estimates and covariance matrix.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels with shape (n_obs,)
    scores : np.ndarray[float32, float64]
        model scores with shape (n_obs, n_models) when ``obs_axis=0``
        or (n_models, n_obs) when ``obs_axis=1``
    obs_axis : int, default=0
        axis that contains observations in ``scores``
    n_threads : int, default=None
        number of threads, models are processed in parallel. Defaults to 4
        if mmu was compiled with OpenMP and 1 otherwise; -1 uses all cores
        but one. Results do not depend on the number of threads.

    Returns:
    -------
    aucs : np.ndarray[float64]
        DeLong AUC estimates per model
    covariance : np.ndarray[float64]
        DeLong covariance matrix between model AUC estimates
    """
    y = _check_labels(y)
    scores = _check_scores_2d(scores, obs_axis, y.size)
    return _core.delong_covariance(y, scores, n_threads=_delong_n_threads(n_threads))


def delong_test(y, scores1, scores2, alternative="two-sided", n_threads=None):
    """Compute paired DeLong test for two correlated ROC AUCs.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels with shape (n_obs,)
    scores1 : np.ndarray[float32, float64]
        model 1 scores with shape (n_obs,)
    scores2 : np.ndarray[float32, float64]
        model 2 scores with shape (n_obs,)
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        alternative hypothesis: AUC1 != AUC2, AUC1 > AUC2 or AUC1 < AUC2
    n_threads : int, default=None
        number of threads, effectively 1 or 2 as the two models are processed
        in parallel. Defaults to 4 (i.e. 2) if mmu was compiled with OpenMP
        and 1 otherwise; -1 uses all cores but one.

    Returns:
    -------
    DeLongTestResult
        named tuple with fields: auc1, auc2, difference, variance, z, p_value

    Notes:
    -----
    The variance of the difference is computed from the differenced
    structural components which avoids cancellation for strongly correlated
    models. If the estimated variance is zero while the AUCs differ, the test
    is undefined: ``z`` and ``p_value`` are NaN and a ``RuntimeWarning`` is
    issued.
    """
    _check_alternative(alternative)
    y = _check_labels(y)
    scores1 = check_array(scores1, max_dim=1, dtype_check=_convert_to_float)
    scores2 = check_array(scores2, max_dim=1, dtype_check=_convert_to_float)

    if scores1.size != y.size or scores2.size != y.size:
        msg = "`scores1`, `scores2` and `y` must have equal length."
        raise ValueError(msg)

    result = _core.delong_test(y, scores1, scores2, n_threads=_delong_n_threads(n_threads))
    z = result["z"]
    if math.isnan(z):
        _warn_degenerate()
        p_value = math.nan
    else:
        p_value = _p_value_from_z(z, alternative)

    return DeLongTestResult(
        auc1=result["auc1"],
        auc2=result["auc2"],
        difference=result["difference"],
        variance=result["variance"],
        z=z,
        p_value=p_value,
    )


def delong_contrast_test(y, scores, contrast=None, *, alternative="two-sided", obs_axis=0, n_threads=None):
    r"""Test a linear contrast of correlated ROC AUCs.

    Tests :math:`H_0: L\theta = 0` where :math:`\theta` are the AUCs and
    :math:`L` the contrast matrix with shape (n_contrasts, n_models). With a
    single contrast the statistic is
    :math:`z = L\theta / \sqrt{L S L^T}`; with multiple contrasts it is
    :math:`\chi^2 = (L\theta)^T (L S L^T)^{+} (L\theta)` with degrees of
    freedom equal to the rank of :math:`L S L^T`.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels with shape (n_obs,)
    scores : np.ndarray[float32, float64]
        model scores with shape (n_obs, n_models) when ``obs_axis=0``
        or (n_models, n_obs) when ``obs_axis=1``, requires at least two models
    contrast : np.ndarray, default=None
        contrast matrix with shape (n_models,) or (n_contrasts, n_models).
        Defaults to testing that all AUCs are equal, i.e. rows
        :math:`\theta_1 - \theta_r` for :math:`r = 2, \ldots, k`. For two
        models this is equivalent to :func:`delong_test`.
    alternative : {"two-sided", "greater", "less"}, default="two-sided"
        alternative hypothesis for a single contrast: :math:`L\theta \neq 0`,
        :math:`L\theta > 0` or :math:`L\theta < 0`. Multiple contrasts only
        support "two-sided".
    obs_axis : int, default=0
        axis that contains observations in ``scores``
    n_threads : int, default=None
        number of threads used to compute the structural components, see
        :func:`delong_covariance`

    Returns:
    -------
    DeLongContrastResult
        named tuple with fields: aucs, estimate (:math:`L\theta`),
        covariance (:math:`L S L^T`), statistic (z if df == 1 else chi2),
        df, p_value

    Notes:
    -----
    The covariance is computed from the contrasted structural components
    which avoids cancellation for strongly correlated models. If the contrast
    is non-zero in a direction with zero estimated variance, the test is
    undefined: the statistic and p-value are NaN and a ``RuntimeWarning`` is
    issued.
    """
    _check_alternative(alternative)
    y = _check_labels(y)
    scores = _check_scores_2d(scores, obs_axis, y.size)
    n_models = scores.shape[0]

    if contrast is None:
        if n_models < 2:
            msg = "The default contrast requires at least two models."
            raise ValueError(msg)
        contrast = np.hstack([np.ones((n_models - 1, 1)), -np.eye(n_models - 1)])
    else:
        contrast = np.atleast_2d(np.asarray(contrast, dtype=np.float64))
        if contrast.ndim != 2 or contrast.shape[1] != n_models or contrast.shape[0] < 1:
            msg = f"`contrast` must have shape (n_models,) or (n_contrasts, n_models) with n_models={n_models}."
            raise ValueError(msg)
        if not np.all(np.isfinite(contrast)):
            msg = "`contrast` contains non-finite values."
            raise ValueError(msg)

    n_contrasts = contrast.shape[0]
    if n_contrasts > 1 and alternative != "two-sided":
        msg = "One-sided alternatives are only supported for a single contrast."
        raise ValueError(msg)

    aucs, v10, v01 = _core.delong_components(y, scores, n_threads=_delong_n_threads(n_threads))
    estimate = contrast @ aucs
    covariance = _component_covariance(contrast @ v10, contrast @ v01)

    if n_contrasts == 1:
        diff = float(estimate[0])
        var = float(covariance[0, 0])
        if var > 0.0:
            z = diff / math.sqrt(var)
            p_value = _p_value_from_z(z, alternative)
        elif diff != 0.0:
            _warn_degenerate()
            z = p_value = math.nan
        else:
            z = 0.0
            p_value = _p_value_from_z(z, alternative)
        return DeLongContrastResult(aucs, estimate, covariance, z, 1, p_value)

    eigvals, eigvecs = np.linalg.eigh(covariance)
    tol = max(float(eigvals.max()), 0.0) * n_contrasts * np.finfo(np.float64).eps
    keep = eigvals > tol
    df = int(keep.sum())
    proj = eigvecs.T @ estimate

    if np.any(np.abs(proj[~keep]) > _DEGENERATE_ATOL):
        _warn_degenerate()
        return DeLongContrastResult(aucs, estimate, covariance, math.nan, df, math.nan)
    if df == 0:
        return DeLongContrastResult(aucs, estimate, covariance, 0.0, 0, 1.0)

    chi2 = float(np.sum(proj[keep] ** 2 / eigvals[keep]))
    p_value = float(sts.chi2.sf(chi2, df))
    return DeLongContrastResult(aucs, estimate, covariance, chi2, df, p_value)


def auc_confidence_interval(y, scores, alpha=0.05, *, method="wald", obs_axis=0, n_threads=None):
    r"""Compute DeLong confidence intervals for (marginal) ROC AUCs.

    If the variance estimate of a model is zero, e.g. for perfect separation,
    its interval collapses to the AUC and a ``RuntimeWarning`` is issued.

    Parameters
    ----------
    y : np.ndarray[bool, int32, int64, float32, float64]
        true labels with shape (n_obs,)
    scores : np.ndarray[float32, float64]
        model scores with shape (n_obs,) for a single model, or
        (n_obs, n_models) when ``obs_axis=0`` / (n_models, n_obs) when
        ``obs_axis=1``
    alpha : float, default=0.05
        significance level, the interval has coverage ``1 - alpha``
    method : {"wald", "logit"}, default="wald"
        ``"wald"``: :math:`\hat\theta \pm z_{1-\alpha/2}\,\widehat{se}`,
        clipped to [0, 1].
        ``"logit"``: Wald interval on the logit scale with delta-method
        standard error :math:`\widehat{se} / (\hat\theta(1 - \hat\theta))`,
        transformed back; always within (0, 1) and better behaved for AUCs
        close to 0 or 1. With small samples and a high AUC the Wald interval
        under-covers noticeably (e.g. ~86% nominal 95% at AUC 0.96 with
        30 positives and 45 negatives) whereas the logit interval stays
        close to nominal.
    obs_axis : int, default=0
        axis that contains observations in ``scores`` when it is 2D
    n_threads : int, default=None
        number of threads used to compute the structural components, see
        :func:`delong_covariance`

    Returns:
    -------
    AUCConfidenceInterval
        named tuple with fields: auc, lower, upper, std_err. Floats for 1D
        ``scores``, arrays with shape (n_models,) otherwise.
    """
    if method not in _CI_METHODS:
        msg = f"`method` must be one of {_CI_METHODS}."
        raise ValueError(msg)
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (float, int, np.floating, np.integer))
        or not (0.0 < alpha < 1.0)
    ):
        msg = "`alpha` must be a float in (0, 1)."
        raise ValueError(msg)

    y = _check_labels(y)
    single = np.ndim(scores) == 1
    if single:
        scores = check_array(scores, max_dim=1, dtype_check=_convert_to_float)
        if scores.size != y.size:
            msg = "`scores` and `y` must have equal length."
            raise ValueError(msg)
        scores = np.ascontiguousarray(scores, dtype=np.float64).reshape(1, -1)
    else:
        scores = _check_scores_2d(scores, obs_axis, y.size)

    aucs, v10, v01 = _core.delong_components(y, scores, n_threads=_delong_n_threads(n_threads))
    c10 = _centred(v10)
    c01 = _centred(v01)
    m, n = v10.shape[1], v01.shape[1]
    variance = np.einsum("ij,ij->i", c10, c10) / (m * (m - 1.0)) + np.einsum("ij,ij->i", c01, c01) / (n * (n - 1.0))
    std_err = np.sqrt(variance)

    zq = float(sts.norm.ppf(1.0 - alpha / 2.0))
    degenerate = std_err == 0.0
    if np.any(degenerate):
        warnings.warn(
            "DeLong variance estimate is zero for at least one model; its interval collapses to the AUC.",
            RuntimeWarning,
            stacklevel=2,
        )

    if method == "wald":
        lower = np.clip(aucs - zq * std_err, 0.0, 1.0)
        upper = np.clip(aucs + zq * std_err, 0.0, 1.0)
    else:
        lower = aucs.copy()
        upper = aucs.copy()
        # A zero variance includes AUC in {0, 1}, where the logit is undefined.
        ok = ~degenerate
        theta = aucs[ok]
        logit = np.log(theta / (1.0 - theta))
        se_logit = std_err[ok] / (theta * (1.0 - theta))
        lower[ok] = 1.0 / (1.0 + np.exp(-(logit - zq * se_logit)))
        upper[ok] = 1.0 / (1.0 + np.exp(-(logit + zq * se_logit)))

    if single:
        return AUCConfidenceInterval(float(aucs[0]), float(lower[0]), float(upper[0]), float(std_err[0]))
    return AUCConfidenceInterval(aucs, lower, upper, std_err)
