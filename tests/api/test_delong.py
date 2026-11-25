import math

import numpy as np
import pytest
from scipy.stats import chi2 as chi2_dist
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import mmu
from mmu.lib import _core


def _reference_components(y, scores_by_model):
    """O(m * n) reference for AUCs and structural components V10 (positives), V01 (negatives)."""
    positive_mask = y != 0
    negative_mask = ~positive_mask

    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())

    if positive_count < 2 or negative_count < 2:
        msg = "reference DeLong requires at least two observations in each class"
        raise ValueError(msg)

    n_models = scores_by_model.shape[0]

    aucs = np.empty(n_models, dtype=np.float64)
    v10 = np.empty((n_models, positive_count), dtype=np.float64)
    v01 = np.empty((n_models, negative_count), dtype=np.float64)

    for model in range(n_models):
        scores = scores_by_model[model]
        scores_pos = scores[positive_mask]
        scores_neg = scores[negative_mask]

        delta = scores_pos[:, None] - scores_neg[None, :]
        indicators = (delta > 0.0).astype(np.float64) + 0.5 * (delta == 0.0)

        aucs[model] = indicators.mean()
        v10[model] = indicators.mean(axis=1)
        v01[model] = indicators.mean(axis=0)

    return aucs, v10, v01


def _reference_delong(y, scores_by_model):
    aucs, v10, v01 = _reference_components(y, scores_by_model)
    positive_count = v10.shape[1]
    negative_count = v01.shape[1]
    n_models = aucs.size

    covariance = np.empty((n_models, n_models), dtype=np.float64)

    for lhs in range(n_models):
        for rhs in range(n_models):
            cov_pos = np.cov(v10[lhs], v10[rhs], ddof=1)[0, 1]
            cov_neg = np.cov(v01[lhs], v01[rhs], ddof=1)[0, 1]
            covariance[lhs, rhs] = cov_pos / positive_count + cov_neg / negative_count

    return aucs, covariance


def _reference_delong_test(y, scores1, scores2):
    aucs, v10, v01 = _reference_components(y, np.vstack([scores1, scores2]))

    difference = aucs[0] - aucs[1]
    variance = np.var(v10[0] - v10[1], ddof=1) / v10.shape[1] + np.var(v01[0] - v01[1], ddof=1) / v01.shape[1]

    if variance > 0.0:
        z = difference / np.sqrt(variance)
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    elif difference != 0.0:
        z = np.nan
        p_value = np.nan
    else:
        z = 0.0
        p_value = 1.0

    return {
        "auc1": aucs[0],
        "auc2": aucs[1],
        "difference": difference,
        "variance": variance,
        "z": z,
        "p_value": p_value,
    }


def _make_data(seed=1834, n_obs=400):
    rng = np.random.default_rng(seed)

    # Keep drawing until both classes have enough observations.
    while True:
        y = rng.binomial(1, 0.45, n_obs).astype(np.int64)
        if (y == 1).sum() >= 2 and (y == 0).sum() >= 2:
            break

    shared_noise = rng.normal(0.0, 0.8, n_obs)
    scores1 = 1.15 * y + shared_noise + rng.normal(0.0, 0.45, n_obs)
    scores2 = 0.95 * y + shared_noise + rng.normal(0.0, 0.60, n_obs)
    scores3 = 0.70 * y + shared_noise + rng.normal(0.0, 0.75, n_obs)

    return y, scores1, scores2, scores3


def test_delong_covariance_matches_reference():
    y, scores1, scores2, scores3 = _make_data()
    scores = np.column_stack([scores1, scores2, scores3])

    aucs, covariance = mmu.delong_covariance(y, scores, obs_axis=0)
    ref_aucs, ref_covariance = _reference_delong(y, scores.T)

    assert np.allclose(aucs, ref_aucs, atol=1e-12, rtol=1e-10)
    assert np.allclose(covariance, ref_covariance, atol=1e-12, rtol=1e-10)

    aucs_alt, covariance_alt = mmu.delong_covariance(scores=scores.T, y=y, obs_axis=1)
    assert np.allclose(aucs_alt, ref_aucs, atol=1e-12, rtol=1e-10)
    assert np.allclose(covariance_alt, ref_covariance, atol=1e-12, rtol=1e-10)


def test_delong_test_matches_reference():
    y, scores1, scores2, _ = _make_data(seed=7165)

    result = mmu.delong_test(y, scores1, scores2)
    reference = _reference_delong_test(y, scores1, scores2)

    for key, value in reference.items():
        assert key in result._fields
        assert np.allclose(getattr(result, key), value, atol=1e-12, rtol=1e-10)


def test_delong_test_input_validation():
    y, scores1, scores2, _ = _make_data(seed=441)

    with pytest.raises(ValueError):
        mmu.delong_test(y, scores1[:-1], scores2)

    y_invalid = np.zeros_like(y)
    y_invalid[0] = 1

    with pytest.raises(ValueError):
        mmu.delong_test(y_invalid, scores1, scores2)


def _rank_reference_test_variance(y, scores1, scores2):
    """Sun & Xu midrank components in numpy, for sizes where the O(m * n) reference is too slow."""
    pos = y == 1
    m, n = int(pos.sum()), int((~pos).sum())

    def components(s):
        tz = rankdata(s)
        return (tz[pos] - rankdata(s[pos])) / n, 1.0 - (tz[~pos] - rankdata(s[~pos])) / m

    a10, a01 = components(scores1)
    b10, b01 = components(scores2)
    return np.var(a10 - b10, ddof=1) / m + np.var(a01 - b01, ddof=1) / n


def test_delong_test_variance_no_cancellation():
    # Nearly identical models: S11 + S22 - 2 * S12 loses ~8 significant digits here.
    rng = np.random.default_rng(1)
    n_obs = 20_000
    y = rng.binomial(1, 0.5, n_obs).astype(np.int64)
    scores1 = rng.normal(size=n_obs) + y
    scores2 = scores1.copy()
    idx = rng.choice(n_obs, 3, replace=False)
    scores2[idx] += rng.normal(size=3) * 0.05

    result = mmu.delong_test(y, scores1, scores2)
    expected = _rank_reference_test_variance(y, scores1, scores2)
    assert expected > 0.0
    assert result.variance == pytest.approx(expected, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("seed", range(25))
def test_delong_matches_reference_random_with_ties(seed):
    rng = np.random.default_rng(seed)
    n_obs = int(rng.integers(8, 200))
    n_models = int(rng.integers(1, 5))
    y = np.zeros(n_obs, dtype=np.int64)
    y[: n_obs // 2] = 1
    rng.shuffle(y)
    if seed % 2 == 0:
        scores = rng.integers(0, 4, (n_obs, n_models)).astype(np.float64)
    else:
        scores = np.round(rng.normal(size=(n_obs, n_models)) + y[:, None], 1)

    aucs, covariance = mmu.delong_covariance(y, scores)
    ref_aucs, ref_covariance = _reference_delong(y, scores.T)
    assert np.allclose(aucs, ref_aucs, atol=1e-12, rtol=1e-10)
    assert np.allclose(covariance, ref_covariance, atol=1e-12, rtol=1e-10)
    for model in range(n_models):
        assert aucs[model] == pytest.approx(roc_auc_score(y, scores[:, model]), abs=1e-12)

    if n_models >= 2:
        result = mmu.delong_test(y, scores[:, 0], scores[:, 1])
        reference = _reference_delong_test(y, scores[:, 0], scores[:, 1])
        for key, value in reference.items():
            assert np.allclose(getattr(result, key), value, atol=1e-12, rtol=1e-10, equal_nan=True)


def test_delong_test_identical_models():
    y, scores1, _, _ = _make_data(seed=3)
    result = mmu.delong_test(y, scores1, scores1.copy())
    assert result.difference == 0.0
    assert result.variance == 0.0
    assert result.z == 0.0
    assert result.p_value == 1.0


def test_delong_test_degenerate_variance_returns_nan():
    # Constant model (AUC 0.5) vs perfect model (AUC 1): differenced components are constant.
    y = np.array([1] * 5 + [0] * 5, dtype=np.int64)
    scores_constant = np.ones(10)
    scores_perfect = y.astype(np.float64)

    with pytest.warns(RuntimeWarning, match="variance estimate is zero"):
        result = mmu.delong_test(y, scores_constant, scores_perfect)

    assert result.difference == pytest.approx(-0.5)
    assert result.variance == 0.0
    assert math.isnan(result.z)
    assert math.isnan(result.p_value)


def test_delong_core_rejects_non_finite():
    y, scores1, scores2, _ = _make_data(seed=5)
    scores1 = scores1.copy()
    scores1[3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        _core.delong_test(y, scores1, scores2)
    with pytest.raises(ValueError, match="non-finite"):
        _core.delong_covariance(y, np.ascontiguousarray(np.vstack([scores1, scores2])))


@pytest.mark.parametrize("n_threads", [1, 2, 3, 8, -1])
def test_delong_results_independent_of_n_threads(n_threads):
    rng = np.random.default_rng(11)
    n_obs, n_models = 5_000, 7
    y = rng.binomial(1, 0.4, n_obs).astype(np.int64)
    scores = np.round(rng.normal(size=(n_obs, n_models)) + y[:, None], 2)

    aucs_ref, cov_ref = mmu.delong_covariance(y, scores, n_threads=1)
    aucs, cov = mmu.delong_covariance(y, scores, n_threads=n_threads)
    np.testing.assert_array_equal(aucs, aucs_ref)
    np.testing.assert_array_equal(cov, cov_ref)

    test_ref = mmu.delong_test(y, scores[:, 0], scores[:, 1], n_threads=1)
    test = mmu.delong_test(y, scores[:, 0], scores[:, 1], n_threads=n_threads)
    assert test == test_ref


def test_delong_invalid_n_threads():
    y, scores1, scores2, _ = _make_data(seed=13)
    with pytest.raises(ValueError, match="n_threads"):
        mmu.delong_test(y, scores1, scores2, n_threads=0)
    with pytest.raises(TypeError, match="n_threads"):
        mmu.delong_covariance(y, np.column_stack([scores1, scores2]), n_threads=2.0)
    with pytest.raises(ValueError, match="n_threads"):
        _core.delong_test(y, scores1, scores2, n_threads=0)


# --- Phase 2: components, one-sided alternatives, contrasts and confidence intervals ---


def test_delong_components_match_reference():
    rng = np.random.default_rng(21)
    n_obs = 150
    y = rng.binomial(1, 0.4, n_obs).astype(np.int64)
    scores = np.round(rng.normal(size=(3, n_obs)) + y, 1)

    aucs, v10, v01 = _core.delong_components(y, scores)
    ref_aucs, ref_v10, ref_v01 = _reference_components(y, scores)
    assert v10.shape == (3, y.sum())
    assert v01.shape == (3, n_obs - y.sum())
    np.testing.assert_allclose(aucs, ref_aucs, atol=1e-14)
    np.testing.assert_allclose(v10, ref_v10, atol=1e-14)
    np.testing.assert_allclose(v01, ref_v01, atol=1e-14)


def test_delong_test_alternatives():
    y, scores1, scores2, _ = _make_data(seed=7165)
    two = mmu.delong_test(y, scores1, scores2)
    greater = mmu.delong_test(y, scores1, scores2, alternative="greater")
    less = mmu.delong_test(y, scores1, scores2, alternative="less")

    assert greater.z == less.z == two.z
    assert greater.p_value == pytest.approx(0.5 * math.erfc(two.z / math.sqrt(2.0)), abs=0, rel=1e-14)
    assert greater.p_value + less.p_value == pytest.approx(1.0)
    assert two.p_value == pytest.approx(2.0 * min(greater.p_value, less.p_value))

    with pytest.raises(ValueError, match="alternative"):
        mmu.delong_test(y, scores1, scores2, alternative="bigger")


def test_delong_test_one_sided_degenerate_is_nan():
    y = np.array([1] * 5 + [0] * 5, dtype=np.int64)
    with pytest.warns(RuntimeWarning, match="variance estimate is zero"):
        result = mmu.delong_test(y, np.ones(10), y.astype(np.float64), alternative="less")
    assert math.isnan(result.p_value)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_delong_contrast_two_models_equals_delong_test(alternative):
    y, scores1, scores2, _ = _make_data(seed=99)
    expected = mmu.delong_test(y, scores1, scores2, alternative=alternative)
    result = mmu.delong_contrast_test(y, np.column_stack([scores1, scores2]), alternative=alternative)

    assert result.df == 1
    assert result.estimate[0] == pytest.approx(expected.difference, rel=1e-12)
    assert result.covariance[0, 0] == pytest.approx(expected.variance, rel=1e-10)
    assert result.statistic == pytest.approx(expected.z, rel=1e-10)
    assert result.p_value == pytest.approx(expected.p_value, rel=1e-10)


def test_delong_contrast_default_matches_reference_chi2():
    y, scores1, scores2, scores3 = _make_data(seed=5)
    scores = np.column_stack([scores1, scores2, scores3])
    result = mmu.delong_contrast_test(y, scores)

    aucs, cov = _reference_delong(y, scores.T)
    contrast = np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0]])
    d = contrast @ aucs
    sigma = contrast @ cov @ contrast.T
    chi2 = float(d @ np.linalg.solve(sigma, d))

    assert result.df == 2
    np.testing.assert_allclose(result.aucs, aucs, atol=1e-14)
    np.testing.assert_allclose(result.estimate, d, atol=1e-14)
    np.testing.assert_allclose(result.covariance, sigma, rtol=1e-9, atol=1e-15)
    assert result.statistic == pytest.approx(chi2, rel=1e-9)
    assert result.p_value == pytest.approx(chi2_dist.sf(chi2, 2), rel=1e-9)


def test_delong_contrast_single_row_matches_reference():
    y, scores1, scores2, scores3 = _make_data(seed=8)
    scores = np.column_stack([scores1, scores2, scores3])
    contrast = np.array([1.0, -0.5, -0.5])
    result = mmu.delong_contrast_test(y, scores, contrast=contrast, alternative="greater")

    aucs, cov = _reference_delong(y, scores.T)
    z = (contrast @ aucs) / math.sqrt(contrast @ cov @ contrast)
    assert result.df == 1
    assert result.statistic == pytest.approx(z, rel=1e-9)
    assert result.p_value == pytest.approx(0.5 * math.erfc(z / math.sqrt(2.0)), rel=1e-9)


def test_delong_contrast_rank_deficient():
    y, scores1, scores2, scores3 = _make_data(seed=17)
    scores = np.column_stack([scores1, scores2])

    # Linearly dependent rows: chi2 with df=1 equals z**2.
    z = mmu.delong_test(y, scores1, scores2).z
    result = mmu.delong_contrast_test(y, scores, contrast=[[1.0, -1.0], [2.0, -2.0]])
    assert result.df == 1
    assert result.statistic == pytest.approx(z**2, rel=1e-9)

    # Duplicate model: the theta_1 - theta_2 direction has zero variance and zero estimate.
    dup = np.column_stack([scores1, scores1, scores3])
    result = mmu.delong_contrast_test(y, dup)
    z13 = mmu.delong_test(y, scores1, scores3).z
    assert result.df == 1
    assert result.statistic == pytest.approx(z13**2, rel=1e-9)


def test_delong_contrast_degenerate_is_nan():
    y = np.array([1] * 5 + [0] * 5, dtype=np.int64)
    scores = np.column_stack([np.ones(10), y.astype(np.float64), np.linspace(0, 1, 10)])
    with pytest.warns(RuntimeWarning, match="variance estimate is zero"):
        result = mmu.delong_contrast_test(y, scores[:, :2])
    assert math.isnan(result.statistic)
    assert math.isnan(result.p_value)


def test_delong_contrast_input_validation():
    y, scores1, scores2, scores3 = _make_data(seed=2)
    scores = np.column_stack([scores1, scores2, scores3])

    with pytest.raises(ValueError, match="contrast"):
        mmu.delong_contrast_test(y, scores, contrast=[1.0, -1.0])
    with pytest.raises(ValueError, match="non-finite"):
        mmu.delong_contrast_test(y, scores, contrast=[1.0, np.nan, -1.0])
    with pytest.raises(ValueError, match="One-sided"):
        mmu.delong_contrast_test(y, scores, alternative="greater")
    with pytest.raises(ValueError, match="at least two models"):
        mmu.delong_contrast_test(y, scores1[:, None])
    with pytest.raises(TypeError, match="obs_axis"):
        mmu.delong_contrast_test(y, scores, obs_axis=True)


def test_auc_confidence_interval_wald_and_logit():
    y, scores1, scores2, _ = _make_data(seed=31)
    scores = np.column_stack([scores1, scores2])
    aucs, cov = _reference_delong(y, scores.T)
    se = np.sqrt(np.diag(cov))
    zq = 1.959963984540054

    wald = mmu.auc_confidence_interval(y, scores)
    np.testing.assert_allclose(wald.auc, aucs, atol=1e-14)
    np.testing.assert_allclose(wald.std_err, se, rtol=1e-9)
    np.testing.assert_allclose(wald.lower, aucs - zq * se, rtol=1e-9)
    np.testing.assert_allclose(wald.upper, aucs + zq * se, rtol=1e-9)

    logit = mmu.auc_confidence_interval(y, scores, method="logit")
    lg = np.log(aucs / (1 - aucs))
    se_lg = se / (aucs * (1 - aucs))
    np.testing.assert_allclose(logit.lower, 1 / (1 + np.exp(-(lg - zq * se_lg))), rtol=1e-9)
    np.testing.assert_allclose(logit.upper, 1 / (1 + np.exp(-(lg + zq * se_lg))), rtol=1e-9)

    single = mmu.auc_confidence_interval(y, scores1, alpha=0.1)
    assert isinstance(single.lower, float)
    assert single.auc == pytest.approx(aucs[0], abs=1e-14)
    assert single.upper - single.lower == pytest.approx(2 * 1.6448536269514722 * se[0], rel=1e-9)


def test_auc_confidence_interval_bounds_and_degenerate():
    rng = np.random.default_rng(4)
    y = np.array([1] * 6 + [0] * 6, dtype=np.int64)
    near_perfect = y + 0.1 * rng.normal(size=12) + np.r_[np.zeros(11), 2.0]

    wald = mmu.auc_confidence_interval(y, near_perfect)
    logit = mmu.auc_confidence_interval(y, near_perfect, method="logit")
    assert 0.0 <= wald.lower <= wald.upper <= 1.0
    assert 0.0 < logit.lower < logit.auc < logit.upper < 1.0

    with pytest.warns(RuntimeWarning, match="variance estimate is zero"):
        perfect = mmu.auc_confidence_interval(y, y.astype(np.float64), method="logit")
    assert perfect.auc == perfect.lower == perfect.upper == 1.0
    assert perfect.std_err == 0.0


def test_auc_confidence_interval_input_validation():
    y, scores1, _, _ = _make_data(seed=1)
    with pytest.raises(ValueError, match="method"):
        mmu.auc_confidence_interval(y, scores1, method="exact")
    for alpha in (0.0, 1.0, -0.1, True):
        with pytest.raises(ValueError, match="alpha"):
            mmu.auc_confidence_interval(y, scores1, alpha=alpha)
    with pytest.raises(ValueError, match="equal length"):
        mmu.auc_confidence_interval(y, scores1[:-1])


@pytest.mark.parametrize("method", ["wald", "logit"])
def test_auc_confidence_interval_coverage(method):
    # Binormal model with unit variances: true AUC = Phi(delta / sqrt(2)).
    rng = np.random.default_rng(2024)
    delta = 1.0
    true_auc = 0.5 * math.erfc(-delta / 2.0)
    n_reps, n_pos, n_neg = 400, 60, 90
    y = np.r_[np.ones(n_pos), np.zeros(n_neg)].astype(np.int64)
    covered = 0
    for _ in range(n_reps):
        scores = np.r_[rng.normal(delta, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)]
        ci = mmu.auc_confidence_interval(y, scores, method=method, n_threads=1)
        covered += ci.lower <= true_auc <= ci.upper
    # Binomial sd of the coverage estimate is ~0.011.
    assert 0.92 <= covered / n_reps <= 0.98


# -----------------------------------------------------------------------------
# Phase 3: external reference values and properties
# -----------------------------------------------------------------------------

# aSAH data set shipped with R's pROC (Robin et al., 2011); outcome "Poor" is
# the positive class. Reference values were generated with R 4.6.2, pROC 1.19.1:
#   r <- roc(aSAH$outcome, aSAH$<marker>, levels=c("Good", "Poor"), direction="<")
#   auc(r); var(r, method="delong"); cov(r1, r2, method="delong")
#   ci.auc(r, method="delong", conf.level=...)
#   roc.test(r1, r2, method="delong", paired=TRUE, alternative=...)
# fmt: off
_ASAH_Y = np.array(
    [
        0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
        1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
        1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 1, 0, 0, 0
    ],
    dtype=np.int64,
)
_ASAH_WFNS = np.array(
    [
        1, 1, 1, 1, 3, 2, 5, 4, 1, 2, 5, 2, 5, 5, 1, 2, 5, 2, 2, 5, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 3, 4, 2, 1, 4, 5,
        4, 1, 4, 2, 5, 5, 1, 1, 2, 4, 1, 1, 2, 5, 5, 2, 1, 2, 5, 1, 2, 2, 2, 1, 5, 1, 1, 2, 5, 2, 1, 4, 5, 2, 4, 4,
        1, 4, 1, 2, 1, 4, 2, 1, 2, 3, 2, 4, 2, 1, 4, 5, 1, 1, 5, 4, 2, 2, 1, 1, 1, 3, 1, 2, 5, 2, 2, 5, 5, 5, 2, 4,
        4, 5, 1, 1, 1
    ],
    dtype=np.float64,
)
_ASAH_S100B = np.array(
    [
        0.13, 0.14, 0.1, 0.04, 0.13, 0.1, 0.47, 0.16, 0.18, 0.1, 0.12, 0.1, 0.44, 0.71, 0.04, 0.08, 0.49, 0.04,
        0.07, 0.33, 0.09, 0.09, 0.07, 0.11, 0.07, 0.17, 0.07, 0.11, 0.13, 0.19, 0.05, 0.16, 0.41, 0.14, 0.34, 0.35,
        0.48, 0.09, 0.96, 0.25, 0.5, 0.46, 0.16, 0.07, 0.43, 0.45, 0.11, 0.08, 0.09, 0.86, 0.52, 0.08, 0.06, 0.13,
        2.07, 0.1, 0.14, 0.15, 0.07, 0.06, 0.77, 0.05, 0.09, 0.3, 0.03, 0.09, 0.04, 0.23, 0.7, 0.09, 0.27, 0.71,
        0.08, 0.26, 0.08, 0.16, 0.09, 0.13, 0.1, 0.08, 0.11, 0.33, 0.11, 0.28, 0.07, 0.1, 0.32, 0.22, 0.07, 0.05,
        0.24, 0.38, 0.1, 0.15, 0.08, 0.14, 0.1, 0.07, 0.04, 0.19, 0.56, 0.14, 0.58, 0.32, 0.82, 0.74, 0.15, 0.47,
        0.17, 0.44, 0.15, 0.5, 0.48
    ],
    dtype=np.float64,
)
_ASAH_NDKA = np.array(
    [
        3.01, 8.54, 8.09, 10.42, 17.4, 12.75, 6.0, 13.2, 15.54, 6.01, 15.96, 17.86, 5.18, 8.9, 13.41, 20.75, 11.6,
        16.11, 32.37, 54.82, 32.41, 49.94, 40.34, 9.47, 6.29, 12.53, 6.54, 6.3, 80.3, 12.8, 9.8, 9.81, 9.85, 18.21,
        5.03, 14.04, 21.93, 8.02, 7.42, 8.38, 6.59, 9.63, 13.12, 7.96, 14.34, 41.43, 7.63, 7.06, 12.59, 13.56, 3.87,
        9.44, 7.66, 12.98, 419.19, 27.19, 22.27, 9.95, 21.22, 11.73, 10.4, 58.83, 6.39, 11.09, 12.22, 13.67, 17.21,
        22.63, 12.9, 9.7, 21.57, 8.23, 72.57, 15.54, 10.51, 10.6, 14.57, 5.19, 9.63, 4.61, 21.48, 17.3, 12.71, 9.44,
        11.07, 19.46, 10.83, 5.37, 11.97, 7.75, 9.83, 10.55, 28.49, 8.53, 12.57, 12.9, 46.83, 11.68, 12.67, 9.01,
        9.57, 34.06, 11.72, 14.26, 47.61, 11.67, 24.58, 10.33, 13.87, 15.89, 22.43, 6.79, 13.45
    ],
    dtype=np.float64,
)
# fmt: on

_PROC_AUC = (0.82367886178861793, 0.73136856368563685, 0.61195799457994604)
_PROC_VAR = (0.0014699147088236266, 0.0026686824571724374, 0.0031908105493913025)
_PROC_COV = {(0, 1): 0.0011961556737675452, (0, 2): -0.00053296785676243776, (1, 2): -0.00075616493805657895}
_PROC_CI95 = (
    (0.74853488781945288, 0.89882283575778299),
    (0.63011821176162264, 0.83261891560965107),
    (0.50124499927170263, 0.72267098988818901),
)
_PROC_WFNS_CI90 = (0.76061605088919537, 0.88674167268804049)
_PROC_TEST = {
    (0, 1): (
        2.2089835914409082,
        {"two-sided": 0.027175782229188105, "greater": 0.013587891114594053, "less": 0.98641210888540598},
    ),
    (0, 2): (
        2.7977759186890387,
        {"two-sided": 0.0051455797069109776, "greater": 0.0025727898534554888, "less": 0.99742721014654456},
    ),
    (1, 2): (
        1.3907700257355771,
        {"two-sided": 0.16429517522305448, "greater": 0.08214758761152724, "less": 0.91785241238847282},
    ),
}
_PROC_RTOL = 1e-10


def _asah_scores():
    return np.column_stack([_ASAH_WFNS, _ASAH_S100B, _ASAH_NDKA])


def test_delong_covariance_matches_proc():
    aucs, cov = mmu.delong_covariance(_ASAH_Y, _asah_scores())
    np.testing.assert_allclose(aucs, _PROC_AUC, rtol=_PROC_RTOL)
    np.testing.assert_allclose(np.diag(cov), _PROC_VAR, rtol=_PROC_RTOL)
    for (i, j), value in _PROC_COV.items():
        assert cov[i, j] == pytest.approx(value, rel=_PROC_RTOL, abs=0.0)
        assert cov[j, i] == cov[i, j]


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize("pair", sorted(_PROC_TEST))
def test_delong_test_matches_proc(pair, alternative):
    scores = _asah_scores()
    z_ref, p_ref = _PROC_TEST[pair]
    result = mmu.delong_test(_ASAH_Y, scores[:, pair[0]], scores[:, pair[1]], alternative=alternative)
    assert result.auc1 == pytest.approx(_PROC_AUC[pair[0]], rel=_PROC_RTOL)
    assert result.auc2 == pytest.approx(_PROC_AUC[pair[1]], rel=_PROC_RTOL)
    assert result.z == pytest.approx(z_ref, rel=_PROC_RTOL)
    assert result.p_value == pytest.approx(p_ref[alternative], rel=_PROC_RTOL)

    contrast = np.zeros(3)
    contrast[list(pair)] = (1.0, -1.0)
    via_contrast = mmu.delong_contrast_test(_ASAH_Y, scores, contrast=contrast, alternative=alternative)
    assert via_contrast.statistic == pytest.approx(z_ref, rel=_PROC_RTOL)
    assert via_contrast.p_value == pytest.approx(p_ref[alternative], rel=_PROC_RTOL)


def test_auc_confidence_interval_matches_proc():
    ci = mmu.auc_confidence_interval(_ASAH_Y, _asah_scores())
    np.testing.assert_allclose(np.column_stack([ci.lower, ci.upper]), _PROC_CI95, rtol=_PROC_RTOL)
    np.testing.assert_allclose(ci.std_err, np.sqrt(_PROC_VAR), rtol=_PROC_RTOL)

    ci90 = mmu.auc_confidence_interval(_ASAH_Y, _ASAH_WFNS, alpha=0.1)
    assert (ci90.lower, ci90.upper) == pytest.approx(_PROC_WFNS_CI90, rel=_PROC_RTOL)


def test_delong_invariant_to_observation_permutation():
    y, scores1, scores2, scores3 = _make_data(seed=61)
    scores = np.column_stack([scores1, scores2, scores3])
    perm = np.random.default_rng(3).permutation(y.size)

    aucs, cov = mmu.delong_covariance(y, scores)
    aucs_p, cov_p = mmu.delong_covariance(y[perm], scores[perm])
    np.testing.assert_allclose(aucs_p, aucs, rtol=1e-13)
    np.testing.assert_allclose(cov_p, cov, rtol=1e-11)


def test_delong_invariant_to_monotone_transform():
    y, scores1, scores2, _ = _make_data(seed=62)
    # Round to create ties; a strictly increasing map preserves order and ties exactly.
    scores1 = np.round(scores1, 1)
    base = mmu.delong_test(y, scores1, scores2)
    transformed = mmu.delong_test(y, np.exp(3.0 * scores1) - 7.0, np.arctan(scores2))
    assert transformed == base


def test_delong_test_swapping_models():
    y, scores1, scores2, _ = _make_data(seed=63)
    ab = mmu.delong_test(y, scores1, scores2)
    ba = mmu.delong_test(y, scores2, scores1)
    assert (ba.auc1, ba.auc2) == (ab.auc2, ab.auc1)
    assert ba.difference == -ab.difference
    assert ba.variance == pytest.approx(ab.variance, rel=1e-13)
    assert ba.z == pytest.approx(-ab.z, rel=1e-13)
    assert ba.p_value == pytest.approx(ab.p_value, rel=1e-13)

    greater = mmu.delong_test(y, scores1, scores2, alternative="greater")
    less = mmu.delong_test(y, scores2, scores1, alternative="less")
    assert less.p_value == pytest.approx(greater.p_value, rel=1e-13)


def test_delong_contrast_chi2_equals_z_squared():
    y, scores1, scores2, _ = _make_data(seed=64)
    z = mmu.delong_test(y, scores1, scores2).z
    # Two rows force the chi-square branch even though the rank is one.
    result = mmu.delong_contrast_test(y, np.column_stack([scores1, scores2]), contrast=[[1.0, -1.0], [-1.0, 1.0]])
    assert result.df == 1
    assert result.statistic == pytest.approx(z**2, rel=1e-10)
    assert result.p_value == pytest.approx(math.erfc(abs(z) / math.sqrt(2.0)), rel=1e-9)


@pytest.mark.parametrize("n_models", [2, 3, 5])
def test_delong_contrast_default_df(n_models):
    rng = np.random.default_rng(n_models)
    y = rng.binomial(1, 0.4, 300).astype(np.int64)
    scores = y[:, None] + rng.normal(size=(300, n_models))
    result = mmu.delong_contrast_test(y, scores)
    assert result.df == n_models - 1
    assert result.estimate.shape == (n_models - 1,)
    assert result.covariance.shape == (n_models - 1, n_models - 1)


def test_delong_fully_tied_model():
    y, scores1, _, _ = _make_data(seed=65)
    const = np.full(y.size, 0.3)
    aucs, cov = mmu.delong_covariance(y, np.column_stack([const, scores1]))
    assert aucs[0] == 0.5
    assert cov[0, 0] == 0.0
    assert cov[0, 1] == 0.0
    assert cov[1, 1] > 0.0

    # The comparison against a constant model is non-degenerate.
    result = mmu.delong_test(y, scores1, const)
    assert result.variance == pytest.approx(cov[1, 1], rel=1e-12)
    assert np.isfinite(result.z)


@pytest.mark.parametrize("n_levels", [2, 3, 5])
def test_delong_discrete_scores_match_sklearn(n_levels):
    rng = np.random.default_rng(100 + n_levels)
    y = rng.binomial(1, 0.5, 250).astype(np.int64)
    scores = np.column_stack([np.clip(y + rng.integers(-1, n_levels, 250), 0, n_levels) for _ in range(3)]).astype(
        np.int64
    )
    aucs, cov = mmu.delong_covariance(y, scores)
    ref_aucs, ref_cov = _reference_delong(y, scores.T.astype(np.float64))
    for model in range(3):
        assert aucs[model] == pytest.approx(roc_auc_score(y, scores[:, model]), abs=1e-12)
    np.testing.assert_allclose(aucs, ref_aucs, atol=1e-14)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-10, atol=1e-15)


def test_delong_input_dtypes():
    y, scores1, scores2, _ = _make_data(seed=66)
    expected = mmu.delong_test(y, scores1, scores2)

    for y_alt in (y.astype(bool), y.astype(np.float64), y.astype(np.int32), y.tolist()):
        assert mmu.delong_test(y_alt, scores1, scores2) == expected

    s1_32, s2_32 = scores1.astype(np.float32), scores2.astype(np.float32)
    expected_32 = mmu.delong_test(y, s1_32.astype(np.float64), s2_32.astype(np.float64))
    assert mmu.delong_test(y, s1_32, s2_32) == expected_32

    scores = np.column_stack([scores1, scores2])
    aucs, _ = mmu.delong_covariance(y.astype(bool), scores.tolist())
    np.testing.assert_array_equal(aucs, [expected.auc1, expected.auc2])
    ci = mmu.auc_confidence_interval(y.astype(bool), scores.astype(np.float32))
    assert ci.auc.shape == (2,)


def test_delong_single_model_covariance():
    y, scores1, _, _ = _make_data(seed=67)
    aucs, cov = mmu.delong_covariance(y, scores1[:, None])
    ref_aucs, ref_cov = _reference_delong(y, scores1[None, :])
    assert aucs.shape == (1,)
    assert cov.shape == (1, 1)
    np.testing.assert_allclose(cov, ref_cov, rtol=1e-10)
    assert aucs[0] == pytest.approx(ref_aucs[0], abs=1e-14)


@pytest.mark.parametrize("n_positive", [0, 1])
def test_delong_requires_two_per_class(n_positive):
    y = np.zeros(20, dtype=np.int64)
    y[:n_positive] = 1
    scores = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ValueError, match="at least two observations in each class"):
        mmu.delong_test(y, scores[:, 0], scores[:, 1])
    with pytest.raises(ValueError, match="at least two observations in each class"):
        mmu.delong_covariance(y, scores)
    with pytest.raises(ValueError, match="at least two observations in each class"):
        mmu.delong_covariance(1 - y, scores)


def test_delong_rejects_non_binary_labels():
    y, scores1, scores2, _ = _make_data(seed=68)
    y_bad = y.copy()
    y_bad[0] = 2
    with pytest.raises(ValueError, match="binary"):
        mmu.delong_test(y_bad, scores1, scores2)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_delong_rejects_non_finite_scores(bad):
    y, scores1, scores2, _ = _make_data(seed=69)
    scores1 = scores1.copy()
    scores1[3] = bad
    with pytest.raises(ValueError):  # noqa: PT011
        mmu.delong_test(y, scores1, scores2)
    with pytest.raises(ValueError):  # noqa: PT011
        mmu.delong_covariance(y, np.column_stack([scores1, scores2]))
