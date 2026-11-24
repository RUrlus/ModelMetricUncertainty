"""Binomial references, mapping, shape contracts and shared plot levels."""

import itertools

import mmu
import numpy as np
import pandas as pd
import pytest
from mmu.commons.levels import _parse_levels, _plot_levels
from mmu.methods import binary_metrics_confidence_intervals as intervals
from mmu.methods import clopper_pearson, jeffreys, wilson
from mmu.metrics import col_index, col_names
from scipy.stats import beta, binomtest, chi2, norm

PRIMITIVES = [clopper_pearson, wilson, jeffreys]
METHODS = ["clopper-pearson", "wilson", "jeffreys"]


@pytest.mark.parametrize(
    ("estimator", "method", "kwargs"),
    [(clopper_pearson, "exact", {}), (wilson, "wilson", {}), (wilson, "wilsoncc", {"continuity": True})],
)
@pytest.mark.parametrize("level", [0.1, 0.68, 0.9, 0.95, 0.99])
@pytest.mark.parametrize(("side", "alternative"), [("two-sided", "two-sided"), ("lower", "greater"), ("upper", "less")])
def test_scipy_reference(estimator, method, kwargs, level, side, alternative):
    for n in [1, 2, 5, 10, 100, 1000]:
        for x in sorted({0, 1, n // 2, n - 1, n}):
            reference = binomtest(x, n, alternative=alternative).proportion_ci(confidence_level=level, method=method)
            expected = (reference.low, reference.high)
            if side != "two-sided":
                expected = expected[0 if side == "lower" else 1]
            np.testing.assert_allclose(estimator(x, n, levels=level, side=side, **kwargs), expected, atol=1e-12)


def test_jeffreys_reference_and_boundaries():
    # Published central posterior quantiles, plus our documented endpoint modification.
    np.testing.assert_allclose(jeffreys(5, 10), [0.2235286702527052, 0.7764713297472948])
    for x, n in [(1, 10), (6, 12), (99, 100)]:
        np.testing.assert_allclose(jeffreys(x, n), beta.interval(0.95, x + 0.5, n - x + 0.5))
    assert jeffreys(0, 10)[0] == 0.0
    assert jeffreys(10, 10)[1] == 1.0


def test_wilson_large_counts():
    n = 1e200
    with np.errstate(over="raise", invalid="raise"):
        lower, upper = wilson(0, n)
    assert lower == 0.0
    assert upper == pytest.approx(norm.isf(0.025) ** 2 / n, rel=1e-12, abs=0)


@pytest.mark.parametrize("estimator", PRIMITIVES)
def test_counts_broadcast_and_levels_axis(estimator):
    x = np.array([[0], [1]])
    n = np.array([1, 5, 10])
    levels = [0.95, 0.8, 0.95]
    lower, upper = estimator(x, n, levels=levels)
    assert lower.shape == upper.shape == (2, 3, 3)
    for i, j, k in itertools.product(range(2), range(3), range(3)):
        np.testing.assert_allclose([lower[i, j, k], upper[i, j, k]], estimator(x[i, 0], n[j], levels=levels[k]))
    assert isinstance(estimator(1, 2)[0], float)
    assert estimator(1, 2, levels=[0.95])[0].shape == (1,)


@pytest.mark.parametrize("estimator", PRIMITIVES)
@pytest.mark.parametrize("side", ["two-sided", "lower", "upper"])
def test_undefined_counts_and_extreme_bounds(estimator, side):
    assert np.isnan(estimator(0, 0, side=side)).all()
    result = estimator([0, 1, 0, 10], [1, 1, 10, 10], side=side)
    if side == "two-sided":
        lower, upper = result
        assert lower[0] == lower[2] == 0.0
        assert upper[1] == upper[3] == 1.0
        assert np.all(lower <= upper)
    else:
        assert np.isfinite(result).all()


@pytest.mark.parametrize("estimator", PRIMITIVES)
@pytest.mark.parametrize(("x", "n"), [(-1, 4), (5, 4), (0.5, 4), (1, 4.5), (1, -2), (np.nan, 3), (0, np.inf)])
def test_invalid_counts(estimator, x, n):
    with pytest.raises(ValueError, match="counts|must not exceed"):
        estimator(x, n)


@pytest.mark.parametrize("estimator", PRIMITIVES)
@pytest.mark.parametrize("value", [True, "2", 1j, None])
def test_nonnumeric_counts(estimator, value):
    with pytest.raises(TypeError):
        estimator(value, 3)


@pytest.mark.parametrize("estimator", PRIMITIVES)
@pytest.mark.parametrize("levels", [0, -1, 0.0, 1.0, 1.1, np.nan, np.inf, [], [[0.95]], [0.9, np.nan], 100])
def test_invalid_levels(estimator, levels):
    with pytest.raises(ValueError, match="levels"):
        estimator(1, 2, levels=levels)


@pytest.mark.parametrize("levels", [True, "0.95", None, [False], [1j]])
def test_invalid_level_types(levels):
    with pytest.raises(TypeError):
        clopper_pearson(1, 2, levels=levels)


@pytest.mark.parametrize("estimator", PRIMITIVES)
def test_invalid_side_and_no_confidence_alias(estimator):
    with pytest.raises(ValueError, match="side"):
        estimator(1, 2, side="both")
    with pytest.raises(TypeError):
        estimator(1, 2, confidence=0.95)


@pytest.mark.parametrize("estimator", PRIMITIVES)
def test_monotonicity(estimator):
    lower, upper = estimator([5, 50, 500], [10, 100, 1000])
    assert np.all(np.diff(upper - lower) < 0)
    lower, upper = estimator(5, 10, levels=[0.8, 0.9, 0.95])
    assert np.all(np.diff(upper - lower) > 0)


@pytest.mark.parametrize(("method", "estimator"), zip(METHODS, PRIMITIVES))
@pytest.mark.parametrize("side", ["two-sided", "lower", "upper"])
def test_all_metric_mappings(method, estimator, side):
    cm = [37, 3, 11, 19]
    x = [37, 19, 37, 19, 3, 11, 56]
    n = [48, 22, 40, 30, 40, 30, 70]
    indices = [0, 1, 2, 3, 6, 7, 8]
    result = intervals(cm, method=method, side=side)
    assert result.shape == ((10, 2) if side == "two-sided" else (10,))
    assert np.isnan(result[[4, 5, 9]]).all()
    for index, successes, trials in zip(indices, x, n):
        np.testing.assert_allclose(result[index], estimator(successes, trials, side=side))


@pytest.mark.parametrize("cm", [*itertools.product([0, 3], repeat=4), (10, 0, 5, 0), (10, 5, 0, 0), (1, 0, 0, 0)])
@pytest.mark.parametrize("method", METHODS)
def test_degenerate_confusion_matrices(cm, method):
    result = intervals(cm, method=method)
    tn, fp, fn, tp = cm
    denominators = np.array([tn + fn, tp + fp, tn + fp, tp + fn, 0, 0, tn + fp, tp + fn, sum(cm), 0])
    assert np.isnan(result[denominators == 0]).all()
    assert np.isfinite(result[denominators > 0]).all()
    assert np.all(result[denominators > 0, 0] <= result[denominators > 0, 1])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("side", ["two-sided", "lower", "upper"])
def test_complement_identities(method, side):
    result = intervals([17, 3, 7, 23], levels=[0.9, 0.95], method=method, side=side)
    other_side = {"lower": "upper", "upper": "lower", "two-sided": "two-sided"}[side]
    other = intervals([17, 3, 7, 23], levels=[0.9, 0.95], method=method, side=other_side)
    for left, right in [(2, 6), (3, 7)]:
        expected = 1.0 - (other[..., right, ::-1] if side == "two-sided" else other[..., right])
        np.testing.assert_allclose(result[..., left, :] if side == "two-sided" else result[..., left], expected)


def test_selection_all_aliases_and_order():
    names = [name for name, index in col_index.items() if index not in (4, 5, 9)]
    result = intervals([13, 7, 9, 21], metrics=names)
    all_metrics = intervals([13, 7, 9, 21])
    np.testing.assert_allclose(result, all_metrics[[col_index[name] for name in names]])
    np.testing.assert_allclose(intervals([13, 7, 9, 21], metrics=["recall", "npv", "recall"]), all_metrics[[3, 0, 3]])


@pytest.mark.parametrize("metrics", [["neg.f1"], ["pos.f1_score"], ["mcc"], ["missing"]])
def test_invalid_metric_selection(metrics):
    with pytest.raises(ValueError, match="F1 or MCC|Unknown metric"):
        intervals([1, 2, 3, 4], metrics=metrics)


def test_string_metric_selection():
    with pytest.raises(TypeError, match="sequence"):
        intervals([1, 2, 3, 4], metrics="recall")


@pytest.mark.parametrize(
    "cm", [0, [1, 2, 3], np.ones((3, 2)), np.ones((2, 3, 5)), [-1, 2, 3, 4], [1, 2, 0.5, 4], [1, 2, np.inf, 4]]
)
def test_invalid_confusion_matrix(cm):
    with pytest.raises(ValueError, match="conf_mat|counts"):
        intervals(cm)


@pytest.mark.parametrize("shape", [(4,), (2, 2), (3, 4), (2, 3, 4)])
@pytest.mark.parametrize("side", ["two-sided", "lower", "upper"])
@pytest.mark.parametrize("levels", [0.95, [0.95], [0.95, 0.8, 0.95]])
def test_driver_shapes_and_dataframe(shape, side, levels):
    cm = np.arange(np.prod(shape)).reshape(shape)
    batch = () if shape == (2, 2) else shape[:-1]
    level_shape = () if np.ndim(levels) == 0 else (len(levels),)
    result = intervals(cm, levels=levels, side=side, metrics=["recall", "npv"])
    expected_shape = (*batch, *level_shape, 2, *((2,) if side == "two-sided" else ()))
    assert result.shape == expected_shape
    frame = intervals(cm, levels=levels, side=side, metrics=["recall", "npv"], return_df=True)
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns.get_level_values("metric").unique()) == ["pos.recall", "neg.precision"]
    np.testing.assert_allclose(frame.to_numpy(), result.reshape(frame.shape), equal_nan=True)
    if level_shape:
        assert frame.index.names == ["sample", "level"]
        np.testing.assert_array_equal(frame.index.get_level_values("level")[: len(levels)], levels)
    for i, row in enumerate(cm.reshape(-1, 4)):
        np.testing.assert_allclose(
            result.reshape(-1, *level_shape, 2, *((2,) if side == "two-sided" else ()))[i],
            intervals(row, levels=levels, side=side, metrics=["recall", "npv"]),
            equal_nan=True,
        )


def test_dataframe_default_columns_empty_metrics_and_batches():
    frame = intervals([1, 2, 3, 4], return_df=True)
    assert list(frame.columns) == list(itertools.product(col_names, ["lower", "upper"]))
    assert intervals([1, 2, 3, 4], metrics=[]).shape == (0, 2)
    assert intervals([1, 2, 3, 4], metrics=[], return_df=True).shape == (1, 0)
    assert intervals(np.empty((0, 4)), return_df=True).shape == (0, 20)
    assert intervals(np.empty((0, 4)), levels=[0.95, 0.9], return_df=True).shape == (0, 20)


@pytest.mark.parametrize("method", ["clopper-pearson", "jeffreys"])
@pytest.mark.parametrize("side", ["two-sided", "lower", "upper"])
@pytest.mark.parametrize("levels", [0.95, [0.9, 0.95]])
def test_continuity_ignored_for_other_methods(method, side, levels):
    cm = np.array([[13, 7, 9, 21], [0, 0, 0, 0]])
    np.testing.assert_allclose(
        intervals(cm, method=method, side=side, levels=levels, continuity=True),
        intervals(cm, method=method, side=side, levels=levels, continuity=False),
        equal_nan=True,
    )


def test_driver_options():
    np.testing.assert_allclose(
        intervals([13, 7, 9, 21], metrics=["recall"], method="wilson", continuity=True)[0],
        wilson(21, 30, continuity=True),
    )
    for method in ["hybrid", "unknown"]:
        with pytest.raises(ValueError, match="method"):
            intervals([1, 2, 3, 4], method=method)
    with pytest.raises(TypeError):
        wilson(1, 2, continuity="yes")
    with pytest.raises(TypeError):
        intervals([1, 2, 3, 4], continuity="yes")
    with pytest.raises(TypeError):
        intervals([1, 2, 3, 4], min_count=10)
    with pytest.raises(ValueError, match="overflowed"), np.errstate(over="raise"):
        intervals([1e308] * 4)


@pytest.mark.parametrize("levels", [1, np.int32(2), [1, 2, 3], np.array([1, 2], dtype=np.uint32)])
def test_sigma_equivalence(levels):
    probabilities = 2.0 * (norm.cdf(levels) - 0.5)
    for estimator in PRIMITIVES:
        np.testing.assert_allclose(estimator(7, 10, levels=levels), estimator(7, 10, levels=probabilities))
    critical, _ = _plot_levels(levels)
    np.testing.assert_allclose(critical, chi2.ppf(np.unique(probabilities), 2))
    np.testing.assert_allclose(_parse_levels(levels)[1], probabilities)


@pytest.mark.parametrize("levels", [True, "0.95", [], [[1, 2]], 0, 1.0, np.nan, 100])
def test_plot_levels_validation(levels):
    with pytest.raises((TypeError, ValueError), match="levels"):
        _plot_levels(levels)


def test_plot_labels_and_sorting():
    _, labels = _plot_levels(None)
    assert labels == [r"$1\sigma$ CI", r"$2\sigma$ CI", r"$3\sigma$ CI"]
    _, labels = _plot_levels([3, 1, 2, 2])
    assert labels == [r"1$\sigma$ CI", r"2$\sigma$ CI", r"3$\sigma$ CI"]
    _, labels = _plot_levels([0.95, 0.8, 0.95])
    assert labels == ["80.0% CI", "95.0% CI"]


@pytest.mark.parametrize("levels", [None, 2, 0.95, [3, 1, 2, 2], [0.95, 0.8, 0.95]])
def test_point_and_curve_plot_level_integration(levels):
    import matplotlib.pyplot as plt

    point = mmu.PRU.from_confusion_matrix(np.array([37, 3, 11, 19]), n_bins=20)
    curve = mmu.PRCU.from_confusion_matrices(np.array([[37, 3, 11, 19], [33, 7, 6, 24]]), n_bins=20)
    expected, _ = _plot_levels(levels)
    try:
        for estimator in [point, curve]:
            estimator.plot(levels=levels)
            np.testing.assert_allclose(estimator.critical_values_plot, expected)
    finally:
        plt.close("all")


def test_public_exports():
    for name in ["clopper_pearson", "wilson", "jeffreys", "binary_metrics_confidence_intervals", "bmetrics_ci"]:
        assert getattr(mmu, name) is getattr(mmu.methods, name)
        assert name in mmu.__all__
        assert name in mmu.methods.__all__
    assert mmu.bmetrics_ci is intervals


@pytest.mark.parametrize("side", ["two-sided", "lower", "upper"])
@pytest.mark.parametrize("levels", [0.95, [0.9, 0.95]])
def test_stacked_2x2_confusion_matrices(side, levels):
    cm = np.arange(24).reshape(2, 3, 2, 2)
    np.testing.assert_allclose(
        intervals(cm, levels=levels, side=side),
        intervals(cm.reshape(2, 3, 4), levels=levels, side=side),
        equal_nan=True,
    )


@pytest.mark.parametrize("n_runs", [2, 3, 5])
@pytest.mark.parametrize("levels", [0.95, [0.9, 0.95]])
def test_runs_thresholds_layout(n_runs, levels):
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, (200, n_runs))
    scores = rng.random((200, n_runs))
    thresholds = np.array([0.2, 0.4, 0.6, 0.8])
    cm = mmu.confusion_matrices_runs_thresholds(y, scores, thresholds)
    assert cm.shape == (thresholds.size, 4, n_runs)

    result = intervals(cm, levels=levels)
    assert result.shape[:2] == (thresholds.size, n_runs)
    for t, r in itertools.product(range(thresholds.size), range(n_runs)):
        np.testing.assert_allclose(result[t, r], intervals(cm[t, :, r], levels=levels), equal_nan=True)
    np.testing.assert_allclose(intervals(cm, levels=levels, axis=1), result, equal_nan=True)
    frame = intervals(cm, levels=levels, return_df=True)
    np.testing.assert_allclose(frame.to_numpy(), result.reshape(frame.shape), equal_nan=True)


def test_ambiguous_layout_requires_axis():
    rng = np.random.default_rng(12)
    cm = rng.integers(0, 20, (3, 4, 4))
    with pytest.raises(ValueError, match="ambiguous"):
        intervals(cm)
    np.testing.assert_allclose(intervals(cm, axis=1)[1, 2], intervals(cm[1, :, 2]), equal_nan=True)
    np.testing.assert_allclose(intervals(cm, axis=-1)[1, 2], intervals(cm[1, 2]), equal_nan=True)


@pytest.mark.parametrize(("axis", "error"), [(0, ValueError), (5, ValueError), (1.0, TypeError), (True, TypeError)])
def test_invalid_axis(axis, error):
    with pytest.raises(error, match="axis"):
        intervals(np.ones((3, 4), dtype=int), axis=axis)


@pytest.mark.parametrize("levels", [[1, 0.95], (2, 0.9, 0.8)])
def test_mixed_level_conventions(levels):
    with pytest.raises(TypeError, match="must not mix"):
        clopper_pearson(1, 2, levels=levels)


def test_float_level_of_one_message():
    with pytest.raises(ValueError, match="use integers to specify standard deviations"):
        clopper_pearson(1, 2, levels=1.0)
