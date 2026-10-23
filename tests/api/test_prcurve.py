import os
import numpy as np
import pytest
import sklearn.metrics as skm
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mmu
from mmu.commons._testing import PRCU_skm, ROCCU_skm, generate_test_labels, greater_equal_tol


def test_PR_MU_reference():

    fpath = os.path.abspath(os.path.dirname(__file__))
    conf_mats = np.load(os.path.join(fpath, "pr_curve_reference_conf_mats.npy"))
    ref_scores = np.load(os.path.join(fpath, "pr_curve_reference.npy"))

    curve_err = mmu.PRCU().from_confusion_matrices(conf_mats)
    assert np.allclose(curve_err.chi2_scores, ref_scores)


@pytest.mark.parametrize("curve_class,skm_func", [(mmu.PRCU, PRCU_skm), (mmu.ROCCU, ROCCU_skm)])
def test_MU_from_scores(curve_class, skm_func):
    """Test BaseCurveUncertainty.from_scores"""
    np.random.seed(412)
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    proba, _, y = generate_test_labels(N=500)
    yhat = greater_equal_tol(proba, thresholds[100])

    sk_conf_mat = skm.confusion_matrix(y, yhat)
    err = curve_class.from_scores(y=y, scores=proba, thresholds=thresholds)
    assert err.conf_mats is not None
    assert err.conf_mats.dtype == np.dtype(np.int64)

    y, x = skm_func(y, yhat)

    assert err.chi2_scores.shape == (err.x_grid.size, err.x_grid.size)
    assert np.isclose(err.y[100], y[1])
    assert np.isclose(err.x[100], x[1])
    assert np.array_equal(err.conf_mats[100], sk_conf_mat.flatten())


@pytest.mark.parametrize("curve_class,skm_func", [(mmu.PRCU, PRCU_skm), (mmu.ROCCU, ROCCU_skm)])
def test_MU_from_confusion_matrices(curve_class, skm_func):
    """Test BaseCurveUncertainty.from_confusion_matrices"""
    np.random.seed(412)
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    proba, _, y = generate_test_labels(N=500)
    yhat = greater_equal_tol(proba, thresholds[100])

    sk_conf_mat = skm.confusion_matrix(y, yhat)
    conf_mats = mmu.confusion_matrices_thresholds(y, proba, thresholds)
    err = curve_class.from_confusion_matrices(conf_mats=conf_mats)
    assert err.conf_mats is not None
    assert err.conf_mats.dtype == np.dtype(np.int64)

    y, x = skm_func(y, yhat)

    assert err.chi2_scores.shape == (err.x_grid.size, err.x_grid.size)
    assert np.isclose(err.y[100], y[1])
    assert np.isclose(err.x[100], x[1])
    assert np.array_equal(err.conf_mats[100], sk_conf_mat.flatten())


@pytest.mark.parametrize("curve_class,skm_func", [(mmu.PRCU, PRCU_skm), (mmu.ROCCU, ROCCU_skm)])
def test_MU_from_classifier(curve_class, skm_func):
    """Test BaseCurveUncertainty.from_classifier"""
    # generate seeds to be used by sklearn
    # do not use this in real scenarios,
    # it's a convenience only used in the tutorial notebooks
    seeds = mmu.commons.utils.SeedGenerator(234)

    # generate 2 class dataset
    X, y = make_classification(n_samples=1000, n_classes=2, random_state=seeds())

    # split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=seeds())
    # fit a model
    model = LogisticRegression(solver="lbfgs")
    model.fit(X_train, y_train)

    # predict probabilities, for the positive outcome only
    y_scores = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(1e-12, 1 - 1e-12, 200)
    yhat = greater_equal_tol(y_scores, thresholds[100])
    sk_conf_mat = skm.confusion_matrix(y_test, yhat)
    err = curve_class.from_classifier(clf=model, X=X_test, y=y_test, thresholds=thresholds)
    assert err.conf_mats is not None
    assert err.conf_mats.dtype == np.dtype(np.int64)

    y, x = skm_func(y_test, yhat)

    assert err.chi2_scores.shape == (err.x_grid.size, err.x_grid.size)
    assert np.isclose(err.y[100], y[1])
    assert np.isclose(err.x[100], x[1])
    assert np.array_equal(err.conf_mats[100], sk_conf_mat.flatten())


@pytest.mark.parametrize("curve_class", [mmu.PRCU, mmu.ROCCU])
def test_curve_plot_with_other_single_curve(curve_class):
    """Test adding a single other curve to a curve plot."""
    np.random.seed(412)
    y_scores_1, _, y_true = generate_test_labels(N=400)
    y_scores_2 = np.clip(y_scores_1 + np.random.normal(0.0, 0.08, y_scores_1.size), 1e-12, 1 - 1e-12)

    curve_1 = curve_class.from_scores(y=y_true, scores=y_scores_1, n_bins=100)
    curve_2 = curve_class.from_scores(y=y_true, scores=y_scores_2, n_bins=100)

    ax = curve_1.plot(other=curve_2)
    assert ax is not None
    assert len(ax.lines) >= 2  # At least 2 curves
    legend = ax.get_legend()
    assert legend is not None


@pytest.mark.parametrize("curve_class", [mmu.PRCU, mmu.ROCCU])
def test_curve_plot_with_other_list(curve_class):
    """Test adding multiple other curves (list) to a curve plot."""
    np.random.seed(412)
    y_scores_1, _, y_true = generate_test_labels(N=400)
    y_scores_2 = np.clip(y_scores_1 + np.random.normal(0.0, 0.08, y_scores_1.size), 1e-12, 1 - 1e-12)
    y_scores_3 = np.clip(y_scores_1 + np.random.normal(0.0, 0.12, y_scores_1.size), 1e-12, 1 - 1e-12)

    curve_1 = curve_class.from_scores(y=y_true, scores=y_scores_1, n_bins=100)
    curve_2 = curve_class.from_scores(y=y_true, scores=y_scores_2, n_bins=100)
    curve_3 = curve_class.from_scores(y=y_true, scores=y_scores_3, n_bins=100)

    ax = curve_1.plot(other=[curve_2, curve_3])
    assert ax is not None
    legend = ax.get_legend()
    assert legend is not None


def test_curve_plot_with_other_invalid_type():
    """Test that invalid other type raises TypeError."""
    np.random.seed(412)
    y_scores_1, _, y_true = generate_test_labels(N=400)
    curve_1 = mmu.PRCU.from_scores(y=y_true, scores=y_scores_1, n_bins=100)

    with pytest.raises(TypeError, match="`other` must be a BaseCurveUncertainty"):
        curve_1.plot(other="invalid")


@pytest.mark.parametrize("curve_class", [mmu.PRCU, mmu.ROCCU])
def test_curve_plot_backward_compatibility(curve_class):
    """Test that existing plot() calls without other still work."""
    np.random.seed(412)
    y_scores_1, _, y_true = generate_test_labels(N=400)
    curve_1 = curve_class.from_scores(y=y_true, scores=y_scores_1, n_bins=100)

    ax = curve_1.plot()
    assert ax is not None
    legend = ax.get_legend()
    assert legend is not None


@pytest.mark.parametrize("curve_class", [mmu.PRCU, mmu.ROCCU])
def test_curve_plot_with_default_label(curve_class):
    """Test that default label behavior uses auto-generated symbol labels."""
    np.random.seed(412)
    y_scores_1, _, y_true = generate_test_labels(N=400)
    curve_1 = curve_class.from_scores(y=y_true, scores=y_scores_1, n_bins=100)

    ax = curve_1.plot(label=None)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    # First entry should contain LaTeX symbols (auto-generated)
    assert any("$" in text for text in legend_texts), "Expected LaTeX symbols in legend"


@pytest.mark.parametrize("curve_class", [mmu.PRCU, mmu.ROCCU])
def test_curve_plot_with_custom_label(curve_class):
    """Test that custom label overrides default legend text."""
    np.random.seed(412)
    y_scores_1, _, y_true = generate_test_labels(N=400)
    curve_1 = curve_class.from_scores(y=y_true, scores=y_scores_1, n_bins=100)

    custom_label = "Model A"
    ax = curve_1.plot(label=custom_label)
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    # First entry should be the custom label
    assert custom_label in legend_texts, f"Expected '{custom_label}' in legend texts: {legend_texts}"


@pytest.mark.parametrize("curve_class", [mmu.PRCU, mmu.ROCCU])
def test_curve_plot_with_custom_label_and_overlay(curve_class):
    """Test that custom labels work with overlaid curves."""
    np.random.seed(412)
    y_scores_1, _, y_true = generate_test_labels(N=400)
    y_scores_2 = np.clip(y_scores_1 + np.random.normal(0.0, 0.08, y_scores_1.size), 1e-12, 1 - 1e-12)

    curve_1 = curve_class.from_scores(y=y_true, scores=y_scores_1, n_bins=100)
    curve_2 = curve_class.from_scores(y=y_true, scores=y_scores_2, n_bins=100)

    # Plot with custom label for primary curve and overlay with custom label
    ax = curve_1.plot(label="Model A", other=curve_2, other_kwargs={"label": "Model B"})
    legend = ax.get_legend()
    assert legend is not None
    legend_texts = [t.get_text() for t in legend.get_texts()]
    assert "Model A" in legend_texts, f"Expected 'Model A' in legend: {legend_texts}"
    assert "Model B" in legend_texts, f"Expected 'Model B' in legend: {legend_texts}"
