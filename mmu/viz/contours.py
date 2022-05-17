import numpy as np
import matplotlib.pyplot as plt

from mmu.viz.utils import _get_color_hexes
from mmu.viz.utils import _create_pr_legend
from mmu.viz.utils import _create_pr_legend_scatter


def _plot_pr_curve_contours(
    precision,
    recall,
    scores,
    prec_grid,
    rec_grid,
    levels,
    labels,
    cmap_name=None,
    ax=None,
    legend_loc=None,
    equal_aspect=False
):
    if cmap_name is None:
        cmap_name = 'Blues'
    if legend_loc is None:
        # likely to be the best place for pr curve
        legend_loc = 'lower center'

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    else:
        fig = ax.get_figure()

    # create meshgrid for plotting
    RX, PY = np.meshgrid(rec_grid, prec_grid)
    colors = _get_color_hexes(cmap_name, n_colors=len(labels))

    # create contours
    ax.contourf(RX, PY, scores, levels=levels, colors=colors, alpha=0.8)  # type: ignore
    # plot precision recall
    ax.plot(recall, precision, c='black', alpha=0.6, zorder=10)  # type: ignore
    ax.set_xlabel('Recall', fontsize=14)  # type: ignore
    ax.set_ylabel('Precision', fontsize=14)  # type: ignore
    ax.tick_params(labelsize=12)  # type: ignore
    ax.set_ylim(0.0, 1.001)  # type: ignore
    ax.set_xlim(0.0, 1.001)  # type: ignore
    if equal_aspect:
        ax.set_aspect('equal')  # type: ignore
    # create custom legend with the correct colours and labels
    handles = _create_pr_legend(colors, labels)
    ax.legend(handles=handles, loc=legend_loc, fontsize=12)  # type: ignore
    fig.tight_layout()
    return ax


def _plot_pr_contours(
    n_bins,
    precision,
    recall,
    scores,
    bounds,
    levels,
    labels,
    cmap_name=None,
    ax=None,
    legend_loc=None,
    equal_aspect=False
):
    if cmap_name is None:
        cmap_name = 'Blues'
    if legend_loc is None:
        # likely to be the best place for pr curve
        legend_loc = 'lower center'

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    else:
        fig = ax.get_figure()

    # create meshgrid for plotting
    prec_grid = np.linspace(bounds[0], bounds[2], num=n_bins)
    rec_grid = np.linspace(bounds[3], bounds[5], num=n_bins)
    RX, PY = np.meshgrid(rec_grid, prec_grid)
    colors = _get_color_hexes(cmap_name, n_colors=len(labels))

    # create contours
    ax.contourf(RX, PY, scores, levels=levels, colors=colors, alpha=0.8)  # type: ignore
    # plot precision recall
    ax.scatter(  # type: ignore
        recall,
        precision,
        color='black',
        marker='x',
        s=50,
        lw=2,
        zorder=len(labels) + 1
    )
    ax.set_xlabel('Recall', fontsize=14)  # type: ignore
    ax.set_ylabel('Precision', fontsize=14)  # type: ignore
    ax.tick_params(labelsize=12)  # type: ignore

    ylim_lb, ylim_ub = ax.get_ylim()  # type: ignore
    ylim_lb = max(-0.001, ylim_lb)
    ylim_ub = min(1.001, ylim_ub)
    ax.set_ylim(ylim_lb, ylim_ub)  # type: ignore

    xlim_lb, xlim_ub = ax.get_xlim()  # type: ignore
    xlim_lb = max(-0.001, xlim_lb)
    xlim_ub = min(1.001, xlim_ub)
    ax.set_xlim(xlim_lb, xlim_ub)  # type: ignore

    if equal_aspect:
        ax.set_aspect('equal')  # type: ignore
    # create custom legend with the correct colours and labels
    handles = _create_pr_legend_scatter(colors, labels, (precision, recall))
    ax.legend(handles=handles, loc=legend_loc, fontsize=12)  # type: ignore
    fig.tight_layout()
    return ax
