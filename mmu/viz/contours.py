import numpy as np
import matplotlib.pyplot as plt

from mmu.viz.utils import _get_color_hexes
from mmu.viz.utils import _create_pr_legend
from mmu.viz.utils import _create_pr_legend_scatter


def _plot_curve_contours(
    y,
    x,
    scores,
    y_grid,
    x_grid,
    levels,
    labels,
    cmap,
    ax,
    alpha,
    legend_loc,
    equal_aspect,
    limit_axis,
    y_label='',
    x_label='',
):
    if cmap is None:
        cmap = "Blues"
    if legend_loc is None:
        # likely to be the best place for pr curve
        legend_loc = "lower center"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    # create meshgrid for plotting
    RX, PY = np.meshgrid(x_grid, y_grid)
    colors = _get_color_hexes(cmap, n_colors=len(labels), keep_alpha=True)

    levels = [0.0] + levels.tolist()
    # create contours
    ax.contourf(RX, PY, scores, levels=levels, colors=colors, alpha=alpha)  # type: ignore
    # plot precision recall
    ax.plot(x, y, c="black", alpha=0.6, zorder=10)  # type: ignore
    ax.set_xlabel(x_label, fontsize=14)  # type: ignore
    ax.set_ylabel(y_label, fontsize=14)  # type: ignore
    ax.tick_params(labelsize=12)  # type: ignore
    if limit_axis:
        ylim_lb, ylim_ub = ax.get_ylim()  # type: ignore
        ylim_lb = max(-0.001, ylim_lb)
        ylim_ub = min(1.001, ylim_ub)
        ax.set_ylim(ylim_lb, ylim_ub)  # type: ignore

        xlim_lb, xlim_ub = ax.get_xlim()  # type: ignore
        xlim_lb = max(-0.001, xlim_lb)
        xlim_ub = min(1.001, xlim_ub)
        ax.set_xlim(xlim_lb, xlim_ub)  # type: ignore
    if equal_aspect:
        ax.set_aspect("equal")  # type: ignore
    # create custom legend with the correct colours and labels
    handles = _create_pr_legend(colors, labels)
    ax.legend(handles=handles, loc=legend_loc, fontsize=12)  # type: ignore
    fig.tight_layout()
    return ax, handles


def _plot_contours(
    n_bins,
    y,
    x,
    scores,
    bounds,
    levels,
    labels,
    cmap,
    ax,
    alpha,
    legend_loc,
    equal_aspect,
    limit_axis,
    y_label='',
    x_label='',
):
    if cmap is None:
        cmap = "Blues"
    if legend_loc is None:
        # likely to be the best place for pr curve
        legend_loc = "lower left"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    # create meshgrid for plotting
    y_grid = np.linspace(bounds[0], bounds[1], num=n_bins)
    x_grid = np.linspace(bounds[2], bounds[3], num=n_bins)
    RX, PY = np.meshgrid(x_grid, y_grid)
    colors, c_marker = _get_color_hexes(
        cmap, n_colors=len(labels), return_marker=True, keep_alpha=True
    )

    # add zero level to contours
    levels = [0.0] + levels.tolist()
    # create contours
    ax.contourf(RX, PY, scores, levels=levels, colors=colors, alpha=alpha)  # type: ignore
    # plot precision recall
    ax.scatter(  # type: ignore
        x,
        y,
        color=c_marker,
        marker="x",
        s=50,
        lw=2,
        zorder=len(labels) + 1,
    )
    ax.set_xlabel(x_label, fontsize=14)  # type: ignore
    ax.set_ylabel(y_label, fontsize=14)  # type: ignore
    ax.tick_params(labelsize=12)  # type: ignore

    if limit_axis:
        ylim_lb, ylim_ub = ax.get_ylim()  # type: ignore
        ylim_lb = max(-0.001, ylim_lb)
        ylim_ub = min(1.001, ylim_ub)
        ax.set_ylim(ylim_lb, ylim_ub)  # type: ignore

        xlim_lb, xlim_ub = ax.get_xlim()  # type: ignore
        xlim_lb = max(-0.001, xlim_lb)
        xlim_ub = min(1.001, xlim_ub)
        ax.set_xlim(xlim_lb, xlim_ub)  # type: ignore

    if equal_aspect:
        ax.set_aspect("equal")  # type: ignore
    # create custom legend with the correct colours and labels
    handles = _create_pr_legend_scatter(colors, c_marker, labels, (y, x))
    ax.legend(handles=handles, loc=legend_loc, fontsize=12)  # type: ignore
    fig.tight_layout()
    return ax, handles
