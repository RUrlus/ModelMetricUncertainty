import numpy as np
import matplotlib.pyplot as plt

from mmu.viz.utils import _get_color_hexes
from mmu.viz.utils import _create_pr_legend


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
    ax.legend(handles=_create_pr_legend(colors, labels), loc=legend_loc)  # type: ignore
    fig.tight_layout()
    return ax
