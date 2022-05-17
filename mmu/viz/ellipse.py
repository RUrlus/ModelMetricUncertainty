import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from mmu.viz.utils import _get_color_hexes
from mmu.viz.utils import _create_pr_legend_scatter


def _get_radii_and_angle(cov_mat):
    # Angle and lambdas
    # based on https://cookierobotics.com/007/ :
    a = cov_mat[1, 1]
    c = cov_mat[0, 0]
    b = cov_mat[1, 0]
    lambda1 = (a+c)/2 + np.sqrt(((a-c)/2)**2 + b**2)
    lambda2 = (a+c)/2 - np.sqrt(((a-c)/2)**2 + b**2)

    def calculate_theta(lambda1, a, b, c):
        if b == 0 and a >= c:
            return 0.
        elif b == 0 and a < c:
            return np.pi / 2.
        else:
            return np.arctan2(lambda1 - a, b)

    theta = np.vectorize(calculate_theta)(lambda1, a, b, c)
    angle = theta / np.pi * 180

    # Radii of the ellipse
    recall_r = np.sqrt(lambda1)
    precision_r = np.sqrt(lambda2)

    return precision_r, recall_r, angle

def _plot_pr_ellipse(
    precision,
    recall,
    cov_mat,
    scales,
    labels,
    ax=None,
    cmap_name=None,
    legend_loc=None,
    equal_aspect=True
):
    if cmap_name is None:
        cmap_name = 'Blues'
    if legend_loc is None:
        # likely to be the best place for a single ellipse
        legend_loc = 'lower left'

    if ax is None:
        fig, ax = plt.subplots(figsize=(12,8))
    else:
        fig = ax.get_figure()

    colors = _get_color_hexes(cmap_name, n_colors=len(labels))

    # we multiply the radius by 2 because width and height are diameters
    scales *= 2
    prec_rad, rec_rad, angle = _get_radii_and_angle(cov_mat)

    n_levels = scales.size
    for i in range(n_levels):
        ellipse = Ellipse(
            (recall, precision),
            width=scales[i] * rec_rad,
            height=scales[i] * prec_rad,
            angle=angle,
            alpha=0.5,
            color=colors[i],
            zorder=n_levels - i
        )
        ax.add_patch(ellipse)  # type: ignore

    ax.scatter(  # type: ignore
        recall,
        precision,
        color='black',
        marker='x',
        s=50,
        lw=2,
        zorder=n_levels + 1
    )

    # we need to hide the right and top spines
    # in order to see the curve at the border:
    ylim_lb, ylim_ub = ax.get_ylim()  # type: ignore
    ylim_lb = max(-0.001, ylim_lb)
    ylim_ub = min(1.001, ylim_ub)
    ax.set_ylim(ylim_lb, ylim_ub)  # type: ignore

    xlim_lb, xlim_ub = ax.get_xlim()  # type: ignore
    xlim_lb = max(-0.001, xlim_lb)
    xlim_ub = min(1.001, xlim_ub)
    ax.set_xlim(xlim_lb, xlim_ub)  # type: ignore

    ax.spines['right'].set_visible(False)  # type: ignore
    ax.spines['top'].set_visible(False)  # type: ignore

    ax.set_xlabel('Recall', fontsize=14)  # type: ignore
    ax.set_ylabel('Precision', fontsize=14)  # type: ignore
    ax.tick_params(labelsize=12)  # type: ignore

    if equal_aspect:
        ax.set_aspect('equal')  # type: ignore
    # create custom legend with the correct colours and labels
    handles = _create_pr_legend_scatter(colors, labels, (precision, recall))
    ax.legend(handles=handles, loc=legend_loc, fontsize=12)  # type: ignore
    fig.tight_layout()

    return ax
