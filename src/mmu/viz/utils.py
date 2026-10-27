import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def _set_plot_style(get_colors=False):
    plt.style.use("ggplot")
    plt.rcParams["text.color"] = "black"
    plt.rcParams["figure.max_open_warning"] = 0
    if get_colors:
        return [i["color"] for i in plt.rcParams["axes.prop_cycle"]]  # type: ignore
    return None


def _get_color_hexes(cmap, n_colors=3, n_offset=3, return_marker=False, keep_alpha=True):
    if not isinstance(cmap, str):
        msg = "`cmap` must be a str"
        raise TypeError(msg)

    t_colors = n_colors + 2 * n_offset + 1
    cmap = plt.get_cmap(cmap, t_colors)
    hexes = [rgb2hex(cmap(i), keep_alpha=keep_alpha) for i in range(cmap.N)]
    if return_marker:
        return hexes[n_offset : n_offset + n_colors][::-1], hexes[-3]
    return hexes[n_offset : n_offset + n_colors][::-1]


def _create_pr_legend(c_hexes, labels, line_color="black", y_label="Precision", x_label="Recall", label=None, linestyle=None):
    # Map metric names to their LaTeX symbols
    label_to_symbol = {
        "Precision": r"\hat{P}",
        "Recall": r"\hat{R}",
        "TPR": r"\widehat{TPR}",
        "FPR": r"\widehat{FPR}",
        "PPN": r"\widehat{PPN}",
    }
    if label is not None:
        # Use custom label if provided
        curve_label = label
    else:
        # Fallback to existing symbol mapping
        y_sym = label_to_symbol.get(y_label, rf"\widehat{{{y_label}}}")
        x_sym = label_to_symbol.get(x_label, rf"\widehat{{{x_label}}}")
        curve_label = rf"${x_sym}, {y_sym}$"

    linestyle_arg = linestyle if linestyle is not None else "-"
    line = [Line2D([0], [0], color=line_color, alpha=0.6, linestyle=linestyle_arg, label=curve_label)]
    patches = [Patch(facecolor=c, edgecolor=c, label=l) for c, l in zip(c_hexes, labels)]
    return line + patches  # type: ignore


def _create_pr_legend_scatter(c_hexes, c_marker, labels, vals, y_label="Precision", x_label="Recall", label=None):
    # Map metric names to their LaTeX symbols
    label_to_symbol = {
        "Precision": r"\hat{P}",
        "Recall": r"\hat{R}",
        "TPR": r"\widehat{TPR}",
        "FPR": r"\widehat{FPR}",
        "PPN": r"\widehat{PPN}",
    }
    if label is not None:
        # Use custom label if provided
        marker_label = label
    else:
        # Fallback to existing symbol mapping
        y_sym = label_to_symbol.get(y_label, rf"\widehat{{{y_label}}}")
        x_sym = label_to_symbol.get(x_label, rf"\widehat{{{x_label}}}")

        if vals:
            marker_label = rf"${x_sym}$" + f"={round(vals[1], 3)}, " + rf"${y_sym}$" + f"={round(vals[0], 3)}"
        else:
            marker_label = rf"${x_sym}$, ${y_sym}$"

    line = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            markeredgewidth=3,
            markersize=8,
            marker="x",
            color=c_marker,
            alpha=0.6,
            label=marker_label,
        )
    ]
    patches = [Patch(facecolor=c, edgecolor=c, label=l) for c, l in zip(c_hexes, labels)]
    return line + patches  # type: ignore
