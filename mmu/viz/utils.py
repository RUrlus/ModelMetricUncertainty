import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.cm as plt_cm

def _set_plot_style(get_colors=False):
    plt.style.use('ggplot')
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['figure.max_open_warning'] = 0
    if get_colors:
        return [i['color'] for i in plt.rcParams['axes.prop_cycle']]  # type: ignore


def _get_color_hexes(cmap_name, n_colors=3, n_offset=3):
    if not isinstance(cmap_name, str):
        raise TypeError('`cmap_name` must be a str')

    t_colors = n_colors + 2 * n_offset + 1
    cmap = plt_cm.get_cmap(cmap_name, t_colors)
    hexes = [rgb2hex(cmap(i), keep_alpha=True) for i in range(cmap.N)]
    return hexes[n_offset:n_offset + n_colors][::-1]

def _create_pr_legend(c_hexes, labels):
    line = [Line2D([0], [0], color='black', alpha=0.6, label=r'$\hat{P}, \hat{R}$')]
    patches = [Patch(facecolor=c, edgecolor=c, label=l) for c, l in zip(c_hexes, labels)]
    return line + patches  # type: ignore
