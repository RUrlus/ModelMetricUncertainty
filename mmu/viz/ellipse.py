import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2, norm

def _get_scaling_factor(n_std=None, alpha=0.95):
    """Compute critical value given a number of std devs or alpha."""
    # Get the scale for 2 degrees of freedom confidence interval
    # We use chi2 because the equation of an ellipse is a sum of squared variable,
    # more details here https://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/
    if n_std is not None:
        alpha = 2. * (norm.cdf(n_std) - 0.5)
    chi2_quantile = chi2.ppf(alpha, 2)
    return np.sqrt(chi2_quantile), alpha


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
    norm_nstd=None,
    alpha=0.95,
    lim=1.0,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    # we multiply the radius by 2 because width and height are diameters
    scale, alpha = _get_scaling_factor(norm_nstd, alpha) * 2
    rec_rad, prec_rad, angle = _get_radii_and_angle(cov_mat)

    ellipse = Ellipse(
        (recall, precision),
        width=scale * rec_rad,
        height=scale * prec_rad,
        angle=angle,
        alpha=0.50,
        color='C0'
    )
    ax.add_patch(ellipse)

    ax.scatter(
        recall,
        precision,
        label=r'$\hat{P}$, $\hat{R}$',
        color='black',
        marker='x',
        s=80,
        lw=3,
        zorder=10
    )
    ax.set_xlim((0, lim))
    ax.set_ylim((0, lim))

    # If limit is <=1.0 we don't need rectangle, but we need to hide the right and top spines
    # in order to see the curve at the border:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall {round(alpha * 100, 3)}% CI')

    return fig
