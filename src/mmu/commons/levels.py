import numpy as np
from scipy.stats import chi2, norm


def _std_to_probability(stds):
    """Convert normal standard deviations to central probability mass."""
    return 2.0 * (norm.cdf(stds) - 0.5)


def _parse_levels(levels):
    """Return original levels and probabilities, retaining scalar/array shape."""
    if isinstance(levels, list | tuple):
        kinds = {np.asarray(level).dtype.kind for level in levels}
        if "f" in kinds and kinds & {"i", "u"}:
            msg = (
                "`levels` must not mix integers (sigmas) and floats (probabilities); "
                "convert sigmas to probabilities or use only one convention"
            )
            raise TypeError(msg)
    values = np.asarray(levels)
    if values.dtype.kind not in "iuf":
        msg = "`levels` must contain integers (sigmas) or floats (probabilities)"
        raise TypeError(msg)
    if values.ndim > 1 or values.size == 0:
        msg = "`levels` must be a scalar or a nonempty one-dimensional array"
        raise ValueError(msg)
    if not np.isfinite(values).all() or np.any(values <= 0):
        msg = "`levels` must be finite and positive"
        raise ValueError(msg)
    probabilities = _std_to_probability(values) if values.dtype.kind in "iu" else values.astype(np.float64)
    if np.any(probabilities >= 1.0):
        if values.dtype.kind == "f":
            msg = (
                "float `levels` are probabilities and must be strictly between 0 and 1; "
                "use integers to specify standard deviations"
            )
        else:
            msg = "integer `levels` (sigmas) are too large, their probability rounds to 1"
        raise ValueError(msg)
    return values, probabilities


def _plot_levels(levels):
    """Return sorted joint chi-squared thresholds and existing plot labels."""
    default = levels is None
    values, _ = _parse_levels([1, 2, 3] if default else levels)
    values = np.unique(values)
    _, probabilities = _parse_levels(values)
    if default:
        labels = [r"$1\sigma$ CI", r"$2\sigma$ CI", r"$3\sigma$ CI"]
    elif values.dtype.kind in "iu":
        labels = [f"{value}" + r"$\sigma$ CI" for value in values]
    else:
        labels = [f"{round(value * 100, 3)}% CI" for value in values]
    return chi2.ppf(probabilities, 2), labels
