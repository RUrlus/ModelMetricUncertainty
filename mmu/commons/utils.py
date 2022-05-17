import numpy as np


def generate_data(
    n_samples=10000, n_sets=1, score_dtype=np.float64, yhat_dtype=np.int64,
    y_dtype=np.int64, random_state=None
):
    """Generate data from a logistic process.

    Parameters
    ----------
    n_samples : int, default=10000
        number of samples to generate for each set
    n_sets : int, default=1
        number of sets to generate, each containing ``n_samples``
    score_dtype : np.dtype, default=np.float64
        dtype for generated probabilities
    yhat_dtype : np.dtype, default=np.int64
        dtype for generated estimated labels
    y_dtype : np.dtype, default=np.int64
        dtype for generated true labels

    Returns
    -------
    score : np.ndarray
        array of shape (n_samples, n_sets) with generated
        probabilities of dtype ``proba_dtype``
    yhat : np.ndarray
        array of shape (n_samples, n_sets) with generated
        estimated labels of dtype ``yhat_dtype``
    y : np.ndarray[]
        array of shape (n_samples, n_sets) with generated
        true labels of dtype ``y_dtype``

    """
    rng = np.random.default_rng(random_state)
    t_samples = n_samples * n_sets
    score = rng.beta(5, 3, size=t_samples).astype(score_dtype)  # type: ignore
    yhat = np.rint(score).astype(yhat_dtype)
    y = rng.binomial(1, score, t_samples).astype(y_dtype)  # type: ignore
    return (
        score.reshape(n_samples, n_sets),
        yhat.reshape(n_samples, n_sets),
        y.reshape(n_samples, n_sets),
    )


class SeedGenerator:
    """Generate sklearn compatible seeds/random_state as uint32's."""
    def _set_seeds(self):
       self.seeds = self.rng.integers(
            1000,
            np.iinfo(np.uint32).max,
            100,
            dtype=np.uint32
        )

    def __init__(self, seed=None):
        """Initialise the class."""
        self.rng = np.random.default_rng(seed)
        self._set_seeds()
        self._idx = 0

    def __call__(self):
        """Draw seed from rng.

        Returns
        -------
        seed : uint64
            seed drawn from rng
        """
        seed = self.seeds[self._idx]
        self._idx += 1
        if self._idx >= 99:
            self._set_seeds()
        return seed
