import numpy as np
from mmu.models.base import ConfusionMatrixBase
from mmu.stan import _bnn_code


class BetaBinomialConfusionMatrix(ConfusionMatrixBase):
    """Model a confusion matrix using a Beta-Binomial."""
    def __init__(self, random_state=None):
        """Initialise the class."""
        self.code = _bnn_code
        self.prior_vars = ['phi', 'tpr', 'tnr']
        self.post_vars = ['y_hat']
        self.predictive_var = 'y_hat'
        self._set_random_state(random_state)

    def fit_predict(
        self,
        X,
        y=None,
        n_samples=10000,
        sample_factor=1.5,
        total_count=None,
        n_cores=None,
        n_warmup=500,
        threshold=0.5,
        sampling_kwargs=None,
    ):
        """Fit Beta Binomial model and sample from the posterior.

        Parameters
        ----------
        X : np.ndarray[np.int64, float64]
            the confusion matrix, should have shape (N, 4) where N is
            the number confusion matrices if ``y``=None, else the probabilities
            for the test set used to compute the confusion matrix
        y : np.ndarray[bool, np.int64], default=None
            the labels for the probabilities, must be set if X are probabilities
            rather than the confusion matrix
        n_samples : int, default=10000
            Number of samples to draw, used both during fit as well as draw
            from posterior.
        sample_factor : float, int, default=1.5
            a factor to increase the number of samples drawn from the posterior
            which increases the probability that the number non-divergent
            samples is as least as big as n_samples.
        total_count : int, default=None
            the number of entries a single sampled confusion matrix should have,
            if None the sum of the first row of X is used.
        n_cores : int, default=None
            the number of cores to use during fit and sampling. Default's
            4 cores if present other the number of cores available. Setting it
            to -1 will use all cores available - 1.
        n_warmup : int, default=500
            number of warmup itertations used during fitting of Stan model
        threshold : float, default=0.5
            the classification threshold used to compute the confusion matrix
            is ``y`` is not None
        sampling_kwargs : dict, default=None
            keyword arguments passed to sample function of Stan

        """
        if y is not None:
            X = self._compute_confusion_matrix(
                X, y, threshold, ensure_1d=True
            )

        else:
            X = self._check_X(X).flatten()

        if total_count is None:
            total_count = int(X.sum())
        elif isinstance(total_count, (int, float)):
            total_count = int(total_count)
        else:
            raise TypeError('``total_count`` must be `None` or `int`')

        data = {
            'n': total_count,
            'y': X.flatten(),
            'tpr_prior': np.ones(2),
            'tnr_prior': np.ones(2),
            'phi_prior': np.ones(2),
        }
        return self._fit_predict(
            data, n_samples, sample_factor, n_cores, n_warmup, sampling_kwargs
        )
