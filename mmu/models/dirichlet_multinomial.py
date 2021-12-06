from mmu.models.base import ConfusionMatrixBase
from mmu.stan import _dm_code
from mmu.stan import _dm_multi_code


class DirichletMultinomialConfusionMatrix(ConfusionMatrixBase):
    """Model a confusion matrix using a Dirichlet-multinomial."""
    def __init__(self, random_state=None):
        """Initialise the class."""
        self.code = _dm_code
        self.prior_vars = ['phi', 'theta']
        self.post_vars = ['y_hat', 'theta_hat']
        self.predictive_var = 'y_hat'
        # exclude transformed variables
        self._set_random_state(random_state)

    def fit_predict(
        self,
        X,
        y=None,
        v=2.0,
        alpha=1.0,
        n_samples=10000,
        sample_factor=1.5,
        total_count=None,
        n_cores=None,
        n_warmup=500,
        threshold=0.5,
        sampling_kwargs=None,
    ):
        """Fit Dirichlet-Multinomial model and sample from the posterior.

        Parameters
        ----------
        X : np.ndarray[np.int64, float64]
            the confusion matrix, should have shape (N, 4) where N is
            the number confusion matrices if ``y``=None, else the probabilities
            for the test set used to compute the confusion matrix
        y : np.ndarray[bool, np.int64], default=None
            the labels for the probabilities, must be set if X are probabilities
            rather than the confusion matrix
        v : float, default=2
            prior strength, should be value between 1 and 2.
        alpha : float, default=1
            concentration parameter hyper-prior dirichlet
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

        if isinstance(v, int):
            v = float(v)
        elif not isinstance(v, float):
            raise TypeError('``v`` must be a float.')

        if isinstance(alpha, int):
            alpha = float(alpha)
        elif not isinstance(alpha, float):
            raise TypeError('``alpha`` must be a float.')

        if total_count is None:
            total_count = int(X.sum())
        elif isinstance(total_count, (int, float)):
            total_count = int(total_count)
        else:
            raise TypeError('``total_count`` must be `None` or `int`')

        data = {
            'alpha': alpha,
            'v': v,
            'total_count': total_count,
            'y': X.flatten(),
        }
        return self._fit_predict(
            data, n_samples, sample_factor, n_cores, n_warmup, sampling_kwargs
        )


class DirichletMultinomialMultiConfusionMatrix(ConfusionMatrixBase):
    """Model a confusion matrix using a Dirichlet-multinomial."""
    def __init__(self, random_state=None):
        """Initialise the class."""
        self.code = _dm_multi_code
        self.prior_vars = ['phi', 'theta']
        self.post_vars = ['y_hat', 'theta_hat']
        self.predictive_var = 'y_hat'
        # exclude transformed variables
        self._set_random_state(random_state)

    def fit_predict(
        self,
        X,
        y=None,
        v=2.0,
        alpha=1.0,
        n_samples=10000,
        sample_factor=1.5,
        total_count=None,
        n_cores=None,
        n_warmup=500,
        threshold=0.5,
        sampling_kwargs=None,
    ):
        """Fit Dirichlet-Multinomial model over multiple observations and sample from the posterior.

        Parameters
        ----------
        X : np.ndarray[np.int64, float64]
            the confusion matrix, should have shape (N, 4) where N is
            the number confusion matrices if ``y``=None, else the probabilities
            for the test set used to compute the confusion matrix
        y : np.ndarray[bool, np.int64], default=None
            the labels for the probabilities, must be set if X are probabilities
            rather than the confusion matrix
        v : float, default=2
            prior strength, should be value between 1 and 2.
        alpha : float, default=1
            concentration parameter hyper-prior dirichlet
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
                X, y, threshold, ensure_1d=False
            )

        else:
            X = self._check_X(X)

        if isinstance(v, int):
            v = float(v)
        elif not isinstance(v, float):
            raise TypeError('``v`` must be a float.')

        if isinstance(alpha, int):
            alpha = float(alpha)
        elif not isinstance(alpha, float):
            raise TypeError('``alpha`` must be a float.')

        if total_count is None:
            total_count = int(X.sum())
        elif isinstance(total_count, (int, float)):
            total_count = int(total_count)
        else:
            raise TypeError('``total_count`` must be `None` or `int`')

        data = {
            'v': v,
            'alpha': alpha,
            'N': int(X.shape[0]),
            'total_count': total_count,
            'y': X,
        }
        return self._fit_predict(
            data, n_samples, sample_factor, n_cores, n_warmup, sampling_kwargs
        )
