import copy
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.special import erfinv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement


def _compute_std_dev_from_quant(x, q=0.99):
    """Compute the standard deviation based on quantile.

    This function computes the standard deviation of a zero
    centred normal distribution of which the `q`th quantile is equal to x.

    Parameters
    ----------
    x : float
        the value of the `q`th quantile
    q : float, optional
        the quantile which should have value `x`, default is 0.99

    Returns
    -------
    float
        the standard deviation of a zero-centred normal distribution
        which `q`th quantile has value `x`

    """
    SQRT_2 = 1.414213562373095048801688724209698079
    return x / (SQRT_2 * erfinv(2 * q - 1))

def _generate_hypercube(samples, dimensions, rng):
    """Returns distinct binary samples of length dimensions.

    References
    ----------
    Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
             G. Louppe, J. Nothman
    License: BSD 3 clause
    Url:     https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_samples_generator.py

    """
    if dimensions > 30:
        return np.hstack(
            [
                rng.randint(2, size=(samples, dimensions - 30)),
                _generate_hypercube(samples, 30, rng),
            ]
        )
    out = sample_without_replacement(2 ** dimensions, samples, random_state=rng).astype(
        dtype=">u4", copy=False
    )
    out = np.unpackbits(out.view(">u1")).reshape((-1, 32))[:, -dimensions:]
    return out


class LogisticGenerator:
    """Generates binary classification models based on LogisticRegression.

    Parameters
    ----------

    We use the same X; y for the test set

    """

    def __init__(
        self,
        betas=None,
        train_frac=0.7,
        noise_sigma=1.0,
        random_state=None
    ):
        """Initialise the class."""
        # statefull random number generator
        self.betas = betas or np.array((0.5, 1.2))
        self.train_frac = train_frac
        self.noise_sigma = noise_sigma
        self._pgen = np.random.default_rng(random_state)
        self._pseed = self._pgen.integers(low=0, high=np.iinfo(np.int64).max)
        self._gen = self._get_generator()

    def _get_generator(self, reset=True):
        if reset:
            return np.random.default_rng(self._pseed)
        seed = self._pgen.integers(low=0, high=np.iinfo(np.int64).max)
        return np.random.default_rng(seed)

    def _reset_generator(self):
        """Reset generator to it's original state."""
        self._gen = np.random.default_rng(self._pseed)

    def _generate_X(self, n_samples):
        X = np.ones((n_samples, 2))
        X[:, 1] = self._gen.uniform(-10., 10.1, size=n_samples)
        return X

    def _compute_linear_estimate(self, X, betas):
        """Generate noise-free input samples."""
        return X.dot(betas)[:, None]

    def _generate_train_test_index(self, n_samples, n_models, train_frac, dummy):
        """Create index for train_test splits."""
        index = np.arange(n_samples)
        n_train = int(np.floor(train_frac * n_samples))

        if dummy:
            train_idx = np.tile(index[:n_train][:, None], (n_train, n_models))
            test_idx = np.tile(index[n_train:][:, None], (n_train, n_models))
            return train_idx, test_idx, n_train

        indices = np.empty((n_samples, n_models), dtype=np.int64)
        for i in range(n_models):
            lgen = self._get_generator(reset=False)
            indices[:, i] = lgen.choice(index, size=(n_samples), replace=False)
        return indices[:n_train], indices[n_train:], n_train

    def _fit_and_predict_proba(self, X_train, y_train, X_test):
        """Fit a LR model on the various X_trains.

        Parameters
        ----------
        X_train : np.ndarray
            the train set input features, where the third dimension represents
            the different models
        y_train : np.ndarray
            the train set labels, where the second dimension represents
            the different models

        Returns
        -------
        models : List[LogisticRegression]
            a list containing the fitted models
        train_proba : np.ndarray
            probabilities for train set
        test_proba : np.ndarray
            probabilities for test set

        """
        n_models = y_train.shape[1]
        models = []
        train_proba = np.zeros((X_train.shape[0], n_models))
        test_proba = np.zeros((X_test.shape[0], n_models))
        for i in range(n_models):
            model = LogisticRegression(penalty='none')
            fit = model.fit(X_train[:, i, :], y_train[:, i].flatten())
            train_proba[:, i] = fit.predict_proba(X_train[:, i, :])[:, 1]
            test_proba[:, i] = fit.predict_proba(X_test[:, i, :])[:, 1]
            models.append(fit)
        return models, train_proba, test_proba

    def transform(
        self,
        n_models=100,
        n_samples=10000,
        enable_sample_noise=True,
        enable_measurement_noise=False,
        betas=None,
        train_frac=None,
        noise_sigma=None,
    ):
        """Generate binary classification models."""
        betas = betas or self.betas
        train_frac = train_frac or self.train_frac
        noise_sigma = noise_sigma or self.noise_sigma

        # ensure consistent behaviour for samples
        self._reset_generator()
        # ground truth inputs
        X = self._generate_X(n_samples)
        linear_estimate = self._compute_linear_estimate(X, betas)
        gt_proba = expit(linear_estimate)
        gt_y = self._gen.binomial(1, gt_proba)

        ground_truth = pd.DataFrame()
        ground_truth['proba'] = gt_proba.flatten()
        ground_truth['y'] = gt_y.flatten()
        ground_truth['linear_estimate'] = linear_estimate.flatten()
        ground_truth['X_0'] = X[:, 0]
        ground_truth['X_1'] = X[:, 1]

        if enable_sample_noise:
            train_idx, test_idx, n_train = self._generate_train_test_index(
                n_samples, n_models, train_frac, dummy=False
            )
        else:
            train_idx, test_idx, n_train = self._generate_train_test_index(
                n_samples, n_models, train_frac, dummy=True
            )

        y_test = gt_y[test_idx]
        X_test = X[test_idx]
        X_train = X[train_idx]

        if enable_measurement_noise:
            noise = self._gen.normal(0, noise_sigma, (n_samples, n_models))
            linear_estimates = linear_estimate + noise
            probas = expit(linear_estimates)
            y_train = self._gen.binomial(1, probas[train_idx])
        else:
            y_train = self._gen.binomial(1, gt_proba[train_idx])

        models, train_proba, test_proba = self._fit_and_predict_proba(
            X_train, y_train, X_test
        )
        train_mask = np.ones(n_samples, dtype=np.bool_)
        train_mask[n_train:] = False;

        X = np.vstack((X_train, X_test))
        labels = np.vstack((y_train, y_test))
        probas = np.vstack((train_proba, test_proba))
        indices = np.vstack((train_idx, test_idx))
        return train_mask, indices, labels, probas, X, models, ground_truth


class ModelGenerator:
    """Generates binary classification models that allow different types of
    uncertainty to be injected.

    This initially creates clusters of points normally distributed (std=1)
    about vertices of an ``n_features``-dimensional hypercube with sides of
    length ``2*class_sep`` and assigns an equal number of clusters to each
    class.

    Parameters
    ----------
    n_features : int, default=4
        The number of features in X.
    n_clusters_per_class : int, default=2
        The number of clusters per class in X.
    weights : None, str, float or array_like, default='random'
        weights can be used to influence the class imbalance or the
        number of samples per cluster.
        - If None each cluster will contain an even number of samples.
        - If a float the weights is taken to be the percentage of samples
        in the positive class. The number of samples per cluster within a class
        will be equal.
        - If array_like the weights are the per cluster weights and must have
          length equal to ``n_clusters_per_class`` * 2. The order is taken to
          be [class_0_cluster_0, class_0_cluster_1, ..., class_1_cluster_0]
        - If 'random' the number of samples per cluster is sampled from a
        Dirichlet with parameters ``alpha_weights``.
    alpha_weights : float, array_like, default=15.
        The alpha parameters for the Dirichlet used to sample the
        cluster weights if ``weights`` is random. If a float the
        value is used for each cluster. If array_like the length
        of the input must be equal to ``n_clusters_per_class`` * 2.
        Note this parameter is ignored when ``weights``!='random'
    class_sep : float, default=1.0
        The factor multiplying the hypercube size.  Larger values spread
        out the clusters/classes and make the classification task easier.
    train_frac : float, default=0.7
        The number of samples to use for training the models
    label_flip : float, default=0.01
        The probability of a label being flipped
    input_shift : float, default=0.1
        How much to shift the inputs, drawn from Normal(0, sigma) where
        sigma is such that the ``quantile`` is equal to ``input_shift``.
        The shift is sampled independently for each feature and cluster
        Note that the shift occurs before the scaling
    input_scale : float, default=0.1
        How much to scale the inputs, drawn from Normal(1, sigma) where
        sigma is such that the ``quantile`` is equal to ``input_scale``.
        The scale is sampled independently for each feature and cluster
        Note that scaling is done after shifting
    model_rotation : float, default=0.3
        How much to rotate the sigmoidal representation of the model.
        Higher values cause models that have flatter or steeper sigmoidal ouputs.
        Each model is assigned a value sampled from Normal(1, sigma) where
        sigma is such that the ``quantile`` is equal to ``model_rotation``.
        Note that rotation is performed before shifting.
    model_shift : float, default=0.3
        How much to horizontally shift the sigmoidal representation of the model.
        Each model is assigned a value sampled from Normal(0, sigma) where
        sigma is such that the ``quantile`` is equal to ``model_shift``.
        Note that shifting is performed after rotation.
    quantile : float, default=0.99
        the quantile which should have values for the above noise parameters.
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    References
    ----------
    This class is a modified version of the sklearn.datasets.make_classification
    function.
    Authors: B. Thirion, G. Varoquaux, A. Gramfort, V. Michel, O. Grisel,
             G. Louppe, J. Nothman
    License: BSD 3 clause

    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.

    """

    def __init__(
        self,
        n_features=4,
        n_clusters_per_class=2,
        weights='random',
        alpha_weights=15.,
        class_sep=1.0,
        train_frac=0.7,
        label_flip=0.01,
        input_shift=0.5,
        input_scale=0.2,
        model_rotation=0.3,
        model_shift=0.3,
        quantile=0.99,
        random_state=None
    ):
        self.n_features = n_features
        # clusters
        self.n_clusters_per_class = n_clusters_per_class
        self.n_clusters = n_clusters_per_class * 2

        self._random_samples_per_class = False
        if weights is None:
            self.weights = None
        elif isinstance(weights, float):
            self.weights = weights
        elif isinstance(weights, str):
            if weights not in {'random', 'rand'}:
                raise ValueError('``weights`` is unrecognised string value')
            self.weights = weights
            self._random_samples_per_class = True
        elif isinstance(weights, (list, tuple, np.ndarray)):
            if len(weights) != self.n_clusters:
                raise ValueError(
                    '``weights`` must be a single value or of length '
                    '2 * ``n_clusters_per_class``'
                )
            self.weights = weights
        else:
            raise TypeError('``weights`` is not of the correct type')
        if isinstance(alpha_weights, (float, int)):
            self.alpha_weights = np.repeat(alpha_weights, self.n_clusters)
        elif isinstance(alpha_weights, (tuple, list, np.ndarray)):
            if len(alpha_weights) == self.n_clusters:
                self.alpha_weights = alpha_weights
            else:
                raise ValueError(
                    '``alpha_weights`` must be a single value or of length '
                    '2 * ``n_clusters_per_class``'
                )
        else:
            raise TypeError('``alpha_weights`` is not of the correct type')

        # parameters
        self.class_sep = class_sep
        self.train_frac = train_frac
        self.label_flip = label_flip
        self.input_shift = input_shift
        self.input_scale = input_scale
        self.model_shift = model_shift
        self.model_rotation = model_rotation
        self.noise_quantile = quantile

        # statefull random number generator
        self._gen = check_random_state(random_state)
        self._gen_orig = copy.deepcopy(self._gen)

    def _generate_centroids(self):
        """Generate the centroids which position the clusters in X.
        The clusters are placed on a hypercube.

        """
        # Build the polytope whose vertices become cluster centroids
        self.centroids = (
            _generate_hypercube(self.n_clusters, self.n_features, self._gen)
            .astype(float, copy=False)
        )
        self.centroids *= 2 * self.class_sep
        self.centroids -= self.class_sep

    def _generate_covariance_matrices(self):
        """Create a covariance matrix for each cluster.

        Note that these
        """
        self.cov_mats_ = np.zeros((
            self.n_features, self.n_features, self.n_clusters
        ))
        for k in range(self.n_clusters):
            mat = 2 * self._gen.rand(self.n_features, self.n_features) - 1
            # ensure standard normal marginals
            np.fill_diagonal(mat, 1.0)
            # make positive semi-definte
            self.cov_mats_[:, :, k] = mat.T * mat

    def _compute_n_samples_per_cluster(
        self, n_samples, weights, alpha_weights, random_samples_per_class, unweighted=False
    ):
        """Compute the number of samples in each cluster

        Parameters
        ----------
        n_samples : int
            the number of samples to generate

        Returns
        -------
        n_samples_per_cluster : List[int]
            number of samples per cluster

        """
        if unweighted or weights is None:
            weights = np.zeros(self.n_clusters) + (1 / self.n_clusters)
        elif random_samples_per_class:
            weights = self._gen.dirichlet(alpha_weights)
        elif isinstance(weights, (list, tuple, np.ndarray)):
            pass
        else:
            weights = np.repeat((1 - weights, weights), 2) / 2

        # Distribute samples among clusters by weight
        expected = weights * n_samples
        n_samples_per_cluster = np.floor(expected).astype(int, copy=False)
        diff = n_samples - n_samples_per_cluster.sum()

        # handle n_samples and weights that result in to few samples being
        # assigned
        if diff > 0:
            assign_order = np.argsort((expected - n_samples_per_cluster))
            for _ in range(diff):
                n_samples_per_cluster[assign_order[-1]] += 1
                assign_order = np.argsort((expected - n_samples_per_cluster))

        # the generator expects the samples to class to alternate between
        # the positive and negative class e.g.
        # [class_0_cluster_0, class_1_cluster_0, class_1_cluster_1, ...]
        # the below reorders the samples for this purpose
        n_samples_per_cluster = np.hstack((
            n_samples_per_cluster[:self.n_clusters_per_class, None],
            n_samples_per_cluster[self.n_clusters_per_class:, None]
        )).flatten()

        return n_samples_per_cluster

    def _generate_ground_truth_inputs(
        self, n_samples, n_samples_per_cluster, seed=None
    ):
        """Generate the ground truth values of X.

        Parameters
        ----------
        n_samples : int, default=250000
            the number of samples to generate
        n_samples_per_cluster : np.ndarray[int]
            the number of samples per cluster
        seed : int, default=None
            the ensure reproducible generation of X

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            The generated samples.

        y : ndarray of shape (n_samples,)
            The integer labels for class membership of each sample.

        """
        y = np.zeros(n_samples, dtype=int)
        # Initially draw features from the standard normal
        if seed is not None:
            # use local generator to be able to reproduce X exactly
            local_generator = np.random.default_rng(seed)
            X = local_generator.normal(0, 1, size=(n_samples, self.n_features))
        else:
            X = self._gen.randn(n_samples, self.n_features)

        # Create each cluster; a variant of make_blobs
        stop = 0
        for k in range(self.n_clusters):
            start, stop = stop, stop + n_samples_per_cluster[k]
            # assign labels
            y[start:stop] = k % 2
            # slice a view of the cluster
            X_k = X[start:stop, :]

            # introduce random covariance
            X_k[...] = np.dot(X_k, self.cov_mats_[:, :, k])
            # shift the cluster to a vertex
            X_k += self.centroids[k, :]
        return X, y

    def _generate_shifted_and_scaled_inputs(
        self,
        n_samples,
        n_samples_per_cluster,
        input_shift,
        input_scale,
        quantile,
        seed
    ):
        """Generate noisy input values of X.

        The seed is used to ensure the same values are drawn for both
        the noise and ground truth inputs before the covariance matrices

        Inputs are drawn from a Normal distribution with:
            a mean drawn from Normal(0, sigma) where sigma is such that the
            ``quantile`` is equal to ``input_shift``
            a std dev that is equal Normal(1, sigma) where sigma is such
            that the ``quantile`` is equal to ``input_scale``

        Parameters
        ----------
        n_samples : int, default
            the number of samples to generate
        n_samples_per_cluster : np.ndarray[int]
            the number of samples per cluster
        seed : int, default=None
            the ensure reproducible generation of X

        Returns
        -------
        X_alt : ndarray of shape (n_samples, n_features)
            The generated samples.

        """
        # Allocate memory for X
        local_generator = np.random.default_rng(seed)
        X = local_generator.normal(0, 1, size=(n_samples, self.n_features))

        input_shift_sigma = _compute_std_dev_from_quant(input_shift, quantile)
        input_scale_sigma = _compute_std_dev_from_quant(input_scale, quantile)

        shift = self._gen.normal(
            0, input_shift_sigma, size=(self.n_clusters, self.n_features)
        )
        scale = self._gen.normal(
            1, input_scale_sigma, size=(self.n_clusters, self.n_features)
        )

        # Create each cluster
        stop = 0
        for k in range(self.n_clusters):
            start, stop = stop, stop + n_samples_per_cluster[k]
            # assign labels
            # slice a view of the cluster
            X_k = X[start:stop, :]
            # scale the data
            X_k *= scale[k, :]
            # shift the data
            X_k += shift[k, :]

            # introduce random covariance
            X_k[...] = np.dot(X_k, self.cov_mats_[:, :, k])
            # shift the cluster to a vertex
            X_k += self.centroids[k, :]
        return X

    def _split_data(self, X, y, n_models, n_samples, train_frac):
        """Split dataset into train, test sets with random split for each model.

        Parameters
        ----------
        X : np.ndarray
            input data
        y : np.ndarray
            input labels
        n_models : int
            number of models to generate
        n_samples : int
            number of samples to generate
        train_frac : float
            fraction of samples to allocate to train set

        Returns
        -------
        X_trains : np.ndarray
            The train data randomly split for each model
        X_tests : np.ndarray
            The test data randomly split for each model
        y_trains : np.ndarray
            The train labels randomly split for each model
        y_tests : np.ndarray
            The test labels randomly split for each model

        """
        train_samples = int(np.floor(n_samples * train_frac))
        test_samples = n_samples - train_samples

        X_trains = np.empty(shape=(train_samples, self.n_features, n_models))
        X_tests = np.empty(shape=(test_samples, self.n_features, n_models))

        y_trains = np.empty(shape=(train_samples, n_models))
        y_tests = np.empty(shape=(test_samples, n_models))

        split_seeds = self._gen.randint(
            0, int(2**32), size=n_models, dtype=np.uint64
        )

        for m in range(n_models):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=train_frac,
                shuffle=True,
                random_state=split_seeds[m]
            )

            X_trains[:, :, m] = X_train
            X_tests[:, :, m] = X_test
            y_trains[:, m] = y_train
            y_tests[:, m] = y_test
        return X_trains, X_tests, y_trains, y_tests

    def _copy_split_data(self, X, y, n_models, train_frac):
        """Split dataset into train, test sets with random split for each model.

        Parameters
        ----------
        X : np.ndarray
            input data
        y : np.ndarray
            input labels
        n_models : int
            number of models to generate
        train_frac : float
            fraction of samples to allocate to train set

        Returns
        -------
        X_trains : np.ndarray
            The train data copied to three dimensional array
        X_tests : np.ndarray
            The test data copied to three dimensional array
        y_trains : np.ndarray
            the train labels copied to two dimensional array
        y_tests : np.ndarray
            the test labels copied to two dimensional array

        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=train_frac,
            shuffle=True,
            random_state=self._gen
        )
        X_trains = np.broadcast_to(
            X_train[..., None], X_train.shape + (n_models,)
        ).copy()
        X_tests = np.broadcast_to(
            X_test[..., None], X_test.shape + (n_models,)
        ).copy()
        y_trains = np.broadcast_to(
            y_train[..., None], y_train.shape+(n_models,)
        ).copy()
        y_tests = np.broadcast_to(
            y_test[..., None], y_test.shape+(n_models,)
        ).copy()
        return X_trains, X_tests, y_trains, y_tests

    def _fit_and_project_models(self, X_trains, y_trains, X_tests, duplicate):
        """Fit a LR model on the various X_trains.

        Parameters
        ----------
        X_trains : np.ndarray
            the train set input features, where the third dimension represents
            the different models
        y_trains : np.ndarray
            the train set labels, where the second dimension represents
            the different models
        duplicate : bool
            don't fit all the models but duplicate the instance

        Returns
        -------
        models : List[LogisticRegression]
            a list containing the fitted models

        """
        n_models = y_trains.shape[1]
        # all the splits are the same so we don't need to fit multiple times
        if duplicate:
            # fit the model
            model = LogisticRegression(penalty='none')
            fit = model.fit(X_trains[:, :, 0], y_trains[:, 0])
            models = [fit] * n_models
            # compute projection on X_train
            projection = (X_trains[:, :, 0] *  fit.coef_).sum(1)
            train_projections = np.broadcast_to(projection[..., None], projection.shape+(n_models,)).copy()
            # compute projection on X_test
            projection = (X_tests[:, :, 0] *  fit.coef_).sum(1)
            test_projections = np.broadcast_to(projection[..., None], projection.shape+(n_models,)).copy()
            return models, train_projections, test_projections

        models = []
        train_projections = np.zeros((X_trains.shape[0], n_models))
        test_projections = np.zeros((X_tests.shape[0], n_models))
        for i in range(n_models):
            model = LogisticRegression(penalty='none')
            fit = model.fit(X_trains[:, :, i], y_trains[:, i])
            train_projections[:, i] = (X_trains[:, :, i] *  fit.coef_).sum(1)
            test_projections[:, i] = (X_tests[:, :, i] *  fit.coef_).sum(1)
            models.append(fit)
        return models, train_projections, test_projections

    def _rotate_and_shift_models(
        self,
        train_projections,
        test_projections,
        model_rotation,
        model_shift,
        quantile,
    ):
        """Rotate and shift model.

        Parameters
        ----------
        train_projections : np.ndarray
            the projections to be modified, second dimension contains the
            different models
        test_projections : np.ndarray
            the projections to be modified, second dimension contains the
            different models
        model_shift : float
            the value of the ``quantile``th quantile of the zero-centred normal distribution
            from which the shift noise is sampled.
        model_rotation : float, default=0.3
            the value of the ``quantile``th quantile of the zero-centred normal distribution
            from which the rotation noise is sampled.
        quantile : float, default=0.99
            the quantile which should have value `shift` and `rotation`

        Returns
        -------
        train_projections : np.ndarray
            the modified projections, second dimension contains the
            different models
        test_projections : np.ndarray
            the modified projections, second dimension contains the
            different models
        train_proba : np.ndarray
            the probabilities based on the modified projections,
            second dimension contains the different models
        test_proba : np.ndarray
            the probabilities based on the modified projections,
            second dimension contains the different models
        """
        n_models = train_projections.shape[1]

        shift_sigma = _compute_std_dev_from_quant(model_shift, quantile)
        shift_noise = self._gen.normal(0.0, shift_sigma, size=(1, n_models))

        rotation_sigma = _compute_std_dev_from_quant(model_rotation, quantile)
        rotation_noise = 1 + self._gen.normal(0.0, rotation_sigma, size=(1, n_models))

        rotated_train_proj = train_projections * rotation_noise
        rotated_test_proj = test_projections * rotation_noise
        train_proba = expit(rotated_train_proj + shift_noise)
        test_proba = expit(rotated_test_proj + shift_noise)
        return train_proba, test_proba

    def fit(self, n_samples=1_000_000):
        """Create ground truth.

        The ground truth consist of a LR model fitted on ``n_samples`` from the
        ground truth input X and labels y which do not contain any noise
        parameters.

        Parameters
        ----------
        n_samples : int, default=1_000_000
            the number of samples to generate

        Returns
        -------
        self : ModelGenerator
            fitted instance of ModelGenerator

        """
        # create the centroids for the clusters
        self._generate_centroids()
        # create the covariance matrices
        self._generate_covariance_matrices()
        # create even number of samples per cluster
        n_samples_per_cluster = self._compute_n_samples_per_cluster(
            n_samples=n_samples,
            weights=self.weights,
            alpha_weights=self.alpha_weights,
            random_samples_per_class=self._random_samples_per_class,
            unweighted=True
        )
        # generate ground truth inputs and labels
        self.gt_X_, self.gt_y_ = self._generate_ground_truth_inputs(
            n_samples=n_samples, n_samples_per_cluster=n_samples_per_cluster
        )
        # fit plain LogisticRegression without regularisation
        model = LogisticRegression(penalty='none')
        self.gt_fit_ = model.fit(self.gt_X_, self.gt_y_)
        # compute pre-sigmoid values
        self.gt_projection_ = (self.gt_X_ * self.gt_fit_.coef_).sum(1)
        # compute probability
        self.gt_proba_ = expit(self.gt_projection_)
        # compute yhat at 0.5 classification threshold
        self.gt_yhat_ = np.rint(self.gt_proba_).astype(int, copy=False)
        return self

    def transform(
        self,
        n_samples=10000,
        n_models=30,
        enable_cluster_imbalances=True,
        enable_cluster_noise=True,
        enable_sample_noise=True,
        enable_label_noise=True,
        enable_model_noise=True,
        weights=None,
        alpha_weights=None,
        train_frac=None,
        label_flip=None,
        input_shift=None,
        input_scale=None,
        model_shift=None,
        model_rotation=None,
        quantile=None,
    ):
        """Generate ``n_models`` represented by ``n_samples``.

        Parameters
        ----------
        n_samples : int, default=10_000
            number of samples to generate for each model
        n_models : int, default=30
            number of models to generate
        enable_cluster_imbalances : bool, default=True
            If ``enable_cluster_imbalances`` is true the number of samples per
            cluster is not even but based on ``weights`` and ``alpha_weights``
            if ``weights`` == 'random'
        enable_cluster_noise : bool, default=True
            If ``enable_cluster_noise`` is true the inputs are drawn from a Normal
            distribution with:
                a mean drawn from Normal(0, sigma) where sigma is such that the
                ``quantile`` is equal to ``input_shift``
                a std dev that is equal 1 + Normal(0, sigma) where sigma is such
                that the ``quantile`` is equal to ``input_scale``
            Else the or subsample of ground truth is used
        enable_sample_noise : bool, default=True
            If ``enable_sample_noise`` is true the model's are fitted on the input
            split between a train and test set where the former is of size
            ``train_frac`` * ``n_samples``. For each model an independent split
            is done. Else the full input is used and each model has the same fit.
        enable_label_noise : bool, default=True
            If ``enable_label_noise`` is true the ground truth labels have a
            ``flip_y`` probability of being flipped.
        enable_model_noise : bool, default=True
            If ``enable_model_noise`` is true the models are shifted and rotated to
            simulate randomness during the fit.
                the shift is a horizontal shift of the probabilities of size drawn
                from a zero-centred normal distribution with std dev. such that the
                ``quantile`` is equal to ``model_shift``.
                the rotation is rotation around the center of the probabilities
                achieved by scaling the pre-sigmoid projection with a value drawn
                from a Normal distribution centred around one with a std dev. such
                that the ``quantile`` is equal to ``model_rotation``.
        weights : None, str, float or array_like, default=None
            weights can be used to influence the class imbalance or the
            number of samples per cluster.
            - If None each cluster will contain an even number of samples.
            - If a float the weights is taken to be the percentage of samples
            in the positive class. The number of samples per cluster within a class
            will be equal.
            - If array_like the weights are the per cluster weights and must have
              length equal to ``n_clusters_per_class`` * 2. The order is taken to
              be [class_0_cluster_0, class_0_cluster_1, ..., class_1_cluster_0]
            - If 'random' the number of samples per cluster is sampled from a
            Dirichlet with parameters ``alpha_weights``.
        alpha_weights : float, array_like, default=None
            The alpha parameters for the Dirichlet used to sample the
            cluster weights if ``weights`` is random. If a float the
            value is used for each cluster. If array_like the length
            of the input must be equal to ``n_clusters_per_class`` * 2.
            Note this parameter is ignored when ``weights``!='random'
        train_frac : float, default=0.7
            The number of samples to use for training the models
        label_flip : float, default=None
            The probability of a label being flipped
        input_shift : float, default=None
            How much to shift the inputs, drawn from Normal(0, sigma) where
            sigma is such that the ``quantile`` is equal to ``input_shift``.
            The shift is sampled independently for each feature and cluster
            Note that the shift occurs before the scaling
        input_scale : float, default=None
            How much to scale the inputs, drawn from Normal(1, sigma) where
            sigma is such that the ``quantile`` is equal to ``input_scale``.
            The scale is sampled independently for each feature and cluster
            Note that scaling is done after shifting
        model_rotation : float, default=None
            How much to rotate the sigmoidal representation of the model.
            Higher values cause models that have flatter or steeper sigmoidal ouputs.
            Each model is assigned a value sampled from Normal(1, sigma) where
            sigma is such that the ``quantile`` is equal to ``model_rotation``.
            Note that rotation is performed before shifting.
        model_shift : float, default=None
            How much to horizontally shift the sigmoidal representation of the model.
            Each model is assigned a value sampled from Normal(0, sigma) where
            sigma is such that the ``quantile`` is equal to ``model_shift``.
            Note that shifting is performed after rotation.
        quantile : float, default=None
            the quantile which should have values for the above noise parameters.


        Returns
        -------
        train_mask : np.ndarray[1d]
            one dimensional array containing a mask that can be used to extract
            the training or test values from the other outputs
        labels : np.ndarray[2d]
            array of size (``n_samples``, ``n_models``) containing the labels
            for each model
        proba : np.ndarray[2d]
            array of size (``n_samples``, ``n_models``) containing the
            the probabilities as estimated by the model
        X : np.ndarray[3d]
            array of size (``n_samples``,  ``n_features``, ``n_models``) containing the
            the inputs for that model
        models : List[LogisticRegression]
            list containing the trained instances from which the probabilities
            are computed
        ground_truth : pd.DataFrame
            DataFrame containing the labels, probabilities and inputs for the
            ground truth model

        """
        # reset the random state to it's original position
        self._gen = self._gen_orig
        random_samples_per_class = False
        weights = weights or self.weights
        if weights is None:
            pass
        elif isinstance(weights, float):
            pass
        elif isinstance(weights, str):
            if weights not in {'random', 'rand'}:
                raise ValueError('``weights`` is unrecognised string value')
            random_samples_per_class = True
        elif isinstance(weights, (list, tuple, np.ndarray)):
            if len(weights) != self.n_clusters:
                raise ValueError(
                    '``weights`` must be a single value or of length '
                    '2 * ``n_clusters_per_class``'
                )
        else:
            raise TypeError('``weights`` is not of the correct type')

        alpha_weights = alpha_weights or self.alpha_weights
        if isinstance(alpha_weights, (float, int)):
            alpha_weights = np.repeat(alpha_weights, self.n_clusters)
        elif isinstance(alpha_weights, (tuple, list, np.ndarray)):
            if len(alpha_weights) == self.n_clusters:
                self.alpha_weights = alpha_weights
            else:
                raise ValueError(
                    '``alpha_weights`` must be a single value or of length '
                    '2 * ``n_clusters_per_class``'
                )
        elif alpha_weights is None:
            pass
        else:
            raise TypeError('``alpha_weights`` is not of the correct type')

        train_frac = train_frac or self.train_frac
        label_flip = label_flip or self.label_flip
        input_shift = input_shift or self.input_shift
        input_scale = input_scale or self.input_scale
        model_shift = model_shift or self.model_shift
        model_rotation = model_rotation or self.model_rotation
        quantile = quantile or self.noise_quantile

        if enable_cluster_imbalances:
            n_samples_per_cluster = self._compute_n_samples_per_cluster(
                n_samples=n_samples,
                weights=weights,
                alpha_weights=alpha_weights,
                random_samples_per_class=random_samples_per_class,
                unweighted=False
            )
        else:
            n_samples_per_cluster = self._compute_n_samples_per_cluster(
                n_samples=n_samples,
                weights=weights,
                alpha_weights=alpha_weights,
                random_samples_per_class=random_samples_per_class,
                unweighted=True
            )

        gt_x_cols = [f'X_{i}' for i in range(self.n_features)]
        if enable_cluster_noise:
            input_seed = self._gen.randint(0, 2**32, dtype=np.uint64)
            X_orig, y = self._generate_ground_truth_inputs(
                n_samples=n_samples,
                n_samples_per_cluster=n_samples_per_cluster,
                seed=input_seed
            )
            X = self._generate_shifted_and_scaled_inputs(
                n_samples=n_samples,
                n_samples_per_cluster=n_samples_per_cluster,
                input_shift=input_shift,
                input_scale=input_scale,
                quantile=quantile,
                seed=input_seed
            )
            # shuffle the data
            idx = np.random.choice(
                np.arange(X.shape[0]), X.shape[0], replace=False
            )
            X_orig = X_orig[idx, :]
            X = X[idx, :]
            y = y[idx]

            ground_truth = pd.DataFrame(X_orig, columns=gt_x_cols)
        else:
            X, y = self._generate_ground_truth_inputs(
                n_samples=n_samples,
                n_samples_per_cluster=n_samples_per_cluster
            )
            # shuffle the data
            idx = self._gen.choice(
                np.arange(X.shape[0]), X.shape[0], replace=False
            )
            X = X[idx, :]
            y = y[idx]
            ground_truth = pd.DataFrame(X, columns=gt_x_cols)

        n_train_samples = np.floor(n_samples * train_frac).astype(int)
        # -- ground truth --
        ground_truth['y'] = y.copy()
        # assign the first floor(n_samples * train_frac) train label
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[n_train_samples:] = False
        gt_projection = (
            np.sum(
                ground_truth.iloc[:, :self.n_features] * self.gt_fit_.coef_,
                1
            )
        )
        ground_truth['proba'] = expit(gt_projection)
        col_order = ['y', 'proba'] + gt_x_cols
        ground_truth = ground_truth[col_order]
        # --

        if enable_label_noise:
            mask = (
                self._gen
                .binomial(1, p=label_flip, size=n_samples)
                .astype(np.bool_, copy=False)
            )
            y[mask] = 1 - y[mask]


        if enable_sample_noise:
            X_trains, X_tests, y_trains, y_tests = self._split_data(
                X, y, n_models, n_samples, train_frac
            )
            models, train_proj, test_proj = self._fit_and_project_models(
                X_trains, y_trains, X_tests, duplicate=False)
        else:
            X_trains, X_tests, y_trains, y_tests = self._copy_split_data(
                X, y, n_models, train_frac
            )
            models, train_proj, test_proj = self._fit_and_project_models(
                X_trains, y_trains, X_tests, duplicate=True)


        if enable_model_noise:
            train_proba, test_proba = (
                self._rotate_and_shift_models(
                    train_projections=train_proj,
                    test_projections=test_proj,
                    model_rotation=model_rotation,
                    model_shift=model_shift,
                    quantile=quantile,
                )
            )
        else:
            train_proba = expit(train_proj)
            test_proba = expit(test_proj)

        X = np.vstack((X_trains, X_tests))
        labels = np.vstack((y_trains, y_tests))
        probas = np.vstack((train_proba, test_proba))
        return train_mask, labels, probas, X, models, ground_truth
