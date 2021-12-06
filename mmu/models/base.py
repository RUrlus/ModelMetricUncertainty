import logging
import multiprocessing

# DO NOT CHANGE, needs to be called before stan
import nest_asyncio
nest_asyncio.apply()
import stan
import arviz as az

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import check_array

from mmu.commons import _set_plot_style
from mmu.metrics import confusion_matrix_proba
from mmu.metrics import binary_metrics_runs
from mmu.metrics import binary_metrics_confusion
from mmu.metrics import col_index as mtr_col_index
from mmu.metrics import col_names as mtr_col_names
from mmu.metrics import compute_hdi as _compute_hdi

_COLORS = _set_plot_style()


class ConfusionMatrixBase:
    """Base class for stan models that model the confusion matrix."""
    def __init__(self):
        """Initialise the class."""
        self.code = ''
        self.prior_vars = []
        self.post_vars = []
        self.predictive_var = ''

    def _set_random_state(self, random_state):
        self._logger = logging.getLogger()
        if random_state is None:
            self._gen = np.random.default_rng(None)
            self.random_state = self._gen.integers(0, np.iinfo(np.int32).max)
        elif isinstance(random_state, int):
            self.random_state = random_state
            self._gen = np.random.default_rng(random_state)
        elif isinstance(random_state, np.random.Generator):
            self._gen = random_state
            self.random_state = self._gen.integers(0, np.iinfo(np.int32).max)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state.random_integers(
                0, np.iinfo(np.int32).max
            )
            self._gen = np.random.default_rng(self.random_state)
        else:
            raise TypeError('``random_state`` has an invalid type.')

    def _check_X(self, X):
        """Check if X meets the conditions."""
        X_ndim = X.ndim
        if X_ndim == 1:
            X = X[None, :]

        return check_array(
            X,
            dtype=np.integer,
            ensure_min_features=4,
            accept_large_sparse=False,
            order='C'
        )

    def _fit_predict(
        self,
        data,
        n_samples,
        sample_factor,
        n_cores,
        n_warmup=500,
        sampling_kwargs=None,
    ):
        """Fit model and sample confusion matrices from the posterior.

        Parameters
        ----------
        data : dict
            the data dictionary passed to stan
        n_samples : int
            Number of samples to draw, used both during fit as well as draw
            from posterior.
        sample_factor : float, int, default=1.5
            a factor to increase the number of samples drawn from the posterior
            which increases the probability that the number non-divergent
            samples is as least as big as n_samples.
        n_cores : int, default=None
            the number of cores to use during fit and sampling. Default's
            4 cores if present other the number of cores available. Setting it
            to -1 will use all cores available - 1.
        n_warmup : int, default=500
            number of warmup itertations used during fitting of Stan model
        sampling_kwargs : dict, default=None
            keyword arguments passed to sample function of Stan

        """
        if not isinstance(n_samples, int):
            raise TypeError('``n_samples`` should be of type `int`')
        if not isinstance(sample_factor, (int, float)):
            raise TypeError(
                '``sample_factor`` should be of type `int` or `float`'
            )

        total_samples = int(np.floor(n_samples * sample_factor))

        if n_cores is None:
            n_cores = min(multiprocessing.cpu_count(), 4)
        elif n_cores == -1:
            n_cores = multiprocessing.cpu_count() - 1
        elif isinstance(n_cores, int):
            pass
        else:
            raise TypeError('``n_cores`` has an invalid type.')

        if sampling_kwargs is None:
            sampling_kwargs = {}
        elif not isinstance(sampling_kwargs, dict):
            raise TypeError('``sampling_kwargs`` must a dict or None')

        self._logger.log(logging.INFO, 'MMU: Compiling stan model.')
        self.model = stan.build(
            self.code,
            data=data,
            random_seed=int(self.random_state)
        )
        m = (
            f'MMU: Fitting and sampling using {n_samples} samples'
            ' and {n_cores} chains.'
        )
        self._logger.log(logging.INFO, m)
        # set the number of samples per chain
        n_samples_per_chain = total_samples // n_cores

        self.fit_ = self.model.sample(
            num_warmup=n_warmup,
            num_samples=n_samples_per_chain,
            num_chains=n_cores,
            **sampling_kwargs
        )

        self.fit_data_ = az.from_pystan(
            posterior=self.fit_,
            posterior_predictive=self.post_vars,
            observed_data=list(data.keys()),
            posterior_model=self.model
        )

        # validate if enough non-divergent samples were drawn
        div_mask = self.fit_data_.sample_stats.get('diverging').values
        self._div_mask = div_mask
        n_samples_drawn = div_mask.size
        n_non_divergent = n_samples_drawn - div_mask.sum()
        if n_non_divergent < n_samples:
            m = (
                f'The number of non-divergent samples ({n_non_divergent}) is'
                ' smaller than the requested samples ({n_samples}).'
                ' You should increase the sample factor.'
            )
            raise RuntimeError(m)

        # generative samples
        y_hat = (
            self.fit_data_
            .posterior_predictive
            .get(self.predictive_var)
            .values
            [~div_mask]
        )

        # subsample the posterior samples
        sidx = self._gen.choice(
            np.arange(n_non_divergent),
            replace=False,
            size=n_samples
        )

        self.sample_conf_mats_ = np.asarray(y_hat[sidx], dtype=np.int64, order='C')
        return self.sample_conf_mats_

    def _compute_confusion_matrix(
        self, proba, y, threshold=0.5, ensure_1d=True
    ):
        """Compute the confusion matrix over the probabilities.

        Parameters
        ----------
        proba : np.ndarray[np.float64]
            the probabilities
        y : np.ndarray[bool, np.int64]
            the true y values
        threshold : float, default=0.5
            the classification threshold

        Returns
        -------
        conf_mat : np.ndarray
            the confusion matrix over X

        """
        proba = check_array(
            proba,
            dtype=[np.float32, np.float64, float],
            ensure_2d=False,
            ensure_min_features=1,
            accept_large_sparse=False,
        )
        y = check_array(
            y,
            dtype=[np.integer, np.bool_],
            ensure_2d=False,
            ensure_min_features=1,
            accept_large_sparse=False,
        )
        if proba.ndim == 2 and proba.shape[1] > 1:
            if ensure_1d:
                raise TypeError('``X`` should be 1 dimensional')
            conf_mat, _ = binary_metrics_runs(
                y=y, proba=proba, threshold=threshold,
            )
        else:
            conf_mat = confusion_matrix_proba(y, proba, threshold).flatten()
        return conf_mat

    def compute_metrics(self, X=None, metrics=None, fill=0.0):
        """Compute metrics over the confusion matrix samples.

        Parameters
        ----------
        X : np.ndarray, default=None
            confusion matrices, of shape (N, 4) where the expected order is:
            [TN, FP, FN, TP]. If None the metrics are computed over the
            over the posterior predictive samples.
        metrics : str, list, default=None
            name(s) of metrics to compute, supported options are:
                * 'neg.precision'
                * 'npv'
                * 'pos.precision'
                * 'ppv'
                * 'neg.recall'
                * 'tnr'
                * 'specificity'
                * 'pos.recall'
                * 'tpr'
                * 'sensitivity'
                * 'neg.f1'
                * 'neg.f1_score'
                * 'pos.f1'
                * 'pos.f1_score'
                * 'fpr'
                * 'fnr'
                * 'accuracy'
                * 'acc'
                * 'mcc'
            if None all the metrics are computed
        fill : float
            value to use when a metric is not defined, e.g. TP is zero

        Returns
        -------
        metrics : np.ndarray
            computed metrics, column order is:
                0. 'neg.precision'
                1. 'pos.precision'
                2. 'neg.recall'
                3. 'pos.recall'
                4. 'neg.f1'
                5. 'pos.f1'
                6. 'fpr'
                7. 'fnr'
                8. 'acc'
                9. 'mcc'
            The array can be easily transformed to a DataFrame with column
            names using ``mmu.metrics_to_dataframe``

        """
        if metrics is None:
            metrics = np.arange(start=0, stop=10, step=1, dtype=np.int64)
        elif isinstance(metrics, str):
            if metrics not in mtr_col_index:
                raise ValueError('``metrics`` is not a supported metric.')
            metrics = [mtr_col_index[metrics], ]
        elif isinstance(metrics, (tuple, list, np.ndarray)):
            metrics_idx = []
            for mtr in metrics:
                if mtr not in mtr_col_index:
                    raise ValueError(f'{mtr} is not a supported metric.')
                metrics_idx.append(mtr_col_index[mtr])
            metrics = metrics_idx
        else:
            raise TypeError('``metrics`` has an unsupported type.')

        if isinstance(fill, int):
            fill = float(fill)
        elif not isinstance(fill, float):
            raise TypeError('``fill`` must be a float.')

        if X is not None:
            y_hat = check_array(
                X,
                dtype=[np.int32, np.int64, int],
                ensure_2d=True,
                ensure_min_features=4,
                accept_large_sparse=False,
            )
            if y_hat.flags['C_CONTIGUOUS'] is False:
                y_hat = np.asarray(y_hat, order='C')
        else:
            if not hasattr(self, 'sample_conf_mats_'):
                m = '`fit_predict` must be run before metrics can be computed.'
                raise RuntimeError(m)
            y_hat = self.sample_conf_mats_

        self.sample_metrics_ = binary_metrics_confusion(y_hat, fill=fill)
        return self.sample_metrics_[:, metrics]


    def plot_posterior_trace(self):
        """Plot traces of posterior samples."""
        return az.plot_trace(self.fit_data_, var_names=self.post_vars)

    def plot_prior_trace(self):
        """Plot traces of posterior samples."""
        return az.plot_trace(self.fit_data_, var_names=self.prior_vars)

    def plot_posterior(self, hdi_prob=0.95,  **plot_kwargs):
        return az.plot_posterior(
            self.fit_data_.posterior,
            var_names=self.post_vars,
            hdi_prob=hdi_prob,
            **plot_kwargs
        )

    def posterior_hdi(self, prob=0.95, **kwargs):
        """Compute Highest Density Interval for the posterior distributions.

        Parameters
        ----------
        prob : float, default=0.95
            the cumulative probability that should be contained in the HDI
        kwargs
            keyword arguments passed to az.hdi

        Returns
        -------
        hdi : np.ndarray
            the HDI for the posterior distributions

        """
        return az.hdi(
            ary=self.fit_data_.posterior.get(self.post_vars).values,
            hdi_prob=prob,
            **kwargs
        )

    def posterior_predictive_hdi(self, prob=0.95, **kwargs):
        """Compute Highest Density Interval for the posterior posterior predictive samples.

        Parameters
        ----------
        prob : float, default=0.95
            the cumulative probability that should be contained in the HDI
        kwargs
            keyword arguments passed to az.hdi

        Returns
        -------
        hdi : np.ndarray
            the HDI for the posterior distributions

        """
        return az.hdi(
            ary=self.fit_data_.posterior.get(self.predictive_var).values,
            hdi_prob=prob,
            **kwargs
        )

    def _plot_hdi(
        self,
        hdis,
        y,
        x_labels,
        prob=0.95,
        use_kde=True,
        plot_kwargs=None,
    ):
        """Plot the HDI of the predictive samples.

        Parameters
        ----------
        hdis : np.ndarray
            2d array containing the highest density intervals
        y : np.ndarray
            array containing samples from the distribution
        x_labels : str, list-like
            name or names for the x axis
        prob : float, default=0.95
            the cumulative probability that should be contained in the HDI
        use_kde = bool, default=True
            use KDE instead of histogram for the distribution
        hdi_kwargs : dict, default=None
            keyword arguments passed to az.hdi
        plot_kwargs : dict, default=None
            keyword arguments passed to hist or kde plotting functions

        Returns
        -------
        fig : plt.Figure
            generated figure
        ax : np.ndarray[plt.axes.AxesSubplot], optional
            array of axes when posterior_predictive_hdi contains multiple rows
        ax : plt.axes.AxesSubplot, optional
            ax when posterior_predictive_hdi contains single row

        """
        plot_kwargs = plot_kwargs or {}

        _hist_kwargs = dict(
            bins='auto',
            histtype='step',
            lw='2',
            density=True,
            color=_COLORS[0],
        )

        _kde_kwargs = dict(
            color=_COLORS[0],
            lw='2',
        )

        n_rows = hdis.shape[0]
        if isinstance(x_labels, str):
            x_labels = [f'{x_labels}_{i}' for i in range(n_rows)]

        if n_rows == 10:
            plot_rows = 5
            plot_cols = 2
        elif n_rows > 4:
            plot_rows = np.ceil(n_rows / 4).astype(np.int64)
            plot_cols = 4
        else:
            plot_rows = 1
            plot_cols = n_rows

        fig, axs = plt.subplots(
            figsize=(9 * plot_cols, 9 * plot_rows),
            nrows=plot_rows,
            ncols=plot_cols,
            sharey=True
        )
        if not isinstance(axs, np.ndarray):
            axs = [axs, ]
        else:
            axs = axs.flatten()

        for i in range(n_rows):
            ax = axs[i]
            view = y[:, i]
            if use_kde:
                _kde_kwargs.update(plot_kwargs)
                _ = sns.kdeplot(
                    view,
                    ax=ax,
                    **_kde_kwargs,
                )
                ax_x, ax_y = ax.get_lines()[0].get_data()
                shade_idx = (ax_x >= hdis[i, 0]) & (ax_x < hdis[i, 1])
                ax.fill_between(
                    x=ax_x[shade_idx],
                    y1=ax_y[shade_idx],
                    alpha=0.3,
                    color=_COLORS[0]
                )
            else:
                _hist_kwargs.update(plot_kwargs)
                _ = ax.hist(
                    view,
                    **_hist_kwargs,
                )

            ax.axvline(
                x=hdis[i, 0],
                color=_COLORS[3],
                ls='--',
                lw=2,
            )

            ax.axvline(
                x=hdis[i, 1],
                color=_COLORS[3],
                ls='--',
                lw=2,
            )
            ax.set_title(
                f'{round(prob * 100, 1)}% HDI: '
                f' [{round(hdis[i, 0], 3)}, {round(hdis[i, 1], 3)}]'
                f'\n mean: {round(view.mean(), 3)},'
                f' median: {round(np.median(view), 3)}'
            )
            ax.set_ylabel('density', fontsize=16)
            ax.set_xlabel(x_labels[i], fontsize=18)
            ax.tick_params(labelsize=14)
        fig.suptitle('HDI Predictive Posterior samples', fontsize=20)
        return fig, axs

    def plot_hdi_predictive_posterior(
        self, prob=0.95, use_kde=True, hdi_kwargs=None, plot_kwargs=None,
    ):
        """Plot the HDI of the predictive samples.

        Parameters
        ----------
        prob : float, default=0.95
            the cumulative probability that should be contained in the HDI
        use_kde = bool, default=True
            use KDE instead of histogram for the distribution
        hdi_kwargs : dict, default=None
            keyword arguments passed to az.hdi
        plot_kwargs : dict, default=None
            keyword arguments passed to hist or kde plotting functions

        Returns
        -------
        fig : plt.Figure
            generated figure
        ax : np.ndarray[plt.axes.AxesSubplot], optional
            array of axes when posterior_predictive_hdi contains multiple rows
        ax : plt.axes.AxesSubplot, optional
            ax when posterior_predictive_hdi contains single row

        """
        hdi_kwargs = hdi_kwargs or {}
        hdis = self.posterior_predictive_hdi(prob, **hdi_kwargs)

        # get the generated samples
        if not hasattr(self, 'fit_data_'):
            raise RuntimeError('``fit_predict`` must have been called.')
        y = (
            self.fit_data_
            .posterior
            .get(self.predictive_var)
            .values
            [~self._div_mask]
        )
        # due to chains its a cube
        y = y.flatten().reshape(y.size // 4, 4)

        return self._plot_hdi(
            hdis, y, self.predictive_var,  prob, use_kde, plot_kwargs
        )


    def compute_hdis(
        self, X=None, metrics=True, prob=0.95, return_metrics=False, **kwargs
    ):
        """Compute the Highest Density Interval.

        Parameters
        ----------
        X : pd.DataFrame, array-like, default=None
            at most two dimensional array where the rows contain the observations
            and the column different distributions. If None the HDI is
            computed over the ``metrics`` or confusion matrix sample.
        metrics : bool, str, list[str], default=True
            over which values to compute the HDI, if X is not None this
            parameter is ignored.
            If True the HDI's is computed over all the metrics or those computed
            using ``compute_metrics``, if str or list of strings only the
            HDIs will be computed for those values. If false the HDI is
            computed for the posterior predictive samples.
        prob : float, default=0.95
            the cumulative probability that should be contained in the HDI
        return_metrics : bool, default=False
            return samples metrics alongside hdi
        kwargs
            keyword arguments passed to az.hdi

        Returns
        -------
        hdi : np.ndarray
            the HID interval for each column in ``X``. The first column is the
            lower bound, the second the upper bound.
        sample_metrics : np.ndarray, optional
            the computed sample metrics, returned if return_metrics=True
        metrics : np.ndarray, optional
            the inde

        """
        if X is not None:
            _compute_hdi(X, prob, **kwargs)

        if metrics is False:
            sample_metrics = self.sample_conf_mats_
        elif not hasattr(self, 'sample_metrics_'):
            sample_metrics = self.compute_metrics()
        else:
            sample_metrics = self.sample_metrics_


        metric_names = None
        if metrics is None or metrics is True:
            metrics = np.arange(start=0, stop=10, step=1, dtype=np.int64)
        elif metrics is False:
            metrics = np.arange(sample_metrics.shape[1], dtype=np.int64)
            metric_names = ['TN', 'FP', 'FN', 'TP']
        elif isinstance(metrics, str):
            if metrics not in mtr_col_index:
                raise ValueError('``metrics`` is not a supported metric.')
            metrics = [mtr_col_index[metrics], ]
        elif isinstance(metrics, (tuple, list, np.ndarray)):
            metrics_idx = []
            for mtr in metrics:
                if mtr not in mtr_col_index:
                    raise ValueError(f'{mtr} is not a supported metric.')
                metrics_idx.append(mtr_col_index[mtr])
            metrics = metrics_idx
        else:
            raise TypeError('``metrics`` has an unsupported type.')


        if metric_names is None:
            metric_names = [mtr_col_names[i] for i in metrics]

        n_metrics = sample_metrics.shape[1]
        hdi = np.zeros((n_metrics, 2))
        for i in range(n_metrics):
            hdi[i, :] = az.hdi(
                ary=sample_metrics[:, i], hdi_prob=prob, **kwargs
            )

        if return_metrics:
            return (
                hdi[metrics, :],
                sample_metrics[:, metrics],
                metric_names
            )
        return hdi[metrics, :]

    def plot_hdi(
        self,
        X=None,
        metrics=True,
        prob=0.95,
        use_kde=True,
        hdi_kwargs=None,
        plot_kwargs=None,
    ):
        """Plot the Higest Density Interval.

        Parameters
        ----------
        X : pd.DataFrame, array-like, default=None
            at most two dimensional array where the rows contain the observations
            and the column different distributions. If None the HDI is
            computed over the ``metrics`` or confusion matrix sample.
        metrics : bool, str, list[str], default=True
            over which values to compute the HDI, if X is not None this
            parameter is ignored.
            If True the HDI's is computed over all the metrics or those computed
            using ``compute_metrics``, if str or list of strings only the
            HDIs will be computed for those values. If false the HDI is
            computed for the posterior predictive samples.
        prob : float, default=0.95
            the cumulative probability that should be contained in the HDI
        use_kde = bool, default=True
            use KDE instead of histogram for the distribution
        hdi_kwargs : dict, default=None
            keyword arguments passed to az.hdi
        plot_kwargs : dict, default=None
            keyword arguments passed to hist or kde plotting functions

        Returns
        -------
        fig : plt.Figure
            generated figure
        ax : np.ndarray[plt.axes.AxesSubplot], optional
            array of axes when posterior_predictive_hdi contains multiple rows
        ax : plt.axes.AxesSubplot, optional
            ax when posterior_predictive_hdi contains single row

        """
        hdi_kwargs = hdi_kwargs or {}
        hdis, sample_metrics, metrics = self.compute_hdis(
            X, metrics, prob, return_metrics=True, **hdi_kwargs
        )
        return self._plot_hdi(
            hdis, sample_metrics, metrics, prob, use_kde, plot_kwargs
        )
