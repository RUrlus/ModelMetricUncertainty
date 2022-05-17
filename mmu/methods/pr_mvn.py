"""Module containing the API for the precision-recall uncertainty modelled through profile log likelihoods."""
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from mmu.commons import check_array
from mmu.commons import _convert_to_int
from mmu.metrics.confmat import confusion_matrix
from mmu.metrics.confmat import confusion_matrix_to_dataframe
from mmu.metrics.confmat import confusion_matrices_to_dataframe
from mmu.metrics.confmat import confusion_matrices
from mmu.viz.ellipse import _plot_pr_ellipse

import mmu.lib._mmu_core as _core


class PrecisionRecallEllipticalUncertainty:
    """Precision-Recall uncertainty modelled as a Bivariate Normal.

    Model's the linearly propogated errors of the confusion matrix as a
    bivariate Normal distribution. Note that this method is not valid
    for low statistic test sets or for precision/recall close to 1.0/0.0.
    In these scenarios the `PrecisionRecallMultinomialUncertainty` class should
    be used.

    Attributes
    ----------
    conf_mat : np.ndarray[int64]
        the confusion_matrix with layout
        [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP
        A DataFrame can be obtained by calling `get_conf_mat`.
    precision : float
        the Positive Predictive Value aka positive precision
    recall : float
        True Positive Rate aka Sensitivity aka positive recall
    cov_mat : np.ndarray[float64]
        the covariance matrix of precision and recall with layout
        [0, 0] = V[P], [0, 1] = COV[P, R], [1, 0] = COV[P, R], [1, 1] = V[R]
        A DataFrame can be obtained by calling `get_cov_mat`.
    train_conf_mats : np.ndarray, optional
        the confusion matrices from the multiple training runs.
        Only set when `add_train_uncertainty` is called.
        A DataFrame can be obtained by calling `get_train_conf_mats`
    train_cov_mat : np.ndarray, optional
        the precision, recall covariance matrix from the multiple training runs.
        Only set when `add_train_uncertainty` is called.
        A DataFrame can be obtained by calling `get_train_cov_mat`.
    total_cov_mat : np.ndarray, optional
        the precision, recall covariance matrix that combines the test and
        training sampling uncertainty.
        Only set when `add_train_uncertainty` is called.
        A DataFrame can be obtained by calling `get_total_cov_mat`.

    Methods
    -------
    from_scores
        compute the uncertainty based on classifier scores and true labels
    from_predictions
        compute the uncertainty based on predictions and true labels
    from_confusion_matrix
        compute the uncertainty based on a confusion matrix
    from_classifier
        compute the uncertainty based on classifier scores from a trained
        classifier and true labels
    add_train_uncertainty
        the the training sampling uncertainty to the test uncertainty
    plot
        plot the joint uncertainty of precision and recall
    get_conf_mat
        get the confusion matrix
    get_cov_mat
        get the covariance matrix of the test uncertainty
    get_train_cov_mat
        get the covariance matrix of the train uncertainty
    get_total_cov_mat
        get the covariance matrix of the combined train and test uncertainty
    get_train_conf_mats
        get the confusion matrices over the training runs

    """
    def __init__(self):
        self.conf_mat = None
        self.precision = None
        self.recall = None
        self.cov_mat = None
        self.train_conf_mats = None
        self.train_cov_mat = None
        self.total_cov_mat = None

    def _parse_threshold(self, threshold):
        if not isinstance(threshold, float) or not (0.0 < threshold < 1.0):
            raise TypeError("`threshold` must be a float in [0, 1]")
        self.threshold = threshold

    def _compute_mvn_cov(self):
        out = _core.pr_mvn_cov(self.conf_mat)
        self.precision = out[0]
        if self.precision < 1e-12:
            warnings.warn("`precision` is close to zero, COV[P, R] is not valid")
        elif (1 - self.precision) < 1e-12:
            warnings.warn("`precision` is close to one, COV[P, R] is not valid")
        self.recall = out[1]
        if self.recall < 1e-12:
            warnings.warn("`recall` is close to zero, COV[P, R] is not valid")
        elif (1 - self.recall) < 1e-12:
            warnings.warn("`recall` is close to one, COV[P, R] is not valid")
        self.cov_mat = out[2:].reshape(2, 2)

    def from_scores(self, y : np.ndarray, score : np.ndarray, threshold : float = 0.5):
        """Compute elliptical uncertainty on precision and recall.

        Parameters
        ----------
        y : np.ndarray
            true labels for observations, supported dtypes are [bool, int32,
            int64, float32, float64]
        score : np.ndarray, default=None
            the classifier score to be evaluated against the `threshold`, i.e.
            `yhat` = `score` >= `threshold`.
            Supported dtypes are float32 and float64.
        threshold : float, default=0.5
            the classification threshold to which the classifier score is evaluated,
            is inclusive.

        """
        self._parse_threshold(threshold)
        self.conf_mat = confusion_matrix(y=y, score=score, threshold=threshold)
        self._compute_mvn_cov()
        return self

    def from_predictions(self, y : np.ndarray, yhat : np.ndarray):
        """Compute elliptical uncertainty on precision and recall.

        Parameters
        ----------
        y : np.ndarray
            true labels for observations, supported dtypes are [bool, int32,
            int64, float32, float64]
        yhat : np.ndarray, default=None
            the predicted labels, the same dtypes are supported as y.

        """
        self.conf_mat = confusion_matrix(y=y, yhat=yhat)
        self._compute_mvn_cov()
        return self

    def from_confusion_matrix(self, conf_mat : np.ndarray):
        """Compute elliptical uncertainty on precision and recall.

        Parameters
        ----------
        conf_mat : np.ndarray,
            confusion matrix as returned by mmu.confusion_matrix, i.e.
            with layout [0, 0] = TN, [0, 1] = FP, [1, 0] = FN, [1, 1] = TP or
            the flattened equivalent. Supported dtypes are int32, int64

        """
        if conf_mat.shape == (2, 2):
            conf_mat = conf_mat.ravel()
        self.conf_mat = check_array(conf_mat, max_dim=1, dtype_check=_convert_to_int)
        self._compute_mvn_cov()
        return self

    def from_classifier(self,
        clf,
        X : np.ndarray,
        y : np.ndarray,
        threshold : float = 0.5
    ):
        """Compute uncertainty from classifier.

        Parameters
        ----------
        clf : sklearn.Predictor
            a trained model with method `predict_proba`, used to compute
            the classifier scores
        X : np.ndarray
            the feature array to be used to compute the classifier scores
        y : np.ndarray
            true labels for observations, supported dtypes are [bool, int32,
            int64, float32, float64]
        threshold : float, default=0.5
            the classification threshold to which the classifier score is evaluated,
            is inclusive.

        """
        self._parse_threshold(threshold)
        if not hasattr(clf, 'predict_proba'):
            raise TypeError("`clf` must have a method `predict_proba`")
        score = clf.predict_proba(X)[:, 1]
        self.conf_mat = confusion_matrix(y=y, score=score, threshold=threshold)
        self._compute_mvn_cov()
        return self

    def add_train_uncertainty(
        self,
        y : np.ndarray,
        yhat : Optional[np.ndarray] = None,
        scores : Optional[np.ndarray] = None,
        threshold : float = 0.5,
        obs_axis : int = 0,
    ):
        self.train_conf_mats = confusion_matrices(
            y, yhat, scores, threshold, obs_axis
        )
        out = _core.precision_recall_2d(self.train_conf_mats)
        self.train_precisions = out[:, 0]
        self.train_recalls = out[:, 1]
        self.train_cov_mat = np.cov(out, rowvar=False)

        if self.cov_mat is None:
            raise RuntimeError(
                "the class needs to be initialised with from_*"
                " before adding train uncertainty."
            )
        self.total_cov_mat = self.cov_mat + self.train_cov_mat

    def get_conf_mat(self) -> pd.DataFrame:
        """Obtain confusion matrix as a DataFrame.

        Returns
        -------
        pd.DataFrame
            the confusion matrix of the test set
        """
        return confusion_matrix_to_dataframe(self.conf_mat)

    def get_train_conf_mats(self) -> pd.DataFrame:
        """Obtain confusion matrices as a DataFrame.

        Returns
        -------
        pd.DataFrame
            the confusion matrices of the training sets
        """
        return confusion_matrices_to_dataframe(self.train_conf_mats)

    def _get_cov_mat(self, cov_mat):
        cov_cols = ['precision', 'recall']
        return pd.DataFrame(cov_mat, index=cov_cols, columns=cov_cols)

    def get_cov_mat(self) -> pd.DataFrame:
        """Obtain covariance matrix of the test set.

        Returns
        -------
        pd.DataFrame
            the covariance matrix
        """
        return self._get_cov_mat(self.cov_mat)

    def get_train_cov_mat(self) -> pd.DataFrame:
        """Obtain covariance matrix of the train set.

        Returns
        -------
        pd.DataFrame
            the covariance matrix
        """
        return self._get_cov_mat(self.train_cov_mat)

    def get_total_cov_mat(self) -> pd.DataFrame:
        """Obtain covariance matrix of the train and test set combined.

        Returns
        -------
        pd.DataFrame
            the covariance matrix
        """
        return self._get_cov_mat(self.total_cov_mat)

    def plot(
        self,
        n_std : Optional[int] = None,
        alpha : float = 0.95,
        uncertainties : str = 'test',
        ax=None,
    ):
        """Plot `alpha` elliptical confidence interval for precision and recall

        Parameters
        ----------
        n_std : int, default=None
            number of standard deviations to include in the confidence interval.
            `n_std` takes precedence over `alpa`.
        alpha : float, default=0.95
            the total density of the confidence interval, is ignored when n_std
            is not None
        uncertainties : str, default='test'
            which uncertainty to plot, 'test' indicates only the sampling
            uncertainty of the test set. 'train' only plots the sampling
            uncertainty of the train set. 'all' plots to toal uncertainty over
            both the train and test sets. Note that 'train' and 'all' require
            that `add_train_uncertainty` has been called.
        ax : matplotlib.axes.Axes, default=None
            Pre-existing axes for the plot

        Returns
        -------
        ax : matplotlib.axes.Axes
            the axis with the ellipse added to it

        """
        if self.cov_mat is None:
            raise RuntimeError("the class needs to be initialised with from_*")

        # -- parse uncertainties
        if uncertainties == 'test':
            cov_mat = self.cov_mat
        elif uncertainties == 'all':
            if self.total_cov_mat is not None:
                cov_mat =  self.total_cov_mat
            else:
                raise RuntimeError(
                    "`add_train_uncertainty` must have been called when"
                    " `uncertainties`=='all'"
                )
        elif uncertainties == 'train':
            if self.train_cov_mat is not None:
                cov_mat = self.cov_mat
            else:
                raise RuntimeError(
                    "`add_train_uncertainty` must have been called when"
                    " `uncertainties`=='train'"
                )
        else:
            raise ValueError(
                "`uncertainties` must be one of {'test', 'train', 'all'}"
            )

        return _plot_pr_ellipse(
            self.precision,
            self.recall,
            cov_mat,
            n_std,
            alpha,
            ax
        )
