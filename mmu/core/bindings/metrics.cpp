/* metrics_bindings.cpp -- Python bindings for metrics.hpp
 *
 * Copyright 2021 Ralph Urlus
 */
#include "metrics.hpp"

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_binary_metrics(py::module &m) {
    m.def(
        "binary_metrics",
        [](
            const py::array_t<bool>& y,
            const py::array_t<bool>& yhat,
            const double fill
        ) {
            return binary_metrics<bool, bool>(y, yhat, fill);
        },
        R"pbdoc(Compute binary classification metrics.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        yhat : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the predicted labels
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<int64_t>& yhat,
            const double fill
        ) {
            return binary_metrics<int64_t, int64_t>(y, yhat, fill);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& yhat,
            const double fill
        ) {
            return binary_metrics<double, double>(y, yhat, fill);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics",
        [](
            const py::array_t<int>& y,
            const py::array_t<int>& yhat,
            const double fill
        ) {
            return binary_metrics<int, int>(y, yhat, fill);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics",
        [](
            const py::array_t<float>& y,
            const py::array_t<float>& yhat,
            const double fill
        ) {
            return binary_metrics<float, float>(y, yhat, fill);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
}

void bind_binary_metrics_score(py::module &m) {
    m.def(
        "binary_metrics_score",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& score,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_score<bool, double>(y, score, threshold, fill);
        },
        R"pbdoc(Compute binary classification metrics for a given threshold.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        score : np.array[np.float64]
            the predicted probability
        threshold : float
            classification threshold
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_score",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& score,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_score<int64_t, double>(y, score, threshold, fill);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_score",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& score,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_score<double, double>(y, score, threshold, fill);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_score",
        [](
            const py::array_t<int>& y,
            const py::array_t<double>& score,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_score<int, double>(y, score, threshold, fill);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_score",
        [](
            const py::array_t<float>& y,
            const py::array_t<double>& score,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_score<float, double>(y, score, threshold, fill);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
}

void bind_binary_metrics_runs(py::module &m) {
    m.def(
        "binary_metrics_runs",
        [](
            py::array_t<bool>& y,
            py::array_t<double>& score,
            double threshold,
            const double fill,
            const int obs_axis
        ) {
            return binary_metrics_runs<bool, double>(y, score, threshold, fill, obs_axis);
        },
        R"pbdoc(Compute binary classification metrics over thresholds.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        score : np.array[np.float64]
            the predicted probability
        threshold : float
            classification threshold
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.
        obs_axis : int
            the axis containing the observations

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.,
        py::arg("obs_axis") = 0
    );
    m.def(
        "binary_metrics_runs",
        [](
            py::array_t<int64_t>& y,
            py::array_t<double>& score,
            double threshold,
            const double fill,
            const int obs_axis
        ) {
            return binary_metrics_runs<int64_t, double>(y, score, threshold, fill, obs_axis);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.,
        py::arg("obs_axis") = 0

    );
    m.def(
        "binary_metrics_runs",
        [](
            py::array_t<double>& y,
            py::array_t<double>& score,
            double threshold,
            const double fill,
            const int obs_axis
        ) {
            return binary_metrics_runs<double, double>(y, score, threshold, fill, obs_axis);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.,
        py::arg("obs_axis") = 0
    );
    m.def(
        "binary_metrics_runs",
        [](
            py::array_t<int>& y,
            py::array_t<double>& score,
            double threshold,
            const double fill,
            const int obs_axis
        ) {
            return binary_metrics_runs<int, double>(y, score, threshold, fill, obs_axis);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.,
        py::arg("obs_axis") = 0

    );
    m.def(
        "binary_metrics_runs",
        [](
            py::array_t<float>& y,
            py::array_t<double>& score,
            double threshold,
            const double fill,
            const int obs_axis
        ) {
            return binary_metrics_runs<float, double>(y, score, threshold, fill, obs_axis);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("threshold"),
        py::arg("fill") = 0.,
        py::arg("obs_axis") = 0
    );
}

void bind_binary_metrics_thresholds(py::module &m) {
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<bool, double>(y, score, thresholds, fill);
        },
        R"pbdoc(Compute binary classification metrics over thresholds.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        score : np.array[np.float64]
            the predicted probability
        thresholds : np.array[np.float64]
            classification thresholds
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<int64_t, double>(y, score, thresholds, fill);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<double, double>(y, score, thresholds, fill);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<int>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<int, double>(y, score, thresholds, fill);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<float>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<float, double>(y, score, thresholds, fill);
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
}

void bind_binary_metrics_runs_thresholds(py::module &m) {
    m.def(
        "_binary_metrics_runs_thresholds",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<bool, double>(
                y, score, thresholds, n_obs, fill
            );
        },
        R"pbdoc(Compute binary classification metrics over runs and thresholds.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        score : np.array[np.float[32/64]]
            the predicted probability
        thresholds : np.array[np.float[32/64]]
            classification thresholds
        n_obs : np.array[np.int64]
            the number of observations per run
        fill : double
            value to fill when a metric is not defined, e.g. divide by zero.

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill") = 0.
    );
    m.def(
        "_binary_metrics_runs_thresholds",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<int64_t, double>(
                y, score, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill") = 0.
    );
    m.def(
        "_binary_metrics_runs_thresholds",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<double, double>(
                y, score, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill") = 0.
    );
    m.def(
        "_binary_metrics_runs_thresholds",
        [](
            const py::array_t<int>& y,
            const py::array_t<double>& score,
            const py::array_t<double>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<int, double>(
                y, score, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill") = 0.
    );
    m.def(
        "_binary_metrics_runs_thresholds",
        [](
            const py::array_t<bool>& y,
            const py::array_t<float>& score,
            const py::array_t<float>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<bool, float>(
                y, score, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill") = 0.
    );
    m.def(
        "_binary_metrics_runs_thresholds",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<float>& score,
            const py::array_t<float>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<int64_t, float>(
                y, score, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill") = 0.
    );
    m.def(
        "_binary_metrics_runs_thresholds",
        [](
            const py::array_t<int>& y,
            const py::array_t<float>& score,
            const py::array_t<float>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<int, float>(
                y, score, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill") = 0.
    );
    m.def(
        "_binary_metrics_runs_thresholds",
        [](
            const py::array_t<float>& y,
            const py::array_t<float>& score,
            const py::array_t<float>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<float, float>(
                y, score, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("score"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill") = 0.
    );
}

void bind_binary_metrics_confusion(py::module &m) {
    m.def(
        "binary_metrics_confusion",
        [](
            const py::array_t<int64_t>& conf_mat,
            const double fill
        ) {
            return binary_metrics_confusion<int64_t>(conf_mat, fill
            );
        },
        R"pbdoc(Compute binary classification metrics over a set of confusion matrices.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        conf_mat : np.array[np.int64]
            the confusion matrices where the rows are the different confusion matrices
            and the columns the entries of the matrix. Array should have C-order.
            Note that the entries are assumed to have the following order:
            [TN, FP, FN, TP]
        fill : double
            value to fill when a metric is not defined, e.g. divide by zero.

        Returns
        -------
        np.array[np.float64]
            metrics array
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 0.
    ),
    m.def(
        "binary_metrics_confusion",
        [](
            const py::array_t<int>& conf_mat,
            const double fill
        ) {
            return binary_metrics_confusion<int>(conf_mat, fill
            );
        },
        py::arg("conf_mat"),
        py::arg("fill") = 0.
    );
}

}  // namespace bindings
}  // namespace mmu
