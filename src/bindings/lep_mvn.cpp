/* lep_mvn.cpp -- Python bindings for lep_mvn
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/lep_mvn.hpp>
#include <mmu/bindings/lep_mvn.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_pr_mvn_var(py::module &m) {
    m.def(
        "pr_mvn_var",
        [](const py::array_t<int64_t>& y, const py::array_t<int64_t>& yhat) {
            return api::pr_var<int64_t, int64_t>(y, yhat);
        },
        R"pbdoc(Compute confusion matrix, Precision, Recall and their variances.

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        yhat : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the predicted labels

        Returns
        -------
        confusion_matrix : np.array[np.int64]
            confusion matrix
        metrics : np.array[np.float64]
            array with columns:
                - precision
                - Var[precision]
                - recall
                - Var[recall]
                - Cov[precision, recall]
        )pbdoc",
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "pr_mvn_var",
        [](const py::array_t<bool>& y, const py::array_t<bool>& yhat) {
            return api::pr_var<bool, bool>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "pr_mvn_var",
        [](const py::array_t<int64_t>& y, const py::array_t<bool>& yhat) {
            return api::pr_var<int64_t, bool>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "pr_mvn_var",
        [](const py::array_t<bool>& y, const py::array_t<int64_t>& yhat) {
            return api::pr_var<bool, int64_t>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "pr_mvn_var",
        [](const py::array_t<double>& y, const py::array_t<double>& yhat) {
            return api::pr_var<double, double>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
}

void bind_pr_curve_mvn_var(py::module &m) {
    m.def(
        "pr_curve_mvn_var",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& scores,
            const py::array_t<double>& thresholds
        ) {
            return api::pr_curve_var<int64_t, double>(y, scores, thresholds);
        },
        R"pbdoc(Compute confusion matrix, Precision, Recall and their variances.

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        scores : np.array[np.float[32/64]]
            the classifier scores
        thresholds : np.array[np.float[32/64]]
            classification thresholds

        Returns
        -------
        confusion_matrix : np.array[np.int64]
            confusion matrix
        metrics : np.array[np.float64]
            array with columns:
                - precision
                - Var[precision]
                - recall
                - Var[recall]
                - Cov[precision, recall]
        )pbdoc",
        py::arg("y"),
        py::arg("scores"),
        py::arg("thresholds")
    );
    m.def(
        "pr_curve_mvn_var",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& scores,
            const py::array_t<double>& thresholds
        ) {
            return api::pr_curve_var<bool, double>(y, scores, thresholds);
        },
        py::arg("y"),
        py::arg("scores"),
        py::arg("thresholds")
    );
    m.def(
        "pr_curve_mvn_var",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& scores,
            const py::array_t<double>& thresholds
        ) {
            return api::pr_curve_var<double, double>(y, scores, thresholds);
        },
        py::arg("y"),
        py::arg("scores"),
        py::arg("thresholds")
    );
}

void bind_pr_mvn_ci(py::module &m) {
    m.def(
        "pr_mvn_ci",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<int64_t>& yhat,
            const double alpha
        ) {
            return api::pr_ci<int64_t, int64_t>(y, yhat, alpha);
        },
        R"pbdoc(Compute confusion matrix, Precision, Recall and their marginal confidence intervals.

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        yhat : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the predicted labels
        alpha : float
            percentage of density in confidence interval

        Returns
        -------
        confusion_matrix : np.array[np.int64]
            confusion matrix
        metrics : np.array[np.float64]
            array with columns:
                - lower-bound CI precision
                - precision
                - upper-bound CI precision
                - lower-bound CI recall
                - recall
                - upper-bound CI recall
        )pbdoc",
        py::arg("y"),
        py::arg("yhat"),
        py::arg("alpha")
    );
    m.def(
        "pr_mvn_ci",
        [](
            const py::array_t<bool>& y,
            const py::array_t<bool>& yhat,
            const double alpha
        ) {
            return api::pr_ci<bool, bool>(y, yhat, alpha);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("alpha")
    );
    m.def(
        "pr_mvn_ci",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<bool>& yhat,
            const double alpha
        ) {
            return api::pr_ci<int64_t, bool>(y, yhat, alpha);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("alpha")
    );
    m.def(
        "pr_mvn_ci",
        [](
            const py::array_t<bool>& y,
            const py::array_t<int64_t>& yhat,
            const double alpha
        ) {
            return api::pr_ci<bool, int64_t>(y, yhat, alpha);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("alpha")
    );
    m.def(
        "pr_mvn_ci",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& yhat,
            const double alpha
        ) {
            return api::pr_ci<double, double>(y, yhat, alpha);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("alpha")
    );
}

void bind_pr_curve_mvn_ci(py::module &m) {
    m.def(
        "pr_curve_mvn_ci",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& scores,
            const py::array_t<double>& thresholds,
            const double alpha
        ) {
            return api::pr_curve_ci<int64_t, double>(y, scores, thresholds, alpha);
        },
        R"pbdoc(Compute confusion matrix, Precision, Recall and their marginal confidence intervals.

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        scores : np.array[np.float[32/64]]
            the classifier scores
        thresholds : np.array[np.float[32/64]]
            classification thresholds
        alpha : float
            percentage of density in confidence interval

        Returns
        -------
        confusion_matrix : np.array[np.int64]
            confusion matrix
        metrics : np.array[np.float64]
            array with columns:
                - lower-bound CI precision
                - precision
                - upper-bound CI precision
                - lower-bound CI recall
                - recall
                - upper-bound CI recall
        )pbdoc",
        py::arg("y"),
        py::arg("scores"),
        py::arg("thresholds"),
        py::arg("alpha")
    );
    m.def(
        "pr_curve_mvn_ci",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& scores,
            const py::array_t<double>& thresholds,
            const double alpha
        ) {
            return api::pr_curve_ci<bool, double>(y, scores, thresholds, alpha);
        },
        py::arg("y"),
        py::arg("scores"),
        py::arg("thresholds"),
        py::arg("alpha")

    );
    m.def(
        "pr_curve_mvn_ci",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& scores,
            const py::array_t<double>& thresholds,
            const double alpha
        ) {
            return api::pr_curve_ci<double, double>(y, scores, thresholds, alpha);
        },
        py::arg("y"),
        py::arg("scores"),
        py::arg("thresholds"),
        py::arg("alpha")

    );
}


}  // namespace bindings
}  // namespace mmu
