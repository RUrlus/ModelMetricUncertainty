/* confusion_matrix_bindings.cpp -- Python bindings for confusion_matrix.hpp metrics
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/confusion_matrix.hpp>
#include <mmu/bindings/confusion_matrix.hpp>

namespace py = pybind11;

/*                  pred
 *                0     1
 *  actual  0    TN    FP
 *          1    FN    TP
 *
 *  Flattened, implies C-contiguous, we have:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace bindings {

void bind_confusion_matrix(py::module &m) {
    m.def(
        "confusion_matrix",
        [](const py::array_t<int64_t>& y, const py::array_t<int64_t>& yhat) {
            return api::confusion_matrix<int64_t, int64_t>(y, yhat);
        },
        R"pbdoc(Compute binary Confusion Matrix.

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
        )pbdoc",
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "confusion_matrix",
        [](const py::array_t<bool>& y, const py::array_t<bool>& yhat) {
            return api::confusion_matrix<bool, bool>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "confusion_matrix",
        [](const py::array_t<int64_t>& y, const py::array_t<bool>& yhat) {
            return api::confusion_matrix<int64_t, bool>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "confusion_matrix",
        [](const py::array_t<bool>& y, const py::array_t<int64_t>& yhat) {
            return api::confusion_matrix<bool, int64_t>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "confusion_matrix",
        [](const py::array_t<double>& y, const py::array_t<double>& yhat) {
            return api::confusion_matrix<double, double>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
}

void bind_confusion_matrix_runs(py::module &m) {
    m.def(
        "confusion_matrix_runs",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<int64_t>& yhat,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<int64_t, int64_t>(y, yhat, obs_axis);
        },
        R"pbdoc(Compute binary Confusion Matrix given probabilities over multiple runs.

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels, must be 2 dimensional
        yhat : np.array[np.float64]
            the estimated labels, must be 2 dimensional and have the same shape as y
        obs_axis : int
            the axis containing the observations, e.g. 0 if (N, K) for K runs

        Returns
        -------
        confusion_matrix : np.array[np.int64]
            confusion matrix
        )pbdoc",
        py::arg("y"),
        py::arg("yhat"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_runs",
        [](
            const py::array_t<bool>& y,
            const py::array_t<bool>& yhat,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<bool, bool>(y, yhat, obs_axis);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_runs",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<bool>& yhat,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<int64_t, bool>(y, yhat, obs_axis);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_runs",
        [](
            const py::array_t<bool>& y,
            const py::array_t<int64_t>& yhat,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<bool, int64_t>(y, yhat, obs_axis);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_runs",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& yhat,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<double, double>(y, yhat, obs_axis);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("obs_axis")
    );
}

void bind_confusion_matrix_score(py::module &m) {
    m.def(
        "confusion_matrix_score",
        [](const py::array_t<int64_t>& y, const py::array_t<double>& score, const double threshold) {
            return api::confusion_matrix<int64_t, double>(y, score, threshold);
        },
        R"pbdoc(Compute binary Confusion Matrix given probabilities.

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        score : np.array[np.float64]
            the classifier scores
        threshold : double
            classification threshold

        Returns
        -------
        confusion_matrix : np.array[np.int64]
            confusion matrix
        )pbdoc",
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_score",
        [](const py::array_t<bool>& y, const py::array_t<double>& score, const double threshold) {
            return api::confusion_matrix<bool, double>(y, score, threshold);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_score",
        [](const py::array_t<double>& y, const py::array_t<double>& score, const double threshold) {
            return api::confusion_matrix<double, double>(y, score, threshold);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_score",
        [](const py::array_t<int64_t>& y, const py::array_t<float>& score, const double threshold) {
            return api::confusion_matrix<int64_t, float>(y, score, threshold);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_score",
        [](const py::array_t<bool>& y, const py::array_t<float>& score, const double threshold) {
            return api::confusion_matrix<bool, double>(y, score, threshold);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_score",
        [](const py::array_t<float>& y, const py::array_t<float>& score, const double threshold) {
            return api::confusion_matrix<float, float>(y, score, threshold);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold")
    );
}

void bind_confusion_matrix_score_runs(py::module &m) {
    m.def(
        "confusion_matrix_score_runs",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& score,
            const double threshold,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<int64_t, double>(y, score, threshold, obs_axis);
        },
        R"pbdoc(Compute binary Confusion Matrix given probabilities over multiple runs.

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels, must be 2 dimensional
        score : np.array[np.float64]
            the classifier scores, must be 2 dimensional and have the same shape as y
        threshold : double
            classification threshold
        obs_axis : int
            the axis containing the observations, e.g. 0 if (N, K) for K runs

        Returns
        -------
        confusion_matrix : np.array[np.int64]
            confusion matrix
        )pbdoc",
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_score_runs",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& score,
            const double threshold,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<bool, double>(y, score, threshold, obs_axis);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_score_runs",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& score,
            const double threshold,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<double, double>(y, score, threshold, obs_axis);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_score_runs",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<float>& score,
            const double threshold,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<int64_t, float>(y, score, threshold, obs_axis);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_score_runs",
        [](
            const py::array_t<bool>& y,
            const py::array_t<float>& score,
            const double threshold,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<float, float>(y, score, threshold, obs_axis);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold"),
        py::arg("obs_axis")
    );
    m.def(
        "confusion_matrix_score_runs",
        [](
            const py::array_t<float>& y,
            const py::array_t<float>& score,
            const double threshold,
            const int obs_axis
        ) {
            return api::confusion_matrix_runs<float, float>(y, score, threshold, obs_axis);
        },
        py::arg("y"),
        py::arg("score").noconvert(),
        py::arg("threshold"),
        py::arg("obs_axis")
    );
}

}  // namespace bindings
}  // namespace mmu
