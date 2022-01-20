/* confusion_matrix_bindings.cpp -- Python bindings for confusion_matrix.hpp metrics
 * Copyright 2021 Ralph Urlus
 */
#include "confusion_matrix.hpp"

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
        [](const py::array_t<bool>& y, const py::array_t<bool>& yhat) {
            return confusion_matrix<bool>(y, yhat);
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
        [](const py::array_t<int64_t>& y, const py::array_t<int64_t>& yhat) {
            return confusion_matrix<int64_t>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "confusion_matrix",
        [](const py::array_t<double>& y, const py::array_t<double>& yhat) {
            return confusion_matrix<double>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "confusion_matrix",
        [](const py::array_t<int>& y, const py::array_t<int>& yhat) {
            return confusion_matrix<int>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
    m.def(
        "confusion_matrix",
        [](const py::array_t<float>& y, const py::array_t<float>& yhat) {
            return confusion_matrix<float>(y, yhat);
        },
        py::arg("y"),
        py::arg("yhat")
    );
}

void bind_confusion_matrix_proba(py::module &m) {
    m.def(
        "confusion_matrix_proba",
        [](const py::array_t<bool>& y, const py::array_t<double>& proba, const double threshold) {
            return confusion_matrix<bool>(y, proba, threshold);
        },
        R"pbdoc(Compute binary Confusion Matrix given probabilities.

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        proba : np.array[np.float64]
            the estimted probabilities
        threshold : double
            classification threshold

        Returns
        -------
        confusion_matrix : np.array[np.int64]
            confusion matrix
        )pbdoc",
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_proba",
        [](const py::array_t<int64_t>& y, const py::array_t<double>& proba, const double threshold) {
            return confusion_matrix<int64_t>(y, proba, threshold);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_proba",
        [](const py::array_t<double>& y, const py::array_t<double>& proba, const double threshold) {
            return confusion_matrix<double>(y, proba, threshold);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_proba",
        [](const py::array_t<int>& y, const py::array_t<double>& proba, const double threshold) {
            return confusion_matrix<int>(y, proba, threshold);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold")
    );
    m.def(
        "confusion_matrix_proba",
        [](const py::array_t<float>& y, const py::array_t<double>& proba, const double threshold) {
            return confusion_matrix<float>(y, proba, threshold);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold")
    );
}

}  // namespace bindings
}  // namespace mmu
