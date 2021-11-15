/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_METRICS_CONFUSION_MATRIX_HPP_
#define MMU_METRICS_CONFUSION_MATRIX_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>  // FIXME
#include <cmath>
#include <string>
#include <limits>
#include <cinttypes>
#include <type_traits>

#include "commons/utils.hpp"

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
namespace details {

// --- from yhat ---
template<class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline void confusion_matrix(const size_t n_obs, T* y, T* yhat, int64_t* const conf_mat) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[static_cast<bool>(*y) * 2 + static_cast<bool>(*yhat)]++; yhat++; y++;
    }
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value, int> = 1>
inline void confusion_matrix(const size_t n_obs, T* y,  T* yhat, int64_t* const conf_mat) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > 1e-12) * 2 + (*yhat > 1e-12)]++; yhat++; y++;
    }
}

// --- from proba and threshold ---
template<class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline void confusion_matrix(
    const size_t n_obs, T* y, double* proba, const double threshold, int64_t* const conf_mat
) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[static_cast<bool>(*y) * 2 + (*proba > threshold)]++; proba++; y++;
    }
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value, int> = 1>
inline void confusion_matrix(
    const size_t n_obs, T* y, double* proba, const double threshold, int64_t* const conf_mat
) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > 1e-12) * 2 + (*proba > threshold)]++; proba++; y++;
    }
}

// --- from proba and threshold ---
template<class T, typename A, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline void confusion_matrix(
    const size_t n_obs, T* y, A* proba, const A threshold, int64_t* const conf_mat
) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[static_cast<bool>(*y) * 2 + (*proba > threshold)]++; proba++; y++;
    }
}

template<class T, typename A, std::enable_if_t<std::is_floating_point<T>::value, int> = 1>
inline void confusion_matrix(
    const size_t n_obs, T* y, A* proba, const A threshold, int64_t* const conf_mat
) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > 1e-12) * 2 + (*proba > threshold)]++; proba++; y++;
    }
}

}  // namespace details


namespace bindings {

template <typename T>
py::array_t<int64_t> confusion_matrix(const py::array_t<T>& y, const py::array_t<T>& yhat) {
    details::check_1d_soft(y, "y");
    details::check_contiguous(y, "y");
    details::check_1d_soft(yhat, "yhat");
    details::check_contiguous(yhat, "yhat");
    details::check_equal_length(y, yhat);

    // note memory is uninitialised
    auto conf_mat = py::array_t<int64_t>({2, 2}, {16, 8});
    int64_t* cm_ptr = reinterpret_cast<int64_t*>(conf_mat.request().ptr);
    // initialise the memory
    memset(cm_ptr, 0, sizeof(int64_t) * 4);

    auto y_ptr = reinterpret_cast<T*>(y.request().ptr);
    auto yhat_ptr = reinterpret_cast<T*>(yhat.request().ptr);
    details::confusion_matrix<T>(y.size(), y_ptr, yhat_ptr, cm_ptr);

    return conf_mat;
}

void bind_confusion_matrix(py::module &m) {
    m.def(
        "confusion_matrix",
        [](const py::array_t<bool>& y, const py::array_t<bool>& yhat) {
            return confusion_matrix<bool>(y, yhat);
        },
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
}

template <typename T>
py::array_t<int64_t> confusion_matrix(const py::array_t<T>& y, const py::array_t<double>& proba, const double threshold) {
    details::check_1d_soft(y, "y");
    details::check_contiguous(y, "y");
    details::check_1d_soft(proba, "proba");
    details::check_contiguous(proba, "proba");
    details::check_equal_length(y, proba);

    // note memory is uninitialised
    auto conf_mat = py::array_t<int64_t>({2, 2}, {16, 8});
    int64_t* cm_ptr = reinterpret_cast<int64_t*>(conf_mat.request().ptr);
    // initialise the memory
    memset(cm_ptr, 0, sizeof(int64_t) * 4);

    auto y_ptr = reinterpret_cast<T*>(y.request().ptr);
    auto proba_ptr = reinterpret_cast<double*>(proba.request().ptr);
    details::confusion_matrix<T>(y.size(), y_ptr, proba_ptr, threshold, cm_ptr);

    return conf_mat;
}

void bind_confusion_matrix_proba(py::module &m) {
    m.def(
        "confusion_matrix_proba",
        [](const py::array_t<bool>& y, const py::array_t<double>& proba, const double threshold) {
            return confusion_matrix<bool>(y, proba, threshold);
        },
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
}

}  // namespace bindings
}  // namespace mmu

#endif  // MMU_METRICS_CONFUSION_MATRIX_HPP_
