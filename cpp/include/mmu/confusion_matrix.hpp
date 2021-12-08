/* confusion_matrix.hpp -- Implementation of binary classification confusion matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef CPP_INCLUDE_MMU_CONFUSION_MATRIX_HPP_
#define CPP_INCLUDE_MMU_CONFUSION_MATRIX_HPP_

/* TODO *
 *
 * - Add function over runs for yhat
 * - Add function over runs for proba with single threshold
 * - Add function over runs for proba over multiple thresholds
 * TODO */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <limits>
#include <cinttypes>
#include <type_traits>

#include <mmu/utils.hpp>

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

/* Fill binary confusion matrix based on true labels y and estimated labels yhat
 *
 * --- NOTE ---
 * - this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * * expects conf_mat to point to zero'd memory
 * --- NOTE ---
 *
 * --- Parameters ---
 * n_obs : minimum length of y and yhat
 * y : true labels
 * yhat : estimated labels
 *
 */
template<class T, std::enable_if_t<std::is_integral<T>::value, int> = 0>
inline void confusion_matrix(const size_t n_obs, T* y, T* yhat, int64_t* const conf_mat) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[static_cast<bool>(*y) * 2 + static_cast<bool>(*yhat)]++; yhat++; y++;
    }
}

/* Fill binary confusion matrix based on true labels y and estimated labels yhat
 *
 * --- NOTE ---
 * - this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * * expects conf_mat to point to zero'd memory
 * --- NOTE ---
 *
 * --- Parameters ---
 * n_obs : minimum length of y and yhat
 * y : true labels
 * yhat : estimated labels
 *
 */
template<class T, std::enable_if_t<std::is_floating_point<T>::value, int> = 1>
inline void confusion_matrix(const size_t n_obs, T* y,  T* yhat, int64_t* const conf_mat) {
    static constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > epsilon) * 2 + (*yhat > epsilon)]++; yhat++; y++;
    }
}

/* Fill binary confusion matrix based on true labels y and estimated probalities
 *
 * --- NOTE ---
 * - this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * * expects conf_mat to point to zero'd memory
 * --- NOTE ---
 *
 * --- Parameters ---
 * n_obs : minimum length of y and yhat
 * y : true labels
 * proba : estimated probalities
 * threshold : inclusive classification threshold
 */
template<class T, typename A, std::enable_if_t<std::is_integral<T>::value, int> = 2>
inline void confusion_matrix(
    const size_t n_obs, T* y, A* proba, const A threshold, int64_t* const conf_mat
) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[static_cast<bool>(*y) * 2 + (*proba >= threshold)]++; proba++; y++;
    }
}

/* Fill binary confusion matrix based on true labels y and estimated probalities
 *
 * --- NOTE ---
 * - this function:
 * * does not handle nullptrs
 * * expects all memory to be contiguous
 * * expects conf_mat to point to zero'd memory
 * --- NOTE ---
 *
 * --- Parameters ---
 * n_obs : minimum length of y and yhat
 * y : true labels
 * proba : estimated probalities
 * threshold : inclusive classification threshold
 */
template<class T, typename A, std::enable_if_t<std::is_floating_point<T>::value, int> = 3>
inline void confusion_matrix(
    const size_t n_obs, T* y, A* proba, const A threshold, int64_t* const conf_mat
) {
    static constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > epsilon) * 2 + (*proba >= threshold)]++; proba++; y++;
    }
}

}  // namespace details


namespace bindings {

/* Compute the confusion matrix given true labels y and estimated labels yhat.
 *
 * --- Parameters ---
 * - y : true labels
 * - yhat : estimated labels
 *
 * --- Returns ---
 * - confusion matrix
 */
template <typename T>
py::array_t<int64_t> confusion_matrix(const py::array_t<T>& y, const py::array_t<T>& yhat) {
    int y_obs_axis = details::check_1d_soft(y, "y");
    details::check_contiguous(y, "y");
    int yhat_obs_axis = details::check_1d_soft(yhat, "yhat");
    details::check_contiguous(yhat, "yhat");
    details::check_equal_length(y, yhat, "y", "yhat", y_obs_axis, yhat_obs_axis);

    auto conf_mat = details::allocate_2d_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = reinterpret_cast<int64_t*>(conf_mat.request().ptr);

    auto y_ptr = reinterpret_cast<T*>(y.request().ptr);
    auto yhat_ptr = reinterpret_cast<T*>(yhat.request().ptr);
    details::confusion_matrix<T>(y.size(), y_ptr, yhat_ptr, cm_ptr);

    return conf_mat;
}

/* Compute the confusion matrix given true labels y and estimated probalities proba.
 *
 * --- Parameters ---
 * - y : true labels
 * - proba : estimated probalities
 * - threshold : inclusive classification threshold
 *
 * --- Returns ---
 * - confusion matrix
 */
template <typename T>
py::array_t<int64_t> confusion_matrix(
    const py::array_t<T>& y, const py::array_t<double>& proba, const double threshold
) {
    int y_obs_axis = details::check_1d_soft(y, "y");
    details::check_contiguous(y, "y");
    int proba_obs_axis = details::check_1d_soft(proba, "proba");
    details::check_contiguous(proba, "proba");
    details::check_equal_length(y, proba, "y", "proba", y_obs_axis, proba_obs_axis);

    auto conf_mat = details::allocate_2d_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = reinterpret_cast<int64_t*>(conf_mat.request().ptr);

    auto y_ptr = reinterpret_cast<T*>(y.request().ptr);
    auto proba_ptr = reinterpret_cast<double*>(proba.request().ptr);
    details::confusion_matrix<T>(y.size(), y_ptr, proba_ptr, threshold, cm_ptr);

    return conf_mat;
}

}  // namespace bindings
}  // namespace mmu

#endif  // CPP_INCLUDE_MMU_CONFUSION_MATRIX_HPP_
