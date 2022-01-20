/* confusion_matrix.hpp -- Implementation of binary classification confusion matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_CORE_INCLUDE_MMU_CONFUSION_MATRIX_HPP_
#define MMU_CORE_INCLUDE_MMU_CONFUSION_MATRIX_HPP_

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
#include <algorithm>
#include <type_traits>

#include "mmu/numpy.hpp"

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
 * - n_obs : minimum length of y and yhat
 * - y : true labels
 * - yhat : estimated labels
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
inline void confusion_matrix(const size_t n_obs, bool* y, bool* yhat, int64_t* const conf_mat) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[*y * 2 + *yhat]++; yhat++; y++;
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
 * - n_obs : minimum length of y and yhat
 * - y : true labels
 * - yhat : estimated labels
 * - conf_mat : allocated and zero'd memory for the confusion matrix
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
 * - n_obs : minimum length of y and yhat
 * - y : true labels
 * - yhat : estimated labels
 * - conf_mat : allocated and zero'd memory for the confusion matrix
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
 * - n_obs : minimum length of y and yhat
 * - y : true labels
 * - proba : estimated probalities
 * - threshold : inclusive classification threshold
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
template<class iT, std::enable_if_t<std::is_integral<iT>::value, int> = 2>
inline void confusion_matrix(
    const size_t n_obs, iT* y, float* proba, const float threshold, int64_t* const conf_mat
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
 * - n_obs : minimum length of y and yhat
 * - y : true labels
 * - proba : estimated probalities
 * - threshold : inclusive classification threshold
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
template<class iT, std::enable_if_t<std::is_integral<iT>::value, int> = 2>
inline void confusion_matrix(
    const size_t n_obs,
    iT* y,
    double* proba,
    const double threshold,
    int64_t* const conf_mat
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
 * - n_obs : minimum length of y and yhat
 * - y : true labels
 * - proba : estimated probalities
 * - threshold : inclusive classification threshold
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
template<class fT, std::enable_if_t<std::is_floating_point<fT>::value, int> = 3>
inline void confusion_matrix(
    const size_t n_obs, fT* y, float* proba, const float threshold, int64_t* const conf_mat
) {
    static constexpr fT epsilon = std::numeric_limits<float>::epsilon();
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > epsilon) * 2 + (*proba >= threshold)]++; proba++; y++;
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
 * - n_obs : minimum length of y and yhat
 * - y : true labels
 * - proba : estimated probalities
 * - threshold : inclusive classification threshold
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
template<class fT, std::enable_if_t<std::is_floating_point<fT>::value, int> = 3>
inline void confusion_matrix(
    const size_t n_obs,
    fT* y,
    double* proba,
    const double threshold,
    int64_t* const conf_mat
) {
    static constexpr fT epsilon = std::numeric_limits<float>::epsilon();
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
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(yhat))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const size_t n_obs = std::min(y.size(), yhat.size());
    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto y_ptr = npy::get_data(y);
    auto yhat_ptr = npy::get_data(yhat);
    details::confusion_matrix<T>(n_obs, y_ptr, yhat_ptr, cm_ptr);

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
template <
    typename iT, typename fT, std::enable_if_t<std::is_floating_point<fT>::value, int> = 0
>
py::array_t<int64_t> confusion_matrix(
    const py::array_t<iT>& y, const py::array_t<fT>& proba, const fT threshold
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(proba))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const size_t n_obs = std::min(y.size(), proba.size());
    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    iT* y_ptr = npy::get_data(y);
    fT* proba_ptr = npy::get_data<fT>(proba);
    details::confusion_matrix<iT>(n_obs, y_ptr, proba_ptr, threshold, cm_ptr);

    return conf_mat;
}

}  // namespace bindings
}  // namespace mmu

#endif  // MMU_CORE_INCLUDE_MMU_CONFUSION_MATRIX_HPP_
