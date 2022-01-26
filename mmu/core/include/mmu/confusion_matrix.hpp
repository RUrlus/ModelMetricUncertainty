/* confusion_matrix.hpp -- Implementation of binary classification confusion matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_CORE_INCLUDE_MMU_CONFUSION_MATRIX_HPP_
#define MMU_CORE_INCLUDE_MMU_CONFUSION_MATRIX_HPP_

/* TODO *
 *
 * - Add function over runs for yhat
 * - Add function over runs for score with single threshold
 * - Add function over runs for score over multiple thresholds
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

#include "mmu/common.hpp"
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
inline void confusion_matrix(
    const size_t n_obs,
    bool* y,
    bool* yhat,
    int64_t* const conf_mat
) {
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
template<typename T1, typename T2, isInt<T1> = true, isInt<T2> = true>
inline void confusion_matrix(
    const size_t n_obs,
    T1* y,
    T2* yhat,
    int64_t* const conf_mat
) {
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
template<typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const size_t n_obs,
    T1* y,
    T2* yhat,
    int64_t* const conf_mat
) {
    static constexpr T1 epsilon_T1 = std::numeric_limits<T1>::epsilon();
    static constexpr T2 epsilon_T2 = std::numeric_limits<T2>::epsilon();
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > epsilon_T1) * 2 + (*yhat > epsilon_T2)]++; yhat++; y++;
    }
}

/* Fill binary confusion matrix based on true labels y and classifier scores
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
 * - score : classifier scores
 * - threshold : inclusive classification threshold
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
template<typename T1, typename T2, isInt<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const size_t n_obs,
    T1* y,
    T2* score,
    const T2 threshold,
    int64_t* const conf_mat
) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[static_cast<bool>(*y) * 2 + (*score >= threshold)]++;
        y++;
        score++;
    }
}

/* Fill binary confusion matrix based on true labels y and classifier scores
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
 * - score : classifier scores
 * - threshold : inclusive classification threshold
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
template<typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const size_t n_obs,
    T1* y,
    T2* score,
    const T2 threshold,
    int64_t* const conf_mat
) {
    static constexpr T1 epsilon = std::numeric_limits<T1>::epsilon();
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > epsilon) * 2 + (*score >= threshold)]++; score++; y++;
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
template <typename T1, typename T2>
py::array_t<int64_t> confusion_matrix(
    const py::array_t<T1>& y,
    const py::array_t<T2>& yhat
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(yhat))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const size_t n_obs = std::min(y.size(), yhat.size());

    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto y_ptr = npy::get_data(y);
    auto yhat_ptr = npy::get_data(yhat);
    details::confusion_matrix<T1, T2>(n_obs, y_ptr, yhat_ptr, cm_ptr);

    return conf_mat;
}

/* Compute the confusion matrix given true labels y and classifier scores.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - threshold : inclusive classification threshold
 *
 * --- Returns ---
 * - confusion matrix
 */
template <typename T1, typename T2, isFloat<T2> = true>
py::array_t<int64_t> confusion_matrix(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const T2 threshold
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const size_t n_obs = std::min(y.size(), score.size());

    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data<T2>(score);
    details::confusion_matrix<T1, T2>(
        n_obs, y_ptr, score_ptr, threshold, cm_ptr
    );

    return conf_mat;
}

}  // namespace bindings
}  // namespace mmu

#endif  // MMU_CORE_INCLUDE_MMU_CONFUSION_MATRIX_HPP_
