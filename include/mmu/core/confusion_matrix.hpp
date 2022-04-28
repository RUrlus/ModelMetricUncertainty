/* confusion_matrix.hpp -- Implementation of binary classification confusion matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_
#define INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_

#include <cmath>
#include <string>
#include <limits>
#include <cinttypes>
#include <algorithm>
#include <type_traits>

#include <mmu/core/common.hpp>

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
namespace core {

template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
bool greater_equal_tol(T1 a, T2 b, double rtol = 1e-05, double atol = 1e-8) {
    return std::abs(a - b) <= (atol + rtol * std::max(std::abs(b), std::abs(a))) || a > b;
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
inline void confusion_matrix(
    const size_t n_obs,
    bool* __restrict y,
    bool* __restrict yhat,
    int64_t* __restrict const conf_mat
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
    T1* __restrict y,
    T2* __restrict yhat,
    int64_t* __restrict const conf_mat
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
    T1* __restrict y,
    T2* __restrict yhat,
    int64_t* __restrict const conf_mat
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
    T1* __restrict y,
    T2* __restrict score,
    const T2 threshold,
    int64_t* __restrict const conf_mat
) {
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[static_cast<bool>(*y) * 2 + greater_equal_tol(*score, threshold)]++;
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
    T1* __restrict y,
    T2* __restrict score,
    const T2 threshold,
    int64_t* __restrict const conf_mat
) {
    static constexpr T1 epsilon = std::numeric_limits<T1>::epsilon();
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > epsilon) * 2 + greater_equal_tol(*score, threshold)]++; score++; y++;
    }
}

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_
