/* confusion_matrix.hpp -- Implementation of binary classification confusion
 * matrix Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_
#define INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include <mmu/core/common.hpp>
#include <mmu/core/random.hpp>

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

/* Check if a is greater or equal to b taking into account floating point noise
 *
 * Returns true if a >= b - tolerance, i.e., a is not significantly less than b.
 *
 * Note that this function is asymmetric for the equality check as it uses
 * the scale of `b` to determine the tolerance.
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline bool greater_equal_tol(
    const T1 a,
    const T2 b,
    const double rtol = 1e-05,
    const double atol = 1e-8) {
    const double delta = a - b;
    const double scaled_tol = atol + rtol * b;
    return delta >= -scaled_tol;
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
    const int64_t n_obs,
    const bool* __restrict y,
    const bool* __restrict yhat,
    int64_t* __restrict const conf_mat) {
    int64_t local_cm[4] = {0, 0, 0, 0};
    const int64_t n_unroll = n_obs - (n_obs % 8);

    for (int64_t i = 0; i < n_unroll; i += 8) {
        local_cm[y[i] * 2 + yhat[i]]++;
        local_cm[y[i+1] * 2 + yhat[i+1]]++;
        local_cm[y[i+2] * 2 + yhat[i+2]]++;
        local_cm[y[i+3] * 2 + yhat[i+3]]++;
        local_cm[y[i+4] * 2 + yhat[i+4]]++;
        local_cm[y[i+5] * 2 + yhat[i+5]]++;
        local_cm[y[i+6] * 2 + yhat[i+6]]++;
        local_cm[y[i+7] * 2 + yhat[i+7]]++;
    }
    for (int64_t i = n_unroll; i < n_obs; i++) {
        local_cm[y[i] * 2 + yhat[i]]++;
    }
    for (int j = 0; j < 4; j++) {
        conf_mat[j] = local_cm[j];
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
template <typename T1, typename T2, isInt<T1> = true, isInt<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict yhat,
    int64_t* __restrict const conf_mat) {
    int64_t local_cm[4] = {0, 0, 0, 0};
    const int64_t n_unroll = n_obs - (n_obs % 8);

    for (int64_t i = 0; i < n_unroll; i += 8) {
        local_cm[static_cast<bool>(y[i]) * 2 + static_cast<bool>(yhat[i])]++;
        local_cm[static_cast<bool>(y[i+1]) * 2 + static_cast<bool>(yhat[i+1])]++;
        local_cm[static_cast<bool>(y[i+2]) * 2 + static_cast<bool>(yhat[i+2])]++;
        local_cm[static_cast<bool>(y[i+3]) * 2 + static_cast<bool>(yhat[i+3])]++;
        local_cm[static_cast<bool>(y[i+4]) * 2 + static_cast<bool>(yhat[i+4])]++;
        local_cm[static_cast<bool>(y[i+5]) * 2 + static_cast<bool>(yhat[i+5])]++;
        local_cm[static_cast<bool>(y[i+6]) * 2 + static_cast<bool>(yhat[i+6])]++;
        local_cm[static_cast<bool>(y[i+7]) * 2 + static_cast<bool>(yhat[i+7])]++;
    }
    for (int64_t i = n_unroll; i < n_obs; i++) {
        local_cm[static_cast<bool>(y[i]) * 2 + static_cast<bool>(yhat[i])]++;
    }
    for (int j = 0; j < 4; j++) {
        conf_mat[j] = local_cm[j];
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
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict yhat,
    int64_t* __restrict const conf_mat) {
    constexpr T1 epsilon_T1 = std::numeric_limits<T1>::epsilon();
    constexpr T2 epsilon_T2 = std::numeric_limits<T2>::epsilon();
    int64_t local_cm[4] = {0, 0, 0, 0};
    const int64_t n_unroll = n_obs - (n_obs % 8);

    for (int64_t i = 0; i < n_unroll; i += 8) {
        local_cm[(y[i] > epsilon_T1) * 2 + (yhat[i] > epsilon_T2)]++;
        local_cm[(y[i+1] > epsilon_T1) * 2 + (yhat[i+1] > epsilon_T2)]++;
        local_cm[(y[i+2] > epsilon_T1) * 2 + (yhat[i+2] > epsilon_T2)]++;
        local_cm[(y[i+3] > epsilon_T1) * 2 + (yhat[i+3] > epsilon_T2)]++;
        local_cm[(y[i+4] > epsilon_T1) * 2 + (yhat[i+4] > epsilon_T2)]++;
        local_cm[(y[i+5] > epsilon_T1) * 2 + (yhat[i+5] > epsilon_T2)]++;
        local_cm[(y[i+6] > epsilon_T1) * 2 + (yhat[i+6] > epsilon_T2)]++;
        local_cm[(y[i+7] > epsilon_T1) * 2 + (yhat[i+7] > epsilon_T2)]++;
    }
    for (int64_t i = n_unroll; i < n_obs; i++) {
        local_cm[(y[i] > epsilon_T1) * 2 + (yhat[i] > epsilon_T2)]++;
    }
    for (int j = 0; j < 4; j++) {
        conf_mat[j] = local_cm[j];
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
template <typename T1, typename T2, isInt<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    int64_t* __restrict const conf_mat) {
    int64_t local_cm[4] = {0, 0, 0, 0};
    const int64_t n_unroll = n_obs - (n_obs % 8);

    for (int64_t i = 0; i < n_unroll; i += 8) {
        local_cm[static_cast<bool>(y[i]) * 2 + greater_equal_tol(score[i], threshold)]++;
        local_cm[static_cast<bool>(y[i+1]) * 2 + greater_equal_tol(score[i+1], threshold)]++;
        local_cm[static_cast<bool>(y[i+2]) * 2 + greater_equal_tol(score[i+2], threshold)]++;
        local_cm[static_cast<bool>(y[i+3]) * 2 + greater_equal_tol(score[i+3], threshold)]++;
        local_cm[static_cast<bool>(y[i+4]) * 2 + greater_equal_tol(score[i+4], threshold)]++;
        local_cm[static_cast<bool>(y[i+5]) * 2 + greater_equal_tol(score[i+5], threshold)]++;
        local_cm[static_cast<bool>(y[i+6]) * 2 + greater_equal_tol(score[i+6], threshold)]++;
        local_cm[static_cast<bool>(y[i+7]) * 2 + greater_equal_tol(score[i+7], threshold)]++;
    }
    for (int64_t i = n_unroll; i < n_obs; i++) {
        local_cm[static_cast<bool>(y[i]) * 2 + greater_equal_tol(score[i], threshold)]++;
    }
    for (int j = 0; j < 4; j++) {
        conf_mat[j] = local_cm[j];
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
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    int64_t* __restrict const conf_mat) {
    constexpr T1 epsilon = std::numeric_limits<T1>::epsilon();
    int64_t local_cm[4] = {0, 0, 0, 0};
    const int64_t n_unroll = n_obs - (n_obs % 8);

    for (int64_t i = 0; i < n_unroll; i += 8) {
        local_cm[(y[i] > epsilon) * 2 + greater_equal_tol(score[i], threshold)]++;
        local_cm[(y[i+1] > epsilon) * 2 + greater_equal_tol(score[i+1], threshold)]++;
        local_cm[(y[i+2] > epsilon) * 2 + greater_equal_tol(score[i+2], threshold)]++;
        local_cm[(y[i+3] > epsilon) * 2 + greater_equal_tol(score[i+3], threshold)]++;
        local_cm[(y[i+4] > epsilon) * 2 + greater_equal_tol(score[i+4], threshold)]++;
        local_cm[(y[i+5] > epsilon) * 2 + greater_equal_tol(score[i+5], threshold)]++;
        local_cm[(y[i+6] > epsilon) * 2 + greater_equal_tol(score[i+6], threshold)]++;
        local_cm[(y[i+7] > epsilon) * 2 + greater_equal_tol(score[i+7], threshold)]++;
    }
    for (int64_t i = n_unroll; i < n_obs; i++) {
        local_cm[(y[i] > epsilon) * 2 + greater_equal_tol(score[i], threshold)]++;
    }
    for (int j = 0; j < 4; j++) {
        conf_mat[j] = local_cm[j];
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
 * - scaled_tol : pre-computed tolerance (atol + rtol * threshold)
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
template <typename T1, typename T2, isInt<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    const T2 scaled_tol,
    int64_t* __restrict const conf_mat) {
    int64_t local_cm[4] = {0, 0, 0, 0};
    const int64_t n_unroll = n_obs - (n_obs % 8);

    for (int64_t i = 0; i < n_unroll; i += 8) {
        local_cm[static_cast<bool>(y[i]) * 2 + (score[i] - threshold >= -scaled_tol)]++;
        local_cm[static_cast<bool>(y[i+1]) * 2 + (score[i+1] - threshold >= -scaled_tol)]++;
        local_cm[static_cast<bool>(y[i+2]) * 2 + (score[i+2] - threshold >= -scaled_tol)]++;
        local_cm[static_cast<bool>(y[i+3]) * 2 + (score[i+3] - threshold >= -scaled_tol)]++;
        local_cm[static_cast<bool>(y[i+4]) * 2 + (score[i+4] - threshold >= -scaled_tol)]++;
        local_cm[static_cast<bool>(y[i+5]) * 2 + (score[i+5] - threshold >= -scaled_tol)]++;
        local_cm[static_cast<bool>(y[i+6]) * 2 + (score[i+6] - threshold >= -scaled_tol)]++;
        local_cm[static_cast<bool>(y[i+7]) * 2 + (score[i+7] - threshold >= -scaled_tol)]++;
    }
    for (int64_t i = n_unroll; i < n_obs; i++) {
        local_cm[static_cast<bool>(y[i]) * 2 + (score[i] - threshold >= -scaled_tol)]++;
    }
    for (int j = 0; j < 4; j++) {
        conf_mat[j] = local_cm[j];
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
 * - scaled_tol : pre-computed tolerance (atol + rtol * threshold)
 * - conf_mat : allocated and zero'd memory for the confusion matrix
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
inline void confusion_matrix(
    const int64_t n_obs,
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    const T2 scaled_tol,
    int64_t* __restrict const conf_mat) {
    constexpr T1 epsilon = std::numeric_limits<T1>::epsilon();
    int64_t local_cm[4] = {0, 0, 0, 0};
    const int64_t n_unroll = n_obs - (n_obs % 8);

    for (int64_t i = 0; i < n_unroll; i += 8) {
        local_cm[(y[i] > epsilon) * 2 + (score[i] - threshold >= -scaled_tol)]++;
        local_cm[(y[i+1] > epsilon) * 2 + (score[i+1] - threshold >= -scaled_tol)]++;
        local_cm[(y[i+2] > epsilon) * 2 + (score[i+2] - threshold >= -scaled_tol)]++;
        local_cm[(y[i+3] > epsilon) * 2 + (score[i+3] - threshold >= -scaled_tol)]++;
        local_cm[(y[i+4] > epsilon) * 2 + (score[i+4] - threshold >= -scaled_tol)]++;
        local_cm[(y[i+5] > epsilon) * 2 + (score[i+5] - threshold >= -scaled_tol)]++;
        local_cm[(y[i+6] > epsilon) * 2 + (score[i+6] - threshold >= -scaled_tol)]++;
        local_cm[(y[i+7] > epsilon) * 2 + (score[i+7] - threshold >= -scaled_tol)]++;
    }
    for (int64_t i = n_unroll; i < n_obs; i++) {
        local_cm[(y[i] > epsilon) * 2 + (score[i] - threshold >= -scaled_tol)]++;
    }
    for (int j = 0; j < 4; j++) {
        conf_mat[j] = local_cm[j];
    }
}

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_
