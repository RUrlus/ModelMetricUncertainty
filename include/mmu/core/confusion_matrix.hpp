/* confusion_matrix.hpp -- Implementation of binary classification confusion matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_
#define INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif

#include <cmath>
#include <string>
#include <limits>
#include <cinttypes>
#include <algorithm>
#include <type_traits>

#include <mmu/core/common.hpp>
#include <mmu/core/multinomial.hpp>

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
 * Note that this function is assymmetric for the equality check as it uses
 * the scale of `b` to determine the tollerance.
 */
template <typename T1, typename T2, isFloat<T1> = true, isFloat<T2> = true>
bool greater_equal_tol(const T1 a, const T2 b, const double rtol = 1e-05, const double atol = 1e-8) {
    double delta = a - b;
    double scaled_tol = atol + rtol * std::abs(b);
    // the first condition checks if a and b are approximately equal
    // the second condition checks if a is greater than b given the tollerance
    return std::abs(delta) <= scaled_tol || delta > scaled_tol;
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
    const bool* __restrict y,
    const bool* __restrict yhat,
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
    const T1* __restrict y,
    const T2* __restrict yhat,
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
    const T1* __restrict y,
    const T2* __restrict yhat,
    int64_t* __restrict const conf_mat
) {
    constexpr T1 epsilon_T1 = std::numeric_limits<T1>::epsilon();
    constexpr T2 epsilon_T2 = std::numeric_limits<T2>::epsilon();
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
    const T1* __restrict y,
    const T2* __restrict score,
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
    const T1* __restrict y,
    const T2* __restrict score,
    const T2 threshold,
    int64_t* __restrict const conf_mat
) {
    constexpr T1 epsilon = std::numeric_limits<T1>::epsilon();
    for (size_t i = 0; i < n_obs; i++) {
        conf_mat[(*y > epsilon) * 2 + greater_equal_tol(*score, threshold)]++; score++; y++;
    }
}

template <typename T, isInt<T> = true>
inline T* generate_confusion_matrices(
    const size_t n_matrices,
    const size_t N,
    const double* probas,
    const uint64_t seed = 0,
    T* samples = nullptr
) {
    if (!samples) {
        samples = reinterpret_cast<T*>(std::malloc(n_matrices * 4 * sizeof(double)));
    }
    pcg64_dxsm rng;
    if (seed == 0) {
        pcg_seed_seq seed_source;
        rng.seed(seed_source);
    } else {
        rng.seed(seed);
    }

    for (size_t i = 0; i < n_matrices; i++) {
        multinomial_rvs<T, 4>(rng, N, probas, samples);
        samples += 4;
    }
    return samples;
}

#if defined(MMU_HAS_OPENMP_SUPPORT)
template <typename T, isInt<T> = true>
inline T* generate_confusion_matrices_mt(
    const size_t n_matrices,
    const size_t N,
    const double* probas,
    const uint64_t seed = 0,
    T* samples = nullptr
) {
    if (!samples) {
        samples = reinterpret_cast<T*>(std::malloc(n_matrices * 4 * sizeof(double)));
    }
    pcg64_dxsm global_rng;
    if (seed == 0) {
        pcg_seed_seq seed_source;
        global_rng.seed(seed_source);
    } else {
        global_rng.seed(seed);
    }
#pragma omp parallel shared(N, probas, samples)
    {
        pcg64_dxsm rng = global_rng;
        rng.set_stream(omp_get_thread_num() + 1);

#pragma omp for
        for (size_t i = 0; i < n_matrices; i++) {
            multinomial_rvs<T, 4>(rng, N, probas, samples + (i * 4));
        }
    }  // omp parallel
    return samples;
}
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_CONFUSION_MATRIX_HPP_
