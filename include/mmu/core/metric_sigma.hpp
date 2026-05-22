/* metric_sigma.hpp -- Compute standard deviation estimates for binary metrics
 * using linear error propagation over Poisson errors.
 * These are used for computing grid bounds in multinomial log-likelihood methods.
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_METRIC_SIGMA_HPP_
#define INCLUDE_MMU_CORE_METRIC_SIGMA_HPP_

#include <cinttypes>
#include <cmath>

#include <mmu/core/common.hpp>

namespace mmu {
namespace core {

namespace pr {

/* Compute precision, recall and their standard deviations.
 * Used for determining grid bounds for profile log-likelihood computation.
 *
 * Sets the following values at metrics index:
 * 0 - precision
 * 1 - precision sigma (standard deviation)
 * 2 - recall
 * 3 - recall sigma (standard deviation)
 */
template <typename T, isInt<T> = true>
inline void metric_sigma(
    const T* __restrict const conf_mat,
    double* __restrict const metrics) {
    /*
     *                  pred
     *                0     1
     *  actual  0    TN    FP
     *          1    FN    TP
     *
     *  Flattened we have:
     *  0 TN
     *  1 FP
     *  2 FN
     *  3 TP
     *
     */
    const int64_t itp = conf_mat[3];
    const auto tp = static_cast<double>(itp);

    const int64_t itp_fn = conf_mat[2] + conf_mat[3];
    const bool tp_fn_nonzero = itp_fn > 0;
    const auto tp_fn = static_cast<double>(itp_fn);

    const int64_t itp_fp = conf_mat[3] + conf_mat[1];
    const bool tp_fp_nonzero = itp_fp > 0;
    const auto tp_fp = static_cast<double>(itp_fp);

    // precision
    double prec;
    double prec_sigma;
    double prec_for_sigma;
    // precision == 1
    if (itp == itp_fp) {
        prec = 1.0;
        prec_for_sigma = static_cast<double>(itp_fp - 1) / tp_fp;
        prec_sigma = std::sqrt((prec_for_sigma * (1 - prec_for_sigma)) / tp_fp);
    } else if (tp_fp_nonzero) {
        prec = tp / (tp_fp);
        prec_sigma = std::sqrt(
            static_cast<double>(conf_mat[3] * conf_mat[1])
            / static_cast<double>(std::pow(tp_fp, 3.0)));
    } else {
        // precision == 0
        prec = 0.0;
        prec_for_sigma = 1.0 / tp_fp;
        prec_sigma = std::sqrt((prec_for_sigma * (1 - prec_for_sigma)) / tp_fp);
    }

    // recall
    double rec;
    double rec_sigma;
    double rec_for_sigma;
    // recall == 1
    if (itp == itp_fn) {
        rec = 1.0;
        rec_for_sigma = static_cast<double>(itp_fn - 1) / tp_fn;
        rec_sigma = std::sqrt((rec_for_sigma * (1 - rec_for_sigma)) / tp_fn);
    } else if (tp_fn_nonzero) {
        rec = tp / (tp_fn);
        rec_sigma = std::sqrt(
            static_cast<double>(conf_mat[3] * conf_mat[2])
            / static_cast<double>(std::pow(tp_fn, 3.0)));
    } else {
        // recall == 0.0
        rec = 0.0;
        rec_for_sigma = 1.0 / tp_fn;
        rec_sigma = std::sqrt((rec_for_sigma * (1 - rec_for_sigma)) / tp_fn);
    }

    metrics[0] = prec;
    metrics[1] = prec_sigma;
    metrics[2] = rec;
    metrics[3] = rec_sigma;
}

// Backward compatibility alias
template <typename T, isInt<T> = true>
inline void bvn_sigma(
    const T* __restrict const conf_mat,
    double* __restrict const metrics) {
    metric_sigma(conf_mat, metrics);
}

}  // namespace pr

namespace roc {

/* Compute TPR, FPR and their standard deviations.
 * Used for determining grid bounds for profile log-likelihood computation.
 *
 * Sets the following values at metrics index:
 * 0 - Y (TPR)
 * 1 - Y sigma (TPR standard deviation)
 * 2 - X (FPR)
 * 3 - X sigma (FPR standard deviation)
 */
template <typename T, isInt<T> = true>
inline void metric_sigma(
    const T* __restrict const conf_mat,
    double* __restrict const metrics) {
    /*
     *                  pred
     *                0     1
     *  actual  0    TN    FP
     *          1    FN    TP
     *
     *  Flattened we have:
     *  0 TN
     *  1 FP
     *  2 FN
     *  3 TP
     *
     */

    // Y: TPR = TP / (TP + FN)
    const int64_t iterm1_Y = conf_mat[3]; // TP
    const bool term1_Y_nonzero = iterm1_Y > 0;
    auto term1_Y = static_cast<double>(iterm1_Y);

    const int64_t iterm2_Y = conf_mat[2]; // FN
    const bool term2_Y_nonzero = iterm2_Y > 0;
    auto term2_Y = static_cast<double>(iterm2_Y);

    double Y = term1_Y / (term1_Y + term2_Y);

    if (!term1_Y_nonzero) {
        term1_Y = 1;
    }
    if (!term2_Y_nonzero) {
        term2_Y = 1;
    }

    double Y_sigma = std::sqrt(
        static_cast<double>(term1_Y * term2_Y)
        / static_cast<double>(std::pow(term1_Y + term2_Y, 3.0)));

    // X: FPR = FP / (FP + TN)
    const int64_t iterm1_X = conf_mat[1]; // FP
    const bool term1_X_nonzero = iterm1_X > 0;
    auto term1_X = static_cast<double>(iterm1_X);

    const int64_t iterm2_X = conf_mat[0]; // TN
    const bool term2_X_nonzero = iterm2_X > 0;
    auto term2_X = static_cast<double>(iterm2_X);

    double X = term1_X / (term1_X + term2_X);

    if (!term1_X_nonzero) {
        term1_X = 1;
    }
    if (!term2_X_nonzero) {
        term2_X = 1;
    }

    double X_sigma = std::sqrt(
        static_cast<double>(term1_X * term2_X)
        / static_cast<double>(std::pow(term1_X + term2_X, 3.0)));

    metrics[0] = Y;
    metrics[1] = Y_sigma;
    metrics[2] = X;
    metrics[3] = X_sigma;
}

// Backward compatibility alias
template <typename T, isInt<T> = true>
inline void bvn_sigma(
    const T* __restrict const conf_mat,
    double* __restrict const metrics) {
    metric_sigma(conf_mat, metrics);
}

}  // namespace roc

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_METRIC_SIGMA_HPP_

