/* error_prop.hpp -- Implementation of varianceand CI of Normal distributions
 * over the Poisson errors of the Confusion Matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_ERROR_PROP_HPP_
#define INCLUDE_MMU_CORE_ERROR_PROP_HPP_

#include <cmath>      // for sqrt
#include <limits>     // for numeric_limits
#include <cinttypes>  // for int64_t
#include <algorithm>  // for max/min
#include <stdexcept>  // for runtime_error
#include <type_traits>  // for enable_if_t

#include <mmu/core/common.hpp>
#include <mmu/core/erfinv.hpp>

namespace mmu {
namespace core {

/* Compute PPF of Normal distribution
 *
 * Parameters
 * ----------
 * mu : mean of distribution
 * sigma : std dev of distribution
 * p : percentile to compue
 */
template <typename T, isFloat<T> = true>
inline double norm_ppf(const T mu, const T sigma, const T p) {
    const double sqrt2 = 1.414213562373095048801688724209698079;
    return mu + sigma * sqrt2 * erfinv<T>(2 * p - 1);
}

/* This function computes the uncertainty on the precision-recall curve
 * using linear error propagation over the Poisson error.
 *
 * Another way of thinking about this as a Multivariate Normal over the Poisson errors.
 * Sets the following values at metrics index:
 *    0 - pos.precision aka Positive Predictive Value
 *    1 - variance of [0]
 *    2 - pos.recall aka True Positive Rate aka Sensitivity
 *    3 - variance of [2]
 *    4 - covariance between precision and recall
 */
template<typename T, isInt<T> = true>
inline void lep_conf_mat(T* const conf_mat, double* const metrics) {
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
    auto FP = static_cast<double>(conf_mat[1]);
    auto FN = static_cast<double>(conf_mat[2]);
    auto TP = static_cast<double>(conf_mat[3]);

    auto TP_FN = static_cast<double>(conf_mat[2] + conf_mat[3]);
    auto TP_FP = static_cast<double>(conf_mat[1] + conf_mat[3]);

    // tpr == recall = TP/P = TP/(TP + FN)
    // precision == positive predictive value = TP/PP = TP/(TP + FP)
    double fn_tp_sq = std::pow(TP_FN, 2.);
    double fp_tp_sq = std::pow(TP_FP, 2.);

    double recall_d_TP = FN / fn_tp_sq;
    double recall_d_FN = - TP / fn_tp_sq;
    double precision_d_TP = FP / fp_tp_sq;
    double precision_d_FP = - TP / fp_tp_sq;

    double TP_var = std::max(TP, 1.0);
    double FN_var = std::max(FN, 1.0);
    double FP_var = std::max(FP, 1.0);

    // precision
    metrics[0] = TP / (TP_FP);
    // # precision variance
    metrics[1] = (
        std::pow(precision_d_TP, 2) * TP_var
        + std::pow(precision_d_FP, 2) * FP_var
    );
    // recall
    metrics[2] = TP / (TP_FN);
    // recall variance
    metrics[3] = (
        std::pow(recall_d_TP, 2) * TP_var
        + std::pow(recall_d_FN, 2) * FN_var
    );
    // covariance
    metrics[4] = recall_d_TP * precision_d_TP * TP_var;
}  // lep_conf_mat

template<typename T, isInt<T> = true>
inline void lep_conf_mat_ci(T* const conf_mat, double* const metrics, const double alpha) {
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
    const double alpha_lb = alpha / 2;
    const double alpha_ub = 1.0 - alpha_ub;
    auto FP = static_cast<double>(conf_mat[1]);
    auto FN = static_cast<double>(conf_mat[2]);
    auto TP = static_cast<double>(conf_mat[3]);

    auto TP_FN = static_cast<double>(conf_mat[2] + conf_mat[3]);
    auto TP_FP = static_cast<double>(conf_mat[1] + conf_mat[3]);

    // tpr == recall = TP/P = TP/(TP + FN)
    // precision == positive predictive value = TP/PP = TP/(TP + FP)
    double fn_tp_sq = std::pow(TP_FN, 2.);
    double fp_tp_sq = std::pow(TP_FP, 2.);

    double recall_d_TP = FN / fn_tp_sq;
    double recall_d_FN = - TP / fn_tp_sq;
    double precision_d_TP = FP / fp_tp_sq;
    double precision_d_FP = - TP / fp_tp_sq;

    double TP_var = std::max(TP, 1.0);
    double FN_var = std::max(FN, 1.0);
    double FP_var = std::max(FP, 1.0);

    // precision
    metrics[1] = TP / (TP_FP);
    double precision_sigma = std::sqrt(
        std::pow(precision_d_TP, 2) * TP_var
        + std::pow(precision_d_FP, 2) * FP_var
    );
    metrics[0] = norm_ppf(metrics[1], precision_sigma, alpha_lb);
    metrics[2] = norm_ppf(metrics[1], precision_sigma, alpha_ub);

    // recall
    metrics[4] = TP / (TP_FN);
    double recall_sigma =  std::sqrt(
        std::pow(recall_d_TP, 2) * TP_var
        + std::pow(recall_d_FN, 2) * FN_var
    );
    metrics[3] = norm_ppf(metrics[4], recall_sigma, alpha_lb);
    metrics[5] = norm_ppf(metrics[4], recall_sigma, alpha_ub);
}  // lep_conf_mat


}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_ERROR_PROP_HPP_
