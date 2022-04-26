/* mvn_error.hpp -- Implementation of variance and CI of Multivariate Normal distributions
 * over the Poisson errors of the Confusion Matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_MVN_ERROR_HPP_
#define INCLUDE_MMU_CORE_MVN_ERROR_HPP_

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
    static const double sqrt2 = 1.414213562373095048801688724209698079;
    return mu + sigma * sqrt2 * erfinv<T>(2 * p - 1);
}

/* This function computes the uncertainty on the precision-recall curve
 * using linear error propagation over the Poisson error.
 *
 * Another way of thinking about this as a Multivariate Normal over the Poisson errors.
 * Sets the following values at metrics index:
 *    0 - pos.precision aka Positive Predictive Value
 *    1 - lower-bound of marginal CI precision
 *    2 - upper-bound of marginal CI precision
 *    3 - pos.recall aka True Positive Rate aka Sensitivity
 *    4 - lower-bound of marginal CI recall
 *    5 - upper-bound of marginal CI recall
 *    6 - variance of precision
 *    7 - covariance precision, recall
 *    8 - variance of recall
 *    9 - covariance precision, recall
 */
template<typename T, isInt<T> = true>
inline void pr_mvn_error(
    T* __restrict const conf_mat,
    double* __restrict const metrics,
    double alpha
) {
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
    const double alpha_ub = 1.0 - alpha_lb;
    const auto FP = static_cast<double>(conf_mat[1]);
    const auto FN = static_cast<double>(conf_mat[2]);
    const auto TP = static_cast<double>(conf_mat[3]);

    const auto TP_FN = static_cast<double>(conf_mat[2] + conf_mat[3]);
    const auto TP_FP = static_cast<double>(conf_mat[1] + conf_mat[3]);

    const double fn_tp_sq = std::pow(TP_FN, 2.);
    const double fp_tp_sq = std::pow(TP_FP, 2.);

    const double recall_d_TP = FN / fn_tp_sq;
    const double recall_d_FN = - TP / fn_tp_sq;
    const double precision_d_TP = FP / fp_tp_sq;
    const double precision_d_FP = - TP / fp_tp_sq;

    const double TP_var = std::max(TP, 1.0);
    const double FN_var = std::max(FN, 1.0);
    const double FP_var = std::max(FP, 1.0);

    // precision
    double prec = TP / (TP_FP);
    double prec_var = (
        std::pow(precision_d_TP, 2) * TP_var
        + std::pow(precision_d_FP, 2) * FP_var
    );
    double prec_sigma = std::sqrt(prec_var);
    metrics[0] = prec;
    metrics[1] = norm_ppf(prec, prec_sigma, alpha_lb);
    metrics[2] = norm_ppf(prec, prec_sigma, alpha_ub);

    // recall
    double rec = TP / (TP_FN);
    double rec_var = (
        std::pow(recall_d_TP, 2) * TP_var
        + std::pow(recall_d_FN, 2) * FN_var
    );
    double rec_sigma = std::sqrt(rec_var);
    metrics[3] = rec;
    metrics[4] = norm_ppf(rec, rec_sigma, alpha_lb);
    metrics[5] = norm_ppf(rec, rec_sigma, alpha_ub);

    // covariance
    double cov = recall_d_TP * precision_d_TP * TP_var;
    metrics[6] = prec_var;
    metrics[7] = cov;
    metrics[8] = cov;
    metrics[9] = rec_var;
}  // pr_mvn_error

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_MVN_ERROR_HPP_
