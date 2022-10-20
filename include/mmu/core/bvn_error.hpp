/* bvn_error.hpp -- Implementation of variance and CI of Bivariate Normal
 * distributions over the Poisson errors of the Confusion Matrix Copyright 2022
 * Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_BVN_ERROR_HPP_
#define INCLUDE_MMU_CORE_BVN_ERROR_HPP_

#include <algorithm>    // for max/min
#include <cinttypes>    // for int64_t
#include <cmath>        // for sqrt
#include <limits>       // for numeric_limits
#include <stdexcept>    // for runtime_error
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

namespace pr {

/* This function computes the uncertainty on the precision-recall curve
 * using linear error propagation over the Poisson error.
 *
 * Another way of thinking about this as a Multivariate Normal over the Poisson
 * errors. Sets the following values at metrics index:
 * 0 - Y: pos.precision aka Positive Predictive Value
 * 1 - X: pos.recall aka True Positive Rate aka Sensitivity
 * 2 - Y variance: precision variance
 * 3 - covariance precision, recall
 * 4 - covariance precision, recall
 * 5 - X variance: recall variance
 */
template <typename T, isInt<T> = true>
inline void bvn_cov(
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
    if (itp == itp_fp) {
        // precision == 1.0
        metrics[0] = 1.0;
        metrics[2] = 0.0;
    } else if (tp_fp_nonzero) {
        metrics[0] = tp / (tp_fp);
        // precision variance:
        metrics[2]
            = (static_cast<double>(conf_mat[3] * conf_mat[1])
               / static_cast<double>(std::pow(tp_fp, 3.0)));
    } else {
        // precision == 0.0
        metrics[0] = 0.0;
        metrics[2] = NAN;
    }

    // recall
    if (itp == itp_fn) {
        // recall == 1.0
        metrics[1] = 1.0;
        metrics[5] = 0.0;
    } else if (tp_fn_nonzero) {
        metrics[1] = tp / (tp_fn);
        // recall variance:
        metrics[5]
            = (static_cast<double>(conf_mat[3] * conf_mat[2])
               / static_cast<double>(std::pow(tp_fn, 3.0)));
    } else {
        // recall == 0.0
        metrics[1] = 0.0;
        metrics[5] = NAN;
    }

    // covariance
    if (itp != itp_fp && itp != itp_fn) {
        metrics[3] = metrics[4]
            = (static_cast<double>(itp * conf_mat[1] * conf_mat[2])
               / (std::pow(tp_fp, 2) * std::pow(tp_fn, 2)));
    } else {
        metrics[3] = metrics[4] = 0.0;
    }
}  // bvn_cov

/* This function computes the uncertainty on the precision-recall curve
 * using linear error propagation over the Poisson error.
 *
 * Another way of thinking about this as a Multivariate Normal over the Poisson
 * errors. Sets the following values at metrics index:
 * 0 - pos.precision aka Positive Predictive Value
 * 1 - lower-bound of marginal CI precision
 * 2 - upper-bound of marginal CI precision
 * 3 - pos.recall aka True Positive Rate aka Sensitivity
 * 4 - lower-bound of marginal CI recall
 * 5 - upper-bound of marginal CI recall
 * 6 - variance of precision
 * 7 - covariance precision, recall
 * 8 - variance of recall
 * 9 - covariance precision, recall
 */
template <typename T, isInt<T> = true>
inline void bvn_error(
    const T* __restrict const conf_mat,
    double* __restrict const metrics,
    double alpha) {
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
    const int64_t itp = conf_mat[3];
    const auto tp = static_cast<double>(itp);

    const int64_t itp_fn = conf_mat[2] + conf_mat[3];
    const bool tp_fn_nonzero = itp_fn > 0;
    const auto tp_fn = static_cast<double>(itp_fn);

    const int64_t itp_fp = conf_mat[3] + conf_mat[1];
    const bool tp_fp_nonzero = itp_fp > 0;
    const auto tp_fp = static_cast<double>(itp_fp);

    // precision
    double val;
    double val_var;
    double val_sigma;
    if (itp == itp_fp) {
        // precision == 1.0
        metrics[0] = metrics[1] = metrics[2] = 1.0;
        metrics[6] = 0.;
    } else if (tp_fp_nonzero) {
        val = tp / (tp_fp);
        // precision variance:
        val_var
            = (static_cast<double>(conf_mat[3] * conf_mat[1])
               / static_cast<double>(std::pow(tp_fp, 3.0)));
        val_sigma = std::sqrt(val_var);
        metrics[0] = val;
        metrics[1] = norm_ppf(val, val_sigma, alpha_lb);
        metrics[2] = norm_ppf(val, val_sigma, alpha_ub);
        metrics[6] = val_var;

    } else {
        // precision == 0.0
        metrics[0] = metrics[1] = metrics[2] = 0.0;
        metrics[6] = NAN;
    }

    // recall
    if (itp == itp_fn) {
        // recall == 1.0
        metrics[3] = metrics[4] = metrics[5] = 1.0;
        metrics[9] = 0.;
    } else if (tp_fn_nonzero) {
        val = tp / (tp_fn);
        // recall variance:
        val_var
            = (static_cast<double>(conf_mat[3] * conf_mat[2])
               / static_cast<double>(std::pow(tp_fn, 3.0)));
        val_sigma = std::sqrt(val_var);
        metrics[3] = val;
        metrics[4] = norm_ppf(val, val_sigma, alpha_lb);
        metrics[5] = norm_ppf(val, val_sigma, alpha_ub);
        metrics[9] = val_var;
    } else {
        // recall == 0.0
        metrics[3] = metrics[4] = metrics[5] = 0.0;
        metrics[9] = NAN;
    }

    // covariance
    metrics[7] = metrics[8]
        = (static_cast<double>(itp * conf_mat[1] * conf_mat[2])
           / (std::pow(tp_fp, 2) * std::pow(tp_fn, 2)));
}  // bvn_error

template <typename T, isInt<T> = true>
inline void bvn_sigma(
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
    // precision == 1
    if (itp == itp_fn) {
        // recall == 1
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
}  // bvn_sigma

}  // namespace pr


namespace roc {

/* This function computes the uncertainty on the ROC curve
 * using linear error propagation over the Poisson error.
 *
 * Another way of thinking about this as a Multivariate Normal over the Poisson
 * errors. Sets the following values at metrics index:
 * 0 - Y: True Positive Rate (TPR) aka pos.recall
 * 1 - X: False Negative Rate (FPR)
 * 2 - Y variance: TPR variance
 * 3 - covariance TPR, FPR
 * 4 - covariance TPR, FPR
 * 5 - X variance: FPR variance
 */
template <typename T, isInt<T> = true>
inline void bvn_cov(
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
    const int64_t ifp = conf_mat[1];
    const auto fp = static_cast<double>(ifp);

    const int64_t itp_fn = conf_mat[2] + conf_mat[3];
    const bool tp_fn_nonzero = itp_fn > 0;
    const auto tp_fn = static_cast<double>(itp_fn);

    const int64_t ifp_tn = conf_mat[1] + conf_mat[0];
    const bool fp_tn_nonzero = ifp_tn > 0;
    const auto fp_tn = static_cast<double>(ifp_tn);

    // Y: TPR, recall
    if (itp == itp_fn) {
        // recall == 1.0
        metrics[0] = 1.0;
        metrics[2] = 0.0;
    } else if (tp_fn_nonzero) {
        metrics[0] = tp / (tp_fn);
        // recall variance:
        metrics[2]
            = (static_cast<double>(conf_mat[3] * conf_mat[2])
               / static_cast<double>(std::pow(tp_fn, 3.0)));
    } else {
        // recall == 0.0
        metrics[0] = 0.0;
        metrics[2] = NAN;
    }

    // X: FPR
    if (ifp == ifp_tn) {
        // FPR == 1.0
        metrics[1] = 1.0;
        metrics[5] = 0.0;
    } else if (fp_tn_nonzero) {
        metrics[1] = fp / (fp_tn);
        // FPR variance:
        metrics[5]
            = (static_cast<double>(conf_mat[1] * conf_mat[0])
               / static_cast<double>(std::pow(fp_tn, 3.0)));
    } else {
        // FPR == 0.0
        metrics[1] = 0.0;
        metrics[5] = NAN;
    }

    // covariance
    metrics[3] = metrics[4] = 0.0;

}  // bvn_cov

/* This function computes the uncertainty on the ROC curve
 * using linear error propagation over the Poisson error.
 *
 * Another way of thinking about this as a Multivariate Normal over the Poisson
 * errors. Sets the following values at metrics index:
 * 0 - TPR aka pos.recall aka True Positive Rate aka Sensitivity
 * 1 - lower-bound of marginal CI TPR
 * 2 - upper-bound of marginal CI TPR
 * 3 - FPR
 * 4 - lower-bound of marginal CI FPR
 * 5 - upper-bound of marginal CI FPR
 * 6 - variance of TPR
 * 7 - covariance TPR, FPR
 * 8 - variance of FPR
 * 9 - covariance TPR, FPR
 */
 // TODO this function is not Unit-tested
template <typename T, isInt<T> = true>
inline void bvn_error(
    const T* __restrict const conf_mat,
    double* __restrict const metrics,
    double alpha) {
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
    const int64_t itp = conf_mat[3];
    const auto tp = static_cast<double>(itp);
    const int64_t ifp = conf_mat[1];
    const auto fp = static_cast<double>(ifp);

    const int64_t itp_fn = conf_mat[2] + conf_mat[3];
    const bool tp_fn_nonzero = itp_fn > 0;
    const auto tp_fn = static_cast<double>(itp_fn);

    const int64_t ifp_tn = conf_mat[1] + conf_mat[0];
    const bool fp_tn_nonzero = ifp_tn > 0;
    const auto fp_tn = static_cast<double>(ifp_tn);

    // Y: TPR, recall
    double val;
    double val_var;
    double val_sigma;
    if (itp == itp_fn) {
        // recall == 1.0
        metrics[0] = metrics[1] = metrics[2] = 1.0;
        metrics[6] = 0.;
    } else if (tp_fn_nonzero) {
        val = tp / (tp_fn);
        // recall variance:
        val_var
            = (static_cast<double>(conf_mat[3] * conf_mat[1])
               / static_cast<double>(std::pow(tp_fn, 3.0)));
        val_sigma = std::sqrt(val_var);
        metrics[0] = val;
        metrics[1] = norm_ppf(val, val_sigma, alpha_lb);
        metrics[2] = norm_ppf(val, val_sigma, alpha_ub);
        metrics[6] = val_var;

    } else {
        // recall == 0
        metrics[0] = metrics[1] = metrics[2] = 0.0;
        metrics[6] = NAN;
    }

    // X: FPR
    if (ifp == ifp_tn) {
        // FPR == 1.0
        metrics[3] = metrics[4] = metrics[5] = 1.0;
        metrics[9] = 0.;
    } else if (fp_tn_nonzero) {
        val = fp / (fp_tn);
        // FPR variance:
        val_var
            = (static_cast<double>(conf_mat[3] * conf_mat[2])
               / static_cast<double>(std::pow(fp_tn, 3.0)));
        val_sigma = std::sqrt(val_var);
        metrics[3] = val;
        metrics[4] = norm_ppf(val, val_sigma, alpha_lb);
        metrics[5] = norm_ppf(val, val_sigma, alpha_ub);
        metrics[9] = val_var;
    } else {
        // FPR == 0.0
        metrics[3] = metrics[4] = metrics[5] = 0.0;
        metrics[9] = NAN;
    }

    // covariance
    metrics[7] = metrics[8] = 0.0;
}  // bvn_error

// Compute sigma for ROC
// Current version is creating term1 and term2 and overwriting them based on conditions.
// TODO: to refactor, with term1 term2 called 2 time for Y and X
template <typename T, isInt<T> = true>
inline void bvn_sigma(
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

    // Y
    const int64_t iterm1_Y = conf_mat[3]; // TP
    const bool term1_Y_nonzero = iterm1_Y > 0;
    auto term1_Y = static_cast<double>(iterm1_Y);

    const int64_t iterm2_Y = conf_mat[2]; // FN
    const bool term2_Y_nonzero = iterm2_Y > 0;
    auto term2_Y = static_cast<double>(iterm2_Y);

    double Y = term1_Y/(term1_Y+term2_Y);

    if (!term1_Y_nonzero) {
        term1_Y = 1;
    }
    if (!term2_Y_nonzero) {
        term2_Y = 1;
    }

    double Y_sigma;
    Y_sigma = std::sqrt(
        static_cast<double>(term1_Y * term2_Y)
        / static_cast<double>(std::pow(term1_Y + term2_Y, 3.0)));


    // X
    const int64_t iterm1_X = conf_mat[1]; // FP
    const bool term1_X_nonzero = iterm1_X > 0;
    auto term1_X = static_cast<double>(iterm1_X);

    const int64_t iterm2_X = conf_mat[0]; // TN
    const bool term2_X_nonzero = iterm2_X > 0;
    auto term2_X = static_cast<double>(iterm2_X);

    double X = term1_X/(term1_X+term2_X);

    if (!term1_X_nonzero) {
        term1_X = 1;
    }
    if (!term2_X_nonzero) {
        term2_X = 1;
    }

    double X_sigma;
    X_sigma = std::sqrt(
        static_cast<double>(term1_X * term2_X)
        / static_cast<double>(std::pow(term1_X + term2_X, 3.0)));

    metrics[0] = Y;
    metrics[1] = Y_sigma;
    metrics[2] = X;
    metrics[3] = X_sigma;
}  // bvn_error

}  // namespace roc


}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_BVN_ERROR_HPP_
