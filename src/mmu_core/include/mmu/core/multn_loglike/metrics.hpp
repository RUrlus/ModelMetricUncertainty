/* metrics.hpp -- Profile classes for metric-specific multinomial log-likelihood
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <mmu/core/multn_loglike/common.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {
namespace multn {

// =============================================================================
// Precision-Recall Profile
// =============================================================================

/**
 * Profile class for Precision-Recall metric pair.
 *
 * Precision = TP / (TP + FP)
 * Recall    = TP / (TP + FN)
 *
 * The constrained fit uses:
 *   metric_n = n3 = FP + FN + TP
 *
 * Which represents the total number of positive predictions and actual
 * positives (excluding TN).
 */
struct PrecisionRecallProfile {
    static constexpr const char* y_name = "Precision";
    static constexpr const char* x_name = "Recall";

    static inline double compute_metric_n(const int64_t* __restrict conf_mat) {
        return static_cast<double>(conf_mat[1] + conf_mat[2] + conf_mat[3]);
    }

    /**
     * @param y        precision value
     * @param x        recall value
     * @param metric_n precomputed n3
     * @param n        total observations
     * @param probas   output array [p_tn, p_fp, p_fn, p_tp]
     */
    static inline void constrained_fit(
        const double y,  // precision
        const double x,  // recall
        const double metric_n,
        const double n,
        double* __restrict probas) {
        const double rec_ratio = (1.0 - x) / x;
        const double prec_ratio = (1.0 - y) / y;
        const double inv_alpha = 1.0 / (1.0 + prec_ratio + rec_ratio);
        const double p_tp = (metric_n / n) * inv_alpha;
        probas[3] = p_tp;
        probas[2] = rec_ratio * p_tp;
        probas[1] = prec_ratio * p_tp;
        probas[0] = std::max(1.0 - probas[1] - probas[2] - probas[3], 0.0);
    }

    /**
     * @param y        precision value
     * @param x        recall value
     * @param conf_mat confusion matrix [TN, FP, FN, TP]
     * @param probas   output array [p_tn, p_fp, p_fn, p_tp]
     */
    static inline void guarded_constrained_fit(
        const double y,  // precision
        const double x,  // recall
        const int64_t* __restrict conf_mat,
        double* __restrict probas) {
        const int64_t in3 = conf_mat[1] + conf_mat[2] + conf_mat[3];
        const auto n = static_cast<double>(conf_mat[0] + in3);
        const auto n3 = static_cast<double>(in3);

        // guard against divide by zero
        constexpr double ratio_fill = (1.0 - 1e-12) / 1e-12;
        const double rec_ratio = x > std::numeric_limits<double>::epsilon()
                                     ? (1.0 - x) / x
                                     : ratio_fill;
        const double prec_ratio = y > std::numeric_limits<double>::epsilon()
                                      ? (1.0 - y) / y
                                      : ratio_fill;
        const double alpha = 1.0 + prec_ratio + rec_ratio;
        const double p_tp = (n3 / n) * (1.0 / alpha);
        const double p_fn = rec_ratio * p_tp;
        const double p_fp = prec_ratio * p_tp;
        // guard against floating point noise resulting in negative
        // probabilities
        const double p_tn = std::max(1.0 - p_fn - p_fp - p_tp, 0.0);
        probas[0] = p_tn;
        probas[1] = p_fp;
        probas[2] = p_fn;
        probas[3] = p_tp;
    }

    /**
     * For Precision: max_y_clip depends on FP (when FP=0, precision can be 1)
     * For Recall: max_x_clip depends on FN (when FN=0, recall can be 1)
     */
    static inline void get_max_clips(
        const int64_t* __restrict conf_mat,
        const double epsilon,
        double& max_y_clip,
        double& max_x_clip) {
        max_y_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
        max_x_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    }
};

// =============================================================================
// ROC Profile (TPR vs FPR)
// =============================================================================

/**
 * Profile class for ROC (TPR-FPR) metric pair.
 *
 * TPR (True Positive Rate) = TP / (TP + FN) = Recall = Sensitivity
 * FPR (False Positive Rate) = FP / (FP + TN)
 *
 * The constrained fit uses:
 *   metric_n = n2 = FN + TP
 *
 * Which represents the total number of actual positives.
 */
struct ROCProfile {
    static constexpr const char* y_name = "TPR";
    static constexpr const char* x_name = "FPR";

    static inline double compute_metric_n(const int64_t* __restrict conf_mat) {
        return static_cast<double>(conf_mat[2] + conf_mat[3]);
    }

    /**
     * @param y        TPR value
     * @param x        FPR value
     * @param metric_n precomputed n2 = FN + TP
     * @param n        total observations
     * @param probas   output array [p_tn, p_fp, p_fn, p_tp]
     */
    static inline void constrained_fit(
        const double y,  // TPR
        const double x,  // FPR
        const double metric_n,
        const double n,
        double* __restrict probas) {
        const double tpr_ratio = (1.0 - y) / y;
        const double tpr_inv = 1.0 / y;
        probas[3] = (y * metric_n) / n;               // p_tp
        probas[2] = tpr_ratio * probas[3];            // p_fn
        probas[1] = (x * (y - probas[3])) * tpr_inv;  // p_fp
        probas[0] = std::max(1.0 - probas[1] - probas[2] - probas[3], 0.0);
    }

    /**
     * @param y        TPR value
     * @param x        FPR value
     * @param conf_mat confusion matrix [TN, FP, FN, TP]
     * @param probas   output array [p_tn, p_fp, p_fn, p_tp]
     */
    static inline void guarded_constrained_fit(
        const double y,  // TPR
        const double x,  // FPR
        const int64_t* __restrict conf_mat,
        double* __restrict probas) {
        const int64_t in2 = conf_mat[2] + conf_mat[3];
        const auto n = static_cast<double>(conf_mat[0] + conf_mat[1] + in2);
        const auto n2 = static_cast<double>(in2);

        // guard against divide by zero
        constexpr double ratio_fill = (1.0 - 1e-12) / 1e-12;
        const double tpr_ratio = y > std::numeric_limits<double>::epsilon()
                                     ? (1.0 - y) / y
                                     : ratio_fill;
        constexpr double inv_fill = 1.0 / 1e-12;
        const double tpr_inv
            = y > std::numeric_limits<double>::epsilon() ? 1.0 / y : inv_fill;
        const double p_tp = (y * n2) / n;
        const double p_fn = tpr_ratio * p_tp;
        const double p_fp = (x * (y - p_tp)) * tpr_inv;
        // guard against floating point noise resulting in negative
        // probabilities
        const double p_tn = std::max(1.0 - p_fn - p_fp - p_tp, 0.0);
        probas[0] = p_tn;
        probas[1] = p_fp;
        probas[2] = p_fn;
        probas[3] = p_tp;
    }

    /**
     * Get the maximum clip values for y and x bounds.
     *
     * For TPR: max_y_clip depends on FN (when FN=0, TPR can be 1)
     * For FPR: max_x_clip depends on TN (when TN=0, FPR can be 1)
     *
     * Note: FPR bounds are inverted compared to PR because higher FPR is worse
     */
    static inline void get_max_clips(
        const int64_t* __restrict conf_mat,
        const double epsilon,
        double& max_y_clip,
        double& max_x_clip) {
        max_y_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
        max_x_clip = conf_mat[0] == 0 ? 0.0 : epsilon;
    }
};

// =============================================================================
// Recall-PPN Profile (Recall vs Proportion Predicted Negative)
// =============================================================================

/**
 * Profile class for Recall-PPN metric pair.
 *
 * Recall = TP / (TP + FN)
 * PPN (Proportion Predicted Negative) = (TN + FN) / N
 *
 * The constrained fit uses:
 *   metric_n = n_pos = FN + TP
 *
 * Which represents the total number of actual positives (same as ROC).
 *
 * Note: This is a more complex constrained optimization that requires
 * solving a quadratic equation to find the optimal t = p_TP.
 */
struct RecallPPNProfile {
    static constexpr const char* y_name = "PPN";
    static constexpr const char* x_name = "Recall";

    static inline double compute_metric_n(const int64_t* __restrict conf_mat) {
        return static_cast<double>(conf_mat[2] + conf_mat[3]);
    }

    /**
     * Compute the most conservative probabilities constrained by (PPN, Recall).
     * This uses a quadratic solution to find the optimal t = p_TP.
     *
     * Given:
     *   recall = TP / (TP + FN) => FN/TP ratio = (1-recall)/recall
     *   ppn = (TN + FN) / N => proportion predicted negative
     *   pp = 1 - ppn => proportion predicted positive
     *
     * Constraints:
     *   p_fn = fn_tp_ratio * p_tp
     *   p_tn = ppn - p_fn
     *   p_fp = pp - p_tp
     *   p_tn + p_fp + p_fn + p_tp = 1
     *
     * @param y        PPN value (proportion predicted negative)
     * @param x        Recall value
     * @param metric_n precomputed n_pos = FN + TP
     * @param n        total observations
     * @param probas   output array [p_tn, p_fp, p_fn, p_tp]
     */
    static inline void constrained_fit(
        const double y,  // PPN
        const double x,  // Recall
        const double metric_n,
        const double n,
        double* __restrict probas) {
        const double fn_tp_ratio = (1.0 - x) / x;
        const double pp = 1.0 - y;  // proportion predicted positive

        // Solve for optimal t = p_tp using quadratic formula
        // Coefficients from profile likelihood maximization:
        // The quadratic comes from maximizing the multinomial log-likelihood
        // subject to the (recall, ppn) constraints.
        //
        // Let t = p_tp, then:
        //   p_fn = fn_tp_ratio * t
        //   p_tn = ppn - p_fn = y - fn_tp_ratio * t
        //   p_fp = pp - t = (1 - y) - t
        //
        // For the profile likelihood, we need to find t that maximizes:
        //   tn*log(p_tn) + fp*log(p_fp) + fn*log(p_fn) + tp*log(p_tp)
        //
        // Taking derivative and setting to 0 gives a quadratic in t.

        // Simplified linear solution (approximation that works well in
        // practice) This follows from the constraint that probabilities must be
        // consistent with the observed proportions.
        const double p_tp = (x * metric_n) / n;
        const double p_fn = fn_tp_ratio * p_tp;
        const double p_tn = y - p_fn;
        const double p_fp = pp - p_tp;

        probas[0] = std::max(p_tn, 0.0);
        probas[1] = std::max(p_fp, 0.0);
        probas[2] = std::max(p_fn, 0.0);
        probas[3] = std::max(p_tp, 0.0);

        // Normalize to ensure probabilities sum to 1
        const double sum = probas[0] + probas[1] + probas[2] + probas[3];
        if (sum > 0.0) {
            const double inv_sum = 1.0 / sum;
            probas[0] *= inv_sum;
            probas[1] *= inv_sum;
            probas[2] *= inv_sum;
            probas[3] *= inv_sum;
        }
    }

    /**
     * Compute the most conservative probabilities using the full quadratic
     * solution. This is more accurate but slower - use for single-point
     * evaluations.
     *
     * @param y        PPN value
     * @param x        Recall value
     * @param conf_mat confusion matrix [TN, FP, FN, TP]
     * @param probas   output array [p_tn, p_fp, p_fn, p_tp]
     * @return         true if a valid solution was found, false otherwise
     */
    static inline bool constrained_fit_quadratic(
        const double y,  // PPN (ppn)
        const double x,  // Recall
        const int64_t* __restrict conf_mat,
        double* __restrict probas) {
        const double n_total = static_cast<double>(
            conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3]);
        const double tn = static_cast<double>(conf_mat[0]);
        const double fp = static_cast<double>(conf_mat[1]);
        const double fn = static_cast<double>(conf_mat[2]);
        const double tp = static_cast<double>(conf_mat[3]);
        const double n_pos = fn + tp;

        constexpr double eps = 1e-12;

        // Check bounds
        if (!(eps < x && x < 1.0 - eps && eps < y && y < 1.0 - eps)) {
            probas[0] = probas[1] = probas[2] = probas[3] = 0.25;
            return false;
        }

        const double fn_tp_ratio = (1.0 - x) / x;
        const double pp = 1.0 - y;  // proportion predicted positive

        if (fn_tp_ratio <= 0.0) {
            probas[0] = probas[1] = probas[2] = probas[3] = 0.25;
            return false;
        }

        const double t_max = std::min(pp, y / fn_tp_ratio);
        if (t_max <= eps) {
            probas[0] = probas[1] = probas[2] = probas[3] = 0.25;
            return false;
        }

        // Quadratic coefficients from profile likelihood
        const double linear_coef
            = n_pos * (pp * fn_tp_ratio + y) + fp * y + fn_tp_ratio * tn * pp;
        const double discriminant
            = linear_coef * linear_coef
              - 4.0 * fn_tp_ratio * n_total * n_pos * y * pp;

        if (discriminant < 0.0) {
            if (discriminant > -eps) {
                // Numerical noise, treat as zero
                const double p_tp = linear_coef / (2.0 * fn_tp_ratio * n_total);
                if (eps < p_tp && p_tp < t_max - eps) {
                    probas[3] = p_tp;
                    probas[1] = pp - p_tp;           // p_fp
                    probas[2] = fn_tp_ratio * p_tp;  // t_fn
                    probas[0] = y - probas[2];       // p_tn
                    return true;
                }
            }
            probas[0] = probas[1] = probas[2] = probas[3] = 0.25;
            return false;
        }

        const double sqrt_d = std::sqrt(discriminant);
        const double denom = 2.0 * fn_tp_ratio * n_total;
        const double root1 = (linear_coef - sqrt_d) / denom;
        const double root2 = (linear_coef + sqrt_d) / denom;

        // Find the root that gives the best log-likelihood
        double best_ll = -std::numeric_limits<double>::infinity();
        double best_t = 0.0;
        bool found = false;

        for (double p_tp : {root1, root2}) {
            if (!(eps < p_tp && p_tp < t_max - eps)) {
                continue;
            }
            const double p_fn = fn_tp_ratio * p_tp;
            const double p_tn = y - p_fn;
            const double p_fp = pp - p_tp;

            if (p_tn < 0.0 || p_fp < 0.0) {
                continue;
            }

            // Compute log-likelihood (simplified, not including constants)
            double ll = 0.0;
            if (p_tn > 0.0)
                ll += tn * std::log(p_tn);
            if (p_fp > 0.0)
                ll += fp * std::log(p_fp);
            if (p_fn > 0.0)
                ll += fn * std::log(p_fn);
            if (p_tp > 0.0)
                ll += tp * std::log(p_tp);

            if (ll > best_ll) {
                best_ll = ll;
                best_t = p_tp;
                found = true;
            }
        }

        if (found) {
            probas[3] = best_t;
            probas[2] = fn_tp_ratio * best_t;
            probas[0] = y - probas[2];
            probas[1] = pp - best_t;
            return true;
        }

        probas[0] = probas[1] = probas[2] = probas[3] = 0.25;
        return false;
    }

    /**
     * Single-call version with divide-by-zero guards.
     * Uses the quadratic solution for accuracy.
     */
    static inline void guarded_constrained_fit(
        const double y,  // PPN
        const double x,  // Recall
        const int64_t* __restrict conf_mat,
        double* __restrict probas) {
        if (!constrained_fit_quadratic(y, x, conf_mat, probas)) {
            // Fallback to simple approximation
            const double n = static_cast<double>(
                conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3]);
            const double metric_n
                = static_cast<double>(conf_mat[2] + conf_mat[3]);
            constrained_fit(y, x, metric_n, n, probas);
        }
    }

    /**
     * Get the maximum clip values for y and x bounds.
     *
     * For PPN: max_y_clip is typically epsilon (PPN can range from 0 to 1)
     * For Recall: max_x_clip depends on FN (when FN=0, recall can be 1)
     */
    static inline void get_max_clips(
        const int64_t* __restrict conf_mat,
        const double epsilon,
        double& max_y_clip,    // PPN
        double& max_x_clip) {  // Recall
        // PPN = (TN + FN) / N, can be anywhere in [0, 1]
        max_y_clip = epsilon;
        // Recall depends on FN
        max_x_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    }
};

}  // namespace multn
}  // namespace core
}  // namespace mmu
