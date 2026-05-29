/* common.hpp -- Common types and helpers for multinomial
 * profile log-likelihood Copyright 2026 Ralph Urlus
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {
namespace multn {

// fill value for the chi2 scres, this values results
// p-values very close to 1
// chi2.ppf(1-1e-14) --> 64.47398179869367
constexpr double MULT_DEFAULT_CHI2_SCORE = 65.0;

/**
 * Fields:
 *   total_n  integer total sample size
 *   n        total sample size as double
 *   metric_n cached metric-specific convenience count
 *
 * Observed counts:
 *   x_tn, x_fp, x_fn, x_tp
 *
 * Unconstrained multinomial probabilities:
 *   p_tn, p_fp, p_fn, p_tp
 *
 * Cached unconstrained deviance:
 *   nll_h0 = -2 log L(unconstrained)
 */
struct prof_loglike_store {
    int64_t total_n = 0;
    double n = 0.0;

    double metric_n = 0.0;

    // Observed counts
    double x_tn = 0.0;
    double x_fp = 0.0;
    double x_fn = 0.0;
    double x_tp = 0.0;

    double p_tn = 0.0;
    double p_fp = 0.0;
    double p_fn = 0.0;
    double p_tp = 0.0;

    double nll_h0 = 0.0;
};

}  // namespace multn
}  // namespace core
}  // namespace mmu
