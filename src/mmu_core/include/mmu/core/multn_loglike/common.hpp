/* common.hpp -- Common types and utilities for multinomial log-likelihood
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_MULTN_LOGLIKE_COMMON_HPP_
#define INCLUDE_MMU_CORE_MULTN_LOGLIKE_COMMON_HPP_

#include <cinttypes>
#include <cmath>
#include <limits>

#include <mmu/core/common.hpp>

namespace mmu {
namespace core {
namespace multn {

/* Store for profile log-likelihood computation.
 * This struct holds all precomputed values needed for efficient
 * iterative computation of the profile log-likelihood.
 */
struct prof_loglike_store {
    int64_t total_n;   // Total observations (integer)
    double metric_n;   // Metric-specific denominator (n2 for ROC, n3 for PR)
    double n;          // Total observations as double
    double x_tn;       // TN as double
    double x_fp;       // FP as double
    double x_fn;       // FN as double
    double x_tp;       // TP as double
    double p_tn;       // Observed probability p(TN)
    double p_fp;       // Observed probability p(FP)
    double p_fn;       // Observed probability p(FN)
    double p_tp;       // Observed probability p(TP)
    double nll_h0;     // Null hypothesis NLL: -2 * log(L_unconstrained)
};

}  // namespace multn
}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_MULTN_LOGLIKE_COMMON_HPP_

