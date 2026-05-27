/* core.hpp -- Generic template functions for multinomial log-likelihood
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <mmu/core/common.hpp>
#include <mmu/core/multn_loglike/common.hpp>
#include <mmu/core/multn_loglike/profiles.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {

// Forward declaration of linspace (from grid_bounds.hpp)
namespace details {
inline void linspace(
    const double start,
    double const end,
    const size_t steps,
    double* values);
}  // namespace details

namespace multn {

template <typename Profile>
inline void set_store(
    const int64_t* __restrict conf_mat,
    prof_loglike_store* store) {
    store->metric_n = Profile::compute_metric_n(conf_mat);
    store->total_n = conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3];
    store->n = static_cast<double>(store->total_n);

    store->x_tn = static_cast<double>(conf_mat[0]);
    store->x_fp = static_cast<double>(conf_mat[1]);
    store->x_fn = static_cast<double>(conf_mat[2]);
    store->x_tp = static_cast<double>(conf_mat[3]);

    store->p_tn = store->x_tn / store->n;
    store->p_fp = store->x_fp / store->n;
    store->p_fn = store->x_fn / store->n;
    store->p_tp = store->x_tp / store->n;

    store->nll_h0 = -2.0
                    * (details::xlogy(store->x_tn, store->p_tn)
                       + details::xlogy(store->x_fp, store->p_fp)
                       + details::xlogy(store->x_fn, store->p_fn)
                       + details::xlogy(store->x_tp, store->p_tp));
}

// =============================================================================
// Profile Log-Likelihood Functions
// =============================================================================

/**
 * Compute the profile log-likelihood ratio statistic.
 * Loop-optimized version using precomputed store.
 *
 * Computes: -2 * ln(L_constrained / L_unconstrained)
 *         = -2 * ln(L_constrained) + 2 * ln(L_unconstrained)
 *         = nll_h1 - nll_h0
 *
 * @tparam Profile  The metric profile class
 * @param y         Y-axis metric value (e.g., precision, TPR, PPN)
 * @param x         X-axis metric value (e.g., recall, FPR)
 * @param store     Precomputed store with nll_h0 and confusion matrix values
 * @param p_h0      Output array for constrained probabilities [p_tn, p_fp,
 * p_fn, p_tp]
 * @return          The profile log-likelihood ratio statistic (chi2 distributed
 * with df=2)
 */
template <typename Profile>
inline double prof_loglike(
    const double y,
    const double x,
    prof_loglike_store* store,
    double* __restrict p_h0) {
    Profile::constrained_fit(y, x, store->metric_n, store->n, p_h0);
    const double nll_h1 = -2.0
                          * (details::xlogy(store->x_tn, p_h0[0])
                             + details::xlogy(store->x_fp, p_h0[1])
                             + details::xlogy(store->x_fn, p_h0[2])
                             + details::xlogy(store->x_tp, p_h0[3]));
    return nll_h1 - store->nll_h0;
}

/**
 * Compute the profile log-likelihood ratio statistic.
 * Single-call version with guards.
 *
 * @tparam Profile  The metric profile class
 * @param y         Y-axis metric value
 * @param x         X-axis metric value
 * @param n         Total observations
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param p_h0      Output array for constrained probabilities
 * @return          The profile log-likelihood ratio statistic
 */
template <typename Profile>
inline double prof_loglike(
    const double y,
    const double x,
    const double n,
    const int64_t* __restrict conf_mat,
    double* __restrict p_h0) {
    const auto x_tn = static_cast<double>(conf_mat[0]);
    const auto x_fp = static_cast<double>(conf_mat[1]);
    const auto x_fn = static_cast<double>(conf_mat[2]);
    const auto x_tp = static_cast<double>(conf_mat[3]);

    // Constitutes the optimal/unconstrained fit of a multinomial
    const double nll_h0
        = -2.0
          * (details::xlogy(x_tn, x_tn / n) + details::xlogy(x_fp, x_fp / n)
             + details::xlogy(x_fn, x_fn / n) + details::xlogy(x_tp, x_tp / n));

    Profile::guarded_constrained_fit(y, x, conf_mat, p_h0);

    const double nll_h1
        = -2.0
          * (details::xlogy(x_tn, p_h0[0]) + details::xlogy(x_fp, p_h0[1])
             + details::xlogy(x_fn, p_h0[2]) + details::xlogy(x_tp, p_h0[3]));
    return nll_h1 - nll_h0;
}

}  // namespace multn
}  // namespace core
}  // namespace mmu
