#pragma once

#include <cmath>
#include <limits>

#include <mmu/core/common.hpp>
#include <mmu/core/multn_loglike/common.hpp>
#include <mmu/core/multn_loglike/profiles.hpp>

namespace mmu {
namespace core {
namespace multn {

namespace details {

/* x * log(num / den) with the convention 0 * log(anything) == 0.
 *
 * `den` is an unconstrained MLE probability x/n, so it is strictly positive
 * whenever x > 0 and the division below is always safe. `num` is a
 * constrained-fit probability and may legitimately be clipped to exactly
 * zero, in which case the term is -inf and the caller treats the point as
 * unreachable.
 */
inline double xlog_ratio(const double x, const double num, const double den) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (num <= 0.0) {
        return -std::numeric_limits<double>::infinity();
    }
    return x * std::log(num / den);
}

}  // namespace details

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

    // Unconstrained MLE probabilities. These are the denominators of the
    // likelihood ratio evaluated in `prof_loglike`.
    store->p_tn = store->x_tn / store->n;
    store->p_fp = store->x_fp / store->n;
    store->p_fn = store->x_fn / store->n;
    store->p_tp = store->x_tp / store->n;
}

/* Deviance of the constrained fit against the unconstrained MLE.
 *
 * Returns -2 * sum_i x_i * log(p_constrained_i / p_mle_i), which is chi2
 * distributed with two degrees of freedom under Wilks' theorem.
 *
 */
template <typename Profile, bool Guarded = false>
inline double prof_loglike(
    const double y,
    const double x,
    const prof_loglike_store& store,
    double* __restrict p_h0) {
    if (!Profile::template constrained_fit<Guarded>(y, x, store, p_h0)) {
        return MULT_DEFAULT_CHI2_SCORE;
    }

    const double log_ratio_sum
        = details::xlog_ratio(store.x_tn, p_h0[0], store.p_tn)
          + details::xlog_ratio(store.x_fp, p_h0[1], store.p_fp)
          + details::xlog_ratio(store.x_fn, p_h0[2], store.p_fn)
          + details::xlog_ratio(store.x_tp, p_h0[3], store.p_tp);

    const double deviance = -2.0 * log_ratio_sum;
    return std::isfinite(deviance) ? deviance : MULT_DEFAULT_CHI2_SCORE;
}

template <typename Profile, bool Guarded = true>
inline double prof_loglike(
    const double y,
    const double x,
    const int64_t* __restrict conf_mat,
    double* __restrict p_h0) {
    prof_loglike_store store;
    set_store<Profile>(conf_mat, &store);
    return prof_loglike<Profile, Guarded>(y, x, store, p_h0);
}

}  // namespace multn
}  // namespace core
}  // namespace mmu
