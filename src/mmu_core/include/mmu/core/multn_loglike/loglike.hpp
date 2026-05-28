#pragma once

#include <mmu/core/common.hpp>
#include <mmu/core/multn_loglike/common.hpp>
#include <mmu/core/multn_loglike/profiles.hpp>

namespace mmu {
namespace core {
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
                    * (mmu::core::details::xlogy(store->x_tn, store->p_tn)
                       + mmu::core::details::xlogy(store->x_fp, store->p_fp)
                       + mmu::core::details::xlogy(store->x_fn, store->p_fn)
                       + mmu::core::details::xlogy(store->x_tp, store->p_tp));
}

template <typename Profile, bool Guarded = false>
inline double prof_loglike(
    const double y,
    const double x,
    const prof_loglike_store& store,
    double* __restrict p_h0) {
    if (!Profile::template constrained_fit<Guarded>(y, x, store, p_h0)) {
        return MULT_DEFAULT_CHI2_SCORE;
    }

    const double nll_h1 = -2.0
                          * (mmu::core::details::xlogy(store.x_tn, p_h0[0])
                             + mmu::core::details::xlogy(store.x_fp, p_h0[1])
                             + mmu::core::details::xlogy(store.x_fn, p_h0[2])
                             + mmu::core::details::xlogy(store.x_tp, p_h0[3]));
    return nll_h1 - store.nll_h0;
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
