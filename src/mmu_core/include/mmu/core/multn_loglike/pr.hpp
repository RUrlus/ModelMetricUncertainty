/* pr.hpp -- Precision-Recall namespace aliases for multinomial log-likelihood
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <mmu/core/multn_loglike/chi2.hpp>
#include <mmu/core/multn_loglike/common.hpp>
#include <mmu/core/multn_loglike/grid_bounds.hpp>
#include <mmu/core/multn_loglike/loglike.hpp>
#include <mmu/core/multn_loglike/profiles.hpp>

namespace mmu {
namespace core {
namespace pr {

using Profile = multn::PrecisionRecallProfile;

inline void set_store(
    const int64_t* __restrict conf_mat,
    multn::prof_loglike_store* store) {
    multn::set_store<Profile>(conf_mat, store);
}

inline double prof_loglike(
    const double prec,
    const double rec,
    const multn::prof_loglike_store& store,
    double* __restrict p_h0) {
    return multn::prof_loglike<Profile, false>(prec, rec, store, p_h0);
}

inline double prof_loglike(
    const double prec,
    const double rec,
    const int64_t* __restrict conf_mat,
    double* __restrict p_h0) {
    return multn::prof_loglike<Profile, true>(prec, rec, conf_mat, p_h0);
}

inline double multn_chi2_score(
    const double prec,
    const double rec,
    const int64_t* __restrict conf_mat,
    const double epsilon = 1e-4) {
    return multn::multn_chi2_score<Profile>(prec, rec, conf_mat, epsilon);
}

inline void multn_chi2_scores(
    const int64_t n_points,
    const double* precs,
    const double* recs,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double epsilon = 1e-4) {
    multn::multn_chi2_scores<Profile>(
        n_points, precs, recs, conf_mat, scores, epsilon);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void multn_chi2_scores_mt(
    const int64_t n_points,
    const double* precs,
    const double* recs,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double epsilon = 1e-4) {
    multn::multn_chi2_scores_mt<Profile>(
        n_points, precs, recs, conf_mat, scores, epsilon);
}
#endif  // MMU_HAS_OPENMP_SUPPORT

inline void get_grid_bounds(
    const int64_t* __restrict conf_mat,
    double* bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    multn::get_grid_bounds<Profile>(conf_mat, bounds, n_sigmas, epsilon);
}

inline void multn_error(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    multn::multn_error<Profile>(
        n_bins, conf_mat, result, bounds, n_sigmas, epsilon);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void multn_error_mt(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int n_threads = 4) {
    multn::multn_error_mt<Profile>(
        n_bins, conf_mat, result, bounds, n_sigmas, epsilon, n_threads);
}
#endif  // MMU_HAS_OPENMP_SUPPORT

inline void multn_grid_error(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    multn::multn_grid_error<Profile>(
        n_prec_bins,
        n_rec_bins,
        prec_grid,
        rec_grid,
        conf_mat,
        scores,
        n_sigmas,
        epsilon);
}

inline void multn_grid_curve_error(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    multn::multn_grid_curve_error<Profile>(
        n_prec_bins,
        n_rec_bins,
        n_conf_mats,
        prec_grid,
        rec_grid,
        conf_mat,
        scores,
        n_sigmas,
        epsilon);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void multn_grid_curve_error_mt(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* __restrict prec_grid,
    const double* __restrict rec_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int64_t n_threads = 4) {
    multn::multn_grid_curve_error_mt<Profile>(
        n_prec_bins,
        n_rec_bins,
        n_conf_mats,
        prec_grid,
        rec_grid,
        conf_mat,
        scores,
        n_sigmas,
        epsilon,
        n_threads);
}
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace pr
}  // namespace core
}  // namespace mmu
