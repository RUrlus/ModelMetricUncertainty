/* roc.hpp -- ROC namespace aliases for multinomial log-likelihood
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
namespace roc {

using Profile = multn::ROCProfile;

inline void set_store(
    const int64_t* __restrict conf_mat,
    multn::prof_loglike_store* store) {
    multn::set_store<Profile>(conf_mat, store);
}

inline double prof_loglike(
    const double tpr,
    const double fpr,
    const multn::prof_loglike_store& store,
    double* __restrict p_h0) {
    return multn::prof_loglike<Profile, false>(tpr, fpr, store, p_h0);
}

inline double prof_loglike(
    const double tpr,
    const double fpr,
    const int64_t* __restrict conf_mat,
    double* __restrict p_h0) {
    return multn::prof_loglike<Profile, true>(tpr, fpr, conf_mat, p_h0);
}

inline double multn_chi2_score(
    const double tpr,
    const double fpr,
    const int64_t* __restrict conf_mat,
    const double epsilon = 1e-4) {
    return multn::multn_chi2_score<Profile>(tpr, fpr, conf_mat, epsilon);
}

inline void multn_chi2_scores(
    const int64_t n_points,
    const double* tprs,
    const double* fprs,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double epsilon = 1e-4) {
    multn::multn_chi2_scores<Profile>(
        n_points, tprs, fprs, conf_mat, scores, epsilon);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void multn_chi2_scores_mt(
    const int64_t n_points,
    const double* tprs,
    const double* fprs,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double epsilon = 1e-4) {
    multn::multn_chi2_scores_mt<Profile>(
        n_points, tprs, fprs, conf_mat, scores, epsilon);
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
    const int64_t n_tpr_bins,
    const int64_t n_fpr_bins,
    const double* __restrict tpr_grid,
    const double* __restrict fpr_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    multn::multn_grid_error<Profile>(
        n_tpr_bins,
        n_fpr_bins,
        tpr_grid,
        fpr_grid,
        conf_mat,
        scores,
        n_sigmas,
        epsilon);
}

inline void multn_grid_curve_error(
    const int64_t n_tpr_bins,
    const int64_t n_fpr_bins,
    const int64_t n_conf_mats,
    const double* __restrict tpr_grid,
    const double* __restrict fpr_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    multn::multn_grid_curve_error<Profile>(
        n_tpr_bins,
        n_fpr_bins,
        n_conf_mats,
        tpr_grid,
        fpr_grid,
        conf_mat,
        scores,
        n_sigmas,
        epsilon);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void multn_grid_curve_error_mt(
    const int64_t n_tpr_bins,
    const int64_t n_fpr_bins,
    const int64_t n_conf_mats,
    const double* __restrict tpr_grid,
    const double* __restrict fpr_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int64_t n_threads = 4) {
    multn::multn_grid_curve_error_mt<Profile>(
        n_tpr_bins,
        n_fpr_bins,
        n_conf_mats,
        tpr_grid,
        fpr_grid,
        conf_mat,
        scores,
        n_sigmas,
        epsilon,
        n_threads);
}
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace roc
}  // namespace core
}  // namespace mmu
