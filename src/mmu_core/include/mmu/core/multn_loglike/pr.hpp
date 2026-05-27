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

// Profile type alias
using Profile = multn::PrecisionRecallProfile;

// =============================================================================
// Store Initialization
// =============================================================================

/**
 * Initialize the profile log-likelihood store from a confusion matrix.
 *
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param store     Pointer to store to initialize
 */
inline void set_store(
    const int64_t* __restrict conf_mat,
    multn::prof_loglike_store* store) {
    multn::set_store<Profile>(conf_mat, store);
}

// =============================================================================
// Profile Log-Likelihood Functions
// =============================================================================

/**
 * Compute the profile log-likelihood ratio statistic.
 * Loop-optimized version using precomputed store.
 *
 * @param prec      Precision value
 * @param rec       Recall value
 * @param store     Precomputed store with nll_h0 and confusion matrix values
 * @param p_h0      Output array for constrained probabilities [p_tn, p_fp,
 * p_fn, p_tp]
 * @return          The profile log-likelihood ratio statistic (chi2 distributed
 * with df=2)
 */
inline double prof_loglike(
    const double prec,
    const double rec,
    multn::prof_loglike_store* store,
    double* __restrict p_h0) {
    return multn::prof_loglike<Profile>(prec, rec, store, p_h0);
}

/**
 * Compute the profile log-likelihood ratio statistic.
 * Single-call version that computes everything from scratch.
 *
 * @param prec      Precision value
 * @param rec       Recall value
 * @param n         Total observations
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param p_h0      Output array for constrained probabilities
 * @return          The profile log-likelihood ratio statistic
 */
inline double prof_loglike(
    const double prec,
    const double rec,
    const double n,
    const int64_t* __restrict conf_mat,
    double* __restrict p_h0) {
    return multn::prof_loglike<Profile>(prec, rec, n, conf_mat, p_h0);
}

// =============================================================================
// Chi2 Score Functions
// =============================================================================

/**
 * Compute the chi2 score for a single (precision, recall) point.
 *
 * @param prec      Precision value
 * @param rec       Recall value
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param epsilon   Clipping value to avoid boundary issues
 * @return          The chi2 score
 */
inline double multn_chi2_score(
    const double prec,
    const double rec,
    const int64_t* __restrict conf_mat,
    const double epsilon = 1e-4) {
    return multn::multn_chi2_score<Profile>(prec, rec, conf_mat, epsilon);
}

/**
 * Compute chi2 scores for multiple (precision, recall) points.
 *
 * @param n_points  Number of points to evaluate
 * @param precs     Array of precision values
 * @param recs      Array of recall values
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param scores    Output array for chi2 scores
 * @param epsilon   Clipping value to avoid boundary issues
 */
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
/**
 * Compute chi2 scores for multiple (precision, recall) points (multi-threaded).
 *
 * @param n_points  Number of points to evaluate
 * @param precs     Array of precision values
 * @param recs      Array of recall values
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param scores    Output array for chi2 scores
 * @param epsilon   Clipping value to avoid boundary issues
 */
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

// =============================================================================
// Grid Error Functions
// =============================================================================

/**
 * Compute grid bounds for Precision-Recall.
 *
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param bounds    Output array [prec_min, prec_max, rec_min, rec_max]
 * @param n_sigmas  Number of sigmas for grid bounds
 * @param epsilon   Clipping value to avoid boundary issues
 */
inline void get_grid_bounds(
    const int64_t* __restrict conf_mat,
    double* bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    multn::get_grid_bounds<Profile>(conf_mat, bounds, n_sigmas, epsilon);
}

/**
 * Compute chi2 scores over a uniform grid.
 * Grid bounds are automatically determined based on metric sigma.
 *
 * @param n_bins    Number of bins in each dimension
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param result    Output array for chi2 scores (n_bins x n_bins)
 * @param bounds    Output array for grid bounds [prec_min, prec_max, rec_min,
 * rec_max]
 * @param n_sigmas  Number of sigmas for grid bounds
 * @param epsilon   Clipping value to avoid boundary issues
 */
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
/**
 * Compute chi2 scores over a uniform grid (multi-threaded).
 *
 * @param n_bins    Number of bins in each dimension
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param result    Output array for chi2 scores (n_bins x n_bins)
 * @param bounds    Output array for grid bounds [prec_min, prec_max, rec_min,
 * rec_max]
 * @param n_sigmas  Number of sigmas for grid bounds
 * @param epsilon   Clipping value to avoid boundary issues
 * @param n_threads Number of threads to use
 */
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

// =============================================================================
// Grid Error Functions with User-Provided Grid
// =============================================================================

/**
 * Compute chi2 scores over a user-provided grid.
 * Only evaluates points within the computed bounds for efficiency.
 *
 * @param n_prec_bins Number of precision bins
 * @param n_rec_bins  Number of recall bins
 * @param prec_grid   Array of precision values
 * @param rec_grid    Array of recall values
 * @param conf_mat    Confusion matrix [TN, FP, FN, TP]
 * @param scores      Output array for chi2 scores (n_prec_bins x n_rec_bins)
 * @param n_sigmas    Number of sigmas for grid bounds
 * @param epsilon     Clipping value to avoid boundary issues
 */
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

/**
 * Compute chi2 scores over a user-provided grid for multiple confusion
 * matrices. Takes the minimum score across all confusion matrices.
 *
 * @param n_prec_bins Number of precision bins
 * @param n_rec_bins  Number of recall bins
 * @param n_conf_mats Number of confusion matrices
 * @param prec_grid   Array of precision values
 * @param rec_grid    Array of recall values
 * @param conf_mat    Array of confusion matrices (n_conf_mats x 4)
 * @param scores      Output array for chi2 scores (n_prec_bins x n_rec_bins)
 * @param n_sigmas    Number of sigmas for grid bounds
 * @param epsilon     Clipping value to avoid boundary issues
 */
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
/**
 * Compute chi2 scores over a user-provided grid for multiple confusion matrices
 * (multi-threaded).
 *
 * @param n_prec_bins Number of precision bins
 * @param n_rec_bins  Number of recall bins
 * @param n_conf_mats Number of confusion matrices
 * @param prec_grid   Array of precision values
 * @param rec_grid    Array of recall values
 * @param conf_mat    Array of confusion matrices (n_conf_mats x 4)
 * @param scores      Output array for chi2 scores (n_prec_bins x n_rec_bins)
 * @param n_sigmas    Number of sigmas for grid bounds
 * @param epsilon     Clipping value to avoid boundary issues
 * @param n_threads   Number of threads to use
 */
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
