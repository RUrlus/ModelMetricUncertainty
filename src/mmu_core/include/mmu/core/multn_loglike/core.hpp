/* core.hpp -- Generic template functions for multinomial log-likelihood
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <array>
#include <memory>

#include <mmu/core/common.hpp>
#include <mmu/core/multn_loglike/common.hpp>
#include <mmu/core/multn_loglike/metrics.hpp>

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
 * Single-call version that computes everything from scratch.
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
    const double metric_n = Profile::compute_metric_n(conf_mat);
    const auto x_tn = static_cast<double>(conf_mat[0]);
    const auto x_fp = static_cast<double>(conf_mat[1]);
    const auto x_fn = static_cast<double>(conf_mat[2]);
    const auto x_tp = static_cast<double>(conf_mat[3]);

    // Constitutes the optimal/unconstrained fit of a multinomial
    const double nll_h0
        = -2.0
          * (details::xlogy(x_tn, x_tn / n) + details::xlogy(x_fp, x_fp / n)
             + details::xlogy(x_fn, x_fn / n) + details::xlogy(x_tp, x_tp / n));

    Profile::constrained_fit(y, x, metric_n, n, p_h0);

    const double nll_h1
        = -2.0
          * (details::xlogy(x_tn, p_h0[0]) + details::xlogy(x_fp, p_h0[1])
             + details::xlogy(x_fn, p_h0[2]) + details::xlogy(x_tp, p_h0[3]));
    return nll_h1 - nll_h0;
}

// =============================================================================
// Chi2 Score Functions
// =============================================================================

/**
 * Compute the chi2 score for a single (y, x) point.
 *
 * @tparam Profile  The metric profile class
 * @param y         Y-axis metric value
 * @param x         X-axis metric value
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param epsilon   Clipping value to avoid boundary issues
 * @return          The chi2 score
 */
template <typename Profile>
inline double multn_chi2_score(
    const double y,
    const double x,
    const int64_t* __restrict conf_mat,
    const double epsilon = 1e-4) {
    std::array<double, 4> probas;
    double* p = probas.data();
    const auto n = static_cast<double>(
        conf_mat[0] + conf_mat[1] + conf_mat[2] + conf_mat[3]);
    const double max_val = 1.0 - epsilon;
    return prof_loglike<Profile>(
        details::clamp(y, epsilon, max_val),
        details::clamp(x, epsilon, max_val),
        n,
        conf_mat,
        p);
}

/**
 * Compute chi2 scores for multiple (y, x) points.
 *
 * @tparam Profile  The metric profile class
 * @param n_points  Number of points to evaluate
 * @param ys        Array of y-axis metric values
 * @param xs        Array of x-axis metric values
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param scores    Output array for chi2 scores
 * @param epsilon   Clipping value to avoid boundary issues
 */
template <typename Profile>
inline void multn_chi2_scores(
    const int64_t n_points,
    const double* ys,
    const double* xs,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double epsilon = 1e-4) {
    std::array<double, 4> probas;
    double* p = probas.data();

    prof_loglike_store nll_store;
    set_store<Profile>(conf_mat, &nll_store);

    const double max_val = 1.0 - epsilon;
    for (int64_t i = 0; i < n_points; ++i) {
        scores[i] = prof_loglike<Profile>(
            details::clamp(ys[i], epsilon, max_val),
            details::clamp(xs[i], epsilon, max_val),
            &nll_store,
            p);
    }
}

#ifdef MMU_HAS_OPENMP_SUPPORT
/**
 * Compute chi2 scores for multiple (y, x) points (multi-threaded).
 *
 * @tparam Profile  The metric profile class
 * @param n_points  Number of points to evaluate
 * @param ys        Array of y-axis metric values
 * @param xs        Array of x-axis metric values
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param scores    Output array for chi2 scores
 * @param epsilon   Clipping value to avoid boundary issues
 */
template <typename Profile>
inline void multn_chi2_scores_mt(
    const int64_t n_points,
    const double* ys,
    const double* xs,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double epsilon = 1e-4) {
    // Shared state (read-only after init)
    prof_loglike_store nll_store;
    set_store<Profile>(conf_mat, &nll_store);
    const double max_val = 1.0 - epsilon;

#pragma omp parallel shared(ys, xs, nll_store, scores)
    {
        // Thread-private memory
        std::array<double, 4> probas;
        double* p = probas.data();
        prof_loglike_store* nll_ptr = &nll_store;

#pragma omp for
        for (int64_t i = 0; i < n_points; ++i) {
            scores[i] = prof_loglike<Profile>(
                details::clamp(ys[i], epsilon, max_val),
                details::clamp(xs[i], epsilon, max_val),
                nll_ptr,
                p);
        }
    }  // omp parallel
}
#endif  // MMU_HAS_OPENMP_SUPPORT

// =============================================================================
// Grid Error Functions
// =============================================================================

// Forward declaration for get_grid_bounds (will be in grid_bounds.hpp)
template <typename Profile>
void get_grid_bounds(
    const int64_t* __restrict conf_mat,
    double* bounds,
    const double n_sigmas,
    const double epsilon);

/**
 * Compute chi2 scores over a uniform grid.
 * Grid bounds are automatically determined based on metric sigma.
 *
 * @tparam Profile  The metric profile class
 * @param n_bins    Number of bins in each dimension
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param result    Output array for chi2 scores (n_bins x n_bins)
 * @param bounds    Output array for grid bounds [y_min, y_max, x_min, x_max]
 * @param n_sigmas  Number of sigmas for grid bounds
 * @param epsilon   Clipping value to avoid boundary issues
 */
template <typename Profile>
inline void multn_error(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    std::array<double, 4> probas;
    double* p = probas.data();

    // Obtain y_start, y_end, x_start, x_end
    get_grid_bounds<Profile>(conf_mat, bounds, n_sigmas, epsilon);
    auto x_grid = std::unique_ptr<double[]>(new double[n_bins]);
    details::linspace(bounds[2], bounds[3], n_bins, x_grid.get());
    const double y_start = bounds[0];
    const double y_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    prof_loglike_store nll_store;
    set_store<Profile>(conf_mat, &nll_store);

    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; i++) {
        double y = y_start + (static_cast<double>(i) * y_delta);
        for (int64_t j = 0; j < n_bins; j++) {
            result[idx] = prof_loglike<Profile>(y, x_grid[j], &nll_store, p);
            idx++;
        }
    }
}

#ifdef MMU_HAS_OPENMP_SUPPORT
/**
 * Compute chi2 scores over a uniform grid (multi-threaded).
 *
 * @tparam Profile  The metric profile class
 * @param n_bins    Number of bins in each dimension
 * @param conf_mat  Confusion matrix [TN, FP, FN, TP]
 * @param result    Output array for chi2 scores (n_bins x n_bins)
 * @param bounds    Output array for grid bounds [y_min, y_max, x_min, x_max]
 * @param n_sigmas  Number of sigmas for grid bounds
 * @param epsilon   Clipping value to avoid boundary issues
 * @param n_threads Number of threads to use
 */
template <typename Profile>
inline void multn_error_mt(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int n_threads = 4) {
    // Obtain y_start, y_end, x_start, x_end
    get_grid_bounds<Profile>(conf_mat, bounds, n_sigmas, epsilon);
    auto x_grid = std::unique_ptr<double[]>(new double[n_bins]);
    details::linspace(bounds[2], bounds[3], n_bins, x_grid.get());
    const double y_start = bounds[0];
    const double y_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    prof_loglike_store nll_store;
    set_store<Profile>(conf_mat, &nll_store);

#pragma omp parallel num_threads(n_threads) shared(x_grid, nll_store, result)
    {
        std::array<double, 4> probas;
        double* p = probas.data();

#pragma omp for
        for (int64_t i = 0; i < n_bins; i++) {
            int64_t idx;
            double y;
            y = y_start + (static_cast<double>(i) * y_delta);
            idx = i * n_bins;
            for (int64_t j = 0; j < n_bins; j++) {
                result[idx + j]
                    = prof_loglike<Profile>(y, x_grid[j], &nll_store, p);
            }
        }
    }  // omp parallel
}
#endif  // MMU_HAS_OPENMP_SUPPORT

// =============================================================================
// Grid Error Functions with User-Provided Grid
// =============================================================================

// Forward declaration for GridBounds class and get_grid_bounds with index
// output
template <typename Profile>
class GenericGridBounds;

template <typename Profile>
void get_grid_bounds(
    const int64_t n_y_bins,
    const int64_t n_x_bins,
    const int64_t* __restrict conf_mat,
    const double* __restrict y_grid,
    const double* __restrict x_grid,
    int64_t* __restrict idx_bounds,
    const double n_sigmas,
    const double epsilon);

/**
 * Compute chi2 scores over a user-provided grid.
 * Only evaluates points within the computed bounds for efficiency.
 *
 * @tparam Profile   The metric profile class
 * @param n_y_bins   Number of y-axis bins
 * @param n_x_bins   Number of x-axis bins
 * @param y_grid     Array of y-axis values
 * @param x_grid     Array of x-axis values
 * @param conf_mat   Confusion matrix [TN, FP, FN, TP]
 * @param scores     Output array for chi2 scores (n_y_bins x n_x_bins)
 * @param n_sigmas   Number of sigmas for grid bounds
 * @param epsilon    Clipping value to avoid boundary issues
 */
template <typename Profile>
inline void multn_grid_error(
    const int64_t n_y_bins,
    const int64_t n_x_bins,
    const double* __restrict y_grid,
    const double* __restrict x_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // Initialize scores to high value (results in p-values close to 0)
    std::fill(scores, scores + n_y_bins * n_x_bins, MULT_DEFAULT_CHI2_SCORE);

    std::array<double, 4> probas;
    double* p = probas.data();

    std::array<int64_t, 4> bounds;
    int64_t* idx_bounds = bounds.data();

    prof_loglike_store nll_store;

    // Obtain the indexes over which to loop
    get_grid_bounds<Profile>(
        n_y_bins,
        n_x_bins,
        conf_mat,
        y_grid,
        x_grid,
        idx_bounds,
        n_sigmas,
        epsilon);
    const int64_t y_idx_min = idx_bounds[0];
    const int64_t y_idx_max = idx_bounds[1];
    const int64_t x_idx_min = idx_bounds[2];
    const int64_t x_idx_max = idx_bounds[3];

    set_store<Profile>(conf_mat, &nll_store);

    for (int64_t i = y_idx_min; i < y_idx_max; i++) {
        double y = y_grid[i];
        int64_t odx = i * n_x_bins;
        for (int64_t j = x_idx_min; j < x_idx_max; j++) {
            double score = prof_loglike<Profile>(y, x_grid[j], &nll_store, p);
            int64_t idx = odx + j;
            // Log likelihoods are always positive
            if (score < scores[idx]) {
                scores[idx] = score;
            }
        }
    }
}

/**
 * Compute chi2 scores over a user-provided grid for multiple confusion
 * matrices. Takes the minimum score across all confusion matrices.
 *
 * @tparam Profile    The metric profile class
 * @param n_y_bins    Number of y-axis bins
 * @param n_x_bins    Number of x-axis bins
 * @param n_conf_mats Number of confusion matrices
 * @param y_grid      Array of y-axis values
 * @param x_grid      Array of x-axis values
 * @param conf_mat    Array of confusion matrices (n_conf_mats x 4)
 * @param scores      Output array for chi2 scores (n_y_bins x n_x_bins)
 * @param n_sigmas    Number of sigmas for grid bounds
 * @param epsilon     Clipping value to avoid boundary issues
 */
template <typename Profile>
inline void multn_grid_curve_error(
    const int64_t n_y_bins,
    const int64_t n_x_bins,
    const int64_t n_conf_mats,
    const double* __restrict y_grid,
    const double* __restrict x_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // Initialize scores to high value
    std::fill(scores, scores + n_y_bins * n_x_bins, MULT_DEFAULT_CHI2_SCORE);

    std::array<double, 4> probas;
    double* p = probas.data();

    prof_loglike_store nll_store;

    GenericGridBounds<Profile> bounds(
        n_y_bins, n_x_bins, n_sigmas, epsilon, y_grid, x_grid);

    for (int64_t k = 0; k < n_conf_mats; k++) {
        // Update to new conf_mat
        set_store<Profile>(conf_mat, &nll_store);
        bounds.compute_bounds(conf_mat);

        for (int64_t i = bounds.y_idx_min; i < bounds.y_idx_max; i++) {
            double y = y_grid[i];
            const int64_t odx = i * n_x_bins;
            for (int64_t j = bounds.x_idx_min; j < bounds.x_idx_max; j++) {
                double score
                    = prof_loglike<Profile>(y, x_grid[j], &nll_store, p);
                if (score < scores[odx + j]) {
                    scores[odx + j] = score;
                }
            }
        }
        // Increment pointer
        conf_mat += 4;
    }
}

#ifdef MMU_HAS_OPENMP_SUPPORT
/**
 * Compute chi2 scores over a user-provided grid for multiple confusion matrices
 * (multi-threaded). Takes the minimum score across all confusion matrices and
 * threads.
 *
 * @tparam Profile    The metric profile class
 * @param n_y_bins    Number of y-axis bins
 * @param n_x_bins    Number of x-axis bins
 * @param n_conf_mats Number of confusion matrices
 * @param y_grid      Array of y-axis values
 * @param x_grid      Array of x-axis values
 * @param conf_mat    Array of confusion matrices (n_conf_mats x 4)
 * @param scores      Output array for chi2 scores (n_y_bins x n_x_bins)
 * @param n_sigmas    Number of sigmas for grid bounds
 * @param epsilon     Clipping value to avoid boundary issues
 * @param n_threads   Number of threads to use
 */
template <typename Profile>
inline void multn_grid_curve_error_mt(
    const int64_t n_y_bins,
    const int64_t n_x_bins,
    const int64_t n_conf_mats,
    const double* __restrict y_grid,
    const double* __restrict x_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int64_t n_threads = 4) {
    const int64_t n_elem = n_y_bins * n_x_bins;
    const int64_t t_elem = n_elem * n_threads;
    auto thread_scores = std::unique_ptr<double[]>(new double[t_elem]);

    // Initialize all thread scores to high value
    std::fill(
        thread_scores.get(),
        thread_scores.get() + t_elem,
        MULT_DEFAULT_CHI2_SCORE);

#pragma omp parallel num_threads(n_threads) \
    shared(n_y_bins,                        \
               n_x_bins,                    \
               n_conf_mats,                 \
               y_grid,                      \
               x_grid,                      \
               conf_mat,                    \
               n_sigmas,                    \
               epsilon)
    {
        double* thread_block
            = thread_scores.get() + (omp_get_thread_num() * n_elem);

        std::array<double, 4> probas;
        double* p = probas.data();

        prof_loglike_store nll_store;

        GenericGridBounds<Profile> bounds(
            n_y_bins, n_x_bins, n_sigmas, epsilon, y_grid, x_grid);

#pragma omp for
        for (int64_t k = 0; k < n_conf_mats; k++) {
            const int64_t* lcm = conf_mat + (k * 4);
            // Update to new conf_mat
            set_store<Profile>(lcm, &nll_store);
            bounds.compute_bounds(lcm);

            for (int64_t i = bounds.y_idx_min; i < bounds.y_idx_max; i++) {
                double y = y_grid[i];
                int64_t odx = i * n_x_bins;
                for (int64_t j = bounds.x_idx_min; j < bounds.x_idx_max; j++) {
                    double score
                        = prof_loglike<Profile>(y, x_grid[j], &nll_store, p);
                    int64_t idx = odx + j;
                    if (score < thread_block[idx]) {
                        thread_block[idx] = score;
                    }
                }
            }
        }
    }  // omp parallel

    // Collect the scores (reduce across threads)
    auto offsets = std::unique_ptr<int64_t[]>(new int64_t[n_threads]);
    for (int64_t j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    for (int64_t i = 0; i < n_elem; i++) {
        double min_score = MULT_DEFAULT_CHI2_SCORE;
        for (int64_t j = 0; j < n_threads; j++) {
            double tscore = thread_scores[i + offsets[j]];
            if (tscore < min_score) {
                min_score = tscore;
            }
        }
        scores[i] = min_score;
    }
}
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace multn
}  // namespace core
}  // namespace mmu
