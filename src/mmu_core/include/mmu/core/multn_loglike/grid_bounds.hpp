/* grid_bounds.hpp -- Utility functions for multinomial log-likelihood grids
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <mmu/config.hpp>
#include <mmu/core/multn_loglike/profiles.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 *
 * bounds layout:
 *  0 y_min
 *  1 y_max
 *  2 x_min
 *  3 x_max
 */

namespace mmu {
namespace core {
namespace multn {

namespace details {

inline void linspace(
    const double start,
    const double end,
    const size_t steps,
    double* values) {
    if (steps == 0) {
        throw std::runtime_error("`steps` must be greater than zero.");
    } else if (steps == 1) {
        values[0] = start;
        return;
    }

    const double delta = (end - start) / static_cast<double>(steps - 1);
    const size_t last = steps - 1;

    values[0] = start;
    values[last] = end;
    for (size_t i = 1; i < last; ++i) {
        values[i] = start + (delta * static_cast<double>(i));
    }
}

template <typename T>
inline T clamp(const T v, const T lo, const T hi) {
    return std::min(std::max(v, lo), hi);
}

/* Index of the first grid point strictly greater than `value`, minus one. */
inline int64_t lower_index(
    const double* __restrict grid,
    const int64_t n_bins,
    const double value) {
    const int64_t idx = std::upper_bound(grid, grid + n_bins, value) - grid;
    if (idx >= n_bins) {
        return 0;
    }
    return idx > 0 ? idx - 1 : 0;
}

/* Index one past the first grid point strictly greater than `value`.
 *
 * Mirrors `lower_index`: returns `n_bins` when no grid point exceeds `value`,
 * and clamps to `n_bins` otherwise. `from` restricts the search to the
 * already-determined lower bound, as the original loop did.
 */
inline int64_t upper_index(
    const double* __restrict grid,
    const int64_t n_bins,
    const double value,
    const int64_t from) {
    const int64_t begin = from > 0 ? from : 0;
    const int64_t idx
        = std::upper_bound(grid + begin, grid + n_bins, value) - grid;
    if (idx >= n_bins) {
        return n_bins;
    }
    return idx + 1 <= n_bins ? idx + 1 : n_bins;
}

/* Defensive guard against a non-finite grid.
 *
 * `Profile::bvn_sigma` is required to return finite values for every possible
 * confusion matrix. If that contract is ever broken the NaN propagates
 * silently through the whole chi2 grid and surfaces as an empty plot rather
 * than an error, so we fail loudly here instead.
 *
 * Only safe to call outside an OpenMP region.
 */
inline void assert_finite_bounds(const double* __restrict bounds) {
    if (!(std::isfinite(bounds[0]) && std::isfinite(bounds[1])
          && std::isfinite(bounds[2]) && std::isfinite(bounds[3]))) {
        throw std::runtime_error(
            "Encountered a non-finite grid bound; the confusion matrix is "
            "likely degenerate.");
    }
}

}  // namespace details

// =============================================================================
// Continuous value bounds for auto-generated grids
// =============================================================================

template <typename Profile>
inline void get_grid_bounds(
    const int64_t* __restrict conf_mat,
    double* __restrict bounds,
    const double n_sigmas = MULT_DEFAULT_N_SIGMAS,
    double epsilon = MULT_DEFAULT_EPSILON) {
    double max_y_clip = epsilon;
    double max_x_clip = epsilon;
    Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);

    std::array<double, 4> metrics;
    Profile::bvn_sigma(conf_mat, metrics.data());

    const double y = metrics[0];
    const double y_sigma = metrics[1];
    const double x = metrics[2];
    const double x_sigma = metrics[3];

    const double ns_y_sigma = n_sigmas * y_sigma;
    const double ns_x_sigma = n_sigmas * x_sigma;

    bounds[0] = std::max(y - ns_y_sigma, epsilon);
    bounds[1] = std::min(y + ns_y_sigma, 1.0 - max_y_clip);
    bounds[2] = std::max(x - ns_x_sigma, epsilon);
    bounds[3] = std::min(x + ns_x_sigma, 1.0 - max_x_clip);

    details::assert_finite_bounds(bounds);
}

// =============================================================================
// Index bounds helper for user-provided grids
// =============================================================================

template <typename Profile>
class GridBounds {
    const int64_t n_y_bins;
    const int64_t n_x_bins;
    const double n_sigmas;
    const double epsilon;

    const double* __restrict ys;
    const double* __restrict xs;

    double max_y_clip;
    double max_x_clip;

    double y_min;
    double y_max;
    double x_min;
    double x_max;

    double ns_y_sigma;
    double ns_x_sigma;

    std::array<double, 4> metrics;

   public:
    int64_t y_idx_min = 0;
    int64_t y_idx_max = 0;
    int64_t x_idx_min = 0;
    int64_t x_idx_max = 0;

    GridBounds(
        const int64_t n_y_bins,
        const int64_t n_x_bins,
        const double n_sigmas,
        const double epsilon,
        const double* __restrict ys,
        const double* __restrict xs)
        : n_y_bins{n_y_bins},
          n_x_bins{n_x_bins},
          n_sigmas{n_sigmas},
          epsilon{epsilon},
          ys{ys},
          xs{xs},
          max_y_clip{epsilon},
          max_x_clip{epsilon},
          y_min{0.0},
          y_max{0.0},
          x_min{0.0},
          x_max{0.0},
          ns_y_sigma{0.0},
          ns_x_sigma{0.0},
          metrics{} {}

    inline void compute_bounds(const int64_t* __restrict conf_mat) {
        Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);
        Profile::bvn_sigma(conf_mat, metrics.data());

        ns_y_sigma = n_sigmas * metrics[1];
        y_max = std::min(metrics[0] + ns_y_sigma, 1.0 - max_y_clip);
        y_min = std::max(metrics[0] - ns_y_sigma, epsilon);

        y_idx_min = details::lower_index(ys, n_y_bins, y_min);
        y_idx_max = details::upper_index(ys, n_y_bins, y_max, y_idx_min);

        ns_x_sigma = n_sigmas * metrics[3];
        x_max = std::min(metrics[2] + ns_x_sigma, 1.0 - max_x_clip);
        x_min = std::max(metrics[2] - ns_x_sigma, epsilon);

        x_idx_min = details::lower_index(xs, n_x_bins, x_min);
        x_idx_max = details::upper_index(xs, n_x_bins, x_max, x_idx_min);
    }
};

template <typename Profile>
inline void get_grid_bounds(
    const int64_t y_bins,
    const int64_t x_bins,
    const int64_t* __restrict conf_mat,
    const double* __restrict ys,
    const double* __restrict xs,
    int64_t* __restrict result,
    const double n_sigmas = MULT_DEFAULT_N_SIGMAS,
    double epsilon = MULT_DEFAULT_EPSILON) {
    double max_y_clip = epsilon;
    double max_x_clip = epsilon;
    Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);

    std::array<double, 4> metrics;
    Profile::bvn_sigma(conf_mat, metrics.data());

    const double ns_y_sigma = n_sigmas * metrics[1];
    const double y_max = std::min(metrics[0] + ns_y_sigma, 1.0 - max_y_clip);
    const double y_min = std::max(metrics[0] - ns_y_sigma, epsilon);

    result[0] = details::lower_index(ys, y_bins, y_min);
    result[1] = details::upper_index(ys, y_bins, y_max, result[0]);

    const double ns_x_sigma = n_sigmas * metrics[3];
    const double x_max = std::min(metrics[2] + ns_x_sigma, 1.0 - max_x_clip);
    const double x_min = std::max(metrics[2] - ns_x_sigma, epsilon);

    result[2] = details::lower_index(xs, x_bins, x_min);
    result[3] = details::upper_index(xs, x_bins, x_max, result[2]);
}

}  // namespace multn
}  // namespace core
}  // namespace mmu
