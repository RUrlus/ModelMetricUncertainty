#pragma once

#include <algorithm>
#include <cstdint>

#include <mmu/core/multn_loglike/profiles.hpp>

namespace mmu {
namespace core {
namespace multn {

template <typename Profile>
inline void get_grid_bounds(
    const int64_t* __restrict conf_mat,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    double y_obs, x_obs;
    Profile::metric_values(conf_mat, y_obs, x_obs);

    double y_sigma, x_sigma;
    Profile::metric_sigmas(conf_mat, y_sigma, x_sigma);

    double max_y_clip = epsilon;
    double max_x_clip = epsilon;
    Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);

    const double y_lo = epsilon;
    const double y_hi = 1.0 - max_y_clip;
    const double x_lo = epsilon;
    const double x_hi = 1.0 - max_x_clip;

    bounds[0]
        = mmu::core::details::clamp(y_obs - n_sigmas * y_sigma, y_lo, y_hi);
    bounds[1]
        = mmu::core::details::clamp(y_obs + n_sigmas * y_sigma, y_lo, y_hi);
    bounds[2]
        = mmu::core::details::clamp(x_obs - n_sigmas * x_sigma, x_lo, x_hi);
    bounds[3]
        = mmu::core::details::clamp(x_obs + n_sigmas * x_sigma, x_lo, x_hi);
}

template <typename Profile>
struct GridBounds {
    const int64_t n_y_bins;
    const int64_t n_x_bins;
    const double n_sigmas;
    const double epsilon;
    const double* __restrict y_grid;
    const double* __restrict x_grid;

    int64_t y_idx_min = 0;
    int64_t y_idx_max = 0;
    int64_t x_idx_min = 0;
    int64_t x_idx_max = 0;

    GridBounds(
        const int64_t n_y_bins,
        const int64_t n_x_bins,
        const double n_sigmas,
        const double epsilon,
        const double* __restrict y_grid,
        const double* __restrict x_grid)
        : n_y_bins(n_y_bins),
          n_x_bins(n_x_bins),
          n_sigmas(n_sigmas),
          epsilon(epsilon),
          y_grid(y_grid),
          x_grid(x_grid) {}

    inline void compute_bounds(const int64_t* __restrict conf_mat) {
        double bounds[4];
        get_grid_bounds<Profile>(conf_mat, bounds, n_sigmas, epsilon);

        y_idx_min = static_cast<int64_t>(
            std::lower_bound(y_grid, y_grid + n_y_bins, bounds[0]) - y_grid);
        y_idx_max = static_cast<int64_t>(
            std::upper_bound(y_grid, y_grid + n_y_bins, bounds[1]) - y_grid);

        x_idx_min = static_cast<int64_t>(
            std::lower_bound(x_grid, x_grid + n_x_bins, bounds[2]) - x_grid);
        x_idx_max = static_cast<int64_t>(
            std::upper_bound(x_grid, x_grid + n_x_bins, bounds[3]) - x_grid);
    }
};

}  // namespace multn
}  // namespace core
}  // namespace mmu
