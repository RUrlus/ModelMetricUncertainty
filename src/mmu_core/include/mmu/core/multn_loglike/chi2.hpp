#pragma once

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <mmu/config.hpp>
#include <mmu/core/common.hpp>
#include <mmu/core/multn_loglike/grid_bounds.hpp>
#include <mmu/core/multn_loglike/loglike.hpp>
#include <mmu/core/multn_loglike/profiles.hpp>

namespace mmu {
namespace core {
namespace multn {

template <typename Profile>
inline double multn_chi2_score(
    const double y,
    const double x,
    const int64_t* __restrict conf_mat,
    double epsilon = MULT_DEFAULT_EPSILON) {
    std::array<double, 4> probas;
    double* p = probas.data();

    double max_y_clip = epsilon;
    double max_x_clip = epsilon;
    Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);

    const double y_eval
        = mmu::core::details::clamp(y, epsilon, 1.0 - max_y_clip);
    const double x_eval
        = mmu::core::details::clamp(x, epsilon, 1.0 - max_x_clip);

    return prof_loglike<Profile, true>(y_eval, x_eval, conf_mat, p);
}

template <typename Profile>
inline void multn_chi2_scores(
    const int64_t n_points,
    const double* ys,
    const double* xs,
    const int64_t* __restrict conf_mat,
    double* scores,
    double epsilon = MULT_DEFAULT_EPSILON) {
    std::array<double, 4> probas;
    double* p = probas.data();

    prof_loglike_store store;
    set_store<Profile>(conf_mat, &store);

    double max_y_clip = epsilon;
    double max_x_clip = epsilon;
    Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);
    const double y_max = 1.0 - max_y_clip;
    const double x_max = 1.0 - max_x_clip;

    for (int64_t i = 0; i < n_points; ++i) {
        scores[i] = prof_loglike<Profile, false>(
            mmu::core::details::clamp(ys[i], epsilon, y_max),
            mmu::core::details::clamp(xs[i], epsilon, x_max),
            store,
            p);
    }
}

/* Multi-threaded point evaluation.
 */
#ifdef MMU_HAS_OPENMP_SUPPORT
template <typename Profile>
inline void multn_chi2_scores_mt(
    const int64_t n_points,
    const double* ys,
    const double* xs,
    const int64_t* __restrict conf_mat,
    double* scores,
    double epsilon = MULT_DEFAULT_EPSILON) {
    prof_loglike_store store;
    set_store<Profile>(conf_mat, &store);

    double max_y_clip = epsilon;
    double max_x_clip = epsilon;
    Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);
    const double y_max = 1.0 - max_y_clip;
    const double x_max = 1.0 - max_x_clip;

#pragma omp parallel if (n_points > MMU_OMP_MIN_POINTS) \
    shared(ys, xs, scores, store, n_points, epsilon, y_max, x_max)
    {
        std::array<double, 4> probas;
        double* p = probas.data();

#pragma omp for schedule(static)
        for (int64_t i = 0; i < n_points; ++i) {
            scores[i] = prof_loglike<Profile, false>(
                mmu::core::details::clamp(ys[i], epsilon, y_max),
                mmu::core::details::clamp(xs[i], epsilon, x_max),
                store,
                p);
        }
    }
}
#endif

template <typename Profile>
inline void multn_error(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = MULT_DEFAULT_N_SIGMAS,
    double epsilon = MULT_DEFAULT_EPSILON) {
    std::array<double, 4> probas;
    double* p = probas.data();

    get_grid_bounds<Profile>(conf_mat, bounds, n_sigmas, epsilon);

    auto x_grid = std::unique_ptr<double[]>(new double[n_bins]);
    details::linspace(bounds[2], bounds[3], n_bins, x_grid.get());

    const double y_start = bounds[0];
    const double y_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    prof_loglike_store store;
    set_store<Profile>(conf_mat, &store);

    int64_t idx = 0;
    for (int64_t i = 0; i < n_bins; ++i) {
        const double y = y_start + static_cast<double>(i) * y_delta;
        for (int64_t j = 0; j < n_bins; ++j) {
            result[idx++]
                = prof_loglike<Profile, false>(y, x_grid[j], store, p);
        }
    }
}

#ifdef MMU_HAS_OPENMP_SUPPORT
template <typename Profile>
inline void multn_error_mt(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = MULT_DEFAULT_N_SIGMAS,
    double epsilon = MULT_DEFAULT_EPSILON,
    const int n_threads = 4) {
    get_grid_bounds<Profile>(conf_mat, bounds, n_sigmas, epsilon);

    auto x_grid = std::unique_ptr<double[]>(new double[n_bins]);
    details::linspace(bounds[2], bounds[3], n_bins, x_grid.get());

    const double y_start = bounds[0];
    const double y_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    prof_loglike_store store;
    set_store<Profile>(conf_mat, &store);

#pragma omp parallel num_threads(n_threads) shared(x_grid, store, result)
    {
        std::array<double, 4> probas;
        double* p = probas.data();

#pragma omp for schedule(static)
        for (int64_t i = 0; i < n_bins; ++i) {
            const double y = y_start + static_cast<double>(i) * y_delta;
            const int64_t row = i * n_bins;
            for (int64_t j = 0; j < n_bins; ++j) {
                result[row + j]
                    = prof_loglike<Profile, false>(y, x_grid[j], store, p);
            }
        }
    }
}
#endif

template <typename Profile>
inline void multn_grid_error(
    const int64_t n_y_bins,
    const int64_t n_x_bins,
    const double* __restrict y_grid,
    const double* __restrict x_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = MULT_DEFAULT_N_SIGMAS,
    double epsilon = MULT_DEFAULT_EPSILON) {
    std::fill(scores, scores + n_y_bins * n_x_bins, MULT_DEFAULT_CHI2_SCORE);

    std::array<double, 4> probas;
    double* p = probas.data();

    GridBounds<Profile> bounds(
        n_y_bins, n_x_bins, n_sigmas, epsilon, y_grid, x_grid);
    bounds.compute_bounds(conf_mat);

    prof_loglike_store store;
    set_store<Profile>(conf_mat, &store);

    for (int64_t i = bounds.y_idx_min; i < bounds.y_idx_max; ++i) {
        const double y = y_grid[i];
        const int64_t row = i * n_x_bins;
        for (int64_t j = bounds.x_idx_min; j < bounds.x_idx_max; ++j) {
            const double score
                = prof_loglike<Profile, false>(y, x_grid[j], store, p);
            const int64_t idx = row + j;
            if (score < scores[idx]) {
                scores[idx] = score;
            }
        }
    }
}

template <typename Profile>
inline void multn_grid_curve_error(
    const int64_t n_y_bins,
    const int64_t n_x_bins,
    const int64_t n_conf_mats,
    const double* __restrict y_grid,
    const double* __restrict x_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = MULT_DEFAULT_N_SIGMAS,
    double epsilon = MULT_DEFAULT_EPSILON) {
    std::fill(scores, scores + n_y_bins * n_x_bins, MULT_DEFAULT_CHI2_SCORE);

    std::array<double, 4> probas;
    double* p = probas.data();

    prof_loglike_store store;
    GridBounds<Profile> bounds(
        n_y_bins, n_x_bins, n_sigmas, epsilon, y_grid, x_grid);

    for (int64_t k = 0; k < n_conf_mats; ++k) {
        set_store<Profile>(conf_mat, &store);
        bounds.compute_bounds(conf_mat);

        for (int64_t i = bounds.y_idx_min; i < bounds.y_idx_max; ++i) {
            const double y = y_grid[i];
            const int64_t row = i * n_x_bins;
            for (int64_t j = bounds.x_idx_min; j < bounds.x_idx_max; ++j) {
                const double score
                    = prof_loglike<Profile, false>(y, x_grid[j], store, p);
                const int64_t idx = row + j;
                if (score < scores[idx]) {
                    scores[idx] = score;
                }
            }
        }
        conf_mat += 4;
    }
}

#ifdef MMU_HAS_OPENMP_SUPPORT

namespace details {

/* The sub-rectangle of the output grid that one confusion matrix touches. */
struct GridRegion {
    int64_t y_min;
    int64_t y_max;
    int64_t x_min;
    int64_t x_max;
};

}  // namespace details

/* Minimum-chi2 surface over a curve of confusion matrices. */
template <typename Profile>
inline void multn_grid_curve_error_mt(
    const int64_t n_y_bins,
    const int64_t n_x_bins,
    const int64_t n_conf_mats,
    const double* __restrict y_grid,
    const double* __restrict x_grid,
    const int64_t* __restrict conf_mat,
    double* __restrict scores,
    const double n_sigmas = MULT_DEFAULT_N_SIGMAS,
    double epsilon = MULT_DEFAULT_EPSILON,
    const int64_t n_threads = 4) {
    if (n_conf_mats < 1 || n_y_bins < 1 || n_x_bins < 1) {
        std::fill(
            scores, scores + (n_y_bins * n_x_bins), MULT_DEFAULT_CHI2_SCORE);
        return;
    }

    // -- Phase 1: cache each matrix's grid region and likelihood store.
    // Every iteration writes only its own slot, so there is no conflict.
    std::vector<prof_loglike_store> stores(static_cast<size_t>(n_conf_mats));
    std::vector<details::GridRegion> regions(static_cast<size_t>(n_conf_mats));

#pragma omp parallel num_threads(n_threads) \
    shared(stores, regions, conf_mat, y_grid, x_grid, n_conf_mats)
    {
        GridBounds<Profile> bounds(
            n_y_bins, n_x_bins, n_sigmas, epsilon, y_grid, x_grid);

#pragma omp for schedule(static)
        for (int64_t k = 0; k < n_conf_mats; ++k) {
            const int64_t* lcm = conf_mat + (k * 4);
            set_store<Profile>(lcm, &stores[k]);
            bounds.compute_bounds(lcm);
            regions[k] = details::GridRegion{
                bounds.y_idx_min,
                bounds.y_idx_max,
                bounds.x_idx_min,
                bounds.x_idx_max};
        }
    }

    // -- Phase 2: each tile owns a disjoint band of rows and writes directly.
    const int64_t n_blocks = std::min<int64_t>(n_y_bins, n_threads * 8);
    const details::GridRegion* const region_ptr = regions.data();
    const prof_loglike_store* const store_ptr = stores.data();

#pragma omp parallel num_threads(n_threads) \
    shared(scores, region_ptr, store_ptr, x_grid, y_grid, n_conf_mats)
    {
        std::array<double, 4> probas;
        double* p = probas.data();

#pragma omp for schedule(dynamic)
        for (int64_t b = 0; b < n_blocks; ++b) {
            const int64_t row_begin = (b * n_y_bins) / n_blocks;
            const int64_t row_end = ((b + 1) * n_y_bins) / n_blocks;
            if (row_begin >= row_end) {
                continue;
            }

            std::fill(
                scores + (row_begin * n_x_bins),
                scores + (row_end * n_x_bins),
                MULT_DEFAULT_CHI2_SCORE);

            for (int64_t k = 0; k < n_conf_mats; ++k) {
                const details::GridRegion& r = region_ptr[k];
                const int64_t i_lo = r.y_min > row_begin ? r.y_min : row_begin;
                const int64_t i_hi = r.y_max < row_end ? r.y_max : row_end;
                if (i_lo >= i_hi) {
                    continue;
                }

                const prof_loglike_store& store = store_ptr[k];
                for (int64_t i = i_lo; i < i_hi; ++i) {
                    const double y = y_grid[i];
                    double* __restrict row_ptr = scores + (i * n_x_bins);
                    for (int64_t j = r.x_min; j < r.x_max; ++j) {
                        const double score = prof_loglike<Profile, false>(
                            y, x_grid[j], store, p);
                        if (score < row_ptr[j]) {
                            row_ptr[j] = score;
                        }
                    }
                }
            }
        }
    }
}
#endif

}  // namespace multn
}  // namespace core
}  // namespace mmu
