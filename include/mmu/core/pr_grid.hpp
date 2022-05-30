/* pr_grid.hpp -- Utility functions used in multn_loglike and bvn_grid
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_PR_GRID_HPP_
#define INCLUDE_MMU_CORE_PR_GRID_HPP_

#include <algorithm>
#include <array>
#include <mmu/core/common.hpp>
#include <mmu/core/bvn_error.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {
namespace details {

inline void linspace(const double start, double const end, const size_t steps, double* values) {
    if (steps == 0) {
        throw std::runtime_error("`steps` must be greater than zero.");
    } else if (steps == 1) {
        auto v = new double[steps];
        v[0] = static_cast<double>(start);
        return;
    }
    const double delta = (end - start) / static_cast<double>(steps - 1);
    const size_t N = steps - 1;
    values[0] = start;
    values[N] = end;
    for (size_t i = 1; i < N; ++i) {
        values[i] = start + (delta * i);
    }
    return;
}

/* Determine grid for precision and recall based on their marginal std
 * deviations assuming a Multivariate Normal
 */
inline void get_pr_grid_bounds(
    const int64_t* __restrict conf_mat,
    double* bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    const double max_prec_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
    const double max_rec_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    // computes prec, prec_sigma, rec, rec_sigma accounting for edge cases
    std::array<double, 4> prec_rec;
    pr_bvn_sigma(conf_mat, prec_rec.data());

    const double ns_prec_sigma = n_sigmas * prec_rec[1];
    const double ns_rec_sigma = n_sigmas * prec_rec[3];

    bounds[0] = std::max(prec_rec[0] - ns_prec_sigma, epsilon);
    bounds[1] = std::min(prec_rec[0] + ns_prec_sigma, 1 - max_prec_clip);
    bounds[2] = std::max(prec_rec[2] - ns_rec_sigma, epsilon);
    bounds[3] = std::min(prec_rec[2] + ns_rec_sigma, 1. - max_rec_clip);
}  // get_pr_grid_bounds

class PrGridBounds {
    const int64_t n_prec_bins;
    const int64_t n_rec_bins;
    const double n_sigmas;
    const double epsilon;
    double max_prec_clip;
    double max_rec_clip;
    double prec_max;
    double prec_min;
    double rec_max;
    double rec_min;
    double ns_prec_sigma;
    double ns_rec_sigma;
    const double* __restrict precs;
    const double* __restrict recs;
    std::array<double, 4> prec_rec;

   public:
    int64_t prec_idx_min = 0;
    int64_t prec_idx_max = 0;
    int64_t rec_idx_min = 0;
    int64_t rec_idx_max = 0;

    PrGridBounds(
        const int64_t n_prec_bins,
        const int64_t n_rec_bins,
        const double n_sigmas,
        const double epsilon,
        const double* __restrict precs,
        const double* __restrict recs)
        : n_prec_bins{n_prec_bins},
          n_rec_bins{n_rec_bins},
          n_sigmas{n_sigmas},
          epsilon{epsilon},
          precs{precs},
          recs{recs} {}

    void compute_bounds(const int64_t* __restrict conf_mat) {
        max_prec_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
        max_rec_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
        // computes prec, prec_sigma, rec, rec_sigma accounting for edge cases
        pr_bvn_sigma(conf_mat, prec_rec.data());

        ns_prec_sigma = n_sigmas * prec_rec[1];
        prec_max = std::min(prec_rec[0] + ns_prec_sigma, 1 - max_prec_clip);
        prec_min = std::max(prec_rec[0] - ns_prec_sigma, epsilon);
        prec_idx_min = 0;
        prec_idx_max = n_prec_bins;

        int64_t i;
        for (i = 0; i < n_prec_bins; i++) {
            if (prec_min < precs[i]) {
                prec_idx_min = i - 1;
                break;
            }
        }
        prec_idx_min = prec_idx_min > 0 ? prec_idx_min : 0;

        for (i = prec_idx_min; i < n_rec_bins; i++) {
            if (prec_max < precs[i]) {
                prec_idx_max = i + 1;
                break;
            }
        }
        prec_idx_max = prec_idx_max <= n_prec_bins ? prec_idx_max : n_prec_bins;

        ns_rec_sigma = n_sigmas * prec_rec[3];
        rec_max = std::min(prec_rec[2] + ns_rec_sigma, 1. - max_rec_clip);
        rec_min = std::max(prec_rec[2] - ns_rec_sigma, epsilon);
        rec_idx_min = 0;
        rec_idx_max = n_rec_bins;
        for (i = 0; i < n_rec_bins; i++) {
            if (rec_min < recs[i]) {
                rec_idx_min = i - 1;
                break;
            }
        }
        rec_idx_min = rec_idx_min > 0 ? rec_idx_min : 0;

        for (i = rec_idx_min; i < n_rec_bins; i++) {
            if (rec_max < recs[i]) {
                rec_idx_max = i + 1;
                break;
            }
        }
        rec_idx_max = rec_idx_max <= n_rec_bins ? rec_idx_max : n_rec_bins;
    }
};

inline void get_pr_grid_bounds(
    const int64_t prec_bins,
    const int64_t rec_bins,
    const int64_t* __restrict conf_mat,
    const double* __restrict precs,
    const double* __restrict recs,
    int64_t* result,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    const double max_prec_clip = conf_mat[1] == 0 ? 0.0 : epsilon;
    const double max_rec_clip = conf_mat[2] == 0 ? 0.0 : epsilon;
    // computes prec, prec_sigma, rec, rec_sigma accounting for edge cases
    std::array<double, 4> prec_rec;
    pr_bvn_sigma(conf_mat, prec_rec.data());

    const double ns_prec_sigma = n_sigmas * prec_rec[1];
    const double prec_max = std::min(prec_rec[0] + ns_prec_sigma, 1 - max_prec_clip);
    const double prec_min = std::max(prec_rec[0] - ns_prec_sigma, epsilon);
    int64_t prec_idx_min = 0;
    int64_t prec_idx_max = prec_bins;
    for (int64_t i = 0; i < prec_bins; i++) {
        if (prec_min < precs[i]) {
            prec_idx_min = i - 1;
            break;
        }
    }
    result[0] = prec_idx_min > 0 ? prec_idx_min : 0;

    for (int64_t i = prec_idx_min; i < rec_bins; i++) {
        if (prec_max < precs[i]) {
            prec_idx_max = i + 1;
            break;
        }
    }
    result[1] = prec_idx_max <= prec_bins ? prec_idx_max : prec_bins;

    const double ns_rec_sigma = n_sigmas * prec_rec[3];
    const double rec_max = std::min(prec_rec[2] + ns_rec_sigma, 1. - max_rec_clip);
    const double rec_min = std::max(prec_rec[2] - ns_rec_sigma, epsilon);
    int64_t rec_idx_min = 0;
    int64_t rec_idx_max = rec_bins;
    for (int64_t i = 0; i < rec_bins; i++) {
        if (rec_min < recs[i]) {
            rec_idx_min = i - 1;
            break;
        }
    }
    result[2] = rec_idx_min > 0 ? rec_idx_min : 0;

    for (int64_t i = rec_idx_min; i < rec_bins; i++) {
        if (rec_max < recs[i]) {
            rec_idx_max = i + 1;
            break;
        }
    }
    result[3] = rec_idx_max <= rec_bins ? rec_idx_max : rec_bins;
}  // get_pr_grid_bounds

}  // namespace details
}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_PR_GRID_HPP_
