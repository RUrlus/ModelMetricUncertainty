/* grid_bounds.hpp -- Template-based grid bounds for multinomial log-likelihood
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <mmu/core/common.hpp>
#include <mmu/core/multn_loglike/common.hpp>
#include <mmu/core/multn_loglike/profiles.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {

namespace details {

inline void linspace(
    const double start,
    double const end,
    const size_t steps,
    double* values) {
    if (steps == 0) {
        throw std::runtime_error("`steps` must be greater than zero.");
    } else if (steps == 1) {
        values[0] = static_cast<double>(start);
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

}  // namespace details

namespace multn {

/**
 * Compute metric value and standard deviation for grid bounds.
 * Uses linear error propagation over Poisson errors.
 *
 * Specialization for PrecisionRecallProfile:
 *   metrics[0] = precision
 *   metrics[1] = precision_sigma
 *   metrics[2] = recall
 *   metrics[3] = recall_sigma
 */
template <typename Profile>
inline void compute_metric_sigma(
    const int64_t* __restrict conf_mat,
    double* __restrict metrics);

template <>
inline void compute_metric_sigma<PrecisionRecallProfile>(
    const int64_t* __restrict conf_mat,
    double* __restrict metrics) {
    const int64_t itp = conf_mat[3];
    const auto tp = static_cast<double>(itp);

    const int64_t itp_fn = conf_mat[2] + conf_mat[3];
    const bool tp_fn_nonzero = itp_fn > 0;
    const auto tp_fn = static_cast<double>(itp_fn);

    const int64_t itp_fp = conf_mat[3] + conf_mat[1];
    const bool tp_fp_nonzero = itp_fp > 0;
    const auto tp_fp = static_cast<double>(itp_fp);

    // precision
    double prec;
    double prec_sigma;
    double prec_for_sigma;
    // precision == 1
    if (itp == itp_fp) {
        prec = 1.0;
        prec_for_sigma = static_cast<double>(itp_fp - 1) / tp_fp;
        prec_sigma = std::sqrt((prec_for_sigma * (1 - prec_for_sigma)) / tp_fp);
    } else if (tp_fp_nonzero) {
        prec = tp / (tp_fp);
        prec_sigma = std::sqrt(
            static_cast<double>(conf_mat[3] * conf_mat[1])
            / static_cast<double>(std::pow(tp_fp, 3.0)));
    } else {
        // precision == 0
        prec = 0.0;
        prec_for_sigma = 1.0 / tp_fp;
        prec_sigma = std::sqrt((prec_for_sigma * (1 - prec_for_sigma)) / tp_fp);
    }

    // recall
    double rec;
    double rec_sigma;
    double rec_for_sigma;
    // recall == 1
    if (itp == itp_fn) {
        rec = 1.0;
        rec_for_sigma = static_cast<double>(itp_fn - 1) / tp_fn;
        rec_sigma = std::sqrt((rec_for_sigma * (1 - rec_for_sigma)) / tp_fn);
    } else if (tp_fn_nonzero) {
        rec = tp / (tp_fn);
        rec_sigma = std::sqrt(
            static_cast<double>(conf_mat[3] * conf_mat[2])
            / static_cast<double>(std::pow(tp_fn, 3.0)));
    } else {
        // recall == 0.0
        rec = 0.0;
        rec_for_sigma = 1.0 / tp_fn;
        rec_sigma = std::sqrt((rec_for_sigma * (1 - rec_for_sigma)) / tp_fn);
    }

    metrics[0] = prec;
    metrics[1] = prec_sigma;
    metrics[2] = rec;
    metrics[3] = rec_sigma;
}

// ROC (TPR-FPR) specialization
template <>
inline void compute_metric_sigma<ROCProfile>(
    const int64_t* __restrict conf_mat,
    double* __restrict metrics) {
    // Y: TPR = TP / (TP + FN)
    const int64_t iterm1_Y = conf_mat[3];  // TP
    const bool term1_Y_nonzero = iterm1_Y > 0;
    auto term1_Y = static_cast<double>(iterm1_Y);

    const int64_t iterm2_Y = conf_mat[2];  // FN
    const bool term2_Y_nonzero = iterm2_Y > 0;
    auto term2_Y = static_cast<double>(iterm2_Y);

    double Y = term1_Y / (term1_Y + term2_Y);

    if (!term1_Y_nonzero) {
        term1_Y = 1;
    }
    if (!term2_Y_nonzero) {
        term2_Y = 1;
    }

    double Y_sigma = std::sqrt(
        static_cast<double>(term1_Y * term2_Y)
        / static_cast<double>(std::pow(term1_Y + term2_Y, 3.0)));

    // X: FPR = FP / (FP + TN)
    const int64_t iterm1_X = conf_mat[1];  // FP
    const bool term1_X_nonzero = iterm1_X > 0;
    auto term1_X = static_cast<double>(iterm1_X);

    const int64_t iterm2_X = conf_mat[0];  // TN
    const bool term2_X_nonzero = iterm2_X > 0;
    auto term2_X = static_cast<double>(iterm2_X);

    double X = term1_X / (term1_X + term2_X);

    if (!term1_X_nonzero) {
        term1_X = 1;
    }
    if (!term2_X_nonzero) {
        term2_X = 1;
    }

    double X_sigma = std::sqrt(
        static_cast<double>(term1_X * term2_X)
        / static_cast<double>(std::pow(term1_X + term2_X, 3.0)));

    metrics[0] = Y;
    metrics[1] = Y_sigma;
    metrics[2] = X;
    metrics[3] = X_sigma;
}

// Recall-PPN specialization
template <>
inline void compute_metric_sigma<PPNRecallProfile>(
    const int64_t* __restrict conf_mat,
    double* __restrict metrics) {
    const int64_t tn = conf_mat[0];
    const int64_t fp = conf_mat[1];
    const int64_t fn = conf_mat[2];
    const int64_t tp = conf_mat[3];
    const int64_t n_total = tn + fp + fn + tp;
    const auto n = static_cast<double>(n_total);

    // Y: PPN = (TN + FN) / N
    const auto ppn = static_cast<double>(tn + fn) / n;
    // PPN sigma using binomial standard error approximation
    const double ppn_sigma = std::sqrt(ppn * (1.0 - ppn) / n);

    // X: Recall = TP / (TP + FN)
    const int64_t n_pos = fn + tp;
    const auto n_pos_d = static_cast<double>(n_pos);

    double recall;
    double recall_sigma;
    if (n_pos > 0) {
        recall = static_cast<double>(tp) / n_pos_d;
        // Use binomial formula for sigma
        const auto tp_d = static_cast<double>(tp);
        const auto fn_d = static_cast<double>(fn);
        if (tp > 0 && fn > 0) {
            recall_sigma = std::sqrt((tp_d * fn_d) / std::pow(n_pos_d, 3.0));
        } else if (tp == n_pos) {
            // recall == 1
            const double recall_for_sigma
                = static_cast<double>(n_pos - 1) / n_pos_d;
            recall_sigma = std::sqrt(
                (recall_for_sigma * (1 - recall_for_sigma)) / n_pos_d);
        } else {
            // recall == 0
            const double recall_for_sigma = 1.0 / n_pos_d;
            recall_sigma = std::sqrt(
                (recall_for_sigma * (1 - recall_for_sigma)) / n_pos_d);
        }
    } else {
        recall = 0.0;
        recall_sigma = 0.0;
    }

    metrics[0] = ppn;
    metrics[1] = ppn_sigma;
    metrics[2] = recall;
    metrics[3] = recall_sigma;
}

// =============================================================================
// Grid Bounds Functions
// =============================================================================

/**
 * Compute grid bounds for uniform grid.
 * Returns bounds as [y_min, y_max, x_min, x_max].
 *
 * @tparam Profile   The metric profile class
 * @param conf_mat   Confusion matrix [TN, FP, FN, TP]
 * @param bounds     Output array [y_min, y_max, x_min, x_max]
 * @param n_sigmas   Number of sigmas for grid bounds
 * @param epsilon    Clipping value to avoid boundary issues
 */
template <typename Profile>
inline void get_grid_bounds(
    const int64_t* __restrict conf_mat,
    double* bounds,
    const double n_sigmas,
    const double epsilon) {
    double max_y_clip, max_x_clip;
    Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);

    // Compute metric values and sigmas
    std::array<double, 4> metric_vals;
    compute_metric_sigma<Profile>(conf_mat, metric_vals.data());

    const double ns_y_sigma = n_sigmas * metric_vals[1];
    const double ns_x_sigma = n_sigmas * metric_vals[3];

    bounds[0] = std::max(metric_vals[0] - ns_y_sigma, epsilon);
    bounds[1] = std::min(metric_vals[0] + ns_y_sigma, 1.0 - max_y_clip);
    bounds[2] = std::max(metric_vals[2] - ns_x_sigma, epsilon);
    bounds[3] = std::min(metric_vals[2] + ns_x_sigma, 1.0 - max_x_clip);
}

/**
 * Compute grid index bounds for a user-provided grid.
 * Returns bounds as [y_idx_min, y_idx_max, x_idx_min, x_idx_max].
 *
 * @tparam Profile   The metric profile class
 * @param n_y_bins   Number of y-axis bins
 * @param n_x_bins   Number of x-axis bins
 * @param conf_mat   Confusion matrix [TN, FP, FN, TP]
 * @param y_grid     Array of y-axis values
 * @param x_grid     Array of x-axis values
 * @param result     Output array [y_idx_min, y_idx_max, x_idx_min, x_idx_max]
 * @param n_sigmas   Number of sigmas for grid bounds
 * @param epsilon    Clipping value to avoid boundary issues
 */
template <typename Profile>
inline void get_grid_bounds(
    const int64_t n_y_bins,
    const int64_t n_x_bins,
    const int64_t* __restrict conf_mat,
    const double* __restrict y_grid,
    const double* __restrict x_grid,
    int64_t* result,
    const double n_sigmas,
    const double epsilon) {
    double max_y_clip, max_x_clip;
    Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);

    // Compute metric values and sigmas
    std::array<double, 4> metric_vals;
    compute_metric_sigma<Profile>(conf_mat, metric_vals.data());

    const double ns_y_sigma = n_sigmas * metric_vals[1];
    const double y_max
        = std::min(metric_vals[0] + ns_y_sigma, 1.0 - max_y_clip);
    const double y_min = std::max(metric_vals[0] - ns_y_sigma, epsilon);

    int64_t y_idx_min = 0;
    int64_t y_idx_max = n_y_bins;
    for (int64_t i = 0; i < n_y_bins; i++) {
        if (y_min < y_grid[i]) {
            y_idx_min = i - 1;
            break;
        }
    }
    result[0] = y_idx_min > 0 ? y_idx_min : 0;

    for (int64_t i = y_idx_min; i < n_x_bins; i++) {
        if (y_max < y_grid[i]) {
            y_idx_max = i + 1;
            break;
        }
    }
    result[1] = y_idx_max <= n_y_bins ? y_idx_max : n_y_bins;

    const double ns_x_sigma = n_sigmas * metric_vals[3];
    const double x_max
        = std::min(metric_vals[2] + ns_x_sigma, 1.0 - max_x_clip);
    const double x_min = std::max(metric_vals[2] - ns_x_sigma, epsilon);

    int64_t x_idx_min = 0;
    int64_t x_idx_max = n_x_bins;
    for (int64_t i = 0; i < n_x_bins; i++) {
        if (x_min < x_grid[i]) {
            x_idx_min = i - 1;
            break;
        }
    }
    result[2] = x_idx_min > 0 ? x_idx_min : 0;

    for (int64_t i = x_idx_min; i < n_x_bins; i++) {
        if (x_max < x_grid[i]) {
            x_idx_max = i + 1;
            break;
        }
    }
    result[3] = x_idx_max <= n_x_bins ? x_idx_max : n_x_bins;
}

template <typename Profile>
class GenericGridBounds {
    const int64_t n_y_bins;
    const int64_t n_x_bins;
    const double n_sigmas;
    const double epsilon;
    double max_y_clip;
    double max_x_clip;
    double y_max;
    double y_min;
    double x_max;
    double x_min;
    double ns_y_sigma;
    double ns_x_sigma;
    const double* y_grid;
    const double* x_grid;
    std::array<double, 4> metric_vals;

   public:
    int64_t y_idx_min = 0;
    int64_t y_idx_max = 0;
    int64_t x_idx_min = 0;
    int64_t x_idx_max = 0;

    GenericGridBounds(
        const int64_t n_y_bins,
        const int64_t n_x_bins,
        const double n_sigmas,
        const double epsilon,
        const double* __restrict y_grid,
        const double* __restrict x_grid)
        : n_y_bins{n_y_bins},
          n_x_bins{n_x_bins},
          n_sigmas{n_sigmas},
          epsilon{epsilon},
          y_grid{y_grid},
          x_grid{x_grid} {}

    void compute_bounds(const int64_t* __restrict conf_mat) {
        Profile::get_max_clips(conf_mat, epsilon, max_y_clip, max_x_clip);

        // Compute metric values and sigmas
        compute_metric_sigma<Profile>(conf_mat, metric_vals.data());

        ns_y_sigma = n_sigmas * metric_vals[1];
        y_max = std::min(metric_vals[0] + ns_y_sigma, 1.0 - max_y_clip);
        y_min = std::max(metric_vals[0] - ns_y_sigma, epsilon);
        y_idx_min = 0;
        y_idx_max = n_y_bins;

        int64_t i;
        for (i = 0; i < n_y_bins; i++) {
            if (y_min < y_grid[i]) {
                y_idx_min = i - 1;
                break;
            }
        }
        y_idx_min = y_idx_min > 0 ? y_idx_min : 0;

        for (i = y_idx_min; i < n_x_bins; i++) {
            if (y_max < y_grid[i]) {
                y_idx_max = i + 1;
                break;
            }
        }
        y_idx_max = y_idx_max <= n_y_bins ? y_idx_max : n_y_bins;

        ns_x_sigma = n_sigmas * metric_vals[3];
        x_max = std::min(metric_vals[2] + ns_x_sigma, 1.0 - max_x_clip);
        x_min = std::max(metric_vals[2] - ns_x_sigma, epsilon);
        x_idx_min = 0;
        x_idx_max = n_x_bins;
        for (i = 0; i < n_x_bins; i++) {
            if (x_min < x_grid[i]) {
                x_idx_min = i - 1;
                break;
            }
        }
        x_idx_min = x_idx_min > 0 ? x_idx_min : 0;

        for (i = x_idx_min; i < n_x_bins; i++) {
            if (x_max < x_grid[i]) {
                x_idx_max = i + 1;
                break;
            }
        }
        x_idx_max = x_idx_max <= n_x_bins ? x_idx_max : n_x_bins;
    }
};

}  // namespace multn
}  // namespace core
}  // namespace mmu
