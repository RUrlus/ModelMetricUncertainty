/* bvn_grid.hpp -- Implementation of Bivariate Normal uncertainty over the grid
 * Copyright 2022 Ralph Urlus
 *
 *  /Slide 4 at
 *  https://www2.stat.duke.edu/courses/Spring12/sta104.1/Lectures/Lec22.pdf
 *  Assume that both Precision and Recall are Normally distributed:
 *
 *  P \sim N(\mu_{P}, \sigma_{P})\\
 *  R \sim N(\mu_{R}, \sigma_{R})
 *
 *  Assuming a Bivariate Normal joint density between Precision and Recall with
 *
 *  \rho = \frac{\sigma_{P,R}}{\sigma_{P}\sigma_{R}}
 *
 *  We have:
 *  R = \sigma_{R}Z_{1} + \mu_{R}\\
 *  P = \sigma_{P}\left(\rho Z_{1} + \sqrt{1-\rho^{2}}\right) + \mu_{P}
 *
 *  Which we can be rewritten s.t.
 *  Z_{1} = \frac{R-\mu_{R}}{\sigma_{R}}\\
 *  Z_{2} = \frac{\frac{P-\mu_{P}}{\sigma_{P}}-\rho Z_{1}}{\sqrt{1-\rho^{2}}}
 *
 *  Where we have:
 *  Z_{1}^{2} + Z_{2}^{2} \sim \chi^{2}(2)
 *
 */
#ifndef INCLUDE_MMU_CORE_BVN_GRID_HPP_
#define INCLUDE_MMU_CORE_BVN_GRID_HPP_

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include <mmu/core/common.hpp>
#include <mmu/core/metrics.hpp>
#include <mmu/core/bvn_error.hpp>
#include <mmu/core/pr_grid.hpp>
#include <mmu/core/random.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {

inline void bvn_uncertainty_over_grid(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* prec_rec_cov,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(scores, scores + n_prec_bins * n_rec_bins, 1e4);
    // -- memory allocation --
    std::array<int64_t, 4> bounds;
    int64_t* idx_bounds = bounds.data();
    // -- memory allocation --

    // obtain the indexes over which to loop
    // sets prec_idx_min, prec_idx_max, rec_idx_min, rec_idx_max
    details::get_pr_grid_bounds(n_prec_bins, n_rec_bins, conf_mat, prec_grid, rec_grid, idx_bounds, n_sigmas, epsilon);
    const int64_t prec_idx_min = idx_bounds[0];
    const int64_t prec_idx_max = idx_bounds[1];
    const int64_t rec_idx_min = idx_bounds[2];
    const int64_t rec_idx_max = idx_bounds[3];

    pr_bvn_cov(conf_mat, prec_rec_cov);
    const double prec_mu = prec_rec_cov[0];
    const double rec_mu = prec_rec_cov[1];
    const double prec_simga = std::sqrt(prec_rec_cov[2]);
    const double rec_simga = std::sqrt(prec_rec_cov[5]);
    const double rho = prec_rec_cov[3] / (prec_simga * rec_simga);
    const double rho_rhs = std::sqrt(1 - std::pow(rho, 2));

    auto z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto z1_sq = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto rho_z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    double z_tmp;
    for (int64_t i = rec_idx_min; i < rec_idx_max; i++) {
        // compute Z1
        z_tmp = (rec_grid[i] - rec_mu) / rec_simga;
        z1[i] = z_tmp;
        rho_z1[i] = rho * z_tmp;
        z1_sq[i] = std::pow(z_tmp, 2);
    }

    int64_t idx;
    int64_t odx;
    double z2;
    double score;
    double prec_score;
    for (int64_t i = prec_idx_min; i < prec_idx_max; i++) {
        odx = i * n_rec_bins;
        prec_score = (prec_grid[i] - prec_mu) / prec_simga;
        for (int64_t j = rec_idx_min; j < rec_idx_max; j++) {
            z2 = (prec_score - rho_z1[j]) / rho_rhs;
            score = z1_sq[j] + std::pow(z2, 2);
            idx = odx + j;
            // log likelihoods and thus always positive
            if (score < scores[idx]) {
                scores[idx] = score;
            }
        }
    }
}  // bvn_uncertainty_over_grid

inline void bvn_uncertainty_over_grid_thresholds(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* prec_rec_cov,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(scores, scores + n_prec_bins * n_rec_bins, 1e4);

    // -- memory allocation --
    auto z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto z1_sq = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto rho_z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    double z_tmp;

    auto bounds = details::PrGridBounds(n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid);

    int64_t idx;
    int64_t odx;
    double z2;
    double score;
    double prec_score;
    double prec_mu;
    double rec_mu;
    double prec_simga;
    double rec_simga;
    double rho;
    double rho_rhs;
    // -- memory allocation --

    for (int64_t k = 0; k < n_conf_mats; k++) {
        // update to new conf_mat
        bounds.compute_bounds(conf_mat);

        // compute covariance matrix and mean
        pr_bvn_cov(conf_mat, prec_rec_cov);
        prec_mu = prec_rec_cov[0];
        rec_mu = prec_rec_cov[1];
        prec_simga = std::sqrt(prec_rec_cov[2]);
        rec_simga = std::sqrt(prec_rec_cov[5]);

        // short circuit regions where uncertainty will be zero
        if (
            prec_simga < std::numeric_limits<double>::epsilon()
            || rec_simga < std::numeric_limits<double>::epsilon()
        ) {
            conf_mat += 4;
            prec_rec_cov += 6;
            continue;
        }

        rho = prec_rec_cov[3] / (prec_simga * rec_simga);
        rho_rhs = std::sqrt(1 - std::pow(rho, 2));

        // compute Z1 and variants of it
        for (int64_t i = bounds.rec_idx_min; i < bounds.rec_idx_max; i++) {
            // compute Z1
            z_tmp = (rec_grid[i] - rec_mu) / rec_simga;
            z1[i] = z_tmp;
            rho_z1[i] = rho * z_tmp;
            z1_sq[i] = std::pow(z_tmp, 2);
        }

        for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max; i++) {
            odx = i * n_rec_bins;
            prec_score = (prec_grid[i] - prec_mu) / prec_simga;
            for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max; j++) {
                z2 = (prec_score - rho_z1[j]) / rho_rhs;
                score = z1_sq[j] + std::pow(z2, 2);
                idx = odx + j;
                // log likelihoods and thus always positive
                if (score < scores[idx]) {
                    scores[idx] = score;
                }
            }
        }
        // increment ptr
        conf_mat += 4;
        prec_rec_cov += 6;
    }
}  // bvn_uncertainty_over_grid_thresholds

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void bvn_uncertainty_over_grid_thresholds_mt(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* prec_rec_cov,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int n_threads = 4) {
    const int64_t n_elem = n_prec_bins * n_rec_bins;
    const int64_t t_elem = n_elem * n_threads;
    auto thread_scores = std::unique_ptr<double[]>(new double[t_elem]);

    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(thread_scores.get(), thread_scores.get() + t_elem, 1e4);
#pragma omp parallel num_threads(n_threads) \
    shared(n_prec_bins, n_rec_bins, n_conf_mats, prec_grid, rec_grid, conf_mat, n_sigmas, epsilon, thread_scores, prec_rec_cov)
    {
        double* thread_block = thread_scores.get() + (omp_get_thread_num() * n_elem);

        // -- memory allocation --
        auto z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
        auto z1_sq = std::unique_ptr<double[]>(new double[n_rec_bins]);
        auto rho_z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
        double z_tmp;

        auto bounds = details::PrGridBounds(n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid);

        int64_t idx;
        int64_t odx;
        double z2;
        double score;
        double prec_score;
        double prec_mu;
        double rec_mu;
        double prec_simga;
        double rec_simga;
        double rho;
        double rho_rhs;
        double* prc_ptr;
        const int64_t* lcm;
        // -- memory allocation --

#pragma omp for
        for (int64_t k = 0; k < n_conf_mats; k++) {
            lcm = conf_mat + (k * 4);
            prc_ptr = prec_rec_cov + (k * 6);
            // update to new conf_mat
            bounds.compute_bounds(lcm);

            // compute covariance matrix and mean
            pr_bvn_cov(lcm, prc_ptr);
            prec_mu = prc_ptr[0];
            rec_mu = prc_ptr[1];
            prec_simga = std::sqrt(prc_ptr[2]);
            rec_simga = std::sqrt(prc_ptr[5]);
            // short circuit regions where uncertainty will be zero
            if (
                prec_simga < std::numeric_limits<double>::epsilon()
                || rec_simga < std::numeric_limits<double>::epsilon()
            ) {
                continue;
            }
            rho = prc_ptr[3] / (prec_simga * rec_simga);
            rho_rhs = std::sqrt(1 - std::pow(rho, 2));

            // compute Z1 and variants of it
            for (int64_t i = bounds.rec_idx_min; i < bounds.rec_idx_max; i++) {
                // compute Z1
                z_tmp = (rec_grid[i] - rec_mu) / rec_simga;
                z1[i] = z_tmp;
                rho_z1[i] = rho * z_tmp;
                z1_sq[i] = std::pow(z_tmp, 2);
            }

            for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max; i++) {
                odx = i * n_rec_bins;
                prec_score = (prec_grid[i] - prec_mu) / prec_simga;
                for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max; j++) {
                    z2 = (prec_score - rho_z1[j]) / rho_rhs;
                    score = z1_sq[j] + std::pow(z2, 2);
                    idx = odx + j;
                    // log likelihoods and thus always positive
                    if (score < thread_block[idx]) {
                        thread_block[idx] = score;
                    }
                }
            }
        }
    }  // omp parallel

    // compute the stride offsets for the threads
    auto offsets = std::unique_ptr<int64_t[]>(new int64_t[n_threads]);
    for (int64_t j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    // collect the scores
    double tscore;
    double min_score;
    // We loop through the grid in a flat order
    // for each point in the grid we check which thread
    // has the lowest score
    for (int64_t i = 0; i < n_elem; i++) {
        min_score = 1e4;
        for (int64_t j = 0; j < n_threads; j++) {
            tscore = thread_scores[i + offsets[j]];
            if (tscore < min_score) {
                min_score = tscore;
            }
        }
        scores[i] = min_score;
    }
}  // bvn_uncertainty_over_grid_thresholds_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

inline void bvn_uncertainty_over_grid_thresholds_wtrain(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    const double* __restrict train_cov,
    double* prec_rec_cov,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(scores, scores + n_prec_bins * n_rec_bins, 1e4);

    // -- memory allocation --
    auto z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto z1_sq = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto rho_z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    double z_tmp;

    auto bounds = details::PrGridBounds(n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid);

    int64_t idx;
    int64_t odx;
    double z2;
    double score;
    double prec_score;
    double prec_mu;
    double rec_mu;
    double prec_simga;
    double rec_simga;
    double rho;
    double rho_rhs;
    // -- memory allocation --

    for (int64_t k = 0; k < n_conf_mats; k++) {
        // update to new conf_mat
        bounds.compute_bounds(conf_mat);

        // compute covariance matrix and mean
        pr_bvn_cov(conf_mat, prec_rec_cov);
        prec_mu = prec_rec_cov[0];
        rec_mu = prec_rec_cov[1];
        prec_simga = std::sqrt(prec_rec_cov[2] + train_cov[0]);
        rec_simga = std::sqrt(prec_rec_cov[5] + train_cov[3]);

        // short circuit regions where uncertainty will be zero
        if (
            prec_simga < std::numeric_limits<double>::epsilon()
            || rec_simga < std::numeric_limits<double>::epsilon()
        ) {
            conf_mat += 4;
            train_cov += 4;
            prec_rec_cov += 6;
            continue;
        }

        rho = (prec_rec_cov[3] + train_cov[1]) / (prec_simga * rec_simga);
        rho_rhs = std::sqrt(1 - std::pow(rho, 2));

        // compute Z1 and variants of it
        for (int64_t i = bounds.rec_idx_min; i < bounds.rec_idx_max; i++) {
            // compute Z1
            z_tmp = (rec_grid[i] - rec_mu) / rec_simga;
            z1[i] = z_tmp;
            rho_z1[i] = rho * z_tmp;
            z1_sq[i] = std::pow(z_tmp, 2);
        }

        for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max; i++) {
            odx = i * n_rec_bins;
            prec_score = (prec_grid[i] - prec_mu) / prec_simga;
            for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max; j++) {
                z2 = (prec_score - rho_z1[j]) / rho_rhs;
                score = z1_sq[j] + std::pow(z2, 2);
                idx = odx + j;
                // log likelihoods and thus always positive
                if (score < scores[idx]) {
                    scores[idx] = score;
                }
            }
        }
        // increment ptr
        conf_mat += 4;
        train_cov += 4;
        prec_rec_cov += 6;
    }
}  // bvn_uncertainty_over_grid_thresholds_wtrain

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void bvn_uncertainty_over_grid_thresholds_wtrain_mt(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    const double* __restrict train_cov,
    double* prec_rec_cov,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int n_threads = 4) {
    const int64_t n_elem = n_prec_bins * n_rec_bins;
    const int64_t t_elem = n_elem * n_threads;
    auto thread_scores = std::unique_ptr<double[]>(new double[t_elem]);

    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(thread_scores.get(), thread_scores.get() + t_elem, 1e4);
#pragma omp parallel num_threads(n_threads) \
    shared(n_prec_bins, n_rec_bins, n_conf_mats, prec_grid, rec_grid, conf_mat, n_sigmas, epsilon, thread_scores, train_cov, prec_rec_cov)
    {
        double* thread_block = thread_scores.get() + (omp_get_thread_num() * n_elem);

        // -- memory allocation --
        auto z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
        auto z1_sq = std::unique_ptr<double[]>(new double[n_rec_bins]);
        auto rho_z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
        double z_tmp;

        auto bounds = details::PrGridBounds(n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid);

        int64_t idx;
        int64_t odx;
        double z2;
        double score;
        double prec_score;
        double prec_mu;
        double rec_mu;
        double prec_simga;
        double rec_simga;
        double rho;
        double rho_rhs;
        const int64_t* lcm;
        const double* tcov;
        double* prc_ptr;
        // -- memory allocation --
        int64_t k_offset;
#pragma omp for
        for (int64_t k = 0; k < n_conf_mats; k++) {
            k_offset = k * 4;
            lcm = conf_mat + k_offset;
            tcov = train_cov + k_offset;
            prc_ptr = prec_rec_cov + (k * 6);
            // update to new conf_mat
            bounds.compute_bounds(lcm);

            // compute covariance matrix and mean
            pr_bvn_cov(lcm, prc_ptr);
            prec_mu = prc_ptr[0];
            rec_mu = prc_ptr[1];
            prec_simga = std::sqrt(prc_ptr[2] + tcov[0]);
            rec_simga = std::sqrt(prc_ptr[5] + tcov[3]);
            // short circuit regions where uncertainty will be zero
            if (
                prec_simga < std::numeric_limits<double>::epsilon()
                || rec_simga < std::numeric_limits<double>::epsilon()
            ) {
                continue;
            }
            rho = (prc_ptr[3] + tcov[1]) / (prec_simga * rec_simga);
            rho_rhs = std::sqrt(1 - std::pow(rho, 2));

            // compute Z1 and variants of it
            for (int64_t i = bounds.rec_idx_min; i < bounds.rec_idx_max; i++) {
                // compute Z1
                z_tmp = (rec_grid[i] - rec_mu) / rec_simga;
                z1[i] = z_tmp;
                rho_z1[i] = rho * z_tmp;
                z1_sq[i] = std::pow(z_tmp, 2);
            }

            for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max; i++) {
                odx = i * n_rec_bins;
                prec_score = (prec_grid[i] - prec_mu) / prec_simga;
                for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max; j++) {
                    z2 = (prec_score - rho_z1[j]) / rho_rhs;
                    score = z1_sq[j] + std::pow(z2, 2);
                    idx = odx + j;
                    // log likelihoods and thus always positive
                    if (score < thread_block[idx]) {
                        thread_block[idx] = score;
                    }
                }
            }
        }
    }  // omp parallel

    // compute the stride offsets for the threads
    auto offsets = std::unique_ptr<int64_t[]>(new int64_t[n_threads]);
    for (int64_t j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    // collect the scores
    double tscore;
    double min_score;
    // We loop through the grid in a flat order
    // for each point in the grid we check which thread
    // has the lowest score
    for (int64_t i = 0; i < n_elem; i++) {
        min_score = 1e4;
        for (int64_t j = 0; j < n_threads; j++) {
            tscore = thread_scores[i + offsets[j]];
            if (tscore < min_score) {
                min_score = tscore;
            }
        }
        scores[i] = min_score;
    }
}  // bvn_uncertainty_over_grid_thresholds_wtrain_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_BVN_GRID_HPP_
