/* mvn_grid.hpp -- Implementation of Bivariate Normal uncertainty over the grid
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
#ifndef INCLUDE_MMU_CORE_MVN_GRID_HPP_
#define INCLUDE_MMU_CORE_MVN_GRID_HPP_

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <limits>
#include <memory>
#include <utility>

#include <mmu/core/common.hpp>
#include <mmu/core/mvn_error.hpp>
#include <mmu/core/random.hpp>
#include <mmu/core/metrics.hpp>
#include <mmu/core/pr_grid.hpp>

/* conf_mat layout:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace core {
namespace details {

#define MMU_PI 3.141592653589793238462643383279502884  /* pi */
#define MMU_2PI 6.283185307179586231995926937088  /* 2pi */

inline double bivariate_normal_pdf(
    const double x,
    const double y,
    const double x_mu,
    const double y_mu,
    const double x_sigma,
    const double y_sigma,
    const double x_sigma2,
    const double y_sigma2,
    const double rho
) {
    // https://www2.stat.duke.edu/courses/Spring12/sta104.1/Lectures/Lec22.pdf
    const double rho2 = std::pow(rho, 2);

    const double x_m_x_mu = x - x_mu;
    const double x_block = std::pow(x_m_x_mu, 2) / x_sigma2;
    const double cov_x_block = x_m_x_mu / x_sigma;

    const double y_m_y_mu = y - y_mu;
    const double y_block = std::pow(y_m_y_mu, 2) / y_sigma2;
    const double cov_y_block = y_m_y_mu / y_sigma;

    const double cov_block = 2. * rho * cov_x_block * cov_y_block;

    const double exp_frac = -1. / (2. * (1. - rho2));
    const double rhs = std::exp(exp_frac * (x_block + y_block - cov_block));
    const double lhs = 1.0 / (MMU_2PI * x_sigma * y_sigma * std::sqrt(1.0 - rho2));
    return lhs * rhs;
}
}  // namespace details

inline void mvn_uncertainty_over_grid(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4
) {
    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(scores, scores + n_prec_bins * n_rec_bins, 1e4);
    // -- memory allocation --
    std::array<int64_t, 4> bounds;
    int64_t* idx_bounds = bounds.data();

    std::array<double, 6> prec_rec_cov;
    double* prc_ptr = prec_rec_cov.data();

    // -- memory allocation --

    // obtain the indexes over which to loop
    // sets prec_idx_min, prec_idx_max, rec_idx_min, rec_idx_max
    details::get_pr_grid_bounds(
        n_prec_bins, n_rec_bins, conf_mat,
        prec_grid, rec_grid, idx_bounds,
        n_sigmas, epsilon
    );
    const int64_t prec_idx_min = idx_bounds[0];
    const int64_t prec_idx_max = idx_bounds[1];
    const int64_t rec_idx_min = idx_bounds[2];
    const int64_t rec_idx_max = idx_bounds[3];

    pr_mvn_cov(conf_mat, prc_ptr);
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
}  // mvn_uncertainty_over_grid

inline void mvn_uncertainty_over_grid_thresholds(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4
) {
    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(scores, scores + n_prec_bins * n_rec_bins, 1e4);

    // -- memory allocation --
    std::array<double, 6> prec_rec_cov;
    double* prc_ptr = prec_rec_cov.data();

    auto z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto z1_sq = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto rho_z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    double z_tmp;

    auto bounds = details::PrGridBounds(
        n_prec_bins,
        n_rec_bins,
        n_sigmas,
        epsilon,
        prec_grid,
        rec_grid
    );

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
        pr_mvn_cov(conf_mat, prc_ptr);
        prec_mu = prec_rec_cov[0];
        rec_mu = prec_rec_cov[1];
        prec_simga = std::sqrt(prec_rec_cov[2]);
        rec_simga = std::sqrt(prec_rec_cov[5]);
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
    }
}  // mvn_uncertainty_over_grid_thresholds

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void mvn_uncertainty_over_grid_thresholds_mt(
    const int64_t n_prec_bins,
    const int64_t n_rec_bins,
    const int64_t n_conf_mats,
    const double* prec_grid,
    const double* rec_grid,
    const int64_t* __restrict conf_mat,
    double* scores,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int n_threads = 4
) {
    const int64_t n_elem = n_prec_bins * n_rec_bins;
    const int64_t t_elem = n_elem * n_threads;
    auto thread_scores = std::unique_ptr<double[]>(new double[t_elem]);

    // give scores a high enough initial value that the chi2 p-values will be close to zero
    std::fill(thread_scores.get(), thread_scores.get() + t_elem, 1e4);
#pragma omp parallel num_threads(n_threads) shared(n_prec_bins, n_rec_bins, n_conf_mats, prec_grid, rec_grid, conf_mat, n_sigmas, epsilon, thread_scores)
    {
    double* thread_block = thread_scores.get() + (omp_get_thread_num() * n_elem);

    // -- memory allocation --
    std::array<double, 6> prec_rec_cov;
    double* prc_ptr = prec_rec_cov.data();

    auto z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto z1_sq = std::unique_ptr<double[]>(new double[n_rec_bins]);
    auto rho_z1 = std::unique_ptr<double[]>(new double[n_rec_bins]);
    double z_tmp;

    auto bounds = details::PrGridBounds(
        n_prec_bins,
        n_rec_bins,
        n_sigmas,
        epsilon,
        prec_grid,
        rec_grid
    );

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
    // -- memory allocation --

#pragma omp for
    for (int64_t k = 0; k < n_conf_mats; k++) {
        lcm = conf_mat + (k * 4);
        // update to new conf_mat
        bounds.compute_bounds(lcm);

        // compute covariance matrix and mean
        pr_mvn_cov(lcm, prc_ptr);
        prec_mu = prec_rec_cov[0];
        rec_mu = prec_rec_cov[1];
        prec_simga = std::sqrt(prec_rec_cov[2]);
        rec_simga = std::sqrt(prec_rec_cov[5]);
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
                if (score < thread_block[idx]) {
                    thread_block[idx] = score;
                }
            }
        }
    }
    } // omp parallel

    // collect the scores
    auto offsets = std::unique_ptr<int64_t[]>(new int64_t[n_threads]);
    for (int64_t j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    double tscore;
    double min_score;
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
}  // mvn_uncertainty_over_grid_thresholds_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_MVN_GRID_HPP_
