/* pr_dirich_multn.hpp -- Implementation Bayesian Precision-Recall posterior PDF
with Dirichlet-Multinomial prior. Copyright 2022 Max Baak, Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_PR_DIRICHLET_HPP_
#define INCLUDE_MMU_CORE_PR_DIRICHLET_HPP_

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <mmu/core/common.hpp>
#include <mmu/core/grid_bounds.hpp>
#include <mmu/core/metrics.hpp>

namespace mmu {
namespace core {
namespace pr {

constexpr double DIRICH_MULT_DEFAULT_CHI2_SCORE = 10000.0;

template <typename T, isFloat<T> = 0>
class NegLogDirichMultnPdf {
    const T alpha_sum;
    const T alpha_1_m1;
    const T alpha_2_m1;

   public:
    explicit NegLogDirichMultnPdf(const T* __restrict alphas)
        : alpha_sum{alphas[1] + alphas[2] + alphas[3]},
          alpha_1_m1{alphas[1] - 1.0},
          alpha_2_m1{alphas[2] - 1.0} {}

    T operator()(const T prec, const T rec) {
        const T inv_gamma = 1. / (-1. + (1. / rec) + (1. / prec));
        const T log_fact = -2. * (std::log(rec) + std::log(prec));
        const T log_pow_prec = alpha_1_m1 * std::log((1. - prec) / prec);
        const T log_pow_rec = alpha_2_m1 * std::log((1. - rec) / rec);
        const T log_pow_inv_gamma = alpha_sum * std::log(inv_gamma);
        return -2.
               * (log_fact + log_pow_prec + log_pow_rec + log_pow_inv_gamma);
    }
};

template <typename T, isFloat<T> = 0>
inline void neg_log_dirich_multn_pdf(
    const int64_t size,
    const T* __restrict probas,
    const T* __restrict alphas,
    T* __restrict result) {
    const T alpha_sum = alphas[1] + alphas[2] + alphas[3];
    const T alpha_1_m1 = alphas[1] - 1;
    const T alpha_2_m1 = alphas[2] - 1;
    for (int64_t i = 0; i < size; i++) {
        double prec;
        double rec;
        precision_recall_probas(probas, prec, rec);
        double inv_gamma = 1. / (-1. + (1. / rec) + (1. / prec));

        double log_fact = -2. * (std::log(rec) + std::log(prec));
        double log_pow_prec = alpha_1_m1 * std::log((1. - prec) / prec);
        double log_pow_rec = alpha_2_m1 * std::log((1. - rec) / rec);
        double log_pow_inv_gamma = alpha_sum * std::log(inv_gamma);
        *result
            = -2. * (log_fact + log_pow_prec + log_pow_rec + log_pow_inv_gamma);
        result++;
        probas += 4;
    }
}  // neg_log_dirich_multn_pdf

template <typename T, isFloat<T> = 0>
inline void neg_log_dirich_multn_pdf_mt(
    const int64_t size,
    const T* __restrict probas,
    const T* __restrict alphas,
    T* __restrict result,
    const int n_threads) {
    const T alpha_sum = alphas[1] + alphas[2] + alphas[3];
    const T alpha_1_m1 = alphas[1] - 1;
    const T alpha_2_m1 = alphas[2] - 1;
#pragma omp parallel for num_threads(n_threads) default(none) \
    shared(probas, result, size, alpha_sum, alpha_1_m1, alpha_2_m1)
    for (int64_t i = 0; i < size; i++) {
        double prec;
        double rec;
        precision_recall_probas(&probas[i * 4], prec, rec);
        double inv_gamma = 1. / (-1. + (1. / rec) + (1. / prec));

        double log_fact = -2. * (std::log(rec) + std::log(prec));
        double log_pow_prec = alpha_1_m1 * std::log((1. - prec) / prec);
        double log_pow_rec = alpha_2_m1 * std::log((1. - rec) / rec);
        double log_pow_inv_gamma = alpha_sum * std::log(inv_gamma);
        result[i]
            = -2. * (log_fact + log_pow_prec + log_pow_rec + log_pow_inv_gamma);
    }
}  // neg_log_dirich_multn_pdf_mt

inline void dirich_multn_error(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4) {
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> alphas;
    auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
    // -- memory allocation --

    alphas[0] = static_cast<double>(conf_mat[0]) + 0.5;
    alphas[1] = static_cast<double>(conf_mat[1]) + 0.5;
    alphas[2] = static_cast<double>(conf_mat[2]) + 0.5;
    alphas[3] = static_cast<double>(conf_mat[3]) + 0.5;

    auto pdf = NegLogDirichMultnPdf<double>(alphas.data());

    // obtain prec_start, prec_end, rec_start, rec_end
    get_grid_bounds(conf_mat, bounds, n_sigmas, epsilon);
    details::linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
    const double prec_start = bounds[0];
    const double prec_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

    for (int64_t i = 0; i < n_bins; i++) {
        double prec = prec_start + (static_cast<double>(i) * prec_delta);
        for (int64_t j = 0; j < n_bins; j++) {
            *result = pdf(prec, rec_grid[j]);
            result++;
        }
    }
}  // dirich_multn_error_mt

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void dirich_multn_error_mt(
    const int64_t n_bins,
    const int64_t* __restrict conf_mat,
    double* __restrict result,
    double* __restrict bounds,
    const double n_sigmas = 6.0,
    const double epsilon = 1e-4,
    const int n_threads = 4) {
    // -- memory allocation --
    // memory to be used by constrained_fit_cmp
    std::array<double, 4> alphas;
    auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
    // -- memory allocation --

    alphas[0] = static_cast<double>(conf_mat[0]) + 0.5;
    alphas[1] = static_cast<double>(conf_mat[1]) + 0.5;
    alphas[2] = static_cast<double>(conf_mat[2]) + 0.5;
    alphas[3] = static_cast<double>(conf_mat[3]) + 0.5;

    auto global_pdf = NegLogDirichMultnPdf<double>(alphas.data());

    // obtain prec_start, prec_end, rec_start, rec_end
    get_grid_bounds(conf_mat, bounds, n_sigmas, epsilon);
    details::linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
    const double prec_start = bounds[0];
    const double prec_delta
        = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

#pragma omp parallel num_threads(n_threads) shared(rec_grid, global_pdf, result)
    {
        auto pdf = global_pdf;
#pragma omp for
        for (int64_t i = 0; i < n_bins; i++) {
            double prec = prec_start + (static_cast<double>(i) * prec_delta);
            int64_t idx = i * n_bins;
            for (int64_t j = 0; j < n_bins; j++) {
                result[idx + j] = pdf(prec, rec_grid[j]);
            }
        }
    }  // omp parallel
}  // dirich_multn_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

#ifdef MMU_HAS_OPENMP_SUPPORT
inline void dirich_multn_grid_curve_error_mt(
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
    const int64_t n_elem = n_prec_bins * n_rec_bins;
    const int64_t t_elem = n_elem * n_threads;
    auto thread_scores = std::unique_ptr<double[]>(new double[t_elem]);

    // give scores a high enough initial value that the chi2 p-values will be
    // close to zero
    std::fill(
        thread_scores.get(),
        thread_scores.get() + t_elem,
        DIRICH_MULT_DEFAULT_CHI2_SCORE);
#pragma omp parallel num_threads(n_threads) shared( \
    n_prec_bins,                                    \
    n_rec_bins,                                     \
    n_conf_mats,                                    \
    prec_grid,                                      \
    rec_grid,                                       \
    conf_mat,                                       \
    n_sigmas,                                       \
    epsilon)
    {
        double* thread_block
            = thread_scores.get() + (omp_get_thread_num() * n_elem);

        // -- memory allocation --
        // memory to be used by constrained_fit_cmp
        std::array<double, 4> alphas;
        auto bounds = GridBounds(
            n_prec_bins, n_rec_bins, n_sigmas, epsilon, prec_grid, rec_grid);
        // -- memory allocation --

#pragma omp for
        for (int64_t k = 0; k < n_conf_mats; k++) {
            const int64_t* lcm = conf_mat + (k * 4);
            alphas[0] = static_cast<double>(conf_mat[0]) + 0.5;
            alphas[1] = static_cast<double>(conf_mat[1]) + 0.5;
            alphas[2] = static_cast<double>(conf_mat[2]) + 0.5;
            alphas[3] = static_cast<double>(conf_mat[3]) + 0.5;

            auto pdf = NegLogDirichMultnPdf<double>(alphas.data());

            // update to new conf_mat
            bounds.compute_bounds(lcm);

            for (int64_t i = bounds.prec_idx_min; i < bounds.prec_idx_max;
                 i++) {
                double prec = prec_grid[i];
                int64_t odx = i * n_rec_bins;
                for (int64_t j = bounds.rec_idx_min; j < bounds.rec_idx_max;
                     j++) {
                    double score = pdf(prec, rec_grid[j]);
                    int64_t idx = odx + j;
                    if (score < thread_block[idx]) {
                        thread_block[idx] = score;
                    }
                }
            }
        }
    }  // omp parallel

    // collect the scores
    auto offsets = std::unique_ptr<int64_t[]>(new int64_t[n_threads]);
    for (int64_t j = 0; j < n_threads; j++) {
        offsets[j] = j * n_elem;
    }

    for (int64_t i = 0; i < n_elem; i++) {
        // give scores a high enough initial value that the chi2 p-values will
        // be close to zero i.e. ppf(1-1e-14) --> 64.47398179869367
        double min_score = DIRICH_MULT_DEFAULT_CHI2_SCORE;
        for (int64_t j = 0; j < n_threads; j++) {
            double tscore = thread_scores[i + offsets[j]];
            if (tscore < min_score) {
                min_score = tscore;
            }
        }
        scores[i] = min_score;
    }
}  // multn_grid_curve_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

// Implementation missing the Chi2 ppf

// inline void dirich_multn_error(
//     const int64_t n_samples,
//     const int64_t n_bins,
//     const int64_t* __restrict conf_mat,
//     const double* __restrict ref_samples,
//     double* __restrict result,
//     double* __restrict bounds,
//     const double n_sigmas = 6.0,
//     const double epsilon = 1e-4) {
//     // -- memory allocation --
//     // memory to be used by constrained_fit_cmp
//     std::array<double, 4> alphas;
//     auto ref_x = std::unique_ptr<double[]>(new double[n_samples]);
//     auto ref_x_ptr = ref_x.get();
//     auto ref_y = std::unique_ptr<double[]>(new double[n_samples]);
//     auto ref_y_ptr = ref_y.get();
//     auto rec_grid = std::unique_ptr<double[]>(new double[n_bins]);
//     // -- memory allocation --

//     alphas[0] = static_cast<double>(conf_mat[0]) + 0.5;
//     alphas[1] = static_cast<double>(conf_mat[1]) + 0.5;
//     alphas[2] = static_cast<double>(conf_mat[2]) + 0.5;
//     alphas[3] = static_cast<double>(conf_mat[3]) + 0.5;

//     auto nsamples = static_cast<double>(n_samples);
//     for (int64_t i = 0; i < n_samples; i++) {
//         ref_y_ptr[i] = (static_cast<double>(i) + 0.5) / nsamples;
//     }
//     // computes the ref_x scores
//     neg_log_dirich_multn_pdf(n_samples, ref_samples, alphas.data(),
//     ref_x_ptr);
//     // sort the ref_x scores
//     std::sort(ref_x_ptr, ref_x_ptr + n_samples);
//     auto interp = LinearInterp<double>(
//         static_cast<int>(n_samples), 0.0, 100.0, ref_x_ptr, ref_y_ptr);

//     auto pdf = NegLogDirichMultnPdf<double>(alphas.data());

//     // obtain prec_start, prec_end, rec_start, rec_end
//     get_grid_bounds(conf_mat, bounds, n_sigmas, epsilon);
//     details::linspace(bounds[2], bounds[3], n_bins, rec_grid.get());
//     const double prec_start = bounds[0];
//     const double prec_delta
//         = (bounds[1] - bounds[0]) / static_cast<double>(n_bins - 1);

//     for (int64_t i = 0; i < n_bins; i++) {
//         double prec = prec_start + (static_cast<double>(i) * prec_delta);
//         for (int64_t j = 0; j < n_bins; j++) {
//             *result = interp(pdf(prec, rec_grid[j]));
//             result++;
//         }
//     }
// }  // dirich_multn_error

}  // namespace pr
}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_PR_DIRICHLET_HPP_
