/* pr_dirich_multn.hpp -- Implementation Bayesian Precision-Recall posterior PDF
with Dirichlet-Multinomial prior. Copyright 2022 Max Baak, Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_PR_DIRICHLET_HPP_
#define INCLUDE_MMU_CORE_PR_DIRICHLET_HPP_

#include <array>
#include <cmath>
#include <cstdint>

#include <mmu/core/common.hpp>
#include <mmu/core/metrics.hpp>

namespace mmu {
namespace core {
namespace pr {

template <typename T, isFloat<T> = 0>
class NegLogDirichMultnPdf {
    const T alpha_sum;
    const T alpha_1_m1;
    const T alpha_2_m1;

   public:
    explicit NegLogDirichMultnPdf(const T* __restrict alphas)
        : alpha_sum{alphas[0] + alphas[1] + alphas[2] + alphas[3]},
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

}  // namespace pr
}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_PR_DIRICHLET_HPP_
