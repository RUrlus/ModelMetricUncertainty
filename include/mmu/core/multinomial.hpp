/* multinomial.hpp -- Implementation multinomial rvs.
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_MULTINOMIAL_HPP_
#define INCLUDE_MMU_CORE_MULTINOMIAL_HPP_

#include <random>     // for binomial_distribution

#include <mmu/core/common.hpp>

namespace mmu {
namespace core {

/* Generate multinomial samples using the conditional binomial method.
 * References
 * ----------
 * C.S. David, The computer generation of multinomial random variates,
 * Comp. Stat. Data Anal. 16 (1993) 205-217
 */
template <typename T, isInt<T> = true>
void multinomial_rvs(
    pcg64_dxsm& rng,
    const size_t K,
    const T N,
    const double proba[],
    T sample[]
) {
    size_t k;
    double norm = 0.0;
    double p_generated = 0.0;
    T n_generated = 0;

    using BinomDist = std::binomial_distribution<T>;
    BinomDist binom_rvs(1, 0.0);

    for (k = 0; k < K; k++) {
        norm += proba[k];
    }

    for (k = 0; k < K; k++) {
        if (proba[k] > 0.0) {
            binom_rvs.param(typename BinomDist::param_type(N - n_generated, proba[k] / (norm - p_generated)));
            sample[k] = binom_rvs(rng);
        } else {
            sample[k] = 0;
        }
        p_generated += proba[k];
        n_generated += sample[k];
    }
}  // multinomial_rvs

template <typename T, isInt<T> = true, const size_t K>
void multinomial_rvs(
    pcg64_dxsm& rng,
    const T N,
    const double proba[],
    T sample[]
) {
    size_t k;
    double norm = 0.0;
    double p_generated = 0.0;
    T n_generated = 0;

    using BinomDist = std::binomial_distribution<T>;
    BinomDist binom_rvs(1, 0.0);

    for (k = 0; k < K; k++) {
        norm += proba[k];
    }

    for (k = 0; k < K; k++) {
        if (proba[k] > 0.0) {
            binom_rvs.param(typename BinomDist::param_type(N - n_generated, proba[k] / (norm - p_generated)));
            sample[k] = binom_rvs(rng);
        } else {
            sample[k] = 0;
        }
        p_generated += proba[k];
        n_generated += sample[k];
    }
}  // multinomial_rvs

}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_MULTINOMIAL_HPP_
