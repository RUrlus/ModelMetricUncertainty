/* random.hpp -- Implementation of API for random number generators
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_RANDOM_HPP_
#define INCLUDE_MMU_CORE_RANDOM_HPP_

#include <mmu/core/common.hpp>
#include <mmu/core/distributions.hpp>

namespace mmu {
namespace core {
namespace random {

inline void binomial_rvs(
    const int64_t n_samples,
    const int64_t n,
    const double p,
    int64_t* result,
    const uint64_t seed = 0,
    const uint64_t stream = 0
) {
    pcg64_dxsm rng;
    if (seed == 0) {
        pcg_seed_seq seed_source;
        rng.seed(seed_source);
    } else if (stream != 0) {
        rng.seed(seed, stream);
    } else {
        rng.seed(seed);
    }

    auto binom_store = details::s_binomial_t();
    details::s_binomial_t* sptr = &binom_store;

    for (int64_t i = 0; i < n_samples; i++) {
        result[i] = details::random_binomial(rng, p, n, sptr);
    }
}  // binomial_rvs

inline void multinomial_rvs(
    const int64_t n_samples,
    const int64_t n,
    const int64_t d,
    double* p,
    int64_t* result,
    const uint64_t seed = 0,
    const uint64_t stream = 0
) {
    pcg64_dxsm rng;
    if (seed == 0) {
        pcg_seed_seq seed_source;
        rng.seed(seed_source);
    } else if (stream != 0) {
        rng.seed(seed, stream);
    } else {
        rng.seed(seed);
    }

    auto binom_store = details::s_binomial_t();
    details::s_binomial_t* sptr = &binom_store;

    for (int64_t i = 0; i < n_samples; i++) {
        details::random_multinomial(rng, n, result, p, d, sptr);
        result += d;
    }
}  // binomial_rvs

}  // namespace random
}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_RANDOM_HPP_
