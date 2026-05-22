/* random.hpp -- Implementation of API for random number generators
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_CORE_RANDOM_HPP_
#define INCLUDE_MMU_CORE_RANDOM_HPP_

#include <mmu/core/common.hpp>
#include <mmu/core/distributions.hpp>

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif

#include <array>

namespace mmu {
namespace core {
namespace random {

inline void binomial_rvs(
    const int64_t n_samples,
    const int64_t n,
    const double p,
    int64_t* result,
    const uint64_t seed = 0,
    const uint64_t stream = 0) {
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
    const uint64_t stream = 0) {
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

inline int64_t* generate_confusion_matrices(
    const size_t n_matrices,
    const size_t N,
    const double* probas,
    const uint64_t seed = 0,
    const uint64_t stream = 0,
    int64_t* result = nullptr) {
    if (!result) {
        const size_t n_elem = n_matrices * 4;
        result = new int64_t[n_elem];
        zero_array(result, n_elem);
    }

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

    for (size_t i = 0; i < n_matrices; i++) {
        random::details::random_multinomial(rng, N, result, probas, 4, sptr);
        result += 4;
    }
    return result;
}

#if defined(MMU_HAS_OPENMP_SUPPORT)
inline int64_t* generate_confusion_matrices_mt(
    const int64_t n_matrices,
    const int64_t N,
    const double* probas,
    const uint64_t seed = 0,
    int64_t* result = nullptr) {
    if (!result) {
        const int64_t n_elem = n_matrices * 4;
        result = new int64_t[n_elem];
        zero_array(result, n_elem);
    }
    random::pcg_seed_seq seed_source;
    std::array<uint64_t, 2> gen_seeds;
    seed_source.generate(gen_seeds.begin(), gen_seeds.end());
#pragma omp parallel shared(N, probas, result, gen_seeds)
    {
        random::pcg64_dxsm rng;
        if (seed == 0) {
            rng.seed(gen_seeds[0], omp_get_thread_num());
        } else {
            rng.seed(seed, omp_get_thread_num());
        }

        auto binom_store = details::s_binomial_t();
        details::s_binomial_t* sptr = &binom_store;

#pragma omp for
        for (int64_t i = 0; i < n_matrices; i++) {
            random::details::random_multinomial(
                rng, N, result + (i * 4), probas, 4, sptr);
        }
    }  // omp parallel
    return result;
}
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace random
}  // namespace core
}  // namespace mmu

#endif  // INCLUDE_MMU_CORE_RANDOM_HPP_
