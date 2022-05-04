/* numpy.hpp -- Bindings to test the numpy utility functions
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_TESTS_MATH_HPP_
#define INCLUDE_MMU_TESTS_MATH_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <mmu/core/common.hpp>
#include <mmu/core/erfinv.hpp>
#include <mmu/core/mvn_error.hpp>
#include <mmu/core/multinomial.hpp>

namespace py = pybind11;

namespace mmu_tests {

template <typename T>
py::array_t<T> multinomial_rvs(
    const int64_t n_samples,
    const T N,
    const py::array_t<double>& probas,
    const uint64_t seed
) {
    size_t K = probas.size();
    const double* proba = probas.data();
    py::array_t<T> samples({static_cast<size_t>(n_samples), K});
    T* sample = samples.mutable_data();

    mmu::pcg64_dxsm rng(seed);
    for (int64_t i = 0; i < n_samples; i++) {
        mmu::core::multinomial_rvs(rng, K, N, proba, sample + (i * K));
    }
    return samples;
}

void bind_erfinv(py::module &m);
void bind_norm_ppf(py::module &m);
void bind_multinomial_rvs(py::module &m);

}  // namespace mmu_tests

#endif  // INCLUDE_MMU_TESTS_MATH_HPP_
