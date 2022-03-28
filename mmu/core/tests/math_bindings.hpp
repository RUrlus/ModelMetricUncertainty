/* numpy_bindings.hpp -- Bindings to test the numpy utility functions
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_CORE_TESTS_MATH_BINDINGS_HPP_
#define MMU_CORE_TESTS_MATH_BINDINGS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <mmu/common.hpp>
#include <mmu/erfinv.hpp>
#include <mmu/error_prop.hpp>

namespace py = pybind11;

namespace mmu_tests {

void bind_erfinv(py::module &m) {
    m.def(
        "erfinv",
        [](double x) {
            return mmu::details::erfinv<double>(x);
        }
    );
}

void bind_norm_ppf(py::module &m) {
    m.def(
        "norm_ppf",
        [](double mu, double sigma, double p) {
            return mmu::details::norm_ppf<double>(mu, sigma, p);
        }
    );
}

}  // namespace mmu_tests

#endif  // MMU_CORE_TESTS_MATH_BINDINGS_HPP_
