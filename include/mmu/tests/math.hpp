/* numpy.hpp -- Bindings to test the numpy utility functions
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_TESTS_MATH_HPP_
#define INCLUDE_MMU_TESTS_MATH_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/core/bvn_error.hpp>
#include <mmu/core/common.hpp>
#include <mmu/core/erfinv.hpp>
#include <mmu/core/random.hpp>
#include <string>

namespace py = pybind11;

namespace mmu_tests {

void bind_erfinv(py::module& m);
void bind_norm_ppf(py::module& m);
void bind_binomial_rvs(py::module& m);
void bind_multinomial_rvs(py::module& m);

}  // namespace mmu_tests

#endif  // INCLUDE_MMU_TESTS_MATH_HPP_
