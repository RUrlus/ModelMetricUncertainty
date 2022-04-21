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
#include <mmu/core/error_prop.hpp>

namespace py = pybind11;

namespace mmu_tests {

void bind_erfinv(py::module &m);
void bind_norm_ppf(py::module &m);

}  // namespace mmu_tests

#endif  // INCLUDE_MMU_TESTS_MATH_HPP_
