/* numpy_bindings.hpp -- Bindings to test the numpy utility functions
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_TESTS_NUMPY_HPP_
#define INCLUDE_MMU_TESTS_NUMPY_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/numpy.hpp>
#include <mmu/core/common.hpp>
#include <string>

namespace py = pybind11;

namespace mmu_tests {

void bind_is_contiguous(py::module& m);
void bind_is_c_contiguous(py::module& m);
void bind_is_f_contiguous(py::module& m);
void bind_is_well_behaved(py::module& m);
void bind_test_get_data(py::module& m);
void bind_zero_array(py::module& m);
void bind_zero_array_fixed(py::module& m);
void bind_allocate_confusion_matrix(py::module& m);
void bind_allocate_n_confusion_matrices(py::module& m);
}  // namespace mmu_tests

#endif  // INCLUDE_MMU_TESTS_NUMPY_HPP_
