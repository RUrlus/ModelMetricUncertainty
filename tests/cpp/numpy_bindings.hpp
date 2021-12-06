/* numpy_bindings.hpp -- Bindings to test the numpy utility functions
 * Copyright 2021 Ralph Urlus
 */
#ifndef TESTS_CPP_NUMPY_BINDINGS_HPP_
#define TESTS_CPP_NUMPY_BINDINGS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>
#include <mmu/numpy.hpp>

namespace py = pybind11;

namespace mmu_tests {

void bind_is_contiguous(py::module &m) {
    m.def(
        "is_contiguous",
        [](const py::array_t<double>& arr) {
            return mmu::details::is_contiguous(arr);
        }
    );
}

void bind_is_c_contiguous(py::module &m) {
    m.def(
        "is_c_contiguous",
        [](const py::array_t<double>& arr) {
            return mmu::details::is_c_contiguous(arr);
        }
    );
}

void bind_is_f_contiguous(py::module &m) {
    m.def(
        "is_f_contiguous",
        [](const py::array_t<double>& arr) {
            return mmu::details::is_f_contiguous(arr);
        }
    );
}

void bind_check_shape_order(py::module &m) {
    m.def(
        "check_shape_order",
        [](py::array_t<double>& arr, const std::string name, const int obs_axis) {
            return mmu::details::check_shape_order<double>(arr, name, obs_axis);
        }
    );
}

void bind_assert_shape_order(py::module &m) {
    m.def(
        "assert_shape_order",
        [](const py::array_t<double>& arr, const std::string name, const ssize_t expected) {
            return mmu::details::assert_shape_order(arr, name, expected);
        }
    );
}

}  // namespace mmu_tests

#endif  // TESTS_CPP_NUMPY_BINDINGS_HPP_
