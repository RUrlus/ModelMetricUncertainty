/* utils_bindings.hpp -- Bindings to test the utility functions
 * Copyright 2021 Ralph Urlus
 */
#ifndef TESTS_CPP_UTILS_BINDINGS_HPP_
#define TESTS_CPP_UTILS_BINDINGS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>

#include <mmu/utils.hpp>

namespace py = pybind11;


namespace mmu_tests {

void bind_check_1d_soft(py::module &m) {
    m.def(
        "check_1d_soft",
        [](const py::array_t<double>& x) {
            return mmu::details::check_1d_soft(x, "x");
        }
    );
}

void bind_check_equal_length(py::module &m) {
    m.def(
        "check_equal_length",
        [](
            const py::array_t<double>& x,
            const py::array_t<double>& y
        ) {
            return mmu::details::check_equal_length(x, y, "x", "y");
        }
    );
}

void bind_check_contiguous(py::module &m) {
    m.def(
        "check_contiguous",
        [](
            const py::array_t<double>& x
        ) {
            return mmu::details::check_contiguous(x, "x");
        }
    );
}

void bind_check_equal_shape(py::module &m) {
    m.def(
        "check_equal_shape",
        [](
            const py::array_t<double>& x,
            const py::array_t<double>& y
        ) {
            return mmu::details::check_equal_shape(x, y,  "x", "y");
        }
    );
}

}  // namespace mmu_tests

#endif  // TESTS_CPP_UTILS_BINDINGS_HPP_
