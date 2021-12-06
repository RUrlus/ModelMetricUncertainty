/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef CPP_INCLUDE_MMU_UTILS_HPP_
#define CPP_INCLUDE_MMU_UTILS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>

#include <mmu/numpy.hpp>

namespace py = pybind11;


namespace mmu {
namespace details {

template <typename T>
inline void check_1d_soft(const py::array_t<T>& arr, const std::string& name) {
    if (arr.ndim() > 1 && arr.shape(1) > 1) {
        throw std::runtime_error(name + " should be one dimensional");
    }
}

template <typename T, typename V>
inline void check_equal_length(const py::array_t<T>& x, const py::array_t<V>& y) {
    if (x.shape(0) != y.shape(0)) {
        throw std::runtime_error("arrays should have equal number of rows");
    }
}

template <typename T>
inline void check_contiguous(const py::array_t<T>& arr, const std::string& name) {
    if (!is_contiguous<T>(arr)) {
        throw std::runtime_error(name + " should be C or F contiguous");
    }
}

template <typename T, typename V>
inline void check_equal_shape(const py::array_t<T>& x, const py::array_t<V>& y, const std::string& x_name, const std::string& y_name) {
    size_t x_dim = x.ndim();
    size_t y_dim = y.ndim();
    bool pass = false;
    if (x_dim == y_dim) {
        for (size_t i = 0; i < x_dim; i++) {
            pass = x.shape(i) == y.shape(i);
        }
    }
    if (pass > 0) {
        return;
    }
    throw std::runtime_error(x_name + " and " + y_name + " should have equal shape");
}

}  // namespace details
}  // namespace mmu

#endif  // CPP_INCLUDE_MMU_UTILS_HPP_
