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
inline int check_1d_soft(const py::array_t<T>& arr, const std::string& name) {
    ssize_t n_dim = arr.ndim();
    if (n_dim == 1) {
        return 0;
    }
    if (n_dim == 2) {
        if (arr.shape(1) == 1) {
            return 0;
        }
        if (arr.shape(0) == 1) {
            return 1;
        }
    }
    throw std::runtime_error(name + " should be one dimensional");
}

template <typename T, typename V>
inline void check_equal_length(
    const py::array_t<T>& x,
    const py::array_t<V>& y,
    const std::string& x_name,
    const std::string& y_name,
    const int obs_axis_x = 0,
    const int obs_axis_y = 0
) {
    if (x.shape(obs_axis_x) != y.shape(obs_axis_y)) {
        throw std::runtime_error(
            x_name + " and " + y_name + " should have equal number of observations"
        );
    }
}

template <typename T>
inline void check_contiguous(const py::array_t<T>& arr, const std::string& name) {
    if (!is_contiguous<T>(arr)) {
        throw std::runtime_error(name + " should be C or F contiguous");
    }
}

template <typename T, typename V>
inline void check_equal_shape(
    const py::array_t<T>& x, const py::array_t<V>& y,
    const std::string& x_name, const std::string& y_name
) {
    size_t x_dim = x.ndim();
    size_t y_dim = y.ndim();
    bool pass = false;
    if (x_dim == y_dim) {
        for (size_t i = 0; i < x_dim; i++) {
            pass = x.shape(i) == y.shape(i);
        }
    }
    if (pass == 0) {
        throw std::runtime_error(x_name + " and " + y_name + " should have equal shape");
    }
}

}  // namespace details
}  // namespace mmu

#endif  // CPP_INCLUDE_MMU_UTILS_HPP_
