/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_COMMONS_UTILS_HPP_
#define MMU_COMMONS_UTILS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>

#include "numpy.hpp"

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

}  // namespace details
}  // namespace mmu

#endif  // MMU_COMMONS_UTILS_HPP_
