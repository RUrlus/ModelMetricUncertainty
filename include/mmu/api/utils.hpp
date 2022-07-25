/* utils.hpp -- Utility function around type checking of py::array_t
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_UTILS_HPP_
#define INCLUDE_MMU_API_UTILS_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>

#include <mmu/api/numpy.hpp>
#include <mmu/core/common.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {
namespace details {

/* Check if arr is 1D or two 1D with the second axis containing a single index.
 * I.e. arr.shape ==
 * * (n, )
 * * (1, n)
 * * (n, 1)
 *
 * Throws RuntimeError if condition is not met.
 *
 * --- Parameters ---
 * - arr : the array to validate
 * - name : the name of the parameter
 */
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

/* Check x and y have the same length where we account for row and column
 * orientation. We only consider the obs_axis_ for each array, the obs_axis_ is
 * the 0 for an array shaped (n, m)
 *
 * Throws RuntimeError if condition is not met.
 */
template <typename T, typename V>
inline void check_equal_length(
    const py::array_t<T>& x,
    const py::array_t<V>& y,
    const std::string& x_name,
    const std::string& y_name,
    const int obs_axis_x = 0,
    const int obs_axis_y = 0) {
    if (x.shape(obs_axis_x) != y.shape(obs_axis_y)) {
        throw std::runtime_error(
            x_name + " and " + y_name
            + " should have equal number of observations");
    }
}

template <typename T>
inline void check_contiguous(
    const py::array_t<T>& arr,
    const std::string& name) {
    if (!npy::is_contiguous<T>(arr)) {
        throw std::runtime_error(name + " should be C or F contiguous");
    }
}

template <typename T, typename V>
inline void check_equal_shape(
    const py::array_t<T>& x,
    const py::array_t<V>& y,
    const std::string& x_name,
    const std::string& y_name) {
    int x_dim = x.ndim();
    int y_dim = y.ndim();
    int pass = 0;
    if (x_dim == y_dim) {
        for (int i = 0; i < x_dim; i++) {
            pass += x.shape(i) == y.shape(i);
        }
    }
    if (pass != x_dim) {
        throw std::runtime_error(
            x_name + " and " + y_name + " should have equal shape");
    }
}

/* Check if order matches shape of the array and copy otherwhise.
 * We expect the observations (rows or columns) to be contiguous in memory.
 *
 * --- Parameters ---
 * - arr : the array to validate
 * - name : the name of the parameter
 *
 * --- Returns ---
 * - arr : the input array or the input array with the correct memory order
 */
template <typename T>
inline py::array_t<T> ensure_shape_order(
    py::array_t<T>& arr,
    const std::string& name,
    const int obs_axis = 0) {
    const ssize_t n_dim = arr.ndim();
    if (n_dim > 2) {
        throw std::runtime_error(name + " must be at most two dimensional.");
    }
    if (obs_axis == 0) {
        if (!is_f_contiguous(arr)) {
            return py::array_t<T, py::array::f_style | py::array::forcecast>(
                arr);
        }
        return arr;
    } else if (obs_axis == 1) {
        if (!is_c_contiguous(arr)) {
            return py::array_t<T, py::array::c_style | py::array::forcecast>(
                arr);
        }
        return arr;
    } else {
        throw std::runtime_error("``obs_axis`` must be zero or one.");
    }
}  // ensure_shape_order

/* Check if order matches shape of the array and the shape is as expected.
 * We expect the observations (rows or columns) to be contiguous in memory.
 *
 * Array can be one or two dimensional.
 * - If 1D it should have size == ``expected``
 * - If 2D it should be:
 *     * C-Contiguous if shape (n, ``expected``)
 *     * F-Contiguous if shape (``expected``, n)
 *
 * --- Parameters ---
 * - arr : the array to validate
 * - name : the name of the parameter
 * - expected : the size we expect of one the two dimensions to have
 */
template <typename T>
inline bool is_correct_shape_order(
    const py::array_t<T>& arr,
    ssize_t expected) {
    ssize_t n_dim = arr.ndim();
    bool state = false;
    if (n_dim == 1 && arr.size() == expected) {
        state = npy::is_c_contiguous(arr);
    } else if (n_dim == 2) {
        if (arr.shape(1) == expected) {
            state = is_c_contiguous(arr);
        } else if (arr.shape(0) == expected) {
            state = is_f_contiguous(arr);
        }
    }
    return state;
}  // check_shape_order

}  // namespace details
}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_UTILS_HPP_
