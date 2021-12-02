/* numpy.hpp -- Utility functions used to inferace with Numpy arrays
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_COMMONS_NUMPY_HPP_
#define MMU_COMMONS_NUMPY_HPP_

#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/ndarraytypes.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <string>

namespace py = pybind11;

namespace mmu {
namespace ccore {

inline bool is_f_contiguous(const PyObject* src) {
    return PyArray_CHKFLAGS(
        reinterpret_cast<const PyArrayObject*>(src),
        NPY_ARRAY_F_CONTIGUOUS
    );
}

inline bool is_f_contiguous(const PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_F_CONTIGUOUS);
}

inline bool is_c_contiguous(const PyObject* src) {
    return PyArray_CHKFLAGS(
        reinterpret_cast<const PyArrayObject*>(src),
        NPY_ARRAY_C_CONTIGUOUS
    );
}

inline bool is_c_contiguous(const PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_C_CONTIGUOUS);
}

inline bool is_contiguous(const PyObject* src) {
    auto arr = reinterpret_cast<const PyArrayObject*>(src);
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS)
    );
}

inline bool is_contiguous(const PyArrayObject* arr) {
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS)
    );
}

}  // namespace ccore
}  // namespace mmu

namespace mmu {
namespace details {

template <typename T>
inline bool is_f_contiguous(const py::array_t<T>& arr) {
    return ccore::is_f_contiguous(arr.ptr());
}

template <typename T>
inline bool is_c_contiguous(const py::array_t<T>& arr) {
    return ccore::is_c_contiguous(arr.ptr());
}

template <typename T>
inline bool is_contiguous(const py::array_t<T>& arr) {
    return ccore::is_contiguous(arr.ptr());
}

/*
 * Check if order matches shape of the array.
 * We expect the observations (rows or columns) to be contiguous in memory.
 *
 * Parameters
 * ----------
 * arr : the array to validate
 * name : the name of the parameter
 *
 * Returns
 * -------
 * the input array or the input array with the correct memory order
 */
template <typename T>
inline py::array_t<T> check_shape_order(
    py::array_t<T>& arr, const std::string name, const int obs_axis = 0
) {
    const ssize_t n_dim = arr.ndim();
    if (n_dim > 2) {
        throw std::runtime_error(name + " must be at most two dimensional.");
    }
    if (obs_axis == 0) {
        if (!is_f_contiguous(arr)) {
            return py::array_t<T, py::array::f_style | py::array::forcecast>(arr);
        }
        return arr;
    } else if (obs_axis == 1) {
        if (!is_c_contiguous(arr)) {
            return py::array_t<T, py::array::c_style | py::array::forcecast>(arr);
        }
        return arr;
    } else {
        throw std::runtime_error("``obs_axis`` must be one or two.");
    }
} // check_shape_order

}  // namespace details
}  // namespace mmu

#endif  // MMU_COMMONS_NUMPY_HPP_
