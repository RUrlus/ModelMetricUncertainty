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


}  // namespace details
}  // namespace mmu

#endif  // MMU_COMMONS_NUMPY_HPP_
