/* numpy.hpp -- Utility functions used to inferace with Numpy arrays
 * Copyright 2021 Ralph Urlus
 */
#ifndef CPP_INCLUDE_MMU_NUMPY_HPP_
#define CPP_INCLUDE_MMU_NUMPY_HPP_

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


/* allocate 2x2 shaped array and zero it*/
template <typename T>
inline py::array_t<T> allocate_2d_confusion_matrix() {
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<T>({2, 2}, {16, 8});
    static constexpr size_t block_size = sizeof(T) * 4;
    // zero the memory of the confusion_matrix
    memset(reinterpret_cast<T*>(conf_mat.request().ptr), 0, block_size);
    return conf_mat;
}

/* allocate n_matrices x 4 shaped array and zero it*/
template <typename T>
inline py::array_t<T> allocate_confusion_matrix(const ssize_t n_matrices) {
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<T>({n_matrices, static_cast<ssize_t>(4)});
    static constexpr size_t block_size = sizeof(T) * 4;
    // zero the memory of the confusion_matrix
    memset(reinterpret_cast<T*>(conf_mat.request().ptr), 0, n_matrices * block_size);
    return conf_mat;
}
}  // namespace details
}  // namespace mmu

#endif  // CPP_INCLUDE_MMU_NUMPY_HPP_
