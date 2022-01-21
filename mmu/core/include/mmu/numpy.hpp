/* numpy.hpp -- Utility functions used to inferace with Numpy arrays
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_CORE_INCLUDE_MMU_NUMPY_HPP_
#define MMU_CORE_INCLUDE_MMU_NUMPY_HPP_

/* pybind11 include required even if not explicitly used
 * to prevent link with pythonXX_d.lib on Win32
 * (cf Py_DEBUG defined in numpy headers and https://github.com/pybind/pybind11/issues/1295)
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // for py::array_t
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <Python.h>  // for PyObject
#include <numpy/arrayobject.h>  // for PyArrayObject
#include <numpy/ndarraytypes.h> // for PyArray_*

#include <cstring>  // for memset

#include <mmu/common.hpp>

namespace py = pybind11;

namespace mmu {
namespace npc {

inline bool is_f_contiguous(PyObject* src) {
    return PyArray_CHKFLAGS(
        reinterpret_cast<PyArrayObject*>(src),
        NPY_ARRAY_F_CONTIGUOUS
    );
}

inline bool is_f_contiguous(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_F_CONTIGUOUS);
}

inline bool is_c_contiguous(PyObject* src) {
    return PyArray_CHKFLAGS(
        reinterpret_cast<PyArrayObject*>(src),
        NPY_ARRAY_C_CONTIGUOUS
    );
}

inline bool is_c_contiguous(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_C_CONTIGUOUS);
}

inline bool is_contiguous(PyObject* src) {
    auto arr = reinterpret_cast<PyArrayObject*>(src);
    auto obj_fields = reinterpret_cast<PyArrayObject_fields*>(arr);
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS)
    );
}

inline bool is_contiguous(PyArrayObject* arr) {
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS)
    );
}

inline bool is_well_behaved(PyObject* src) {
    auto arr = reinterpret_cast<PyArrayObject*>(src);
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS)
    );
}

inline bool is_well_behaved(PyArrayObject* arr) {
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED & NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED & NPY_ARRAY_F_CONTIGUOUS)
    );
}

inline void* get_data(PyObject* src) {
    return PyArray_DATA(reinterpret_cast<PyArrayObject*>(src));
}

inline void* get_data(PyArrayObject* arr) {
    return PyArray_DATA(arr);
}
}  // namespace npc

namespace npy {

template <typename T>
inline bool is_f_contiguous(const py::array_t<T>& arr) {
    return npc::is_f_contiguous(arr.ptr());
}

template <typename T>
inline bool is_c_contiguous(const py::array_t<T>& arr) {
    return npc::is_c_contiguous(arr.ptr());
}

template <typename T>
inline bool is_contiguous(const py::array_t<T>& arr) {
    return npc::is_contiguous(arr.ptr());
}

template <typename T>
inline bool is_well_behaved(const py::array_t<T>& arr) {
    return npc::is_well_behaved(arr.ptr());
}

template <typename T>
inline T* get_data(const py::array_t<T>& arr) {
    return reinterpret_cast<T*>(npc::get_data(arr.ptr()));
}

template <typename T>
inline void zero_array(py::array_t<T>& arr) {
    // zero the memory
    memset(get_data(arr), 0, arr.nbytes());
}

template <typename T, size_t n_elem>
inline void zero_array(py::array_t<T>& arr) {
    // zero the memory
    memset(get_data(arr), 0, sizeof(T) * n_elem);
}

/* allocate 2x2 shaped array and zero it*/
template <typename T>
inline py::array_t<T> allocate_confusion_matrix() {
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<T>({2, 2}, {16, 8});
    zero_array<T, 4>(conf_mat);
    return conf_mat;
}

/* allocate n_matrices x 4 shaped array and zero it*/
template <typename T>
inline py::array_t<T> allocate_n_confusion_matrices(
    const ssize_t n_matrices
) {
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<T>({n_matrices, static_cast<ssize_t>(4)});
    // zero the memory of the confusion_matrix
    zero_array<T>(conf_mat);
    return conf_mat;
}
}  // namespace npy
}  // namespace mmu

#endif  // MMU_CORE_INCLUDE_MMU_NUMPY_HPP_
