/* numpy.hpp -- Utility functions used to inferace with Numpy arrays
 * Copyright 2026 Ralph Urlus
 */
#pragma once

/* pybind11 include required even if not explicitly used
 * to prevent link with pythonXX_d.lib on Win32
 * (cf Py_DEBUG defined in numpy headers and
 * https://github.com/pybind/pybind11/issues/1295)
 */
#include <pybind11/numpy.h>  // for py::array_t
#include <pybind11/pybind11.h>
#define NPY_NO_DEPRECATED_API NPY_1_18_API_VERSION
#include <Python.h>              // for PyObject
#include <numpy/arrayobject.h>   // for PyArrayObject
#include <numpy/ndarraytypes.h>  // for PyArray_*
#include <numpy/npy_math.h>      // for isfinite

#include <algorithm>  // for min_element, max_element, sort
#include <cstring>    // for memset
#include <vector>

#include <mmu/core/common.hpp>

namespace py = pybind11;

namespace mmu {
namespace npc {

inline bool is_f_contiguous(PyObject* src) {
    return PyArray_CHKFLAGS(
        reinterpret_cast<PyArrayObject*>(src), NPY_ARRAY_F_CONTIGUOUS);
}

inline bool is_f_contiguous(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_F_CONTIGUOUS);
}

inline bool is_c_contiguous(PyObject* src) {
    return PyArray_CHKFLAGS(
        reinterpret_cast<PyArrayObject*>(src), NPY_ARRAY_C_CONTIGUOUS);
}

inline bool is_c_contiguous(PyArrayObject* src) {
    return PyArray_CHKFLAGS(src, NPY_ARRAY_C_CONTIGUOUS);
}

inline bool is_contiguous(PyObject* src) {
    auto arr = reinterpret_cast<PyArrayObject*>(src);
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS));
}

inline bool is_contiguous(PyArrayObject* arr) {
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_F_CONTIGUOUS));
}

inline bool is_aligned(PyObject* src) {
    auto arr = reinterpret_cast<PyArrayObject*>(src);
    return PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED);
}

inline bool is_aligned(PyArrayObject* arr) {
    return PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED);
}

inline bool is_well_behaved(PyObject* src) {
    auto arr = reinterpret_cast<PyArrayObject*>(src);
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS));
}

inline bool is_well_behaved(PyArrayObject* arr) {
    return (
        PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED | NPY_ARRAY_F_CONTIGUOUS));
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
inline bool is_aligned(const py::array_t<T>& arr) {
    return npc::is_aligned(arr.ptr());
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
    // Strides are derived from `sizeof(T)` rather than hard-coded. The
    // previous `{16, 8}` was correct only because every instantiation
    // happened to be an 8-byte type; it would have silently produced a
    // mis-strided array for any other `T`.
    constexpr auto row_stride = static_cast<py::ssize_t>(sizeof(T) * 2);
    constexpr auto col_stride = static_cast<py::ssize_t>(sizeof(T));
    auto conf_mat = py::array_t<T>({2, 2}, {row_stride, col_stride});
    zero_array<T, 4>(conf_mat);
    return conf_mat;
}

/* allocate n_matrices x 4 shaped array and zero it*/
template <typename T>
inline py::array_t<T> allocate_n_confusion_matrices(
    const py::size_t n_matrices) {
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<T>({n_matrices, static_cast<py::size_t>(4)});
    // zero the memory of the confusion_matrix
    zero_array<T>(conf_mat);
    return conf_mat;
}

template <typename T, isInt<T> = true>
inline constexpr bool all_finite(const py::array_t<T>& arr) {
    UNUSED(arr);
    return true;
}

template <typename T, isFloat<T> = true>
inline bool all_finite_strided(const py::array_t<T>& arr) {
    /* Check every element of a non-contiguous and/or unaligned array.
     *
     * Walks the array with an odometer over the index vector, computing the
     * byte offset from the strides on each step. This is layout agnostic: it
     * is correct for any number of dimensions and for any strides, including
     * the negative ones produced by reversed views.
     */
    auto arr_ptr = reinterpret_cast<PyArrayObject*>(arr.ptr());
    const int ndim = PyArray_NDIM(arr_ptr);
    const npy_intp* strides = PyArray_STRIDES(arr_ptr);
    const npy_intp* dims = PyArray_DIMS(arr_ptr);
    const npy_intp N = static_cast<npy_intp>(arr.size());

    if (N < 1 || ndim < 1) {
        return N < 1;
    }

    std::vector<npy_intp> counter(static_cast<size_t>(ndim), 0);
    const auto base = reinterpret_cast<const char*>(get_data(arr));
    npy_intp acc = 0;

    for (npy_intp i = 0; i < N; ++i) {
        npy_intp offset = 0;
        for (int d = 0; d < ndim; ++d) {
            offset += counter[static_cast<size_t>(d)] * strides[d];
        }
        T value;
        std::memcpy(&value, base + offset, sizeof(T));
        acc += isfinite(value);

        for (int d = ndim - 1; d >= 0; --d) {
            const auto ud = static_cast<size_t>(d);
            if (++counter[ud] < dims[d]) {
                break;
            }
            counter[ud] = 0;
        }
    }
    return acc == N;
}

template <typename T, isFloat<T> = true>
inline bool all_finite(const py::array_t<T>& arr) {
    if (!is_well_behaved(arr)) {
        // slow path but one that handles non-contiguous and unaligned data
        return all_finite_strided(arr);
    }
    const int64_t N = arr.size();
    T* data = get_data(arr);
    if (N < 100000) {
        int64_t acc1 = 0;
        int64_t acc2 = 0;
        for (int64_t i = 1; i < N; i += 2) {
            acc1 += isfinite(*data);
            data++;
            acc2 += isfinite(*data);
            data++;
        }
        if (N & 1) {
            acc1 += isfinite(*data);
        }
        return (acc1 + acc2) == N;
    }
    int64_t acc = 0;
#pragma omp parallel for reduction(+ : acc)
    for (int64_t i = 0; i < N; i++) {
        acc += isfinite(data[i]);
    }
    return acc == N;
}

template <typename T, isInt<T> = true>
inline bool is_well_behaved_finite(const py::array_t<T>& arr) {
    return is_well_behaved(arr);
}

template <typename T, isFloat<T> = true>
inline bool is_well_behaved_finite(const py::array_t<T>& arr) {
    return is_well_behaved(arr) && all_finite(arr);
}

}  // namespace npy
}  // namespace mmu
