/* numpy.hpp -- Utility functions used to inferace with Numpy arrays
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_NUMPY_HPP_
#define INCLUDE_MMU_API_NUMPY_HPP_

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
#include <utility>    // for swap
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
        PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED & NPY_ARRAY_C_CONTIGUOUS)
        || PyArray_CHKFLAGS(arr, NPY_ARRAY_ALIGNED & NPY_ARRAY_F_CONTIGUOUS));
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
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<T>({2, 2}, {16, 8});
    zero_array<T, 4>(conf_mat);
    return conf_mat;
}

/* allocate n_matrices x 4 shaped array and zero it*/
template <typename T>
inline py::array_t<T> allocate_n_confusion_matrices(const ssize_t n_matrices) {
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<T>({n_matrices, static_cast<ssize_t>(4)});
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
    /* The array is assumed to be non-contiguous so we move in
     * strides.
     * We argsort based on the strides and iterate over the
     * array in the order of smallest stride to biggest.
     *
     * We account for the non-contiguous memory by jumping
     * ahead the next biggest strides minus the steps we have
     * already taken, i.e. say strides are (80, 8) and shape
     * is (10, 3). We add 8 to the pointer three times and
     * and than at (80 - (3 * 8)) to jump to the next block.
     * Add 8 to the pointer three times and etc...
     */
    const size_t N = arr.size();
    const size_t ndim = arr.ndim();
    auto arr_ptr = reinterpret_cast<PyArrayObject*>(arr.ptr());
    npy_intp* strides = PyArray_STRIDES(arr_ptr);
    npy_intp* dims = PyArray_DIMS(arr_ptr);
    // argsort the strides
    std::vector<size_t> idx(3);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.begin() + ndim, [&](size_t i1, size_t i2) {
        return strides[i1] < strides[i2];
    });

    int s0 = strides[idx[0]];
    size_t acc = 0;
    auto ptr = reinterpret_cast<unsigned char*>(get_data(arr));

    if (ndim == 1) {
        for (size_t i = 0; i < N; i++) {
            acc += isfinite(*reinterpret_cast<T*>(ptr));
            ptr += s0;
        }
    } else if (ndim == 2) {
        const size_t n0 = dims[idx[0]];
        const size_t n1 = dims[idx[1]];
        const size_t s1 = strides[idx[1]] - (n0 * s0);
        for (size_t i = 0; i < n1; i++) {
            for (size_t j = 0; j < n0; j++) {
                acc += isfinite(*reinterpret_cast<T*>(ptr));
                ptr += s0;
            }  // row loop
            ptr += s1;
        }  // column loop
    } else if (ndim == 3) {
        const size_t n0 = dims[idx[0]];
        const size_t n1 = dims[idx[1]];
        const size_t n2 = dims[idx[2]];
        const size_t inner_stride_offset = n0 * s0;
        const size_t s1 = strides[idx[1]] - inner_stride_offset;
        const size_t s2 = strides[idx[2]] - n1 * inner_stride_offset;
        for (size_t i = 0; i < n2; i++) {
            for (size_t j = 0; j < n1; j++) {
                for (size_t k = 0; k < n0; k++) {
                    acc += isfinite(*reinterpret_cast<T*>(ptr));
                    ptr += s0;
                }  // row loop
                ptr += s1;
            }  // column loop
            ptr += s2;
        }  // slice loop
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

#endif  // INCLUDE_MMU_API_NUMPY_HPP_
