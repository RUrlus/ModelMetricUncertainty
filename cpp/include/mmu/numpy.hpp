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
        throw std::runtime_error("``obs_axis`` must be zero or one.");
    }
} // check_shape_order

/* Check if order matches shape of the array and the shape is as expected.
 * We expect the observations (rows or columns) to be contiguous in memory.
 *
 * Array can be one or two dimensional.
 * - If 1D it should have size == ``expected``
 * - If 2D it should be:
 *     * C-Contiguous if shape (n, ``expected``)
 *     * F-Contiguous if shape (``expected``, n)
 *
 * --- Exceptions ---
 *
 * - RuntimeError : throws a runtime error if the array does not have the right shape and or order
 *
 *
 * --- Parameters ---
 * - arr : the array to validate
 * - name : the name of the parameter
 * - expected : the size we expect of one the two dimensions to have
 */
template <typename T>
inline void assert_shape_order(
    const py::array_t<T>& arr, const std::string name, ssize_t expected
) {
    ssize_t n_dim = arr.ndim();
    int bad_state = 1;
    if ((n_dim == 1) && arr.size() == expected) {
        bad_state -= details::is_c_contiguous(arr);
    } else if (n_dim == 2) {
        if (arr.shape(1) == expected) {
            bad_state -= details::is_c_contiguous(arr);
        } else if (arr.shape(0) == expected) {
            bad_state -= details::is_f_contiguous(arr);
        }
    }
    if (bad_state != 0) {
        throw std::runtime_error(
            "``" + name + "`` should be C-contiguous and have shape (n, 4) or F-contiguous with shape (4, n)"
        );
    }
}  // check_shape_order

}  // namespace details
}  // namespace mmu

#endif  // CPP_INCLUDE_MMU_NUMPY_HPP_
