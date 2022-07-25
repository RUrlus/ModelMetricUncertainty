/* numpy.cpp -- Bindings to test the numpy utility functions
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/tests/numpy.hpp>

namespace py = pybind11;

namespace mmu_tests {

void bind_is_contiguous(py::module& m) {
    m.def(
        "is_contiguous",
        [](const py::array_t<double>& arr) {
            return mmu::npy::is_contiguous(arr);
        },
        py::arg("arr").noconvert());
}

void bind_is_c_contiguous(py::module& m) {
    m.def(
        "is_c_contiguous",
        [](const py::array_t<double>& arr) {
            return mmu::npy::is_c_contiguous(arr);
        },
        py::arg("arr").noconvert());
}

void bind_is_f_contiguous(py::module& m) {
    m.def(
        "is_f_contiguous",
        [](const py::array_t<double>& arr) {
            return mmu::npy::is_f_contiguous(arr);
        },
        py::arg("arr").noconvert());
}

void bind_is_well_behaved(py::module& m) {
    m.def(
        "is_well_behaved",
        [](const py::array_t<double>& arr) {
            return mmu::npy::is_well_behaved(arr);
        },
        py::arg("arr").noconvert());
}

void bind_test_get_data(py::module& m) {
    m.def(
        "test_get_data",
        [](const py::array_t<double>& arr) {
            return arr.data() == mmu::npy::get_data(arr);
        },
        py::arg("arr").noconvert());
}

void bind_zero_array(py::module& m) {
    m.def(
        "zero_array",
        [](py::array_t<double>& arr) { mmu::npy::zero_array(arr); },
        py::arg("arr").noconvert());
    m.def(
        "zero_array",
        [](py::array_t<int64_t>& arr) { mmu::npy::zero_array(arr); },
        py::arg("arr").noconvert());
}

void bind_zero_array_fixed(py::module& m) {
    m.def(
        "zero_array_fixed",
        [](py::array_t<double>& arr) { mmu::npy::zero_array<double, 12>(arr); },
        py::arg("arr").noconvert());
    m.def(
        "zero_array_fixed",
        [](py::array_t<int64_t>& arr) {
            mmu::npy::zero_array<int64_t, 12>(arr);
        },
        py::arg("arr").noconvert());
}

void bind_allocate_confusion_matrix(py::module& m) {
    m.def("allocate_confusion_matrix", []() {
        return mmu::npy::allocate_confusion_matrix<int64_t>();
    });
    m.def("allocate_confusion_matrix", [](bool return_double) {
        UNUSED(return_double);
        return mmu::npy::allocate_confusion_matrix<double>();
    });
}

void bind_allocate_n_confusion_matrices(py::module& m) {
    m.def("allocate_n_confusion_matrices", [](int64_t n_matrices) {
        return mmu::npy::allocate_n_confusion_matrices<int64_t>(n_matrices);
    });
    m.def(
        "allocate_n_confusion_matrices",
        [](int64_t n_matrices, bool return_double) {
            UNUSED(return_double);
            return mmu::npy::allocate_n_confusion_matrices<double>(n_matrices);
        });
}

}  // namespace mmu_tests
