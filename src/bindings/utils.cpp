/* utils.cpp -- Python bindings for utility functions from mmu/numpy and
 * mmu/utils Copyright 2021 Ralph Urlus
 */

#include <mmu/api/utils.hpp>
#include <mmu/bindings/utils.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_all_finite(py::module& m) {
    m.def(
        "all_finite",
        [](const py::array_t<double>& arr) {
            return npy::all_finite<double>(arr);
        },
        R"pbdoc(Check if all values are finite.

        Parameters
        ----------
        arr : np.array[np.numeric]
            array to be checked

        Returns
        -------
        bool
            true if all values are finite
        )pbdoc",
        py::arg("arr").noconvert());
    m.def(
        "all_finite",
        [](const py::array_t<int64_t>& arr) {
            return npy::all_finite<int64_t>(arr);
        },
        py::arg("arr").noconvert());
    m.def(
        "all_finite",
        [](const py::array_t<int>& arr) { return npy::all_finite<int>(arr); },
        py::arg("arr").noconvert());
    m.def(
        "all_finite",
        [](const py::array_t<float>& arr) {
            return npy::all_finite<float>(arr);
        },
        py::arg("arr").noconvert());
    m.def(
        "all_finite",
        [](const py::array_t<bool>& arr) { return npy::all_finite<bool>(arr); },
        py::arg("arr").noconvert());
}

void bind_is_well_behaved_finite(py::module& m) {
    m.def(
        "is_well_behaved_finite",
        [](const py::array_t<double>& arr) {
            return npy::is_well_behaved_finite<double>(arr);
        },
        R"pbdoc(Check if all values are finite.

        Parameters
        ----------
        arr : np.array[np.numeric]
            array to be checked

        Returns
        -------
        bool
            true if all values are finite
        )pbdoc",
        py::arg("arr").noconvert());
    m.def(
        "is_well_behaved_finite",
        [](const py::array_t<int64_t>& arr) {
            return npy::is_well_behaved_finite<int64_t>(arr);
        },
        py::arg("arr").noconvert());
    m.def(
        "is_well_behaved_finite",
        [](const py::array_t<int>& arr) {
            return npy::is_well_behaved_finite<int>(arr);
        },
        py::arg("arr").noconvert());
    m.def(
        "is_well_behaved_finite",
        [](const py::array_t<float>& arr) {
            return npy::is_well_behaved_finite<float>(arr);
        },
        py::arg("arr").noconvert());
    m.def(
        "is_well_behaved_finite",
        [](const py::array_t<bool>& arr) {
            return npy::is_well_behaved_finite<bool>(arr);
        },
        py::arg("arr").noconvert());
}

}  // namespace bindings
}  // namespace mmu
