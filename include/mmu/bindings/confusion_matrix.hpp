/* confusion_matrix_bindings.hpp -- Python bindings for confusion_matrix.hpp
 * metrics Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_CONFUSION_MATRIX_HPP_
#define INCLUDE_MMU_BINDINGS_CONFUSION_MATRIX_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/confusion_matrix.hpp>
namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_confusion_matrix(py::module& m);
void bind_confusion_matrix_score(py::module& m);
void bind_confusion_matrix_runs(py::module& m);
void bind_confusion_matrix_score_runs(py::module& m);
void bind_confusion_matrix_thresholds(py::module& m);
void bind_confusion_matrix_runs_thresholds(py::module& m);
void bind_confusion_matrix_thresholds_runs(py::module& m);

}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_CONFUSION_MATRIX_HPP_
