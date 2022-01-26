/* confusion_matrix_bindings.hpp -- Python bindings for confusion_matrix.hpp metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_CORE_BINDINGS_CONFUSION_MATRIX_HPP_
#define MMU_CORE_BINDINGS_CONFUSION_MATRIX_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mmu/confusion_matrix.hpp>
namespace py = pybind11;


namespace mmu {
namespace bindings {

void bind_confusion_matrix(py::module &m);
void bind_confusion_matrix_score(py::module &m);
}  // namespace bindings
}  // namespace mmu

#endif  // MMU_CORE_BINDINGS_CONFUSION_MATRIX_HPP_
