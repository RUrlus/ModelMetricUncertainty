/* metrics_bindings.hpp -- Python bindings for metrics.hpp
 *
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_METRICS_HPP_
#define INCLUDE_MMU_BINDINGS_METRICS_HPP_
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/metrics.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_binary_metrics(py::module& m);
void bind_binary_metrics_2d(py::module& m);
void bind_binary_metrics_flattened(py::module& m);

void bind_precision_recall(py::module& m);
void bind_precision_recall_2d(py::module& m);
void bind_precision_recall_flattened(py::module& m);

void bind_ROC(py::module& m);
void bind_ROC_2d(py::module& m);
void bind_ROC_flattened(py::module& m);

}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_METRICS_HPP_
