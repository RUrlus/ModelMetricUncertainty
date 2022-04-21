/* metrics_bindings.hpp -- Python bindings for metrics.hpp
 *
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_METRICS_HPP_
#define INCLUDE_MMU_BINDINGS_METRICS_HPP_
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mmu/api/metrics.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_binary_metrics(py::module &m);
void bind_binary_metrics_score(py::module &m);
void bind_binary_metrics_runs(py::module &m);
void bind_binary_metrics_thresholds(py::module &m);
void bind_binary_metrics_runs_thresholds(py::module &m);
void bind_binary_metrics_confusion(py::module &m);
}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_METRICS_HPP_
