/* delong_bindings.hpp -- Python bindings for DeLong APIs
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/delong.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_delong_components(py::module& m);
void bind_delong_covariance(py::module& m);
void bind_delong_test(py::module& m);

}  // namespace bindings
}  // namespace mmu

