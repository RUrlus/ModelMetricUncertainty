/* utils.hpp -- Python bindings for utility functions from mmu/numpy and
 * mmu/utils Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/numpy.hpp>
namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_all_finite(py::module& m);
void bind_is_well_behaved_finite(py::module& m);

}  // namespace bindings
}  // namespace mmu
