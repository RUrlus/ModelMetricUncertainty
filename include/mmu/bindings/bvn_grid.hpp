/* bvn_grid.hpp -- Python bindings of bvn uncertainty over grid
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_BVN_GRID_HPP_
#define INCLUDE_MMU_BINDINGS_BVN_GRID_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array

#include <mmu/api/bvn_grid.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_bvn_uncertainty_over_grid(py::module& m);
void bind_bvn_uncertainty_over_grid_thresholds(py::module& m);
void bind_bvn_uncertainty_over_grid_thresholds_mt(py::module& m);
void bind_bvn_uncertainty_over_grid_thresholds_wtrain(py::module& m);
void bind_bvn_uncertainty_over_grid_thresholds_wtrain_mt(py::module& m);

}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_BVN_GRID_HPP_
