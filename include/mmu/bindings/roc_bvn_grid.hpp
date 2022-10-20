/* roc_bvn_grid.hpp -- Python bindings of bvn uncertainty over grid for PR
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_ROC_BVN_GRID_HPP_
#define INCLUDE_MMU_BINDINGS_ROC_BVN_GRID_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array

#include <mmu/api/roc_bvn_grid.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace roc {

void bind_bvn_grid_error(py::module& m);
void bind_bvn_grid_curve_error(py::module& m);
void bind_bvn_grid_curve_error_mt(py::module& m);

void bind_bvn_chi2_score(py::module& m);
void bind_bvn_chi2_scores(py::module& m);
void bind_bvn_chi2_scores_mt(py::module& m);

}  // namespace roc
}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_ROC_BVN_GRID_HPP_
