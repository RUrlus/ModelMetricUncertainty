/* pr_dirich_multn.hpp -- Python bindings for Bayesian Precision-Recall
posterior PDF with Dirichlet-Multinomial prior. Copyright 2022 Max Baak, Ralph
Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_PR_DIRICH_MULTN_HPP_
#define INCLUDE_MMU_BINDINGS_PR_DIRICH_MULTN_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array

#include <mmu/api/pr_dirich_multn.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace pr {

void bind_neg_log_dirich_multn_pdf(py::module& m);
void bind_neg_log_dirich_multn_pdf_mt(py::module& m);
void bind_dirich_multn_error(py::module& m);
void bind_dirich_multn_error_mt(py::module& m);
void bind_dirich_multn_grid_curve_error_mt(py::module& m);

}  // namespace pr
}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_PR_DIRICH_MULTN_HPP_
