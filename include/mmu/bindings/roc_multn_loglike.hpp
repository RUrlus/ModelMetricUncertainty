/* roc_multn_loglike.hpp -- Python bindings of multinomial log-likelihood
 * uncertainty Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_ROC_MULTN_LOGLIKE_HPP_
#define INCLUDE_MMU_BINDINGS_ROC_MULTN_LOGLIKE_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array

#include <mmu/api/roc_multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace roc {

void bind_multn_error(py::module& m);
void bind_multn_error_mt(py::module& m);
void bind_multn_grid_error(py::module& m);
void bind_multn_grid_curve_error(py::module& m);
void bind_multn_grid_curve_error_mt(py::module& m);
void bind_multn_sim_error(py::module& m);
void bind_multn_sim_error_mt(py::module& m);
void bind_multn_sim_curve_error_mt(py::module& m);

void bind_multn_chi2_score(py::module& m);
void bind_multn_chi2_scores(py::module& m);
void bind_multn_chi2_scores_mt(py::module& m);

}  // namespace roc
}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_ROC_MULTN_LOGLIKE_HPP_
