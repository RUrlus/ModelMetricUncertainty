/* pr_multn_loglike.hpp -- Python bindings of multinomial log-likelihood
 * uncertainty Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/pr_multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace pr {

void bind_multn_error(py::module& m);
void bind_multn_error_mt(py::module& m);
void bind_multn_grid_error(py::module& m);
void bind_multn_grid_curve_error(py::module& m);
void bind_multn_grid_curve_error_mt(py::module& m);

void bind_multn_chi2_score(py::module& m);
void bind_multn_chi2_scores(py::module& m);
void bind_multn_chi2_scores_mt(py::module& m);

}  // namespace pr
}  // namespace bindings
}  // namespace mmu
