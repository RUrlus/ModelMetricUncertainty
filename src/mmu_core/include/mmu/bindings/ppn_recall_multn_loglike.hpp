/* ppn_recall_multn_loglike.hpp -- Python bindings of multinomial log-likelihood
 * uncertainty for Recall-PPN
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/ppn_recall_multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace ppn_recall {

void bind_multn_error(py::module& m);
void bind_multn_error_mt(py::module& m);
void bind_multn_grid_error(py::module& m);
void bind_multn_grid_curve_error(py::module& m);
void bind_multn_grid_curve_error_mt(py::module& m);

void bind_multn_chi2_score(py::module& m);
void bind_multn_chi2_scores(py::module& m);
void bind_multn_chi2_scores_mt(py::module& m);

void bind_ppn_recall(py::module& m);
void bind_ppn_recall_2d(py::module& m);

}  // namespace ppn_recall
}  // namespace bindings
}  // namespace mmu
