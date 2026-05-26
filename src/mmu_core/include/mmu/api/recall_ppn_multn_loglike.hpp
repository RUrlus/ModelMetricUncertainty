/* recall_ppn_multn_loglike.hpp -- Implementation of Python API of multinomial
 * log-likelihood uncertainty for Recall-PPN
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array
#include <pybind11/stl.h>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>
#include <mmu/core/common.hpp>
#include <mmu/core/multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {
namespace recall_ppn {

py::tuple multn_error(
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon);

#ifdef MMU_HAS_OPENMP_SUPPORT
py::tuple multn_error_mt(
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int n_threads);
#endif  // MMU_HAS_OPENMP_SUPPORT

double multn_chi2_score(
    const double ppn,
    const double recall,
    const i64arr& conf_mat,
    const double epsilon);

f64arr multn_chi2_scores(
    const f64arr& ppns,
    const f64arr& recalls,
    const i64arr& conf_mat,
    const double epsilon);

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_chi2_scores_mt(
    const f64arr& ppns,
    const f64arr& recalls,
    const i64arr& conf_mat,
    const double epsilon);
#endif  // MMU_HAS_OPENMP_SUPPORT

f64arr multn_grid_error(
    const f64arr& ppn_grid,
    const f64arr& recall_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon);

f64arr multn_grid_curve_error(
    const int64_t n_conf_mats,
    const f64arr& ppn_grid,
    const f64arr& recall_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon);

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_grid_curve_error_mt(
    const int64_t n_conf_mats,
    const f64arr& ppn_grid,
    const f64arr& recall_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int64_t n_threads);
#endif  // MMU_HAS_OPENMP_SUPPORT

// Metric computation functions
py::tuple recall_ppn(const i64arr& conf_mat);
f64arr recall_ppn_2d(const i64arr& conf_mats);

}  // namespace recall_ppn
}  // namespace api
}  // namespace mmu
