/* pr_bvn_grid.hpp -- Numpy array wrappers around core/pr_bvn_grid
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_PR_BVN_GRID_HPP_
#define INCLUDE_MMU_API_PR_BVN_GRID_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array
#include <pybind11/stl.h>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>
#include <mmu/core/pr_bvn_grid.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {
namespace pr {

py::tuple bvn_grid_error(
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon);

double bvn_chi2_score(
    const double prec,
    const double rec,
    const i64arr& conf_mat,
    const double epsilon);

f64arr bvn_chi2_scores(
    const f64arr& precs,
    const f64arr& recs,
    const i64arr& conf_mat,
    const double epsilon);

f64arr bvn_chi2_scores_mt(
    const f64arr& precs,
    const f64arr& recs,
    const i64arr& conf_mat,
    const double epsilon);

py::tuple bvn_grid_curve_error(
    const int64_t n_conf_mats,
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon);

#ifdef MMU_HAS_OPENMP_SUPPORT
py::tuple bvn_grid_curve_error_mt(
    const int64_t n_conf_mats,
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int n_threads);
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace pr
}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_PR_BVN_GRID_HPP_
