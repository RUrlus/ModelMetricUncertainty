/* roc_multn_loglike.hpp -- Implementation of Python API of multinomial
 * log-likelihood uncertainty Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_ROC_MULTN_LOGLIKE_HPP_
#define INCLUDE_MMU_API_ROC_MULTN_LOGLIKE_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array
#include <pybind11/stl.h>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>
#include <mmu/core/roc_multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {
namespace roc {

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
    const double prec,
    const double rec,
    const i64arr& conf_mat,
    const double epsilon);

f64arr multn_chi2_scores(
    const f64arr& precs,
    const f64arr& recs,
    const i64arr& conf_mat,
    const double epsilon);

f64arr multn_chi2_scores_mt(
    const f64arr& precs,
    const f64arr& recs,
    const i64arr& conf_mat,
    const double epsilon);

f64arr multn_grid_error(
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon);

f64arr multn_grid_curve_error(
    const int64_t n_conf_mats,
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon);

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_grid_curve_error_mt(
    const int64_t n_conf_mats,
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int64_t n_threads);
#endif  // MMU_HAS_OPENMP_SUPPORT

py::tuple multn_sim_error(
    const int64_t n_sims,
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const uint64_t seed,
    const uint64_t stream);

#ifdef MMU_HAS_OPENMP_SUPPORT
py::tuple multn_sim_error_mt(
    const int64_t n_sims,
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const uint64_t seed,
    const int n_threads);
#endif  // MMU_HAS_OPENMP_SUPPORT
        //
#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_grid_sim_curve_error_mt(
    const int64_t n_sims,
    const int64_t n_conf_mats,
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const uint64_t seed,
    const int64_t n_threads);
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace roc
}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_ROC_MULTN_LOGLIKE_HPP_
