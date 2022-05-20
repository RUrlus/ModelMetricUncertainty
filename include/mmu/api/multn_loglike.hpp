/* multn_loglike.hpp -- Implementation of Python API of multinomial log-likelihood uncertainty
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_MULTN_LOGLIKE_HPP_
#define INCLUDE_MMU_API_MULTN_LOGLIKE_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array
#include <pybind11/stl.h>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>
#include <mmu/core/multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

py::tuple
multinomial_uncertainty(const int64_t n_bins, const i64arr conf_mat, const double n_sigmas, const double epsilon);

f64arr multinomial_uncertainty_over_grid(
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon);

f64arr multinomial_uncertainty_over_grid_thresholds(
    const int64_t n_conf_mats,
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon);

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_uncertainty_over_grid_thresholds_mt(
    const int64_t n_conf_mats,
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int64_t n_threads);
#endif  // MMU_HAS_OPENMP_SUPPORT

f64arr simulated_multinomial_uncertainty(
    const int64_t n_sims,
    const int64_t n_bins,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon,
    const uint64_t seed,
    const uint64_t stream);

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_MULTN_LOGLIKE_HPP_
