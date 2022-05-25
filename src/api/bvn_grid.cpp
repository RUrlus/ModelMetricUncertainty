/* bvn_grid.cpp -- Implementation of Python API of bvn uncertainty over grid
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/bvn_grid.hpp>  // for py::array

namespace py = pybind11;

namespace mmu {
namespace api {

py::tuple bvn_uncertainty_over_grid(
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid)) || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    auto prec_rec_cov = f64arr(6);
    core::bvn_uncertainty_over_grid(
        prec_bins,
        rec_bins,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(prec_rec_cov),
        npy::get_data(scores),
        n_sigmas,
        epsilon);
    return py::make_tuple(prec_rec_cov, scores);
}  // bvn_uncertainty_over_grid

py::tuple bvn_uncertainty_over_grid_thresholds(
    const int64_t n_conf_mats,
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid)) || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    auto prec_rec_cov = f64arr({n_conf_mats, static_cast<int64_t>(6)});
    core::bvn_uncertainty_over_grid_thresholds(
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(prec_rec_cov),
        npy::get_data(scores),
        n_sigmas,
        epsilon);
    return py::make_tuple(prec_rec_cov, scores);
}  // bvn_uncertainty_over_grid

#ifdef MMU_HAS_OPENMP_SUPPORT
py::tuple bvn_uncertainty_over_grid_thresholds_mt(
    const int64_t n_conf_mats,
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int64_t n_threads) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid)) || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    auto prec_rec_cov = f64arr({n_conf_mats, static_cast<int64_t>(6)});
    core::bvn_uncertainty_over_grid_thresholds_mt(
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(prec_rec_cov),
        npy::get_data(scores),
        n_sigmas,
        epsilon,
        n_threads);
    return py::make_tuple(prec_rec_cov, scores);
}  // multinomial_uncertainty_over_grid_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

py::tuple bvn_uncertainty_over_grid_thresholds_wtrain(
    const int64_t n_conf_mats,
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const f64arr train_cov,
    const double n_sigmas,
    const double epsilon) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid)) || (!npy::is_well_behaved(conf_mat)) || (!npy::is_well_behaved(train_cov))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    auto prec_rec_cov = f64arr({n_conf_mats, static_cast<int64_t>(6)});
    core::bvn_uncertainty_over_grid_thresholds_wtrain(
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(train_cov),
        npy::get_data(prec_rec_cov),
        npy::get_data(scores),
        n_sigmas,
        epsilon);
    return py::make_tuple(prec_rec_cov, scores);
}  // bvn_uncertainty_over_grid

#ifdef MMU_HAS_OPENMP_SUPPORT
py::tuple bvn_uncertainty_over_grid_thresholds_wtrain_mt(
    const int64_t n_conf_mats,
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const f64arr train_cov,
    const double n_sigmas,
    const double epsilon,
    const int64_t n_threads) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid)) || (!npy::is_well_behaved(conf_mat)) || (!npy::is_well_behaved(train_cov))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    auto prec_rec_cov = f64arr({n_conf_mats, static_cast<int64_t>(6)});
    core::bvn_uncertainty_over_grid_thresholds_wtrain_mt(
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(train_cov),
        npy::get_data(prec_rec_cov),
        npy::get_data(scores),
        n_sigmas,
        epsilon,
        n_threads);
    return py::make_tuple(prec_rec_cov, scores);
}  // multinomial_uncertainty_over_grid_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace api
}  // namespace mmu
