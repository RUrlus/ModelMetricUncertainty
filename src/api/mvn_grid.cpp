/* mvn_grid.cpp -- Implementation of Python API of mvn uncertainty over grid
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/mvn_grid.hpp> // for py::array

namespace py = pybind11;

namespace mmu {
namespace api {

f64arr mvn_uncertainty_over_grid(
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon
) {
    if (
        (!npy::is_well_behaved(prec_grid))
        || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))
    ) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    core::mvn_uncertainty_over_grid(
        prec_bins,
        rec_bins,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon
    );
     return scores;
}  // mvn_uncertainty_over_grid

f64arr mvn_uncertainty_over_grid_thresholds(
    const int64_t n_conf_mats,
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon
) {
    if (
        (!npy::is_well_behaved(prec_grid))
        || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))
    ) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    core::mvn_uncertainty_over_grid_thresholds(
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon
    );
    return scores;
}  // mvn_uncertainty_over_grid

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr mvn_uncertainty_over_grid_thresholds_mt(
    const int64_t n_conf_mats,
    const f64arr prec_grid,
    const f64arr rec_grid,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int64_t n_threads
) {
    if (
        (!npy::is_well_behaved(prec_grid))
        || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))
    ) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    core::mvn_uncertainty_over_grid_thresholds_mt(
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon,
        n_threads
    );
    return scores;
}  // multinomial_uncertainty_over_grid_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace api
}  // namespace mmu
