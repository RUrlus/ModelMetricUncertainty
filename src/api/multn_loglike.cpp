/* multn_loglike.cpp -- Implementation of Python API of multinomial log-likelihood uncertainty
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/multn_loglike.hpp> // for py::array

namespace py = pybind11;

namespace mmu {
namespace api {

py::tuple multinomial_uncertainty(
    const int64_t n_bins,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon
) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    auto result = f64arr({n_bins, n_bins});
    auto bounds = f64arr({2, 3});
    double* res_ptr = npy::get_data(result);
    double* bnds_ptr = npy::get_data(bounds);
    int64_t* cm_ptr = npy::get_data(conf_mat);
    core::multn_uncertainty(n_bins, cm_ptr, res_ptr, bnds_ptr, n_sigmas, epsilon);
     return py::make_tuple(result, bounds);
}  // multinomial_uncertainty

f64arr multinomial_uncertainty_over_grid(
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
    core::multn_uncertainty_over_grid(
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
}  // multinomial_uncertainty_over_grid

f64arr multinomial_uncertainty_over_grid_thresholds(
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
    core::multn_uncertainty_over_grid_thresholds(
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
}  // multinomial_uncertainty_over_grid

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_uncertainty_over_grid_thresholds_mt(
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
    core::multn_uncertainty_over_grid_thresholds_mt(
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

f64arr simulated_multinomial_uncertainty(
    const int64_t n_sims,
    const int64_t n_bins,
    const i64arr conf_mat,
    const double n_sigmas,
    const double epsilon,
    const uint64_t seed,
    const uint64_t stream
) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    auto scores = f64arr({n_bins, n_bins});
    core::simulate_multn_uncertainty(
        n_sims,
        n_bins,
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon,
        seed,
        stream
    );
     return scores;
}  // multinomial_uncertainty

}  // namespace api
}  // namespace mmu
