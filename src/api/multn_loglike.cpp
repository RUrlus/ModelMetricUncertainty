/* multn_loglike.cpp -- Implementation of Python API of multinomial log-likelihood uncertainty
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/multn_loglike.hpp> // for py::array

namespace py = pybind11;

namespace mmu {
namespace api {

py::tuple multinomial_uncertainty(
    const int64_t n_bins,
    const py::array_t<int64_t> conf_mat,
    const double n_sigmas,
    const double epsilon
) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    auto result = py::array_t<double>({n_bins, n_bins});
    auto bounds = py::array_t<double>({2, 3});
    double* res_ptr = npy::get_data(result);
    double* bnds_ptr = npy::get_data(bounds);
    int64_t* cm_ptr = npy::get_data(conf_mat);
    core::multn_uncertainty(n_bins, cm_ptr, res_ptr, bnds_ptr, n_sigmas, epsilon);
     return py::make_tuple(result, bounds);
}  // multinomial_uncertainty

py::array_t<int64_t> multinomial_uncertainty_over_grid(
    const py::array_t<double> prec_grid,
    const py::array_t<double> rec_grid,
    const py::array_t<int64_t> conf_mat,
    py::array_t<double> scores,
    const double n_sigmas,
    const double epsilon
) {
    if (
        (!npy::is_well_behaved(prec_grid))
        || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))
        || (!npy::is_well_behaved(scores))
    ) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto bounds = py::array_t<int64_t>({2, 2});
    double* scores_ptr = npy::get_data(scores);
    int64_t* bnds_ptr = npy::get_data(bounds);
    int64_t* cm_ptr = npy::get_data(conf_mat);
    double* precs = npy::get_data(prec_grid);
    double* recs = npy::get_data(rec_grid);
    core::multn_uncertainty_over_grid(
        prec_bins, rec_bins, precs, recs, cm_ptr, scores_ptr,
        bnds_ptr, n_sigmas, epsilon
    );
     return bounds;
}  // multinomial_uncertainty_over_grid

void multinomial_uncertainty_over_grid_thresholds(
    const int64_t n_conf_mats,
    const py::array_t<double> prec_grid,
    const py::array_t<double> rec_grid,
    const py::array_t<int64_t> conf_mat,
    py::array_t<double> scores,
    const double n_sigmas,
    const double epsilon
) {
    if (
        (!npy::is_well_behaved(prec_grid))
        || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))
        || (!npy::is_well_behaved(scores))
    ) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    core::multn_uncertainty_over_grid_thresholds(
        prec_grid.size(),
        rec_grid.size(),
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon
    );
}  // multinomial_uncertainty_over_grid

py::tuple simulated_multinomial_uncertainty(
    const int64_t n_sims,
    const int64_t n_bins,
    const py::array_t<int64_t> conf_mat,
    const double n_sigmas,
    const double epsilon,
    const uint64_t seed,
    const uint64_t stream
) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    auto result = py::array_t<double>({n_bins, n_bins});
    auto bounds = py::array_t<double>({2, 2});
    double* res_ptr = npy::get_data(result);
    double* bnds_ptr = npy::get_data(bounds);
    int64_t* cm_ptr = npy::get_data(conf_mat);
    core::simulate_multn_uncertainty(n_sims, n_bins, cm_ptr, res_ptr, bnds_ptr, n_sigmas, epsilon, seed, stream);
     return py::make_tuple(result, bounds);
}  // multinomial_uncertainty

}  // namespace api
}  // namespace mmu
