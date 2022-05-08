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
    const double epsilon,
    const uint64_t seed,
    const uint64_t stream
) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    auto result = py::array_t<double>({n_bins, n_bins});
    auto bounds = py::array_t<double>({2, 3});
    double* res_ptr = npy::get_data(result);
    double* bnds_ptr = npy::get_data(bounds);
    int64_t* cm_ptr = npy::get_data(conf_mat);
    core::multn_uncertainty(n_bins, cm_ptr, res_ptr, bnds_ptr, n_sigmas, epsilon, seed, stream);
     return py::make_tuple(result, bounds);
}  // multinomial_uncertainty

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
