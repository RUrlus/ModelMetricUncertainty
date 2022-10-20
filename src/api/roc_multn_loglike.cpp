/* roc_multn_loglike.cpp -- Implementation of Python API of multinomial
 * log-likelihood uncertainty Copyright 2022 Ralph Urlus
 */
#include <mmu/api/roc_multn_loglike.hpp>  // for py::array

namespace py = pybind11;

namespace mmu {
namespace api {
namespace roc {
// This file is exactly the same as for pr
py::tuple multn_error(
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    auto result = f64arr({n_bins, n_bins});
    auto bounds = f64arr({2, 2});
    double* res_ptr = npy::get_data(result);
    double* bnds_ptr = npy::get_data(bounds);
    int64_t* cm_ptr = npy::get_data(conf_mat);
    core::roc::multn_error(n_bins, cm_ptr, res_ptr, bnds_ptr, n_sigmas, epsilon);
    return py::make_tuple(result, bounds);
}  // multn_error

#ifdef MMU_HAS_OPENMP_SUPPORT
py::tuple multn_error_mt(
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int n_threads) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    auto result = f64arr({n_bins, n_bins});
    auto bounds = f64arr({2, 2});
    double* res_ptr = npy::get_data(result);
    double* bnds_ptr = npy::get_data(bounds);
    int64_t* cm_ptr = npy::get_data(conf_mat);
    core::roc::multn_error_mt(
        n_bins, cm_ptr, res_ptr, bnds_ptr, n_sigmas, epsilon, n_threads);
    return py::make_tuple(result, bounds);
}  // multn_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

double multn_chi2_score(
    const double prec,
    const double rec,
    const i64arr& conf_mat,
    const double epsilon = 1e-4) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.size() != 4) {
        throw std::runtime_error("``conf_mat`` should have length of 4.");
    }
    return core::roc::multn_chi2_score(
        prec, rec, npy::get_data(conf_mat), epsilon);
}

f64arr multn_chi2_scores(
    const f64arr& precs,
    const f64arr& recs,
    const i64arr& conf_mat,
    const double epsilon = 1e-4) {
    if ((!npy::is_well_behaved(conf_mat)) || (!npy::is_well_behaved(precs))
        || (!npy::is_well_behaved(recs))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.size() != 4) {
        throw std::runtime_error("``conf_mat`` should have length of 4.");
    }
    if (precs.size() != recs.size()) {
        throw std::runtime_error(
            "``precs`` and ``recs`` should have equal length.");
    }
    const int64_t n_points = precs.size();
    auto scores = f64arr(n_points);
    core::roc::multn_chi2_scores(
        n_points,
        npy::get_data(precs),
        npy::get_data(recs),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        epsilon);
    return scores;
}

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_chi2_scores_mt(
    const f64arr& precs,
    const f64arr& recs,
    const i64arr& conf_mat,
    const double epsilon = 1e-4) {
    if ((!npy::is_well_behaved(conf_mat)) || (!npy::is_well_behaved(precs))
        || (!npy::is_well_behaved(recs))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.size() != 4) {
        throw std::runtime_error("``conf_mat`` should have length of 4.");
    }
    if (precs.size() != recs.size()) {
        throw std::runtime_error(
            "``precs`` and ``recs`` should have equal length.");
    }
    const int64_t n_points = precs.size();
    auto scores = f64arr(n_points);
    core::roc::multn_chi2_scores_mt(
        n_points,
        npy::get_data(precs),
        npy::get_data(recs),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        epsilon);
    return scores;
}
#endif  // MMU_HAS_OPENMP_SUPPORT

f64arr multn_grid_error(
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    core::roc::multn_grid_error(
        prec_bins,
        rec_bins,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon);
    return scores;
}  // multn_grid_error

f64arr multn_grid_curve_error(
    const int64_t n_conf_mats,
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    core::roc::multn_grid_curve_error(
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon);
    return scores;
}  // multn_grid_curve_error

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_grid_curve_error_mt(
    const int64_t n_conf_mats,
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int64_t n_threads) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    core::roc::multn_grid_curve_error_mt(
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon,
        n_threads);
    return scores;
}  // multn_grid_curve_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

py::tuple multn_sim_error(
    const int64_t n_sims,
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const uint64_t seed,
    const uint64_t stream) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    auto scores = f64arr({n_bins, n_bins});
    auto bounds = f64arr({2, 2});
    core::roc::multn_sim_error(
        n_sims,
        n_bins,
        npy::get_data(conf_mat),
        npy::get_data(scores),
        npy::get_data(bounds),
        n_sigmas,
        epsilon,
        seed,
        stream);
    return py::make_tuple(scores, bounds);
}  // multn_sim_error

#ifdef MMU_HAS_OPENMP_SUPPORT
py::tuple multn_sim_error_mt(
    const int64_t n_sims,
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const uint64_t seed,
    const int n_threads) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    auto bounds = f64arr({2, 2});
    auto scores = f64arr({n_bins, n_bins});
    core::roc::multn_sim_error_mt(
        n_sims,
        n_bins,
        npy::get_data(conf_mat),
        npy::get_data(scores),
        npy::get_data(bounds),
        n_sigmas,
        epsilon,
        seed,
        n_threads);
    return py::make_tuple(scores, bounds);
}  // multn_sim_error_mt
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
    const int64_t n_threads) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    core::roc::multn_sim_curve_error_mt(
        n_sims,
        prec_bins,
        rec_bins,
        n_conf_mats,
        npy::get_data(prec_grid),
        npy::get_data(rec_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon,
        seed,
        n_threads);
    return scores;
}  // multn_grid_curve_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace roc
}  // namespace api
}  // namespace mmu
