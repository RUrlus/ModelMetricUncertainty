/* roc_bvn_grid.cpp -- Implementation of Python API of bvn uncertainty over grid
 * TPR FPR grid Copyright 2022 Ralph Urlus
 */
#include <mmu/api/roc_bvn_grid.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {
namespace roc {
// This file is exactly the same as for pr
py::tuple bvn_grid_error(
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
    auto prec_rec_cov = f64arr(6);
    core::roc::bvn_grid_error(
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
}  // bvn_grid_error

double bvn_chi2_score(
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
    return core::roc::bvn_chi2_score(
        prec, rec, npy::get_data(conf_mat), epsilon);
}

f64arr bvn_chi2_scores(
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
    core::roc::bvn_chi2_scores(
        n_points,
        npy::get_data(precs),
        npy::get_data(recs),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        epsilon);
    return scores;
}

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr bvn_chi2_scores_mt(
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
    core::roc::bvn_chi2_scores_mt(
        n_points,
        npy::get_data(precs),
        npy::get_data(recs),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        epsilon);
    return scores;
}
#endif  // MMU_HAS_OPENMP_SUPPORT

py::tuple bvn_grid_curve_error(
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
    auto prec_rec_cov = f64arr({n_conf_mats, static_cast<int64_t>(6)});
    core::roc::bvn_grid_curve_error(
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
py::tuple bvn_grid_curve_error_mt(
    const int64_t n_conf_mats,
    const f64arr& prec_grid,
    const f64arr& rec_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int n_threads) {
    if ((!npy::is_well_behaved(prec_grid)) || (!npy::is_well_behaved(rec_grid))
        || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    const int64_t prec_bins = prec_grid.size();
    const int64_t rec_bins = rec_grid.size();
    auto scores = f64arr({prec_bins, rec_bins});
    auto prec_rec_cov = f64arr({n_conf_mats, static_cast<int64_t>(6)});
    core::roc::bvn_grid_curve_error_mt(
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
}  // bvn_grid_curve_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

}  // namespace roc
}  // namespace api
}  // namespace mmu
