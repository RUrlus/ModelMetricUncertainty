/* recall_ppn_multn_loglike.cpp -- Implementation of Python API of multinomial
 * log-likelihood uncertainty for Recall-PPN
 * Copyright 2026 Ralph Urlus
 */
#include <mmu/api/recall_ppn_multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {
namespace recall_ppn {

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
    core::recall_ppn::multn_error(
        n_bins, cm_ptr, res_ptr, bnds_ptr, n_sigmas, epsilon);
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
    core::recall_ppn::multn_error_mt(
        n_bins, cm_ptr, res_ptr, bnds_ptr, n_sigmas, epsilon, n_threads);
    return py::make_tuple(result, bounds);
}  // multn_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

double multn_chi2_score(
    const double ppn,
    const double recall,
    const i64arr& conf_mat,
    const double epsilon = 1e-4) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.size() != 4) {
        throw std::runtime_error("``conf_mat`` should have length of 4.");
    }
    return core::recall_ppn::multn_chi2_score(
        ppn, recall, npy::get_data(conf_mat), epsilon);
}

f64arr multn_chi2_scores(
    const f64arr& ppns,
    const f64arr& recalls,
    const i64arr& conf_mat,
    const double epsilon = 1e-4) {
    if ((!npy::is_well_behaved(conf_mat)) || (!npy::is_well_behaved(ppns))
        || (!npy::is_well_behaved(recalls))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.size() != 4) {
        throw std::runtime_error("``conf_mat`` should have length of 4.");
    }
    if (ppns.size() != recalls.size()) {
        throw std::runtime_error(
            "``ppns`` and ``recalls`` should have equal length.");
    }
    const int64_t n_points = ppns.size();
    auto scores = f64arr(n_points);
    core::recall_ppn::multn_chi2_scores(
        n_points,
        npy::get_data(ppns),
        npy::get_data(recalls),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        epsilon);
    return scores;
}

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_chi2_scores_mt(
    const f64arr& ppns,
    const f64arr& recalls,
    const i64arr& conf_mat,
    const double epsilon = 1e-4) {
    if ((!npy::is_well_behaved(conf_mat)) || (!npy::is_well_behaved(ppns))
        || (!npy::is_well_behaved(recalls))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.size() != 4) {
        throw std::runtime_error("``conf_mat`` should have length of 4.");
    }
    if (ppns.size() != recalls.size()) {
        throw std::runtime_error(
            "``ppns`` and ``recalls`` should have equal length.");
    }
    const int64_t n_points = ppns.size();
    auto scores = f64arr(n_points);
    core::recall_ppn::multn_chi2_scores_mt(
        n_points,
        npy::get_data(ppns),
        npy::get_data(recalls),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        epsilon);
    return scores;
}
#endif  // MMU_HAS_OPENMP_SUPPORT

f64arr multn_grid_error(
    const f64arr& ppn_grid,
    const f64arr& recall_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon) {
    if ((!npy::is_well_behaved(ppn_grid))
        || (!npy::is_well_behaved(recall_grid))
        || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    const int64_t ppn_bins = ppn_grid.size();
    const int64_t recall_bins = recall_grid.size();
    auto scores = f64arr({ppn_bins, recall_bins});
    core::recall_ppn::multn_grid_error(
        ppn_bins,
        recall_bins,
        npy::get_data(ppn_grid),
        npy::get_data(recall_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon);
    return scores;
}  // multn_grid_error

f64arr multn_grid_curve_error(
    const int64_t n_conf_mats,
    const f64arr& ppn_grid,
    const f64arr& recall_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon) {
    if ((!npy::is_well_behaved(ppn_grid))
        || (!npy::is_well_behaved(recall_grid))
        || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    const int64_t ppn_bins = ppn_grid.size();
    const int64_t recall_bins = recall_grid.size();
    auto scores = f64arr({ppn_bins, recall_bins});
    core::recall_ppn::multn_grid_curve_error(
        ppn_bins,
        recall_bins,
        n_conf_mats,
        npy::get_data(ppn_grid),
        npy::get_data(recall_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon);
    return scores;
}  // multn_grid_curve_error

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr multn_grid_curve_error_mt(
    const int64_t n_conf_mats,
    const f64arr& ppn_grid,
    const f64arr& recall_grid,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon,
    const int64_t n_threads) {
    if ((!npy::is_well_behaved(ppn_grid))
        || (!npy::is_well_behaved(recall_grid))
        || (!npy::is_well_behaved(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    const int64_t ppn_bins = ppn_grid.size();
    const int64_t recall_bins = recall_grid.size();
    auto scores = f64arr({ppn_bins, recall_bins});
    core::recall_ppn::multn_grid_curve_error_mt(
        ppn_bins,
        recall_bins,
        n_conf_mats,
        npy::get_data(ppn_grid),
        npy::get_data(recall_grid),
        npy::get_data(conf_mat),
        npy::get_data(scores),
        n_sigmas,
        epsilon,
        n_threads);
    return scores;
}  // multn_grid_curve_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

// Metric computation functions
py::tuple recall_ppn(const i64arr& conf_mat) {
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.size() != 4) {
        throw std::runtime_error("``conf_mat`` should have length of 4.");
    }
    const int64_t* cm = npy::get_data(conf_mat);
    const int64_t tn = cm[0];
    const int64_t fp = cm[1];
    const int64_t fn = cm[2];
    const int64_t tp = cm[3];
    const int64_t n = tn + fp + fn + tp;
    const int64_t p = fn + tp;

    double recall
        = p > 0 ? static_cast<double>(tp) / static_cast<double>(p) : 0.0;
    double ppn
        = n > 0 ? static_cast<double>(tn + fn) / static_cast<double>(n) : 0.0;

    return py::make_tuple(ppn, recall);
}

f64arr recall_ppn_2d(const i64arr& conf_mats) {
    if (!npy::is_well_behaved(conf_mats)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mats.ndim() != 2 || conf_mats.shape(1) != 4) {
        throw std::runtime_error("``conf_mats`` should have shape (n, 4).");
    }
    const py::size_t n_mats = conf_mats.shape(0);
    auto result
        = f64arr(std::vector<py::size_t>{n_mats, 2});  // [ppn, recall] columns
    const int64_t* cm = npy::get_data(conf_mats);
    double* res = npy::get_data(result);

    for (py::size_t i = 0; i < n_mats; ++i) {
        const int64_t tn = cm[i * 4 + 0];
        const int64_t fp = cm[i * 4 + 1];
        const int64_t fn = cm[i * 4 + 2];
        const int64_t tp = cm[i * 4 + 3];
        const int64_t n = tn + fp + fn + tp;
        const int64_t p = fn + tp;

        res[i * 2 + 0]
            = n > 0 ? static_cast<double>(tn + fn) / static_cast<double>(n)
                    : 0.0;  // ppn
        res[i * 2 + 1] = p > 0
                             ? static_cast<double>(tp) / static_cast<double>(p)
                             : 0.0;  // recall
    }

    return result;
}

}  // namespace recall_ppn
}  // namespace api
}  // namespace mmu
