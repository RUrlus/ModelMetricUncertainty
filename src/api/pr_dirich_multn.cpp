/* pr_dirich_multn.cpp -- Python wrapper around Bayesian Precision-Recall
posterior PDF with Dirichlet-Multinomial prior. Copyright 2022 Max Baak, Ralph
Urlus
 */
#include <mmu/api/pr_dirich_multn.hpp>

namespace mmu {
namespace api {
namespace pr {

f64arr neg_log_dirich_multn_pdf(const f64arr& probas, const f64arr& alphas) {
    if ((!npy::is_well_behaved(probas)) || (!npy::is_well_behaved(alphas))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (probas.ndim() != 2 || probas.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4)");
    }

    const int64_t n_obs = probas.size() / 4;

    auto neg_log_likes = f64arr(n_obs);
    core::pr::neg_log_dirich_multn_pdf<double>(
        n_obs,
        npy::get_data(probas),
        npy::get_data(alphas),
        npy::get_data(neg_log_likes));
    return neg_log_likes;
}

f64arr neg_log_dirich_multn_pdf_mt(
    const f64arr& probas,
    const f64arr& alphas,
    const int n_threads) {
    if ((!npy::is_well_behaved(probas)) || (!npy::is_well_behaved(alphas))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    if (probas.ndim() != 2 || probas.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4)");
    }

    const int64_t n_obs = probas.size() / 4;

    auto neg_log_likes = f64arr(n_obs);
    core::pr::neg_log_dirich_multn_pdf_mt<double>(
        n_obs,
        npy::get_data(probas),
        npy::get_data(alphas),
        npy::get_data(neg_log_likes),
        n_threads);
    return neg_log_likes;
}

py::tuple dirich_multn_error(
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

    core::pr::dirich_multn_error(
        n_bins,
        npy::get_data(conf_mat),
        npy::get_data(result),
        npy::get_data(bounds),
        n_sigmas,
        epsilon);
    return py::make_tuple(result, bounds);
}  // dirich_multn_error

#ifdef MMU_HAS_OPENMP_SUPPORT
py::tuple dirich_multn_error_mt(
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

    core::pr::dirich_multn_error_mt(
        n_bins,
        npy::get_data(conf_mat),
        npy::get_data(result),
        npy::get_data(bounds),
        n_sigmas,
        epsilon,
        n_threads);
    return py::make_tuple(result, bounds);
}  // dirich_multn_error_mt
#endif  // MMU_HAS_OPENMP_SUPPORT

#ifdef MMU_HAS_OPENMP_SUPPORT
f64arr dirich_multn_grid_curve_error_mt(
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
    core::pr::dirich_multn_grid_curve_error_mt(
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

// py::tuple dirich_multn_error(
//     const int64_t n_bins,
//     const i64arr& conf_mat,
//     const f64arr& ref_samples,
//     const double n_sigmas,
//     const double epsilon) {
//     if (!npy::is_well_behaved(conf_mat)) {
//         throw std::runtime_error(
//             "Encountered non-aligned or non-contiguous array.");
//     }
//     auto result = f64arr({n_bins, n_bins});
//     auto bounds = f64arr({2, 2});
//     const int64_t n_samples = ref_samples.shape(0);
//
//     core::pr::dirich_multn_error(
//         n_samples,
//         n_bins,
//         npy::get_data(conf_mat),
//         npy::get_data(ref_samples),
//         npy::get_data(result),
//         npy::get_data(bounds),
//         n_sigmas,
//         epsilon);
//     return py::make_tuple(result, bounds);
// }  // dirich_multn_error

}  // namespace pr
}  // namespace api
}  // namespace mmu
