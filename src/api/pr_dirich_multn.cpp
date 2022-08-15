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

}  // namespace pr
}  // namespace api
}  // namespace mmu
