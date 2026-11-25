/* delong.cpp -- Implementation of DeLong API
 * Copyright 2026 Ralph Urlus
 */
#include <mmu/api/delong.hpp>

#include <algorithm>
#include <stdexcept>

namespace py = pybind11;

namespace mmu {
namespace api {

namespace {

void check_delong_scores(const i64arr& y, const f64arr& scores, const int n_threads) {
    if (!npy::is_well_behaved(y) || !npy::is_well_behaved(scores)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (y.ndim() != 1) {
        throw std::runtime_error("`y` should be one-dimensional.");
    }
    if (scores.ndim() != 2) {
        throw std::runtime_error("`scores` should have shape (n_models, n_obs).");
    }
    if (!npy::is_c_contiguous(scores)) {
        throw std::runtime_error("`scores` should be C-contiguous with shape (n_models, n_obs).");
    }
    if (n_threads < 1) {
        throw std::invalid_argument("`n_threads` must be at least one.");
    }
    if (scores.shape(1) != y.size()) {
        throw std::runtime_error("`scores` should have shape (n_models, n_obs).");
    }
    // NaN breaks the strict weak ordering required by the rank sort.
    if (!npy::all_finite(scores)) {
        throw std::invalid_argument("`scores` contains non-finite values.");
    }
}

}  // namespace

py::tuple delong_components(const i64arr& y, const f64arr& scores, const int n_threads) {
    check_delong_scores(y, scores, n_threads);

    const int64_t n_obs = y.size();
    const int64_t n_models = scores.shape(0);
    const int64_t* y_ptr = npy::get_data(y);
    const double* scores_ptr = npy::get_data(scores);

    core::DeLongComponents comps;
    {
        py::gil_scoped_release release;
        comps = core::delong_components(y_ptr, scores_ptr, n_obs, n_models, n_threads);
    }

    auto aucs = f64arr(n_models);
    auto v10 = f64arr({n_models, comps.n_positive});
    auto v01 = f64arr({n_models, comps.n_negative});
    std::copy(comps.aucs.begin(), comps.aucs.end(), npy::get_data(aucs));
    std::copy(comps.v10.begin(), comps.v10.end(), npy::get_data(v10));
    std::copy(comps.v01.begin(), comps.v01.end(), npy::get_data(v01));

    return py::make_tuple(aucs, v10, v01);
}

py::tuple delong_covariance(const i64arr& y, const f64arr& scores, const int n_threads) {
    check_delong_scores(y, scores, n_threads);

    const int64_t n_obs = y.size();
    const int64_t n_models = scores.shape(0);

    const int64_t* y_ptr = npy::get_data(y);
    const double* scores_ptr = npy::get_data(scores);
    core::DeLongResult result;
    {
        py::gil_scoped_release release;
        result = core::delong(y_ptr, scores_ptr, n_obs, n_models, n_threads);
    }

    auto aucs = f64arr(n_models);
    auto covariance = f64arr({n_models, n_models});

    std::copy(result.aucs.begin(), result.aucs.end(), npy::get_data(aucs));
    std::copy(
        result.covariance.begin(),
        result.covariance.end(),
        npy::get_data(covariance));

    return py::make_tuple(aucs, covariance);
}

py::dict delong_test(
    const i64arr& y, const f64arr& scores1, const f64arr& scores2, const int n_threads) {
    if (!npy::is_well_behaved(y) || !npy::is_well_behaved(scores1)
        || !npy::is_well_behaved(scores2)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (y.ndim() != 1 || scores1.ndim() != 1 || scores2.ndim() != 1) {
        throw std::runtime_error("`y`, `scores1` and `scores2` should be one-dimensional.");
    }

    if (n_threads < 1) {
        throw std::invalid_argument("`n_threads` must be at least one.");
    }

    const int64_t n_obs = y.size();
    if (scores1.size() != n_obs || scores2.size() != n_obs) {
        throw std::runtime_error("`scores1`, `scores2` and `y` should have equal length.");
    }
    // NaN breaks the strict weak ordering required by the rank sort.
    if (!npy::all_finite(scores1) || !npy::all_finite(scores2)) {
        throw std::invalid_argument("`scores1` and `scores2` must not contain non-finite values.");
    }

    const int64_t* y_ptr = npy::get_data(y);
    const double* scores1_ptr = npy::get_data(scores1);
    const double* scores2_ptr = npy::get_data(scores2);
    core::DeLongTestResult result{};
    {
        py::gil_scoped_release release;
        result = core::delong_test(y_ptr, scores1_ptr, scores2_ptr, n_obs, n_threads);
    }

    py::dict output;
    output["auc1"] = result.auc1;
    output["auc2"] = result.auc2;
    output["difference"] = result.difference;
    output["variance"] = result.variance;
    output["z"] = result.z;
    output["p_value"] = result.p_value;
    return output;
}

}  // namespace api
}  // namespace mmu


