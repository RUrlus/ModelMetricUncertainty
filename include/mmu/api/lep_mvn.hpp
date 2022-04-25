/* error_prop.hpp -- Implementation of varianceand CI of Normal distributions
 * over the Poisson errors of the Confusion Matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_LEP_MVN_HPP_
#define INCLUDE_MMU_API_LEP_MVN_HPP_

#include <pybind11/pybind11.h> // for py::array
#include <pybind11/numpy.h>  // for py::array
#include <pybind11/stl.h>  // for py::tuple

#include <cmath>      // for sqrt
#include <limits>     // for numeric_limits
#include <cinttypes>  // for int64_t
#include <algorithm>  // for max/min
#include <stdexcept>  // for runtime_error
#include <type_traits>  // for enable_if_t

#include <mmu/core/common.hpp>
#include <mmu/core/erfinv.hpp>
#include <mmu/core/confusion_matrix.hpp>
#include <mmu/core/error_prop.hpp>

#include <mmu/api/numpy.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

/* Compute the Precision-Recall curve with Poisson uncertainty.
 *
 * --- Parameters ---
 * - y : true labels
 * - yhat : estimated labels
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics:
         - precision
         - V[precision]
         - recall
         - V[recall]
         - COV[precision, recall]
 */
template <typename T1, typename T2>
inline py::tuple pr_var(
    const py::array_t<T1>& y,
    const py::array_t<T2>& yhat
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(yhat))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const size_t n_obs = std::min(y.size(), yhat.size());
    // allocate memory confusion_matrix
    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);;

    core::confusion_matrix<T1, T2>(
        n_obs, npy::get_data(y), npy::get_data(yhat), cm_ptr
    );

    auto metrics = py::array_t<double>(5);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::lep_conf_mat(cm_ptr, metrics_ptr);
    return py::make_tuple(conf_mat, metrics);
}

template <typename T1, typename T2>
inline py::tuple pr_ci(
    const py::array_t<T1>& y,
    const py::array_t<T2>& yhat,
    const double alpha
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(yhat))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    const size_t n_obs = std::min(y.size(), yhat.size());
    // allocate memory confusion_matrix
    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);;

    core::confusion_matrix<T1, T2>(
        n_obs, npy::get_data(y), npy::get_data(yhat), cm_ptr
    );

    auto metrics = py::array_t<double>(5);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::lep_conf_mat_ci(cm_ptr, metrics_ptr, alpha);
    return py::make_tuple(conf_mat, metrics);
}

/* Compute the Precision-Recall curve with Poisson uncertainty.
 *
 * --- Parameters ---
 * - y : true labels
 * - yhat : estimated labels
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics:
         - precision
         - V[precision]
         - recall
         - V[recall]
         - COV[precision, recall]
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline py::tuple pr_curve_var(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const py::array_t<T2>& thresholds
) {
    if (!(
        npy::is_well_behaved(y)
        && npy::is_well_behaved(score)
        && npy::is_well_behaved(thresholds)
    )) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const size_t n_obs = std::min(y.size(), score.size());
    const ssize_t n_thresholds = thresholds.size();

    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data(score);
    T2* threshold_ptr = npy::get_data(thresholds);

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_thresholds);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    // metrics are all set so don't rely on initialisation
    auto metrics = py::array_t<double>({n_thresholds, static_cast<ssize_t>(5)});
    double* const metrics_ptr = npy::get_data(metrics);

    int64_t* p_cm_ptr;
    double* p_metrics_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_metrics_ptr, y_ptr, score_ptr)
    for (ssize_t i = 0; i < n_thresholds; i++) {
        p_cm_ptr = cm_ptr + (i * 4);
        p_metrics_ptr = metrics_ptr + (i * 5);
        // fill confusion matrix
        core::confusion_matrix<T1, T2>(
            n_obs, y_ptr, score_ptr, threshold_ptr[i], p_cm_ptr
        );
        // compute metrics
        core::lep_conf_mat(p_cm_ptr, p_metrics_ptr);
    }
    return py::make_tuple(conf_mat, metrics);
}

template <typename T1, typename T2>
inline py::tuple pr_curve_ci(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const py::array_t<T2>& thresholds,
    const double alpha
) {
    if (!(
        npy::is_well_behaved(y)
        && npy::is_well_behaved(score)
        && npy::is_well_behaved(thresholds)
    )) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const size_t n_obs = std::min(y.size(), score.size());
    const ssize_t n_thresholds = thresholds.size();

    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data(score);
    T2* threshold_ptr = npy::get_data(thresholds);

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_thresholds);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    // metrics are all set so don't rely on initialisation
    auto metrics = py::array_t<double>({n_thresholds, static_cast<ssize_t>(6)});
    double* const metrics_ptr = npy::get_data(metrics);

    int64_t* p_cm_ptr;
    double* p_metrics_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_metrics_ptr, y_ptr, score_ptr)
    for (ssize_t i = 0; i < n_thresholds; i++) {
        p_cm_ptr = cm_ptr + (i * 4);
        p_metrics_ptr = metrics_ptr + (i * 6);
        // fill confusion matrix
        core::confusion_matrix<T1, T2>(
            n_obs, y_ptr, score_ptr, threshold_ptr[i], p_cm_ptr
        );
        // compute metrics
        core::lep_conf_mat_ci(p_cm_ptr, p_metrics_ptr, alpha);
    }
    // compute metrics
    return py::make_tuple(conf_mat, metrics);
}

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_LEP_MVN_HPP_
