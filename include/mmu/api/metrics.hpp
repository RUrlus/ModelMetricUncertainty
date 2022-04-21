/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_METRICS_HPP_
#define INCLUDE_MMU_API_METRICS_HPP_

/* TODO *
 *
 * - Add function over runs for yhat
 * - Add support for int type in binary_metrics_confusion [requires]
 *
 * TODO */

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
#include <mmu/core/confusion_matrix.hpp>
#include <mmu/core/metrics.hpp>

#include <mmu/api/numpy.hpp>
#include <mmu/api/confusion_matrix.hpp>


namespace py = pybind11;

namespace mmu {
namespace api {

/* Compute the binary metrics given true labels y and estimated labels yhat.
 *
 * --- Parameters ---
 * - y : true labels
 * - yhat : estimated labels
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics
 */
template <typename T1, typename T2>
inline py::tuple binary_metrics(
    const py::array_t<T1>& y,
    const py::array_t<T2>& yhat,
    const double fill
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

    auto metrics = py::array_t<double>(10);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::binary_metrics(cm_ptr, metrics_ptr, fill);
    return py::make_tuple(conf_mat, metrics);
}

/* Compute the binary metrics given true labels y and classifier scores.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - threshold : inclusive classification threshold
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline py::tuple binary_metrics_score(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const T2 threshold,
    const double fill
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const size_t n_obs = std::min(y.size(), score.size());
    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);
    core::confusion_matrix<T1, T2>(
        n_obs, npy::get_data(y), npy::get_data(score), threshold, cm_ptr
    );

    auto metrics = py::array_t<double>(10);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::binary_metrics(cm_ptr, metrics_ptr, fill);
    return py::make_tuple(conf_mat, metrics);
}


/* Compute the binary metrics given the confusion matrix(ces).
 *
 * --- Parameters ---
 * - conf_mat : confusion matrix with entries as {TN, FP, FN, TP}
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics
 */
template<typename T, isInt<T> = true>
py::array_t<double> binary_metrics_confusion(
    const py::array_t<T>& conf_mat,
    const double fill
) {
    // condition checks
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // number of confusion matrices
    const ssize_t n_obs = conf_mat.size() / 4;

    // get conf_mat memory ptr
    T* const cm_ptr = npy::get_data(conf_mat);

    // allocate memory for metrics
    // metrics are all set so don't rely on initialisation
    py::array_t<double> metrics({n_obs, static_cast<ssize_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

    T* p_cm_ptr;
    double* p_metrics_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_metrics_ptr)
    for (ssize_t i = 0; i < n_obs; i++) {
        p_cm_ptr = cm_ptr + (i * 4);
        p_metrics_ptr = metrics_ptr + (i * 10);
        // compute metrics
        core::binary_metrics<T>(p_cm_ptr, p_metrics_ptr, fill);
    }
    return metrics;
}

/* Compute the binary metrics given true labels y and
 * classifier scores over a range of thresholds.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - thresholds : inclusive classification thresholds
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline py::tuple binary_metrics_thresholds(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const py::array_t<T2>& thresholds,
    const double fill
) {
    // condition checks
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
    auto metrics = py::array_t<double>({n_thresholds, static_cast<ssize_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

    int64_t* p_cm_ptr;
    double* p_metrics_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_metrics_ptr, y_ptr, score_ptr)
    for (ssize_t i = 0; i < n_thresholds; i++) {
        p_cm_ptr = cm_ptr + (i * 4);
        p_metrics_ptr = metrics_ptr + (i * 10);
        // fill confusion matrix
        core::confusion_matrix<T1, T2>(
            n_obs, y_ptr, score_ptr, threshold_ptr[i], p_cm_ptr
        );
        // compute metrics
        core::binary_metrics(p_cm_ptr, p_metrics_ptr, fill);
    }
    return py::make_tuple(conf_mat, metrics);
}

/* Compute the binary metrics given true labels y and classifier scores.
 * Where both arrays contain the values for multiple runs/experiments.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - threshold : inclusive classification threshold
 * - fill : values to set when divide by zero is encountered
 * - obs_axis : {0, 1} which axis of the array contains the observations
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline py::tuple binary_metrics_runs(
    py::array_t<T1>& y,
    py::array_t<T2>& score,
    const T2 threshold,
    const double fill,
    const int obs_axis
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    ssize_t n_obs_tmp;
    ssize_t n_runs_tmp;
    const ssize_t n_dim = y.ndim();

    if (n_dim == 2) {
        n_obs_tmp = y.shape(obs_axis);
        n_runs_tmp = y.shape(1-obs_axis);
    } else {
        n_obs_tmp = y.shape(0);
        n_runs_tmp = 1;
    }

    // copy to const help compiler optimisations below
    const size_t n_obs = n_obs_tmp;
    const ssize_t n_runs = n_runs_tmp;

    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data(score);

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_runs);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    // metrics are all set so don't rely on initialisation
    auto metrics = py::array_t<double>({n_runs, static_cast<ssize_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

    int64_t* p_cm_ptr;
    double* p_metrics_ptr;

    T1* p_y_ptr;
    T2* p_score_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_metrics_ptr, p_y_ptr, p_score_ptr)
    for (ssize_t i = 0; i < n_runs; i++) {
        p_y_ptr = y_ptr + (i * n_obs);
        p_score_ptr = score_ptr + (i * n_obs);
        p_cm_ptr = cm_ptr + (i * 4);
        p_metrics_ptr = metrics_ptr + (i * 10);
        // fill confusion matrix
        core::confusion_matrix<T1, T2>(
            n_obs, p_y_ptr, p_score_ptr, threshold, p_cm_ptr
        );
        // compute metrics
        core::binary_metrics(p_cm_ptr, p_metrics_ptr, fill);
    }
    return py::make_tuple(conf_mat, metrics);
}


/* Compute the binary metrics given true labels y and classifier scores over
 * a range of thresholds. Where both arrays contain the values for
 * multiple runs/experiments.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - thresholds : inclusive classification threshold
 * - n_obs : array containing the number of observations for each run/experiment
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline py::tuple binary_metrics_runs_thresholds(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const py::array_t<T2>& thresholds,
    const py::array_t<int64_t>& n_obs,
    const double fill
) {
    // condition checks
    if (!(
        npy::is_well_behaved(y)
        && npy::is_well_behaved(score)
        && npy::is_well_behaved(thresholds)
        && npy::is_well_behaved(n_obs)
    )) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // get ptr
    T1* y_ptr = npy::get_data<T1>(y);
    T2* score_ptr = npy::get_data<T2>(score);
    T2* thresholds_ptr = npy::get_data<T2>(thresholds);
    int64_t* n_obs_ptr = npy::get_data(n_obs);

    const ssize_t n_runs = n_obs.size();
    const ssize_t n_thresholds = thresholds.size();
    const ssize_t max_obs = *std::max_element(n_obs_ptr, n_obs_ptr + n_runs);

    // allocate confusion_matrix
    const ssize_t n_run_tresholds = n_thresholds *  n_runs;
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_run_tresholds);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    // allocate memory for metrics; are all set so don't rely on initialisation
    auto metrics = py::array_t<double>(n_run_tresholds * static_cast<ssize_t>(10));
    double* const metrics_ptr = npy::get_data(metrics);

    // Bookkeeping variables
    const ssize_t cm_offset = n_thresholds * 4;
    const ssize_t metrics_offset = n_thresholds * 10;

    T1* o_y_ptr;
    T2* o_score_ptr;
    ssize_t o_n_obs;
    int64_t* o_cm_ptr;
    double* o_metrics_ptr;
    double* i_metrics_ptr;
    int64_t* i_cm_ptr;


    #pragma omp parallel for private(o_n_obs, o_y_ptr, o_metrics_ptr, o_cm_ptr, i_cm_ptr, i_metrics_ptr)
    for (ssize_t r = 0; r < n_runs; r++) {
        o_n_obs = n_obs_ptr[r];
        o_y_ptr = y_ptr + (r * max_obs);
        o_score_ptr = score_ptr + (r * max_obs);
        o_cm_ptr = cm_ptr + (r * cm_offset);
        o_metrics_ptr = metrics_ptr + (r * metrics_offset);
        for (ssize_t i = 0; i < n_thresholds; i++) {
            i_cm_ptr = o_cm_ptr + (i * 4);
            i_metrics_ptr = o_metrics_ptr + (i * 10);
            // fill confusion matrix
            core::confusion_matrix<T1, T2>(
                o_n_obs,
                o_y_ptr,
                o_score_ptr,
                thresholds_ptr[i],
                i_cm_ptr
            );
            // compute metrics
            core::binary_metrics(i_cm_ptr, i_metrics_ptr, fill);
        }
    }
    return py::make_tuple(conf_mat, metrics);
}

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_METRICS_HPP_
