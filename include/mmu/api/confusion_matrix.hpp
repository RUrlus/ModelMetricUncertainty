/* confusion_matrix.hpp -- Implementation of binary classification confusion matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_CONFUSION_MATRIX_HPP_
#define INCLUDE_MMU_API_CONFUSION_MATRIX_HPP_

/* TODO *
 *
 * - Add function over runs for yhat
 * - Add function over runs for score with single threshold
 * - Add function over runs for score over multiple thresholds
 * TODO */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <limits>
#include <cinttypes>
#include <algorithm>
#include <type_traits>

#include <mmu/core/common.hpp>
#include <mmu/core/confusion_matrix.hpp>
#include <mmu/api/numpy.hpp>

namespace py = pybind11;

/*                  pred
 *                0     1
 *  actual  0    TN    FP
 *          1    FN    TP
 *
 *  Flattened, implies C-contiguous, we have:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace api {

/* Compute the confusion matrix given true labels y and estimated labels yhat.
 *
 * --- Parameters ---
 * - y : true labels
 * - yhat : estimated labels
 *
 * --- Returns ---
 * - confusion matrix
 */
template <typename T1, typename T2>
py::array_t<int64_t> confusion_matrix(
    const py::array_t<T1>& y,
    const py::array_t<T2>& yhat
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(yhat))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const size_t n_obs = std::min(y.size(), yhat.size());

    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto y_ptr = npy::get_data(y);
    auto yhat_ptr = npy::get_data(yhat);
    core::confusion_matrix<T1, T2>(n_obs, y_ptr, yhat_ptr, cm_ptr);

    return conf_mat;
}

/* Compute the confusion matrix given true labels y and classifier scores.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - threshold : inclusive classification threshold
 *
 * --- Returns ---
 * - confusion matrix
 */
template <typename T1, typename T2, isFloat<T2> = true>
py::array_t<int64_t> confusion_matrix(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const T2 threshold
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score))) {
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const size_t n_obs = std::min(y.size(), score.size());

    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data<T2>(score);
    core::confusion_matrix<T1, T2>(
        n_obs, y_ptr, score_ptr, threshold, cm_ptr
    );

    return conf_mat;
}

/* Compute the confusion matrix given true labels y and classifier scores.
 * Where both arrays contain the values for multiple runs/experiments.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - threshold : inclusive classification threshold
 * - obs_axis : {0, 1} which axis of the array contains the observations
 *
 * --- Returns ---
 * - confusion matrix
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline py::array_t<int64_t> confusion_matrix_runs(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const T2 threshold,
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

    int64_t* p_cm_ptr;

    T1* p_y_ptr;
    T2* p_score_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_y_ptr, p_score_ptr)
    for (ssize_t i = 0; i < n_runs; i++) {
        p_y_ptr = y_ptr + (i * n_obs);
        p_score_ptr = score_ptr + (i * n_obs);
        p_cm_ptr = cm_ptr + (i * 4);
        // fill confusion matrix
        core::confusion_matrix<T1, T2>(
            n_obs, p_y_ptr, p_score_ptr, threshold, p_cm_ptr
        );
    }
    return conf_mat;
}

/* Compute the confusion matrix given true labels y and estimated labels yhat.
 * Where both arrays contain the values for multiple runs/experiments.
 *
 * --- Parameters ---
 * - y : true labels
 * - yhat : estimated labels
 * - obs_axis : {0, 1} which axis of the array contains the observations
 *
 * --- Returns ---
 * - confusion matrix
 */
template <typename T1, typename T2>
inline py::array_t<int64_t> confusion_matrix_runs(
    const py::array_t<T1>& y,
    const py::array_t<T2>& yhat,
    const int obs_axis
) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(yhat))) {
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
    T2* yhat_ptr = npy::get_data(yhat);

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_runs);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    int64_t* p_cm_ptr;
    T1* p_y_ptr;
    T2* p_yhat_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_y_ptr, p_yhat_ptr)
    for (ssize_t i = 0; i < n_runs; i++) {
        p_y_ptr = y_ptr + (i * n_obs);
        p_yhat_ptr = yhat_ptr + (i * n_obs);
        p_cm_ptr = cm_ptr + (i * 4);
        // fill confusion matrix
        core::confusion_matrix<T1, T2>(
            n_obs, p_y_ptr, p_yhat_ptr, p_cm_ptr
        );
    }
    return conf_mat;
}

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_CONFUSION_MATRIX_HPP_
