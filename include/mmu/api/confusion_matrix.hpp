/* confusion_matrix.hpp -- Implementation of binary classification confusion
 * matrix Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_CONFUSION_MATRIX_HPP_
#define INCLUDE_MMU_API_CONFUSION_MATRIX_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>
#include <mmu/core/common.hpp>
#include <mmu/core/confusion_matrix.hpp>

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
i64arr confusion_matrix(const py::array_t<T1>& y, const py::array_t<T2>& yhat) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(yhat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const int64_t n_obs = std::min(y.size(), yhat.size());

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
i64arr confusion_matrix(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const T2 threshold) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const int64_t n_obs = std::min(y.size(), score.size());

    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data<T2>(score);
    const double scaled_tol = 1e-8 + 1e-05 * threshold;
    core::confusion_matrix<T1, T2>(
        n_obs, y_ptr, score_ptr, threshold, scaled_tol, cm_ptr);

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
inline i64arr confusion_matrix_runs(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const T2 threshold,
    const int obs_axis) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    int64_t n_obs_tmp;
    int64_t n_runs_tmp;
    const int64_t n_dim = y.ndim();

    if (n_dim == 2) {
        n_obs_tmp = y.shape(obs_axis);
        n_runs_tmp = y.shape(1 - obs_axis);
    } else {
        n_obs_tmp = y.shape(0);
        n_runs_tmp = 1;
    }

    // copy to const help compiler optimisations below
    const int64_t n_obs = n_obs_tmp;
    const int64_t n_runs = n_runs_tmp;

    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data(score);

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_runs);
    int64_t* const cm_ptr = npy::get_data(conf_mat);
    const double scaled_tol = 1e-8 + 1e-05 * threshold;

#pragma omp parallel shared( \
    n_obs, n_runs, y_ptr, score_ptr, threshold, scaled_tol, cm_ptr)
    {
#pragma omp for
        for (int64_t i = 0; i < n_runs; i++) {
            // fill confusion matrix
            core::confusion_matrix<T1, T2>(
                n_obs,
                y_ptr + (i * n_obs),
                score_ptr + (i * n_obs),
                threshold,
                scaled_tol,
                cm_ptr + (i * 4));
        }
    }  // pragma omp parallel
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
inline i64arr confusion_matrix_runs(
    const py::array_t<T1>& y,
    const py::array_t<T2>& yhat,
    const int obs_axis) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(yhat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    int64_t n_obs_tmp;
    int64_t n_runs_tmp;
    const int64_t n_dim = y.ndim();

    if (n_dim == 2) {
        n_obs_tmp = y.shape(obs_axis);
        n_runs_tmp = y.shape(1 - obs_axis);
    } else {
        n_obs_tmp = y.shape(0);
        n_runs_tmp = 1;
    }

    // copy to const help compiler optimisations below
    const int64_t n_obs = n_obs_tmp;
    const int64_t n_runs = n_runs_tmp;

    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* yhat_ptr = npy::get_data(yhat);

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_runs);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

#pragma omp parallel shared(n_obs, n_runs, y_ptr, yhat_ptr, cm_ptr)
    {
#pragma omp for
        for (int64_t i = 0; i < n_runs; i++) {
            core::confusion_matrix<T1, T2>(
                n_obs,
                y_ptr + (i * n_obs),
                yhat_ptr + (i * n_obs),
                cm_ptr + (i * 4));
        }
    }  // pragma omp parallel
    return conf_mat;
}

/* Compute the confusion matrix given true labels y and
 * classifier scores over a range of thresholds.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - thresholds : inclusive classification thresholds
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - confusion matrix
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline i64arr confusion_matrix_thresholds(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const py::array_t<T2>& thresholds) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score)
          && npy::is_well_behaved(thresholds))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    // guard against buffer overruns
    const int64_t n_obs = std::min(y.size(), score.size());
    const int64_t n_thresholds = thresholds.size();

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_thresholds);
    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data(score);
    T2* threshold_ptr = npy::get_data(thresholds);
    int64_t* const cm_ptr = npy::get_data(conf_mat);
#pragma omp parallel shared( \
    n_obs, n_thresholds, y_ptr, score_ptr, threshold_ptr, cm_ptr)
    {
#pragma omp for
        for (int64_t i = 0; i < n_thresholds; i++) {
            // fill confusion matrix
            double scaled_tol = 1e-8 + 1e-05 * threshold_ptr[i];
            core::confusion_matrix<T1, T2>(
                n_obs,
                y_ptr,
                score_ptr,
                threshold_ptr[i],
                scaled_tol,
                cm_ptr + (i * 4));
        }
    }  // pragma omp parallel
    return conf_mat;
}

/* Compute the confusion matrices given true labels y and classifier scores over
 * a range of thresholds. Where both arrays contain the values for
 * multiple runs/experiments.
 *
 * --- Parameters ---
 * - y : true labels
 * - score : classifier scores
 * - thresholds : inclusive classification threshold
 * - n_obs : array containing the number of observations for each run/experiment
 *
 * --- Returns ---
 * - tuple
 *   * confusion matrix
 *   * metrics
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline i64arr confusion_matrix_runs_thresholds(
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const py::array_t<T2>& thresholds,
    const i64arr& n_obs) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score)
          && npy::is_well_behaved(thresholds) && npy::is_well_behaved(n_obs))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    // get ptr
    T1* y_ptr = npy::get_data<T1>(y);
    T2* score_ptr = npy::get_data<T2>(score);
    T2* thresholds_ptr = npy::get_data<T2>(thresholds);
    int64_t* n_obs_ptr = npy::get_data(n_obs);

    const int64_t n_runs = n_obs.size();
    const int64_t n_thresholds = thresholds.size();
    const int64_t max_obs = *std::max_element(n_obs_ptr, n_obs_ptr + n_runs);
    const int64_t cm_offset = n_thresholds * 4;

    // allocate confusion_matrix
    auto conf_mat
        = npy::allocate_n_confusion_matrices<int64_t>(n_thresholds * n_runs);
    // we must zero the array as we are not garuenteed to set all values
    // some runs might have less observations
    npy::zero_array(conf_mat);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

// Bookkeeping variables
#pragma omp parallel shared( \
    n_runs, n_thresholds, y_ptr, score_ptr, thresholds_ptr, n_obs_ptr, cm_ptr)
    {
#pragma omp for
        for (int64_t r = 0; r < n_runs; r++) {
            int64_t o_n_obs = n_obs_ptr[r];
            T1* o_y_ptr = y_ptr + (r * max_obs);
            T2* o_score_ptr = score_ptr + (r * max_obs);
            int64_t* o_cm_ptr = cm_ptr + (r * cm_offset);
            for (int64_t i = 0; i < n_thresholds; i++) {
                // fill confusion matrix
                //
                double scaled_tol = 1e-8 + 1e-05 * thresholds_ptr[i];
                core::confusion_matrix<T1, T2>(
                    o_n_obs,
                    o_y_ptr,
                    o_score_ptr,
                    thresholds_ptr[i],
                    scaled_tol,
                    o_cm_ptr + (i * 4));
            }
        }
    }  // pragma omp parallel
    return conf_mat;
}

/* Compute the confusion matrices given true labels y and classifier scores over
 * a range of thresholds. Where both arrays contain the values for
 * multiple runs/experiments.
 *
 * --- Parameters ---
 * - n_obs : the number of elements in a single run
 * - n_runs : the number of runs performed
 * - y : true labels
 * - score : classifier scores
 * - thresholds : inclusive classification threshold
 *
 * --- Returns ---
 *   * confusion matrix
 */
template <typename T1, typename T2, isFloat<T2> = true>
inline i64arr confusion_matrix_thresholds_runs(
    const int64_t n_obs,
    const int64_t n_runs,
    const py::array_t<T1>& y,
    const py::array_t<T2>& score,
    const py::array_t<T2>& thresholds) {
    // condition checks
    if (!(npy::is_well_behaved(y) && npy::is_well_behaved(score)
          && npy::is_well_behaved(thresholds))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }

    // get ptr
    const T1* const y_ptr = npy::get_data<T1>(y);
    const T2* const score_ptr = npy::get_data<T2>(score);
    const T2* const thresholds_ptr = npy::get_data<T2>(thresholds);

    // allocate confusion_matrix
    const int64_t stride_out = n_runs * 4;
    const int64_t n_thresholds = thresholds.size();
    auto conf_mat
        = npy::allocate_n_confusion_matrices<int64_t>(n_thresholds * n_runs);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

// Bookkeeping variables
#pragma omp parallel shared( \
    n_runs,                  \
    n_obs,                   \
    n_thresholds,            \
    stride_out,              \
    y_ptr,                   \
    score_ptr,               \
    thresholds_ptr,          \
    cm_ptr)
    {
#pragma omp for
        for (int64_t i = 0; i < n_thresholds; i++) {
            int64_t* o_cm_ptr = cm_ptr + (i * stride_out);
            T2 threshold = thresholds_ptr[i];
            double scaled_tol = 1e-8 + 1e-05 * threshold;
            for (int64_t r = 0; r < n_runs; r++) {
                int64_t run_offset = r * n_obs;
                // fill confusion matrix
                core::confusion_matrix<T1, T2>(
                    n_obs,
                    y_ptr + run_offset,
                    score_ptr + run_offset,
                    threshold,
                    scaled_tol,
                    o_cm_ptr + (r * 4));
            }
        }
    }  // pragma omp parallel
    return conf_mat;
}

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_CONFUSION_MATRIX_HPP_
