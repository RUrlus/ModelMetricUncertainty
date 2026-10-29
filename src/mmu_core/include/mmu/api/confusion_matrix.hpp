/* confusion_matrix.hpp -- Implementation of binary classification confusion
 * matrix Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if defined(MMU_HAS_OPENMP_SUPPORT)
#include <omp.h>
#endif  // MMU_HAS_OPENMP_SUPPORT

#include <algorithm>
#include <type_traits>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>
#include <mmu/config.hpp>
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

namespace details {

/* Require two arrays to hold the same number of observations. */
inline void check_equal_length(
    const int64_t n_a,
    const int64_t n_b,
    const char* a_name,
    const char* b_name) {
    if (n_a != n_b) {
        throw std::invalid_argument(
            std::string(a_name) + " and " + b_name
            + " must have the same number of observations.");
    }
}

/* Require an array to be large enough for the run/observation layout. */
inline void check_holds_at_least(
    const int64_t available,
    const int64_t required,
    const char* name) {
    if (available < required) {
        throw std::invalid_argument(
            std::string(name) + " is too small for the requested layout: needs "
            + std::to_string(required) + " elements, has "
            + std::to_string(available) + ".");
    }
}

}  // namespace details

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

    details::check_equal_length(y.size(), yhat.size(), "`y`", "`yhat`");
    const int64_t n_obs = y.size();

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

    details::check_equal_length(y.size(), score.size(), "`y`", "`score`");
    const int64_t n_obs = y.size();

    auto conf_mat = npy::allocate_confusion_matrix<int64_t>();
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data<T2>(score);
    const T2 scaled_tol = core::scaled_tolerance(threshold);
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

    // `n_obs`/`n_runs` are derived from `y` alone but used to stride through
    // `score` as well, so the two must agree in size.
    details::check_equal_length(y.size(), score.size(), "`y`", "`score`");

    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data(score);

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_runs);
    int64_t* const cm_ptr = npy::get_data(conf_mat);
    const T2 scaled_tol = core::scaled_tolerance(threshold);

#pragma omp parallel if (n_runs >= MMU_OMP_MIN_TASKS) \
    shared(n_obs, n_runs, y_ptr, score_ptr, threshold, scaled_tol, cm_ptr)
    {
#pragma omp for schedule(static)
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

    // `n_obs`/`n_runs` are derived from `y` alone but used to stride through
    // `yhat` as well, so the two must agree in size.
    details::check_equal_length(y.size(), yhat.size(), "`y`", "`yhat`");

    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* yhat_ptr = npy::get_data(yhat);

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_runs);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

#pragma omp parallel if (n_runs >= MMU_OMP_MIN_TASKS) \
    shared(n_obs, n_runs, y_ptr, yhat_ptr, cm_ptr)
    {
#pragma omp for schedule(static)
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
    details::check_equal_length(y.size(), score.size(), "`y`", "`score`");
    const int64_t n_obs = y.size();
    const int64_t n_thresholds = thresholds.size();

    // allocate confusion_matrix
    auto conf_mat = npy::allocate_n_confusion_matrices<int64_t>(n_thresholds);
    // get ptr
    T1* y_ptr = npy::get_data(y);
    T2* score_ptr = npy::get_data(score);
    T2* threshold_ptr = npy::get_data(thresholds);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

#ifdef MMU_HAS_OPENMP_SUPPORT
    const int n_threads = omp_get_max_threads();
#else
    const int n_threads = 1;
#endif

    /* Dispatch between the O(n_t * n) direct sweep and the O((n + n_t) log n)
     * sort-based one. See `core::prefer_sorted_sweep`; small sweeps keep the
     * direct, zero-allocation path.
     */
    if (core::prefer_sorted_sweep(n_obs, n_thresholds, n_threads)) {
        if constexpr (std::is_integral<T1>::value) {
            core::confusion_matrix_thresholds_sorted(
                n_obs,
                n_thresholds,
                y_ptr,
                score_ptr,
                threshold_ptr,
                cm_ptr,
                core::details::IntLabelPositive<T1>{});
        } else {
            core::confusion_matrix_thresholds_sorted(
                n_obs,
                n_thresholds,
                y_ptr,
                score_ptr,
                threshold_ptr,
                cm_ptr,
                core::details::FloatLabelPositive<T1>{});
        }
        return conf_mat;
    }

#pragma omp parallel if (n_thresholds >= MMU_OMP_MIN_TASKS) \
    shared(n_obs, n_thresholds, y_ptr, score_ptr, threshold_ptr, cm_ptr)
    {
#pragma omp for schedule(static)
        for (int64_t i = 0; i < n_thresholds; i++) {
            // fill confusion matrix
            const T2 scaled_tol = core::scaled_tolerance(threshold_ptr[i]);
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
    const int64_t max_obs
        = n_runs > 0 ? *std::max_element(n_obs_ptr, n_obs_ptr + n_runs) : 0;
    const int64_t cm_offset = n_thresholds * 4;

    // `y` and `score` are indexed as `ptr + r * max_obs`, so both must hold at
    // least `n_runs * max_obs` elements. `n_obs` is caller supplied and was
    // previously trusted outright.
    details::check_equal_length(y.size(), score.size(), "`y`", "`score`");
    details::check_holds_at_least(y.size(), n_runs * max_obs, "`y`");

    // allocate confusion_matrix
    auto conf_mat
        = npy::allocate_n_confusion_matrices<int64_t>(n_thresholds * n_runs);
    // we must zero the array as we are not garuenteed to set all values
    // some runs might have less observations
    npy::zero_array(conf_mat);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

// Bookkeeping variables
#pragma omp parallel if (n_runs >= MMU_OMP_MIN_TASKS) \
    shared(n_runs,                                    \
               n_thresholds,                          \
               y_ptr,                                 \
               score_ptr,                             \
               thresholds_ptr,                        \
               n_obs_ptr,                             \
               cm_ptr)
    {
#pragma omp for schedule(static)
        for (int64_t r = 0; r < n_runs; r++) {
            int64_t o_n_obs = n_obs_ptr[r];
            T1* o_y_ptr = y_ptr + (r * max_obs);
            T2* o_score_ptr = score_ptr + (r * max_obs);
            int64_t* o_cm_ptr = cm_ptr + (r * cm_offset);
            for (int64_t i = 0; i < n_thresholds; i++) {
                // fill confusion matrix
                const T2 scaled_tol = core::scaled_tolerance(thresholds_ptr[i]);
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

    // `n_obs` and `n_runs` are caller supplied and used to stride through both
    // arrays as `ptr + r * n_obs`.
    details::check_equal_length(y.size(), score.size(), "`y`", "`score`");
    details::check_holds_at_least(y.size(), n_runs * n_obs, "`y`");

    // allocate confusion_matrix
    const int64_t stride_out = n_runs * 4;
    const int64_t n_thresholds = thresholds.size();
    auto conf_mat
        = npy::allocate_n_confusion_matrices<int64_t>(n_thresholds * n_runs);
    int64_t* const cm_ptr = npy::get_data(conf_mat);

// Bookkeeping variables
#pragma omp parallel if (n_thresholds >= MMU_OMP_MIN_TASKS) \
    shared(n_runs,                                          \
               n_obs,                                       \
               n_thresholds,                                \
               stride_out,                                  \
               y_ptr,                                       \
               score_ptr,                                   \
               thresholds_ptr,                              \
               cm_ptr)
    {
#pragma omp for schedule(static)
        for (int64_t i = 0; i < n_thresholds; i++) {
            int64_t* o_cm_ptr = cm_ptr + (i * stride_out);
            T2 threshold = thresholds_ptr[i];
            const T2 scaled_tol = core::scaled_tolerance(threshold);
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
