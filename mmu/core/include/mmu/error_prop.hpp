/* error_prop.hpp -- Implementation of varianceand CI of Normal distributions
 * over the Poisson errors of the Confusion Matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_CORE_INCLUDE_MMU_ERROR_PROP_HPP_
#define MMU_CORE_INCLUDE_MMU_ERROR_PROP_HPP_

#include <pybind11/pybind11.h> // for py::array
#include <pybind11/numpy.h>  // for py::array
#include <pybind11/stl.h>  // for py::tuple

#include <cmath>      // for sqrt
#include <limits>     // for numeric_limits
#include <cinttypes>  // for int64_t
#include <algorithm>  // for max/min
#include <stdexcept>  // for runtime_error
#include <type_traits>  // for enable_if_t

#include "mmu/common.hpp"
#include <mmu/numpy.hpp>
#include <mmu/confusion_matrix.hpp>
#include <mmu/erfinv.hpp>

namespace py = pybind11;

namespace mmu {
namespace details {

/* Compute PPF of Normal distribution
 *
 * mu : mean of distribution
 * sigma : std dev of distribution
 * alpha : percentile to compue
 */
template <typename T, isFloat<T> = true>
inline double norm_ppf(const T mu, const T sigma, const T alpha) {
    const double sqrt2 = 1.414213562373095048801688724209698079;
    return mu + sigma * sqrt2 * erfinv<T>(2 * alpha - 1);
}

/* This function computes the uncertainty on the precision-recall curve
 * using linear error propagation over the Poisson error.
 *
 * Another way of thinking about this as a Multivariate Normal over the Poisson errors.
 * Sets the following values at metrics index:
 *    0 - pos.precision aka Positive Predictive Value
 *    1 - variance of [0]
 *    2 - pos.recall aka True Positive Rate aka Sensitivity
 *    3 - variance of [2]
 *    4 - covariance between precision and recall
 */
template<typename T, isInt<T> = true>
inline void lep_conf_mat(T* const conf_mat, double* const metrics) {
    /*
     *                  pred
     *                0     1
     *  actual  0    TN    FP
     *          1    FN    TP
     *
     *  Flattened we have:
     *  0 TN
     *  1 FP
     *  2 FN
     *  3 TP
     *
     */
    auto FP = static_cast<double>(conf_mat[1]);
    auto FN = static_cast<double>(conf_mat[2]);
    auto TP = static_cast<double>(conf_mat[3]);

    auto TP_FN = static_cast<double>(conf_mat[2] + conf_mat[3]);
    auto TP_FP = static_cast<double>(conf_mat[1] + conf_mat[3]);

    // tpr == recall = TP/P = TP/(TP + FN)
    // precision == positive predictive value = TP/PP = TP/(TP + FP)
    double fn_tp_sq = std::pow(TP_FN, 2.);
    double fp_tp_sq = std::pow(TP_FP, 2.);

    double recall_d_TP = FN / fn_tp_sq;
    double recall_d_FN = - TP / fn_tp_sq;
    double precision_d_TP = FP / fp_tp_sq;
    double precision_d_FP = - TP / fp_tp_sq;

    double TP_var = std::max(TP, 1.0);
    double FN_var = std::max(FN, 1.0);
    double FP_var = std::max(FP, 1.0);

    // precision
    metrics[0] = TP / (TP_FP);
    // # precision variance
    metrics[1] = (
        std::pow(precision_d_TP, 2) * TP_var
        + std::pow(precision_d_FP, 2) * FP_var
    );
    // recall
    metrics[2] = TP / (TP_FN);
    // recall variance
    metrics[3] = (
        std::pow(recall_d_TP, 2) * TP_var
        + std::pow(recall_d_FN, 2) * FN_var
    );
    // covariance
    metrics[4] = recall_d_TP * precision_d_TP * TP_var;
}  // lep_conf_mat

template<typename T, isInt<T> = true>
inline void lep_conf_mat_ci(T* const conf_mat, double* const metrics, const double alpha) {
    /*
     *                  pred
     *                0     1
     *  actual  0    TN    FP
     *          1    FN    TP
     *
     *  Flattened we have:
     *  0 TN
     *  1 FP
     *  2 FN
     *  3 TP
     *
     */
    const double alpha_lb = alpha / 2;
    const double alpha_ub = 1.0 - alpha_ub;
    auto FP = static_cast<double>(conf_mat[1]);
    auto FN = static_cast<double>(conf_mat[2]);
    auto TP = static_cast<double>(conf_mat[3]);

    auto TP_FN = static_cast<double>(conf_mat[2] + conf_mat[3]);
    auto TP_FP = static_cast<double>(conf_mat[1] + conf_mat[3]);

    // tpr == recall = TP/P = TP/(TP + FN)
    // precision == positive predictive value = TP/PP = TP/(TP + FP)
    double fn_tp_sq = std::pow(TP_FN, 2.);
    double fp_tp_sq = std::pow(TP_FP, 2.);

    double recall_d_TP = FN / fn_tp_sq;
    double recall_d_FN = - TP / fn_tp_sq;
    double precision_d_TP = FP / fp_tp_sq;
    double precision_d_FP = - TP / fp_tp_sq;

    double TP_var = std::max(TP, 1.0);
    double FN_var = std::max(FN, 1.0);
    double FP_var = std::max(FP, 1.0);

    // precision
    metrics[1] = TP / (TP_FP);
    double precision_sigma = std::sqrt(
        std::pow(precision_d_TP, 2) * TP_var
        + std::pow(precision_d_FP, 2) * FP_var
    );
    metrics[0] = norm_ppf(metrics[1], precision_sigma, alpha_lb);
    metrics[2] = norm_ppf(metrics[1], precision_sigma, alpha_ub);

    // recall
    metrics[4] = TP / (TP_FN);
    double recall_sigma =  std::sqrt(
        std::pow(recall_d_TP, 2) * TP_var
        + std::pow(recall_d_FN, 2) * FN_var
    );
    metrics[3] = norm_ppf(metrics[4], recall_sigma, alpha_lb);
    metrics[5] = norm_ppf(metrics[4], recall_sigma, alpha_ub);
}  // lep_conf_mat


}  // namespace details


namespace bindings {

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

    details::confusion_matrix<T1, T2>(
        n_obs, npy::get_data(y), npy::get_data(yhat), cm_ptr
    );

    auto metrics = py::array_t<double>(5);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    details::lep_conf_mat(cm_ptr, metrics_ptr);
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

    details::confusion_matrix<T1, T2>(
        n_obs, npy::get_data(y), npy::get_data(yhat), cm_ptr
    );

    auto metrics = py::array_t<double>(5);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    details::lep_conf_mat_ci(cm_ptr, metrics_ptr, alpha);
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
        details::confusion_matrix<T1, T2>(
            n_obs, y_ptr, score_ptr, threshold_ptr[i], p_cm_ptr
        );
        // compute metrics
        details::lep_conf_mat(p_cm_ptr, p_metrics_ptr);
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
        details::confusion_matrix<T1, T2>(
            n_obs, y_ptr, score_ptr, threshold_ptr[i], p_cm_ptr
        );
        // compute metrics
        details::lep_conf_mat_ci(p_cm_ptr, p_metrics_ptr, alpha);
    }
    // compute metrics
    return py::make_tuple(conf_mat, metrics);
}

}  // namespace bindings
}  // namespace mmu

#endif  // MMU_CORE_INCLUDE_MMU_ERROR_PROP_HPP_
