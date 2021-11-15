/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_METRICS_METRICS_HPP_
#define MMU_METRICS_METRICS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <string>
#include <limits>
#include <cinttypes>
#include <stdexcept>
#include <type_traits>

#include "commons/utils.hpp"
#include "commons/numpy.hpp"
#include "metrics/confusion_matrix.hpp"


namespace py = pybind11;

namespace mmu {
namespace details {

/*
 * Sets the following values at metrics index:
 *    0 - neg.precision aka Negative Predictive Value
 *    1 - pos.precision aka Positive Predictive Value
 *    2 - neg.recall aka True Negative Rate & Specificity
 *    3 - pos.recall aka True Positive Rate aka Sensitivity
 *    4 - neg.f1 score
 *    5 - pos.f1 score
 *    6 - False Positive Rate
 *    7 - False Negative Rate
 *    8 - Accuracy
 *    9 - MCC
 */
inline void binary_metrics(
    int64_t* const conf_mat, double* const metrics, const double fill = 0.
) {
    // total observations
    auto T = static_cast<double>(
        conf_mat[0]
        + conf_mat[1]
        + conf_mat[2]
        + conf_mat[3]
    );

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

    // real true/positive observations [FN + TP]
    int64_t iP = conf_mat[2] + conf_mat[3];
    bool P_nonzero = iP > 0;
    auto P = static_cast<double>(iP);
    // real false/negative observations [TN + FP]
    int64_t iN = conf_mat[0] + conf_mat[1];
    bool N_nonzero = iN > 0;
    auto N = static_cast<double>(iN);

    auto TN = static_cast<double>(conf_mat[0]);
    auto FP = static_cast<double>(conf_mat[1]);
    auto FN = static_cast<double>(conf_mat[2]);
    auto TP = static_cast<double>(conf_mat[3]);

    int64_t iTP_FP = conf_mat[3] + conf_mat[1];
    bool TP_FP_nonzero = iTP_FP > 0;
    auto TP_FP = static_cast<double>(iTP_FP);

    auto TP_TN = static_cast<double>(conf_mat[3] + conf_mat[0]);

    int64_t iTN_FN = conf_mat[0] + conf_mat[2];
    bool TN_FN_nonzero = iTN_FN > 0;
    auto TN_FN = static_cast<double>(iTN_FN);
    auto FP_FN = static_cast<double>(conf_mat[1] + conf_mat[2]);

    double itm = 0.0;
    double itm_alt = 0.0;
    // metrics 0 - neg.precision aka Negative Predictive Value (NPV)
    metrics[0] = TN_FN_nonzero ? TN / TN_FN : fill;

    // metrics[1]  - pos.precision aka Positive Predictive Value (PPV)
    metrics[1] = TP_FP_nonzero ? TP / TP_FP : fill;

    // metrics[2]  - neg.recall aka True Negative Rate (TNR) & Specificity
    // metrics[6]  - False positive Rate (FPR)
    itm = TN / N;
    metrics[2] = N_nonzero ? itm : fill;
    metrics[6] = N_nonzero ? (1 - itm) : 1.0;

    // metrics[3]  - pos.recall aka True Positive Rate (TPR) aka Sensitivity
    // metrics[7]  - False Negative Rate (FNR)
    itm = TP / P;
    metrics[3] = P_nonzero ? itm : fill;
    metrics[7] = P_nonzero ? (1 - itm) : 1.0;

    // metrics[4]  - Negative F1 score
    itm_alt = 2 * TN;
    itm = itm_alt / (itm_alt + FP_FN);
    metrics[4] = (N_nonzero || TN_FN_nonzero) ? itm : fill;

    // metrics[5]  - Positive F1 score
    itm_alt = 2 * TP;
    itm = itm_alt / (itm_alt + FP_FN);
    metrics[5] = (P_nonzero || TP_FP_nonzero) ? itm : fill;

    // metrics[8]  - Accuracy
    metrics[8] = TP_TN / T;

    // metrics[9]  - MCC
    static constexpr double limit = std::numeric_limits<double>::epsilon();
    itm = TP_FP * P * N * TN_FN;
    metrics[9] = (itm > limit) ? (TP * TN - FP * FN) / std::sqrt(itm) : fill;
}  // binary_metrics


}  // namespace details


namespace bindings {


template <typename T>
inline py::tuple binary_metrics(
    const py::array_t<T>& y,
    const py::array_t<T>& yhat,
    const double fill
) {
    // condition checks
    details::check_contiguous<T>(y, "y");
    details::check_1d_soft(y, "y");
    details::check_1d_soft(yhat, "yhat");
    details::check_contiguous<T>(yhat, "yhat");
    details::check_equal_length<T, T>(y, yhat);

    size_t n_obs = y.size();
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<int64_t>({2, 2}, {16, 8});
    int64_t* const cm_ptr = reinterpret_cast<int64_t*>(conf_mat.request().ptr);
    // initialise confusion_matrix
    memset(cm_ptr, 0, sizeof(int64_t) * 4);

    auto y_ptr = reinterpret_cast<T*>(y.request().ptr);
    auto yhat_ptr = reinterpret_cast<T*>(yhat.request().ptr);
    details::confusion_matrix<T>(n_obs, y_ptr, yhat_ptr, cm_ptr);

    auto metrics = py::array_t<double>(10);
    double* const metrics_ptr = reinterpret_cast<double*>(metrics.request().ptr);

    // compute metrics
    details::binary_metrics(cm_ptr, metrics_ptr, fill);
    return py::make_tuple(conf_mat, metrics);
}

void bind_binary_metrics(py::module &m) {
    m.def(
        "binary_metrics",
        [](
            const py::array_t<bool>& y,
            const py::array_t<bool>& yhat,
            const double fill
        ) {
            return binary_metrics<bool>(y, yhat, fill);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<int64_t>& yhat,
            const double fill
        ) {
            return binary_metrics<int64_t>(y, yhat, fill);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.0
    );
    m.def(
        "binary_metrics",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& yhat,
            const double fill
        ) {
            return binary_metrics<double>(y, yhat, fill);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics",
        [](
            const py::array_t<int>& y,
            const py::array_t<int>& yhat,
            const double fill
        ) {
            return binary_metrics<int>(y, yhat, fill);
        },
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics",
        [](
            const py::array_t<float>& y,
            const py::array_t<float>& yhat,
            const double fill
        ) {
            return binary_metrics<float>(y, yhat, fill);
        },
        R"pbdoc(Compute binary classification metrics.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        yhat : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the predicted labels
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("yhat"),
        py::arg("fill") = 0.
    );
}

template <typename T>
inline py::tuple binary_metrics_proba(
    const py::array_t<T>& y,
    const py::array_t<double>& proba,
    const double threshold,
    const double fill
) {
    // condition checks
    details::check_1d_soft(y, "y");
    details::check_contiguous<T>(y, "y");
    details::check_1d_soft(proba, "proba");
    details::check_contiguous<double>(proba, "proba");
    details::check_equal_length(y, proba);

    size_t n_obs = y.size();
    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<int64_t>({2, 2}, {16, 8});
    int64_t* const cm_ptr = reinterpret_cast<int64_t*>(conf_mat.request().ptr);
    // zero the memory of the confusion_matrix
    memset(cm_ptr, 0, sizeof(int64_t) * 4);

    auto y_ptr = reinterpret_cast<T*>(y.request().ptr);
    auto proba_ptr = reinterpret_cast<double*>(proba.request().ptr);
    details::confusion_matrix<T>(n_obs, y_ptr, proba_ptr, threshold, cm_ptr);

    auto metrics = py::array_t<double>(10);
    double* const metrics_ptr = reinterpret_cast<double*>(metrics.request().ptr);

    // compute metrics
    details::binary_metrics(cm_ptr, metrics_ptr, fill);
    return py::make_tuple(conf_mat, metrics);
}

void bind_binary_metrics_proba(py::module &m) {
    m.def(
        "binary_metrics_proba",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& proba,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_proba<bool>(y, proba, threshold, fill);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_proba",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& proba,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_proba<int64_t>(y, proba, threshold, fill);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_proba",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& proba,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_proba<double>(y, proba, threshold, fill);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_proba",
        [](
            const py::array_t<int>& y,
            const py::array_t<double>& proba,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_proba<int>(y, proba, threshold, fill);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_proba",
        [](
            const py::array_t<float>& y,
            const py::array_t<double>& proba,
            const double threshold,
            const double fill
        ) {
            return binary_metrics_proba<float>(y, proba, threshold, fill);
        },
        R"pbdoc(Compute binary classification metrics for a given threshold.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        proba : np.array[np.float64]
            the predicted probability
        threshold : float
            classification threshold
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("proba"),
        py::arg("threshold"),
        py::arg("fill") = 0.
    );
}

template <typename T>
inline py::tuple binary_metrics_thresholds(
    const py::array_t<T>& y,
    const py::array_t<double>& proba,
    const py::array_t<double>& thresholds,
    const double fill
) {
    // condition checks
    details::check_1d_soft(y, "y");
    details::check_contiguous<T>(y, "y");
    details::check_1d_soft(proba, "proba");
    details::check_contiguous<double>(proba, "proba");
    details::check_1d_soft(thresholds, "thresholds");
    details::check_contiguous<double>(thresholds, "thresholds");
    details::check_equal_length(y, proba);

    const size_t n_obs = y.size();
    const ssize_t n_thresholds = thresholds.size();

    // get ptr
    auto y_ptr = reinterpret_cast<T*>(y.request().ptr);
    auto proba_ptr = reinterpret_cast<double*>(proba.request().ptr);
    auto threshold_ptr = reinterpret_cast<double*>(thresholds.request().ptr);

    // allocate memory confusion_matrix
    auto conf_mat = py::array_t<int64_t>({n_thresholds, static_cast<ssize_t>(4)});
    int64_t* const cm_ptr = reinterpret_cast<int64_t*>(conf_mat.request().ptr);
    static constexpr size_t block_size = sizeof(int64_t) * 4;
    // initialise confusion_matrix to zeros
    memset(cm_ptr, 0, n_thresholds * block_size);

    // metrics are all set so don't rely on initialisation
    auto metrics = py::array_t<double>({n_thresholds, static_cast<ssize_t>(10)});
    double* const metrics_ptr = reinterpret_cast<double*>(metrics.request().ptr);

    int64_t* p_cm_ptr;
    double* p_metrics_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_metrics_ptr)
    for (ssize_t i = 0; i < n_thresholds; i++) {
        p_cm_ptr = cm_ptr + (i * 4);
        p_metrics_ptr = metrics_ptr + (i * 10);
        // fill confusion matrix
        details::confusion_matrix<T>(n_obs, y_ptr, proba_ptr, threshold_ptr[i], p_cm_ptr);
        // compute metrics
        details::binary_metrics(p_cm_ptr, p_metrics_ptr, fill);
    }
    return py::make_tuple(conf_mat, metrics);
}


void bind_binary_metrics_thresholds(py::module &m) {
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<bool>(y, proba, thresholds, fill);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<int64_t>(y, proba, thresholds, fill);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<double>(y, proba, thresholds, fill);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<int>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<int>(y, proba, thresholds, fill);
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
    m.def(
        "binary_metrics_thresholds",
        [](
            const py::array_t<float>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const double fill
        ) {
            return binary_metrics_thresholds<float>(y, proba, thresholds, fill);
        },
        R"pbdoc(Compute binary classification metrics over thresholds.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        proba : np.array[np.float64]
            the predicted probability
        thresholds : np.array[np.float64]
            classification thresholds
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("fill") = 0.
    );
}

template <typename T, typename A>
inline py::tuple binary_metrics_runs_thresholds(
    const py::array_t<T>& y,
    const py::array_t<A>& proba,
    const py::array_t<A>& thresholds,
    const py::array_t<int64_t>& n_obs,
    const double fill
) {
    // condition checks
    details::check_1d_soft(thresholds, "thresholds");
    details::check_contiguous(thresholds, "thresholds");
    details::check_contiguous(y, "y");
    details::check_contiguous(proba, "proba");
    details::check_contiguous(n_obs, "n_obs");
    details::check_equal_shape(y, proba, "y", "proba");

    // get ptr
    auto y_ptr = reinterpret_cast<T*>(y.request().ptr);
    auto proba_ptr = reinterpret_cast<A*>(proba.request().ptr);
    auto thresholds_ptr = reinterpret_cast<A*>(thresholds.request().ptr);
    auto n_obs_ptr = reinterpret_cast<int64_t*>(n_obs.request().ptr);

    const ssize_t n_runs = n_obs.size();
    const ssize_t n_thresholds = thresholds.shape(0);
    const ssize_t max_obs = *std::max_element(n_obs_ptr, n_obs_ptr + n_runs);

    // allocate memory confusion_matrix
    const ssize_t n_run_tresholds = n_thresholds *  n_runs;
    auto conf_mat = py::array_t<int64_t>(n_run_tresholds * static_cast<ssize_t>(4));
    int64_t* const cm_ptr = reinterpret_cast<int64_t*>(conf_mat.request().ptr);
    static constexpr size_t block_size = sizeof(int64_t) * 4;
    // initialise confusion_matrix to zeros
    memset(cm_ptr, 0, n_run_tresholds * block_size);

    // allocate memory for metrics; are all set so don't rely on initialisation
    auto metrics = py::array_t<double>(n_run_tresholds * static_cast<ssize_t>(10));
    double* const metrics_ptr = reinterpret_cast<double*>(metrics.request().ptr);

    // Bookkeeping variables
    const ssize_t cm_offset = n_thresholds * 4;
    const ssize_t metrics_offset = n_thresholds * 10;

    T* outer_y_ptr;
    A* outer_proba_ptr;
    ssize_t outer_n_obs;
    int64_t* outer_cm_ptr;
    double* outer_metrics_ptr;
    double* inner_metrics_ptr;
    int64_t* inner_cm_ptr;


    #pragma omp parallel for private(outer_n_obs, outer_y_ptr, outer_metrics_ptr, outer_cm_ptr, inner_cm_ptr, inner_metrics_ptr)
    for (ssize_t r = 0; r < n_runs; r++) {
        outer_n_obs = n_obs_ptr[r];
        outer_y_ptr = y_ptr + (r * max_obs);
        outer_proba_ptr = proba_ptr + (r * max_obs);
        outer_cm_ptr = cm_ptr + (r * cm_offset);
        outer_metrics_ptr = metrics_ptr + (r * metrics_offset);
        for (ssize_t i = 0; i < n_thresholds; i++) {
            inner_cm_ptr = outer_cm_ptr + (i * 4);
            inner_metrics_ptr = outer_metrics_ptr + (i * 10);
            // fill confusion matrix
            details::confusion_matrix<T>(
                outer_n_obs, outer_y_ptr, outer_proba_ptr, thresholds_ptr[i], inner_cm_ptr
            );
            // compute metrics
            details::binary_metrics(inner_cm_ptr, inner_metrics_ptr, fill);
        }
    }
    return py::make_tuple(conf_mat, metrics);
}

void bind_binary_metrics_runs_thresholds(py::module &m) {
    m.def(
        "_binary_metrics_run_thresholds",
        [](
            const py::array_t<bool>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<bool, double>(
                y, proba, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill")
    );
    m.def(
        "_binary_metrics_run_thresholds",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<int64_t, double>(
                y, proba, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill")
    );
    m.def(
        "_binary_metrics_run_thresholds",
        [](
            const py::array_t<double>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<double, double>(
                y, proba, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill")
    );
    m.def(
        "_binary_metrics_run_thresholds",
        [](
            const py::array_t<int>& y,
            const py::array_t<double>& proba,
            const py::array_t<double>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<int, double>(
                y, proba, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill")
    );
    m.def(
        "_binary_metrics_run_thresholds",
        [](
            const py::array_t<bool>& y,
            const py::array_t<float>& proba,
            const py::array_t<float>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<bool, float>(
                y, proba, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill")
    );
    m.def(
        "_binary_metrics_run_thresholds",
        [](
            const py::array_t<int64_t>& y,
            const py::array_t<float>& proba,
            const py::array_t<float>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<int64_t, float>(
                y, proba, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill")
    );
    m.def(
        "_binary_metrics_run_thresholds",
        [](
            const py::array_t<int>& y,
            const py::array_t<float>& proba,
            const py::array_t<float>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<int, float>(
                y, proba, thresholds, n_obs, fill
            );
        },
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill")
    );
    m.def(
        "_binary_metrics_run_thresholds",
        [](
            const py::array_t<float>& y,
            const py::array_t<float>& proba,
            const py::array_t<float>& thresholds,
            const py::array_t<int64_t>& n_obs,
            const double fill
        ) {
            return binary_metrics_runs_thresholds<float, float>(
                y, proba, thresholds, n_obs, fill
            );
        },
        R"pbdoc(Compute binary classification metrics over runs and thresholds.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        y : np.array[np.bool / np.int[32/64] / np.float[32/64]]
            the ground truth labels
        proba : np.array[np.float[32/64]]
            the predicted probability
        thresholds : np.array[np.float[32/64]]
            classification thresholds
        n_obs : np.array[np.int64]
            the number of observations per run
        fill : double
            value to fill when a metric is not defined, e.g. divide by zero.

        Returns
        -------
        tuple[np.array[np.int64], np.array[np.float64]]
            confusion matrix and metrics array
        )pbdoc",
        py::arg("y"),
        py::arg("proba"),
        py::arg("thresholds"),
        py::arg("n_obs"),
        py::arg("fill")
    );
}
    // metrics are all set so don't rely on initialisation
    auto metrics = py::array_t<double>({n_thresholds, static_cast<ssize_t>(10)});
    double* const metrics_ptr = reinterpret_cast<double*>(metrics.request().ptr);

    int64_t* p_cm_ptr;
    double* p_metrics_ptr;

    #pragma omp parallel for private(p_cm_ptr, p_metrics_ptr)
    for (ssize_t i = 0; i < n_thresholds; i++) {
        p_cm_ptr = cm_ptr + (i * 4);
        p_metrics_ptr = metrics_ptr + (i * 10);
        // fill confusion matrix
        details::confusion_matrix<T>(n_obs, y_ptr, proba_ptr, thresholds_ptr[i], p_cm_ptr);
        // compute metrics
        details::binary_metrics(p_cm_ptr, p_metrics_ptr, fill);
    }
    return py::make_tuple(conf_mat, metrics);
}

}  // namespace bindings
}  // namespace mmu

#endif  // MMU_METRICS_METRICS_HPP_
