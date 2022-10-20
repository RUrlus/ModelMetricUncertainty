/* bvn_error.cpp -- Numpy array wrappers around core/bvn_error
 * Copyright 2022 Ralph Urlus
 */
#include <mmu/api/bvn_error.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {
namespace pr {

/* Compute the Precision-Recall and their uncertainty.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. LB CI precision
 *     2. UB CI precision
 *     3. recall
 *     4. LB CI recall
 *     5. UB CI recall
 *     6. V[precision]
 *     7. COV[precision, recall]
 *     8. V[recall]
 *     9. COV[precision, recall]
 */
f64arr bvn_error(const i64arr& conf_mat, double alpha) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.size() != 4) {
        throw std::runtime_error(
            "`conf_mat` should have shape (2, 2) or (4,).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto metrics = f64arr(10);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::pr::bvn_error(cm_ptr, metrics_ptr, alpha);
    return metrics;
}

/* Compute the Precision-Recall and their uncertainty.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. LB CI precision
 *     2. UB CI precision
 *     3. recall
 *     4. LB CI recall
 *     5. UB CI recall
 *     6. V[precision]
 *     7. COV[precision, recall]
 *     8. V[recall]
 *     9. COV[precision, recall]
 */
f64arr bvn_error_runs(const i64arr& conf_mat, double alpha) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim != 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    const int64_t n_runs = conf_mat.shape(0);

    auto metrics = f64arr({n_runs, static_cast<int64_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

// compute metrics
#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_runs; i++) {
        core::pr::bvn_error(cm_ptr + (i * 4), metrics_ptr + (i * 10), alpha);
    }
    return metrics;
}

/* Compute the Precision-Recall curve and its uncertainty.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. LB CI precision
 *     2. UB CI precision
 *     3. recall
 *     4. LB CI recall
 *     5. UB CI recall
 *     6. V[precision]
 *     7. COV[precision, recall]
 *     8. V[recall]
 *     9. COV[precision, recall]
 */
f64arr curve_bvn_error(const i64arr& conf_mat, double alpha) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    int64_t n_mats = conf_mat.shape(0);
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto metrics = f64arr({n_mats, static_cast<int64_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_mats; i++) {
        core::pr::bvn_error(cm_ptr + (i * 4), metrics_ptr + (i * 10), alpha);
    }
    return metrics;
}

/* Compute the Precision-Recall and their covariance.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. recall
 *     2. V[precision]
 *     3. COV[precision, recall]
 *     4. V[recall]
 *     5. COV[precision, recall]
 */
f64arr bvn_cov(const i64arr& conf_mat) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.size() != 4) {
        throw std::runtime_error(
            "`conf_mat` should have shape (2, 2) or (4,).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto metrics = f64arr(6);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::pr::bvn_cov(cm_ptr, metrics_ptr);
    return metrics;
}

/* Compute the Precision-Recall and their covariance.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. recall
 *     2. V[precision]
 *     3. COV[precision, recall]
 *     4. V[recall]
 *     5. COV[precision, recall]
 */
f64arr bvn_cov_runs(const i64arr& conf_mat) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim != 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    const int64_t n_runs = conf_mat.shape(0);

    auto metrics = f64arr({n_runs, static_cast<int64_t>(6)});
    double* const metrics_ptr = npy::get_data(metrics);

// compute metrics
#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_runs; i++) {
        core::pr::bvn_cov(cm_ptr + (i * 4), metrics_ptr + (i * 6));
    }
    return metrics;
}

/* Compute the Precision-Recall curve and its uncertainty.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. recall
 *     2. V[precision]
 *     3. COV[precision, recall]
 *     4. V[recall]
 *     5. COV[precision, recall]
 */
f64arr curve_bvn_cov(const i64arr& conf_mat) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    int64_t n_mats = conf_mat.shape(0);
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto metrics = f64arr({n_mats, static_cast<int64_t>(6)});
    double* const metrics_ptr = npy::get_data(metrics);

#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_mats; i++) {
        core::pr::bvn_cov(cm_ptr + (i * 4), metrics_ptr + (i * 6));
    }
    return metrics;
}

}  // namespace pr


// TODO: to refactor, all those functions below, pr and roc seems to be the same
// nothing was modified here, it is just calling a different bvn_error() and bvn_cov()
// defined in include/mmu/core/bvn_error.hpp
namespace roc {

/* Compute the Precision-Recall and their uncertainty.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. LB CI precision
 *     2. UB CI precision
 *     3. recall
 *     4. LB CI recall
 *     5. UB CI recall
 *     6. V[precision]
 *     7. COV[precision, recall]
 *     8. V[recall]
 *     9. COV[precision, recall]
 */
f64arr bvn_error(const i64arr& conf_mat, double alpha) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.size() != 4) {
        throw std::runtime_error(
            "`conf_mat` should have shape (2, 2) or (4,).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto metrics = f64arr(10);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::roc::bvn_error(cm_ptr, metrics_ptr, alpha);
    return metrics;
}

/* Compute the Precision-Recall and their uncertainty.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. LB CI precision
 *     2. UB CI precision
 *     3. recall
 *     4. LB CI recall
 *     5. UB CI recall
 *     6. V[precision]
 *     7. COV[precision, recall]
 *     8. V[recall]
 *     9. COV[precision, recall]
 */
f64arr bvn_error_runs(const i64arr& conf_mat, double alpha) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim != 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    const int64_t n_runs = conf_mat.shape(0);

    auto metrics = f64arr({n_runs, static_cast<int64_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

// compute metrics
#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_runs; i++) {
        core::roc::bvn_error(cm_ptr + (i * 4), metrics_ptr + (i * 10), alpha);
    }
    return metrics;
}

/* Compute the Precision-Recall curve and its uncertainty.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. LB CI precision
 *     2. UB CI precision
 *     3. recall
 *     4. LB CI recall
 *     5. UB CI recall
 *     6. V[precision]
 *     7. COV[precision, recall]
 *     8. V[recall]
 *     9. COV[precision, recall]
 */
f64arr curve_bvn_error(const i64arr& conf_mat, double alpha) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    int64_t n_mats = conf_mat.shape(0);
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto metrics = f64arr({n_mats, static_cast<int64_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_mats; i++) {
        core::roc::bvn_error(cm_ptr + (i * 4), metrics_ptr + (i * 10), alpha);
    }
    return metrics;
}

/* Compute the Precision-Recall and their covariance.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. recall
 *     2. V[precision]
 *     3. COV[precision, recall]
 *     4. V[recall]
 *     5. COV[precision, recall]
 */
f64arr bvn_cov(const i64arr& conf_mat) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.size() != 4) {
        throw std::runtime_error(
            "`conf_mat` should have shape (2, 2) or (4,).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto metrics = f64arr(6);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::roc::bvn_cov(cm_ptr, metrics_ptr);
    return metrics;
}

/* Compute the Precision-Recall and their covariance.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. recall
 *     2. V[precision]
 *     3. COV[precision, recall]
 *     4. V[recall]
 *     5. COV[precision, recall]
 */
f64arr bvn_cov_runs(const i64arr& conf_mat) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim != 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    const int64_t n_runs = conf_mat.shape(0);

    auto metrics = f64arr({n_runs, static_cast<int64_t>(6)});
    double* const metrics_ptr = npy::get_data(metrics);

// compute metrics
#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_runs; i++) {
        core::roc::bvn_cov(cm_ptr + (i * 4), metrics_ptr + (i * 6));
    }
    return metrics;
}

/* Compute the Precision-Recall curve and its uncertainty.
 *
 * --- Parameters ---
 * - conf_mat : confusion_matrix
 * - alpha : the density inside the confidence interval
 *
 * --- Returns ---
 * - metrics with columns
 *     0. precision
 *     1. recall
 *     2. V[precision]
 *     3. COV[precision, recall]
 *     4. V[recall]
 *     5. COV[precision, recall]
 */
f64arr curve_bvn_cov(const i64arr& conf_mat) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    int64_t n_mats = conf_mat.shape(0);
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    auto metrics = f64arr({n_mats, static_cast<int64_t>(6)});
    double* const metrics_ptr = npy::get_data(metrics);

#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_mats; i++) {
        core::roc::bvn_cov(cm_ptr + (i * 4), metrics_ptr + (i * 6));
    }
    return metrics;
}

}  // namespace roc

}  // namespace api
}  // namespace mmu
