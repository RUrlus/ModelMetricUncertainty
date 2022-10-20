/* metrics.cpp -- Implementation of binary classification metrics
 * Copyright 2022 Ralph Urlus
 */
#include <mmu/api/metrics.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

/* Compute the precision-recall given a single confusion matrix.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
f64arr precision_recall(const i64arr& conf_mat, const double fill) {
    // condition checks
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    auto metrics = py::array_t<double>(2);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::precision_recall(npy::get_data(conf_mat), metrics_ptr, fill);
    return metrics;
}

/* Compute the precision, recall given N confusion matrices.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
f64arr precision_recall_2d(const i64arr& conf_mat, const double fill) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.ndim() != 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4)");
    }

    const int64_t n_obs = conf_mat.shape(0);
    auto metrics = py::array_t<double>({n_obs, static_cast<int64_t>(2)});

    int64_t* const cm_ptr = npy::get_data(conf_mat);
    double* const metrics_ptr = npy::get_data(metrics);
// compute metrics
#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_obs; i++) {
        core::precision_recall(cm_ptr + (i * 4), metrics_ptr + (i * 2), fill);
    }
    return metrics;
}

/* Compute the precision, recall given N confusion matrices in a flattened
 * shape.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
f64arr precision_recall_flattened(const i64arr& conf_mat, const double fill) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if ((conf_mat.ndim() != 1) || ((conf_mat.size() % 4) != 0)) {
        throw std::runtime_error("`conf_mat` should have shape (N * 4)");
    }

    int64_t n_conf_mats = conf_mat.size() / 4;
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    // allocate memory for metrics; are all set so don't rely on initialisation
    auto metrics = py::array_t<double>(n_conf_mats * static_cast<int64_t>(2));
    double* const metrics_ptr = npy::get_data(metrics);

#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_conf_mats; i++) {
        // compute metrics
        core::precision_recall(cm_ptr + (i * 4), metrics_ptr + (i * 2), fill);
    }
    return metrics;
}

/* Compute the binary metrics given a single confusion matrix.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
f64arr binary_metrics(const i64arr& conf_mat, const double fill) {
    // condition checks
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    auto metrics = py::array_t<double>(10);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::binary_metrics(npy::get_data(conf_mat), metrics_ptr, fill);
    return metrics;
}

/* Compute the binary metrics given N confusion matrices.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
f64arr binary_metrics_2d(const i64arr& conf_mat, const double fill) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.ndim() != 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4)");
    }

    const int64_t n_obs = conf_mat.shape(0);
    int64_t* const cm_ptr = npy::get_data(conf_mat);
    auto metrics = py::array_t<double>({n_obs, static_cast<int64_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

// compute metrics
#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_obs; i++) {
        core::binary_metrics(cm_ptr + (i * 4), metrics_ptr + (i * 10), fill);
    }
    return metrics;
}

/* Compute the binary metrics given N confusion matrices in a flattened shape.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
f64arr binary_metrics_flattened(const i64arr& conf_mat, const double fill) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if ((conf_mat.ndim() != 1) || ((conf_mat.size() % 4) != 0)) {
        throw std::runtime_error("`conf_mat` should have shape (N * 4)");
    }

    int64_t n_conf_mats = conf_mat.size() / 4;
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    // allocate memory for metrics; are all set so don't rely on initialisation
    auto metrics = py::array_t<double>(n_conf_mats * static_cast<int64_t>(10));
    double* const metrics_ptr = npy::get_data(metrics);

#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_conf_mats; i++) {
        // compute metrics
        core::binary_metrics(cm_ptr + (i * 4), metrics_ptr + (i * 10), fill);
    }
    return metrics;
}

/* Compute the ROC metrics given a single confusion matrix.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
// This is the same as PR above, it's just calling a different ROC() defined in include/mmu/core/metrics.hpp
f64arr ROC(const i64arr& conf_mat, const double fill) {
    // condition checks
    if (!npy::is_well_behaved(conf_mat)) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    auto metrics = py::array_t<double>(2);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::ROC(npy::get_data(conf_mat), metrics_ptr, fill);
    return metrics;
}

/* Compute the ROC metrics given N confusion matrices.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
f64arr ROC_2d(const i64arr& conf_mat, const double fill) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if (conf_mat.ndim() != 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4)");
    }

    const int64_t n_obs = conf_mat.shape(0);
    int64_t* const cm_ptr = npy::get_data(conf_mat);
    auto metrics = py::array_t<double>({n_obs, static_cast<int64_t>(2)});
    double* const metrics_ptr = npy::get_data(metrics);

// compute metrics
#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_obs; i++) {
        core::ROC(cm_ptr + (i * 4), metrics_ptr + (i * 2), fill);
    }
    return metrics;
}

/* Compute the ROC metrics given N confusion matrices in a flattened shape.
 *
 * --- Parameters ---
 * - conf_mat : filled confusion matrix
 * - fill : values to set when divide by zero is encountered
 *
 * --- Returns ---
 * - metrics
 */
f64arr ROC_flattened(const i64arr& conf_mat, const double fill) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error(
            "Encountered non-aligned or non-contiguous array.");
    }
    if ((conf_mat.ndim() != 1) || ((conf_mat.size() % 4) != 0)) {
        throw std::runtime_error("`conf_mat` should have shape (N * 4)");
    }

    int64_t n_conf_mats = conf_mat.size() / 4;
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    // allocate memory for metrics; are all set so don't rely on initialisation
    auto metrics = py::array_t<double>(n_conf_mats * static_cast<int64_t>(2));
    double* const metrics_ptr = npy::get_data(metrics);

#pragma omp parallel for shared(cm_ptr, metrics_ptr)
    for (int64_t i = 0; i < n_conf_mats; i++) {
        // compute metrics
        core::ROC(cm_ptr + (i * 4), metrics_ptr + (i * 2), fill);
    }
    return metrics;
}

}  // namespace api
}  // namespace mmu
