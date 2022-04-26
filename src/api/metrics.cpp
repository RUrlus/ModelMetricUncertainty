/* metrics.cpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/metrics.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

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
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
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
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    if (
        (conf_mat.ndim() != 2)
        || ((conf_mat.shape(0) != 4) && (conf_mat.shape(1) != 4))
    ) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4)");
    }

    // guard against buffer overruns
    const size_t n_obs = std::max(conf_mat.shape(0), conf_mat.shape(1));
    int64_t* const cm_ptr = npy::get_data(conf_mat);
    auto metrics = py::array_t<double>({n_obs, static_cast<size_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    #pragma omp parallel for
    for (size_t i = 0; i < n_obs; i++) {
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
        throw std::runtime_error("Encountered non-aligned or non-contiguous array.");
    }
    if ((conf_mat.ndim() != 1) || ((conf_mat.size() % 4) != 0)) {
        throw std::runtime_error("`conf_mat` should have shape (N * 4)");
    }

    size_t n_conf_mats = conf_mat.size() / 4;
    int64_t* const cm_ptr = npy::get_data(conf_mat);

    // allocate memory for metrics; are all set so don't rely on initialisation
    auto metrics = py::array_t<double>(n_conf_mats * static_cast<ssize_t>(10));
    double* const metrics_ptr = npy::get_data(metrics);

    #pragma omp parallel for
    for (size_t i = 0; i < n_conf_mats; i++) {
        // compute metrics
        core::binary_metrics(cm_ptr + (i * 4), metrics_ptr + (i * 10), fill);
    }
    return metrics;
}

}  // namespace api
}  // namespace mmu
