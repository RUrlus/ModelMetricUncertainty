/* error_prop.hpp -- Implementation of varianceand CI of Normal distributions
 * over the Poisson errors of the Confusion Matrix
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/mvn_error.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

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
f64arr pr_mvn_error(
    const i64arr& conf_mat,
    double alpha
) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error("Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.size() != 4) {
        throw std::runtime_error("`conf_mat` should have shape (2, 2) or (4,).");
    }
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);;

    auto metrics = f64arr(10);
    double* const metrics_ptr = npy::get_data(metrics);

    // compute metrics
    core::pr_mvn_error(cm_ptr, metrics_ptr, alpha);
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
f64arr pr_curve_mvn_error(
    const i64arr& conf_mat,
    double alpha
) {
    // condition checks
    if ((!npy::is_aligned(conf_mat)) || (!npy::is_c_contiguous(conf_mat))) {
        throw std::runtime_error("Encountered non-aligned or non-C-contiguous array.");
    }
    size_t ndim = conf_mat.ndim();
    if (ndim > 2 || conf_mat.shape(1) != 4) {
        throw std::runtime_error("`conf_mat` should have shape (N, 4).");
    }
    size_t n_mats = conf_mat.shape(0);
    // allocate memory confusion_matrix
    int64_t* const cm_ptr = npy::get_data(conf_mat);;

    auto metrics = f64arr({n_mats, static_cast<size_t>(10)});
    double* const metrics_ptr = npy::get_data(metrics);

    #pragma omp parallel for
    for (size_t i = 0; i < n_mats; i++) {
        core::pr_mvn_error(cm_ptr + (i * 4), metrics_ptr + (1 * 10), alpha);
    }
    return metrics;
}

}  // namespace api
}  // namespace mmu
