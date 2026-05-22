/* bindings.cpp -- Python bindings for MMU
 * Copyright 2022 Ralph Urlus
 */
#include <pybind11/pybind11.h>

#include <mmu/bindings/confusion_matrix.hpp>
#include <mmu/bindings/metrics.hpp>
#include <mmu/bindings/pr_multn_loglike.hpp>
#include <mmu/bindings/roc_multn_loglike.hpp>
#include <mmu/bindings/utils.hpp>
#include <mmu/core/common.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

PYBIND11_MODULE(EXTENSION_MODULE_NAME, m) {
    // confusion_matrix
    bind_confusion_matrix(m);
    bind_confusion_matrix_score(m);
    bind_confusion_matrix_runs(m);
    bind_confusion_matrix_score_runs(m);
    bind_confusion_matrix_thresholds(m);
    bind_confusion_matrix_runs_thresholds(m);
    bind_confusion_matrix_thresholds_runs(m);
    // metrics
    bind_precision_recall(m);
    bind_precision_recall_2d(m);
    bind_precision_recall_flattened(m);
    bind_binary_metrics(m);
    bind_binary_metrics_2d(m);
    bind_binary_metrics_flattened(m);
    bind_ROC(m);
    bind_ROC_2d(m);
    bind_ROC_flattened(m);
    // npy
    bind_all_finite(m);
    bind_is_well_behaved_finite(m);

    // pr_multn_loglike (Wilks - KEEP)
    pr::bind_multn_error(m);
    pr::bind_multn_grid_error(m);
    pr::bind_multn_grid_curve_error(m);
    pr::bind_multn_chi2_score(m);
    pr::bind_multn_chi2_scores(m);

    // roc_multn_loglike (Wilks - KEEP)
    roc::bind_multn_error(m);
    roc::bind_multn_grid_error(m);
    roc::bind_multn_grid_curve_error(m);
    roc::bind_multn_chi2_score(m);
    roc::bind_multn_chi2_scores(m);

#ifdef MMU_HAS_OPENMP_SUPPORT
    // pr_multn_loglike MT (KEEP)
    pr::bind_multn_error_mt(m);
    pr::bind_multn_grid_curve_error_mt(m);
    pr::bind_multn_chi2_scores_mt(m);

    // roc_multn_loglike MT (KEEP)
    roc::bind_multn_error_mt(m);
    roc::bind_multn_grid_curve_error_mt(m);
    roc::bind_multn_chi2_scores_mt(m);
#endif  // MMU_HAS_OPENMP_SUPPORT

#ifndef OS_WIN
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
#endif

#ifdef MMU_HAS_OPENMP_SUPPORT
    m.attr("_has_openmp_support") = true;
#else
    m.attr("_has_openmp_support") = false;
#endif  // MMU_HAS_OPENMP_SUPPORT
}

}  // namespace bindings
}  // namespace mmu
