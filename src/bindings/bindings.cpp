/* bindings.cpp -- Python bindings for MMU
 * Copyright 2022 Ralph Urlus
 */
#include <pybind11/pybind11.h>

#include <mmu/bindings/bvn_error.hpp>
#include <mmu/bindings/confusion_matrix.hpp>
#include <mmu/bindings/metrics.hpp>
#include <mmu/bindings/pr_bvn_grid.hpp>
#include <mmu/bindings/roc_bvn_grid.hpp>
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

    // pr bvn_error
    pr::bind_bvn_error(m);
    pr::bind_bvn_error_runs(m);
    pr::bind_curve_bvn_error(m);
    pr::bind_bvn_cov(m);
    pr::bind_bvn_cov_runs(m);
    pr::bind_curve_bvn_cov(m);

    // roc bvn_error
    roc::bind_bvn_error(m);
    roc::bind_bvn_error_runs(m);
    roc::bind_curve_bvn_error(m);
    roc::bind_bvn_cov(m);
    roc::bind_bvn_cov_runs(m);
    roc::bind_curve_bvn_cov(m);

    // pr_bvn_grid
    pr::bind_bvn_grid_error(m);
    pr::bind_bvn_grid_curve_error(m);
    pr::bind_bvn_chi2_score(m);
    pr::bind_bvn_chi2_scores(m);

    // roc_bvn_grid
    roc::bind_bvn_grid_error(m);
    roc::bind_bvn_grid_curve_error(m);
    roc::bind_bvn_chi2_score(m);
    roc::bind_bvn_chi2_scores(m);

    // pr_multn_loglike
    pr::bind_multn_error(m);
    pr::bind_multn_grid_error(m);
    pr::bind_multn_grid_curve_error(m);
    pr::bind_multn_sim_error(m);
    pr::bind_multn_chi2_score(m);
    pr::bind_multn_chi2_scores(m);

    // roc_multn_loglike
    roc::bind_multn_error(m);
    roc::bind_multn_grid_error(m);
    roc::bind_multn_grid_curve_error(m);
    roc::bind_multn_sim_error(m);
    roc::bind_multn_chi2_score(m);
    roc::bind_multn_chi2_scores(m);

#ifdef MMU_HAS_OPENMP_SUPPORT
    // pr_bvn_grid
    pr::bind_bvn_grid_curve_error_mt(m);
    pr::bind_bvn_chi2_scores_mt(m);

    // roc_bvn_grid
    roc::bind_bvn_grid_curve_error_mt(m);
    roc::bind_bvn_chi2_scores_mt(m);

    // pr_multn_loglike
    pr::bind_multn_error_mt(m);
    pr::bind_multn_grid_curve_error_mt(m);
    pr::bind_multn_sim_error_mt(m);
    pr::bind_multn_sim_curve_error_mt(m);
    pr::bind_multn_chi2_scores_mt(m);

    // roc_multn_loglike
    roc::bind_multn_error_mt(m);
    roc::bind_multn_grid_curve_error_mt(m);
    roc::bind_multn_sim_error_mt(m);
    roc::bind_multn_sim_curve_error_mt(m);
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
