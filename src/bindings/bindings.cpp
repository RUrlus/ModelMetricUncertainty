/* bindings.cpp -- Python bindings for MMU
 * Copyright 2021 Ralph Urlus
 */
#include <pybind11/pybind11.h>

#include <mmu/bindings/confusion_matrix.hpp>
#include <mmu/bindings/metrics.hpp>
#include <mmu/bindings/multn_loglike.hpp>
#include <mmu/bindings/bvn_error.hpp>
#include <mmu/bindings/bvn_grid.hpp>
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
    // metrics
    bind_precision_recall(m);
    bind_precision_recall_2d(m);
    bind_precision_recall_flattened(m);
    bind_binary_metrics(m);
    bind_binary_metrics_2d(m);
    bind_binary_metrics_flattened(m);
    // npy
    bind_all_finite(m);
    bind_is_well_behaved_finite(m);
    // lep_bvn
    bind_pr_bvn_error(m);
    bind_pr_bvn_error_runs(m);
    bind_pr_curve_bvn_error(m);

    bind_pr_bvn_cov(m);
    bind_pr_bvn_cov_runs(m);
    bind_pr_curve_bvn_cov(m);

    bind_bvn_uncertainty_over_grid(m);
    bind_bvn_uncertainty_over_grid_thresholds(m);
#ifdef MMU_HAS_OPENMP_SUPPORT
    bind_bvn_uncertainty_over_grid_thresholds_mt(m);
#endif  // MMU_HAS_OPENMP_SUPPORT
    // multn_loglike
    bind_multinomial_uncertainty(m);
#ifdef MMU_HAS_OPENMP_SUPPORT
    bind_multinomial_uncertainty_mt(m);
#endif  // MMU_HAS_OPENMP_SUPPORT
    bind_multinomial_uncertainty_over_grid(m);
    bind_multinomial_uncertainty_over_grid_thresholds(m);
#ifdef MMU_HAS_OPENMP_SUPPORT
    bind_multinomial_uncertainty_over_grid_thresholds_mt(m);
#endif  // MMU_HAS_OPENMP_SUPPORT
    bind_simulated_multinomial_uncertainty(m);
#ifdef MMU_HAS_OPENMP_SUPPORT
    bind_simulated_multinomial_uncertainty_mt(m);
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
