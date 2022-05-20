/* bindings.cpp -- Python bindings for MMU
 * Copyright 2021 Ralph Urlus
 */
#include <pybind11/pybind11.h>

#include <mmu/core/common.hpp>
#include <mmu/bindings/confusion_matrix.hpp>
#include <mmu/bindings/metrics.hpp>
#include <mmu/bindings/utils.hpp>
#include <mmu/bindings/mvn_error.hpp>
#include <mmu/bindings/mvn_grid.hpp>
#include <mmu/bindings/multn_loglike.hpp>

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
    // lep_mvn
    bind_pr_mvn_error(m);
    bind_pr_mvn_error_runs(m);
    bind_pr_curve_mvn_error(m);

    bind_pr_mvn_cov(m);
    bind_pr_mvn_cov_runs(m);
    bind_pr_curve_mvn_cov(m);

    bind_mvn_uncertainty_over_grid(m);
    bind_mvn_uncertainty_over_grid_thresholds(m);
#ifdef MMU_HAS_OPENMP_SUPPORT
    bind_mvn_uncertainty_over_grid_thresholds_mt(m);
#endif  // MMU_HAS_OPENMP_SUPPORT
    // multn_loglike
    bind_multinomial_uncertainty(m);
    bind_multinomial_uncertainty_over_grid(m);
    bind_multinomial_uncertainty_over_grid_thresholds(m);
#ifdef MMU_HAS_OPENMP_SUPPORT
    bind_multinomial_uncertainty_over_grid_thresholds_mt(m);
#endif  // MMU_HAS_OPENMP_SUPPORT
    bind_simulated_multinomial_uncertainty(m);

#ifndef OS_WIN
  #ifdef VERSION_INFO
      m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  #else
      m.attr("__version__") = "dev";
  #endif
#endif
}

}  // namespace bindings
}  // namespace mmu
