/* bindings.cpp -- Python bindings for MMU
 * Copyright 2021 Ralph Urlus
 */
#include <pybind11/pybind11.h>

#include "mmu/common.hpp"
#include "confusion_matrix.hpp"
#include "metrics.hpp"
#include "utils.hpp"

namespace py = pybind11;

namespace mmu {
namespace bindings {

PYBIND11_MODULE(EXTENSION_MODULE_NAME, m) {
    bind_confusion_matrix(m);
    bind_confusion_matrix_score(m);
    bind_binary_metrics(m);
    bind_binary_metrics_score(m);
    bind_binary_metrics_thresholds(m);
    bind_binary_metrics_confusion(m);
    bind_binary_metrics_runs(m);
    bind_binary_metrics_runs_thresholds(m);
    bind_all_finite(m);
    bind_is_well_behaved_finite(m);

#if not defined OS_WIN
  #ifdef VERSION_INFO
      m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  #else
      m.attr("__version__") = "dev";
  #endif
#endif
}

}  // namespace bindings
}  // namespace mmu
