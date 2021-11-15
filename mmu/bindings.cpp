/* bindings.cpp -- Python bindings for MMU
 * Copyright 2021 Ralph Urlus
 */
#include <pybind11/pybind11.h>
#include "metrics/metrics.hpp"
#include "metrics/confusion_matrix.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace mmu {
namespace bindings {

PYBIND11_MODULE(EXTENSION_MODULE_NAME, m) {
    bind_confusion_matrix(m);
    bind_confusion_matrix_proba(m);
    bind_binary_metrics(m);
    bind_binary_metrics_proba(m);
    bind_binary_metrics_thresholds(m);
    bind_binary_metrics_confusion(m);
    bind_binary_metrics_runs_thresholds(m);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

}  // namespace bindings
}  // namespace mmu
