/* bindings.cpp -- Python bindings for MMU
 * Copyright 2021 Ralph Urlus
 */
#include <pybind11/pybind11.h>
#include "metrics.hpp"
#include "confusion_matrix.hpp"

#if defined(_WIN32) || defined(WIN32) || defined(__CYGWIN__) || defined(__MINGW32__) || defined(__BORLANDC__)
#define OS_WIN
#endif

// handle error C2059: syntax error: ';'  on windows for this Macro
#if not defined OS_WIN
  #define STRINGIFY(x) #x
  #define MACRO_STRINGIFY(x) STRINGIFY(x)
#endif

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
    bind_binary_metrics_runs(m);
    bind_binary_metrics_runs_thresholds(m);

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
