/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_METRICS_HPP_
#define INCLUDE_MMU_API_METRICS_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array
#include <pybind11/stl.h>       // for py::tuple

#include <algorithm>  // for max/min
#include <cinttypes>  // for int64_t
#include <stdexcept>  // for runtime_error

#include <mmu/core/common.hpp>
#include <mmu/core/confusion_matrix.hpp>
#include <mmu/core/metrics.hpp>

#include <mmu/api/common.hpp>
#include <mmu/api/confusion_matrix.hpp>
#include <mmu/api/numpy.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

f64arr binary_metrics(const i64arr& conf_mat, const double fill);
f64arr binary_metrics_2d(const i64arr& conf_mat, const double fill);
f64arr binary_metrics_flattened(const i64arr& conf_mat, const double fill);

f64arr precision_recall(const i64arr& conf_mat, const double fill);
f64arr precision_recall_2d(const i64arr& conf_mat, const double fill);
f64arr precision_recall_flattened(const i64arr& conf_mat, const double fill);

f64arr ROC(const i64arr& conf_mat, const double fill);
f64arr ROC_2d(const i64arr& conf_mat, const double fill);
f64arr ROC_flattened(const i64arr& conf_mat, const double fill);

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_METRICS_HPP_
