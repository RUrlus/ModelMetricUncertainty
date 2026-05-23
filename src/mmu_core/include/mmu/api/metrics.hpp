/* metrics.hpp -- Implementation of binary classification metrics
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array
#include <pybind11/stl.h>       // for py::tuple

#include <mmu/core/common.hpp>
#include <mmu/core/confusion_matrix.hpp>
#include <mmu/core/metrics.hpp>

#include <mmu/api/common.hpp>
#include <mmu/api/confusion_matrix.hpp>
#include <mmu/api/numpy.hpp>

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
