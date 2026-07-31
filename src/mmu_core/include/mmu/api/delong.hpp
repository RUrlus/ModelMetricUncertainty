/* delong.hpp -- Python API for DeLong AUC covariance and paired test
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>
#include <mmu/core/delong.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

py::tuple delong_components(const i64arr& y, const f64arr& scores, const int n_threads = 1);
py::tuple delong_covariance(const i64arr& y, const f64arr& scores, const int n_threads = 1);
py::dict delong_test(
    const i64arr& y, const f64arr& scores1, const f64arr& scores2, const int n_threads = 1);

}  // namespace api
}  // namespace mmu

