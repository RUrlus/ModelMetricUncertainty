/* mvn_error.hpp -- Implementation of variance and CI of Normal distributions
 * over the Poisson errors of the Confusion Matrix
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_MVN_ERROR_HPP_
#define INCLUDE_MMU_API_MVN_ERROR_HPP_

#include <pybind11/pybind11.h> // for py::array
#include <pybind11/numpy.h>  // for py::array

#include <cmath>      // for sqrt
#include <cinttypes>  // for int64_t
#include <stdexcept>  // for runtime_error

#include <mmu/core/common.hpp>
#include <mmu/core/erfinv.hpp>
#include <mmu/core/mvn_error.hpp>

#include <mmu/api/numpy.hpp>
#include <mmu/api/common.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

f64arr pr_mvn_error(const i64arr& conf_mat, double alpha);
f64arr pr_mvn_error_runs(const i64arr& conf_mat, double alpha);
f64arr pr_curve_mvn_error(const i64arr& conf_mat, double alpha);

f64arr pr_mvn_cov(const i64arr& conf_mat);
f64arr pr_mvn_cov_runs(const i64arr& conf_mat);
f64arr pr_curve_mvn_cov(const i64arr& conf_mat);

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_MVN_ERROR_HPP_
