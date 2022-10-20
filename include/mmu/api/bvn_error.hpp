/* bvn_error.hpp -- Numpy array wrappers around core/bvn_error
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_BVN_ERROR_HPP_
#define INCLUDE_MMU_API_BVN_ERROR_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array

#include <cinttypes>  // for int64_t
#include <cmath>      // for sqrt
#include <stdexcept>  // for runtime_error

#include <mmu/core/bvn_error.hpp>
#include <mmu/core/common.hpp>
#include <mmu/core/erfinv.hpp>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

namespace pr {
f64arr bvn_error(const i64arr& conf_mat, double alpha);
f64arr bvn_error_runs(const i64arr& conf_mat, double alpha);
f64arr curve_bvn_error(const i64arr& conf_mat, double alpha);

f64arr bvn_cov(const i64arr& conf_mat);
f64arr bvn_cov_runs(const i64arr& conf_mat);
f64arr curve_bvn_cov(const i64arr& conf_mat);
}  // namespace pr

namespace roc {
f64arr bvn_error(const i64arr& conf_mat, double alpha);
f64arr bvn_error_runs(const i64arr& conf_mat, double alpha);
f64arr curve_bvn_error(const i64arr& conf_mat, double alpha);

f64arr bvn_cov(const i64arr& conf_mat);
f64arr bvn_cov_runs(const i64arr& conf_mat);
f64arr curve_bvn_cov(const i64arr& conf_mat);
}  // namespace roc

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_BVN_ERROR_HPP_
