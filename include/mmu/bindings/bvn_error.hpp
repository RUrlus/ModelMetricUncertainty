/* bindings/bvn_error.hpp -- Python bindings for bvn_error.hpp
 * Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_BVN_ERROR_HPP_
#define INCLUDE_MMU_BINDINGS_BVN_ERROR_HPP_

#include <mmu/api/bvn_error.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

namespace pr {
void bind_bvn_error(py::module& m);
void bind_bvn_error_runs(py::module& m);
void bind_curve_bvn_error(py::module& m);

void bind_bvn_cov(py::module& m);
void bind_bvn_cov_runs(py::module& m);
void bind_curve_bvn_cov(py::module& m);
}  // namespace pr

namespace roc {
void bind_bvn_error(py::module& m);
void bind_bvn_error_runs(py::module& m);
void bind_curve_bvn_error(py::module& m);

void bind_bvn_cov(py::module& m);
void bind_bvn_cov_runs(py::module& m);
void bind_curve_bvn_cov(py::module& m);
}  // namespace roc

}  // namespace bindings
}  // namespace mmu
#endif  // INCLUDE_MMU_BINDINGS_BVN_ERROR_HPP_
