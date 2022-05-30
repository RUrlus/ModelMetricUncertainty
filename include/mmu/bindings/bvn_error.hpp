/* bindings/bvn_error.hpp -- Python bindings for bvn_error.hpp metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_BVN_ERROR_HPP_
#define INCLUDE_MMU_BINDINGS_BVN_ERROR_HPP_

#include <mmu/api/bvn_error.hpp>

namespace py = pybind11;

/*                  pred
 *                0     1
 *  actual  0    TN    FP
 *          1    FN    TP
 *
 *  Flattened, implies C-contiguous, we have:
 *  0 TN
 *  1 FP
 *  2 FN
 *  3 TP
 */

namespace mmu {
namespace bindings {

void bind_pr_bvn_error(py::module& m);
void bind_pr_bvn_error_runs(py::module& m);
void bind_pr_curve_bvn_error(py::module& m);

void bind_pr_bvn_cov(py::module& m);
void bind_pr_bvn_cov_runs(py::module& m);
void bind_pr_curve_bvn_cov(py::module& m);

}  // namespace bindings
}  // namespace mmu
#endif  // INCLUDE_MMU_BINDINGS_BVN_ERROR_HPP_
