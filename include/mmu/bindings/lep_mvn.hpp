/* confusion_matrix_bindings.cpp -- Python bindings for confusion_matrix.hpp metrics
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_LEP_MVN_HPP_
#define INCLUDE_MMU_BINDINGS_LEP_MVN_HPP_

#include <mmu/api/lep_mvn.hpp>

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

void bind_pr_mvn_var(py::module &m);
void bind_pr_curve_mvn_var(py::module &m);
void bind_pr_mvn_ci(py::module &m);
void bind_pr_curve_mvn_ci(py::module &m);

}  // namespace bindings
}  // namespace mmu
#endif  // INCLUDE_MMU_BINDINGS_LEP_MVN_HPP_
