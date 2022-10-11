/* pr_dirich_multn.hpp -- Bayesian Precision-Recall posterior PDF
with Dirichlet-Multinomial prior. Copyright 2022 Max Baak, Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_PR_DIRICHLET_HPP_
#define INCLUDE_MMU_API_PR_DIRICHLET_HPP_

#include <algorithm>

#include <mmu/api/common.hpp>
#include <mmu/api/numpy.hpp>
#include <mmu/core/pr_dirich_multn.hpp>

namespace mmu {
namespace api {
namespace pr {

f64arr neg_log_dirich_multn_pdf(const f64arr& probas, const f64arr& alphas);

f64arr neg_log_dirich_multn_pdf_mt(
    const f64arr& probas,
    const f64arr& alphas,
    const int n_threads);

py::tuple dirich_multn_error(
    const int64_t n_bins,
    const i64arr& conf_mat,
    const double n_sigmas,
    const double epsilon);

}  // namespace pr
}  // namespace api
}  // namespace mmu
#endif  // INCLUDE_MMU_API_PR_DIRICHLET_HPP_
