/* multn_loglike.hpp -- Python bindings of multinomial log-likelihood uncertainty
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_MULTN_LOGLIKE_HPP_
#define INCLUDE_MMU_BINDINGS_MULTN_LOGLIKE_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array

#include <mmu/api/multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_multinomial_uncertainty(py::module& m);
void bind_multinomial_uncertainty_over_grid(py::module& m);
void bind_multinomial_uncertainty_over_grid_thresholds(py::module& m);
void bind_multinomial_uncertainty_over_grid_thresholds_mt(py::module& m);
void bind_simulated_multinomial_uncertainty(py::module& m);

}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_MULTN_LOGLIKE_HPP_
