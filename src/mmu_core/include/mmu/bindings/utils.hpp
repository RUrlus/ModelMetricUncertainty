/* utils.hpp -- Python bindings for utility functions from mmu/numpy and
 * mmu/utils Copyright 2022 Ralph Urlus
 */
#ifndef INCLUDE_MMU_BINDINGS_UTILS_HPP_
#define INCLUDE_MMU_BINDINGS_UTILS_HPP_

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <mmu/api/numpy.hpp>
namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_all_finite(py::module& m);
void bind_is_well_behaved_finite(py::module& m);

}  // namespace bindings
}  // namespace mmu

#endif  // INCLUDE_MMU_BINDINGS_UTILS_HPP_
