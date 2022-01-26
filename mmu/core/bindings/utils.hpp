/* utils.hpp -- Python bindings for utility functions from mmu/numpy and mmu/utils
 * Copyright 2021 Ralph Urlus
 */
#ifndef MMU_CORE_BINDINGS_UTILS_HPP_
#define MMU_CORE_BINDINGS_UTILS_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <mmu/numpy.hpp>
namespace py = pybind11;


namespace mmu {
namespace bindings {

void bind_all_finite(py::module &m);
void bind_is_well_behaved_finite(py::module &m);

}  // namespace bindings
}  // namespace mmu

#endif  // MMU_CORE_BINDINGS_UTILS_HPP_
