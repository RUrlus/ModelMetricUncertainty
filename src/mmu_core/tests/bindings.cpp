/* bindings.cpp -- Python bindings for MMU tests
 * Copyright 2026 Ralph Urlus
 */
#include <pybind11/pybind11.h>
#include <mmu/tests/numpy.hpp>

namespace mmu_tests {

PYBIND11_MODULE(_mmu_core_tests, m) {
    bind_is_contiguous(m);
    bind_is_c_contiguous(m);
    bind_is_f_contiguous(m);
    bind_is_well_behaved(m);
    bind_test_get_data(m);
    bind_zero_array(m);
    bind_zero_array_fixed(m);
    bind_allocate_confusion_matrix(m);
    bind_allocate_n_confusion_matrices(m);
}

}  // namespace mmu_tests
