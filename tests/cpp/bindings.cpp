/* bindings.cpp -- Python bindings for MMU tests
 * Copyright 2021 Ralph Urlus
 */
#include <pybind11/pybind11.h>
#include "numpy_bindings.hpp"
#include "utils_bindings.hpp"

namespace py = pybind11;

namespace mmu_tests {

PYBIND11_MODULE(_mmu_core_tests, m) {
    bind_is_contiguous(m);
    bind_is_c_contiguous(m);
    bind_is_f_contiguous(m);
    bind_check_shape_order(m);
    bind_assert_shape_order(m);

    bind_check_1d_soft(m);
    bind_check_equal_length(m);
    bind_check_contiguous(m);
    bind_check_equal_shape(m);
}

}  // namespace mmu_tests
