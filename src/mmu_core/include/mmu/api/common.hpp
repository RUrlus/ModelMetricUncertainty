/* common.hpp -- Typedefs and utilities used in the API directory
 * Copyright 2026 Ralph Urlus
 */
#pragma once

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array

#include <mmu/core/common.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

typedef py::array_t<int64_t> i64arr;
typedef py::array_t<double> f64arr;

}  // namespace api
}  // namespace mmu
