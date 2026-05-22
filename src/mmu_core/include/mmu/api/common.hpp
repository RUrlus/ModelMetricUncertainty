/* common.hpp -- Typedefs and utilities used in the API directory
 * Copyright 2021 Ralph Urlus
 */
#ifndef INCLUDE_MMU_API_COMMON_HPP_
#define INCLUDE_MMU_API_COMMON_HPP_

#include <pybind11/numpy.h>     // for py::array
#include <pybind11/pybind11.h>  // for py::array

#include <cinttypes>  // for int64_t

#include <mmu/core/common.hpp>

namespace py = pybind11;

namespace mmu {
namespace api {

typedef py::array_t<int64_t> i64arr;
typedef py::array_t<double> f64arr;

}  // namespace api
}  // namespace mmu

#endif  // INCLUDE_MMU_API_COMMON_HPP_
