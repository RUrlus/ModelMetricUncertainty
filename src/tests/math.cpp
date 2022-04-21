/* nump.cpp -- Bindings to test the numpy utility functions
 * Copyright 2021 Ralph Urlus
 */

#include <mmu/tests/math.hpp>

namespace py = pybind11;

namespace mmu_tests {

void bind_erfinv(py::module &m) {
    m.def(
        "erfinv",
        [](double x) {
            return mmu::core::erfinv<double>(x);
        }
    );
}

void bind_norm_ppf(py::module &m) {
    m.def(
        "norm_ppf",
        [](double mu, double sigma, double p) {
            return mmu::core::norm_ppf<double>(mu, sigma, p);
        }
    );
}

}  // namespace mmu_tests
