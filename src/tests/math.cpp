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

void bind_multinomial_rvs(py::module &m) {
    m.def(
        "multinomial_rvs",
        [](const int64_t n_samples, const int N, const py::array_t<double>& probas, const uint64_t seed) {
            return multinomial_rvs<int>(n_samples, N, probas, seed);
        },
        py::arg("n_samples"),
        py::arg("N"),
        py::arg("probas"),
        py::arg("seed")
    );
    m.def(
        "multinomial_rvs",
        [](const int64_t n_samples, const int64_t N, const py::array_t<double>& probas, const uint64_t seed) {
            return multinomial_rvs<int64_t>(n_samples, N, probas, seed);
        },
        py::arg("n_samples"),
        py::arg("N"),
        py::arg("probas"),
        py::arg("seed")
    );
}

}  // namespace mmu_tests
