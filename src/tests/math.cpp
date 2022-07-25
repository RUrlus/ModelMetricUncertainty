/* nump.cpp -- Bindings to test the numpy utility functions
 * Copyright 2021 Ralph Urlus
 */

#include <mmu/tests/math.hpp>

namespace py = pybind11;

namespace mmu_tests {

void bind_erfinv(py::module& m) {
    m.def("erfinv", [](double x) { return mmu::core::erfinv<double>(x); });
}

void bind_norm_ppf(py::module& m) {
    m.def("norm_ppf", [](double mu, double sigma, double p) {
        return mmu::core::norm_ppf<double>(mu, sigma, p);
    });
}

py::array_t<int64_t> binomial_rvs(
    const int64_t n_samples,
    const int64_t n,
    const double p,
    const uint64_t seed,
    const uint64_t stream) {
    py::array_t<int64_t> result(n_samples);
    int64_t* ptr = result.mutable_data();
    mmu::core::random::binomial_rvs(n_samples, n, p, ptr, seed, stream);
    return result;
}

void bind_binomial_rvs(py::module& m) {
    m.def(
        "binomial_rvs",
        &binomial_rvs,
        py::arg("n_samples"),
        py::arg("n"),
        py::arg("p"),
        py::arg("seed"),
        py::arg("stream"));
}

py::array_t<int64_t> multinomial_rvs(
    const int64_t n_samples,
    const int64_t n,
    py::array_t<double>& p,
    const uint64_t seed,
    const uint64_t stream) {
    py::array_t<int64_t> result(n_samples * p.size());
    int64_t* ptr = result.mutable_data();
    mmu::core::zero_array(ptr, result.size());
    mmu::core::random::multinomial_rvs(
        n_samples, n, p.size(), p.mutable_data(), ptr, seed, stream);
    return result.reshape({static_cast<ssize_t>(n_samples), p.size()});
}

void bind_multinomial_rvs(py::module& m) {
    m.def(
        "multinomial_rvs",
        &multinomial_rvs,
        py::arg("n_samples"),
        py::arg("n"),
        py::arg("p"),
        py::arg("seed"),
        py::arg("stream"));
}

}  // namespace mmu_tests
