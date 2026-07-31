/* delong_bindings.cpp -- Python bindings for DeLong APIs
 * Copyright 2026 Ralph Urlus
 */
#include <mmu/bindings/delong.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_delong_components(py::module& m) {
    m.def(
        "delong_components",
        &api::delong_components,
        R"pbdoc(Compute AUCs and DeLong structural components.

        Parameters
        ----------
        y : np.ndarray[int64]
            binary labels with shape (n_obs,)
        scores : np.ndarray[float64]
            model scores with shape (n_models, n_obs)
        n_threads : int, default=1
            number of OpenMP threads, ignored when compiled without OpenMP

        Returns
        -------
        aucs : np.ndarray[float64]
            AUC estimate per model
        v10 : np.ndarray[float64]
            structural components of the positives, shape (n_models, n_positive)
        v01 : np.ndarray[float64]
            structural components of the negatives, shape (n_models, n_negative)
        )pbdoc",
        py::arg("y"),
        py::arg("scores"),
        py::arg("n_threads") = 1);
}

void bind_delong_covariance(py::module& m) {
    m.def(
        "delong_covariance",
        &api::delong_covariance,
        R"pbdoc(Compute DeLong AUC estimates and covariance matrix.

        Parameters
        ----------
        y : np.ndarray[int64]
            binary labels with shape (n_obs,)
        scores : np.ndarray[float64]
            model scores with shape (n_models, n_obs)
        n_threads : int, default=1
            number of OpenMP threads, ignored when compiled without OpenMP

        Returns
        -------
        aucs : np.ndarray[float64]
            AUC estimate per model
        covariance : np.ndarray[float64]
            DeLong covariance matrix between model AUC estimates
        )pbdoc",
        py::arg("y"),
        py::arg("scores"),
        py::arg("n_threads") = 1);
}

void bind_delong_test(py::module& m) {
    m.def(
        "delong_test",
        &api::delong_test,
        R"pbdoc(Compute paired DeLong test for two correlated ROC AUCs.

        Parameters
        ----------
        y : np.ndarray[int64]
            binary labels with shape (n_obs,)
        scores1 : np.ndarray[float64]
            model 1 scores with shape (n_obs,)
        scores2 : np.ndarray[float64]
            model 2 scores with shape (n_obs,)
        n_threads : int, default=1
            number of OpenMP threads, effectively 1 or 2 as the two models
            are processed in parallel; ignored when compiled without OpenMP

        Returns
        -------
        result : dict[str, float]
            Dictionary with keys: auc1, auc2, difference, variance, z, p_value
        )pbdoc",
        py::arg("y"),
        py::arg("scores1"),
        py::arg("scores2"),
        py::arg("n_threads") = 1);
}

}  // namespace bindings
}  // namespace mmu

