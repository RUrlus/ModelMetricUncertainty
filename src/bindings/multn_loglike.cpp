/* multn_loglike.cpp -- Python bindings of multinomial log-likelihood uncertainty
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/bindings/multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_multinomial_uncertainty(py::module& m) {
    m.def(
        "multinomial_uncertainty",
        &api::multinomial_uncertainty,
        R"pbdoc(Compute multinomial uncertainty.

        Parameters
        ----------
        n_bins : int,
            number of bins in one axis for the precision and recall grid
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        n_sigmas : double, default=6.0
            number std deviations of the marginal distributions to use as
            grid boundaries
        epsilon : double, default=1e-4
            epsilon used to clip recall or precision at the 0, 1 boundaries

        Returns
        -------
        chi2 : np.array[np.float64]
            chi2 score of the profile log-likelihoods over the Precision-Recall grid
        bounds : np.array[np.float64]
            the lower and upper bound of the grids for precision and recall respectively
        )pbdoc",
        py::arg("n_bins"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

void bind_multinomial_uncertainty_over_grid(py::module& m) {
    m.def(
        "multinomial_uncertainty_over_grid",
        &api::multinomial_uncertainty_over_grid,
        R"pbdoc(Compute minimum profile loglike scores for a confusion matrix and PR grid.

        Parameters
        ----------
        precs_grid : np.ndarray[float64]
            the precision space over which to evaluate the uncertainty give the
            confusion matrix
        recs_grid : np.ndarray[float64]
            the recall space over which to evaluate the uncertainty give the
            confusion matrix
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        n_sigmas : double, default=6.0
            number std deviations of the marginal distributions to use as
            grid boundaries
        epsilon : double, default=1e-4
            epsilon used to clip recall or precision at the 0, 1 boundaries

        Returns
        -------
        scores : np.ndarray[float64]
            array with the minimum profile log likelihood scores.
        )pbdoc",
        py::arg("precs_grid"),
        py::arg("recs_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

void bind_multinomial_uncertainty_over_grid_thresholds(py::module& m) {
    m.def(
        "multinomial_uncertainty_over_grid_thresholds",
        &api::multinomial_uncertainty_over_grid_thresholds,
        R"pbdoc(Compute minimum profile loglike scores for each confusion matrix and grid.

        Parameters
        ----------
        n_conf_mats : int
            number of confusion matrices to evaluate
        precs_grid : np.ndarray[float64]
            the precision space over which to evaluate the uncertainty give the
            confusion matrix
        recs_grid : np.ndarray[float64]
            the recall space over which to evaluate the uncertainty give the
            confusion matrix
        conf_mat : np.ndarray[int64]
            the confusion matrices with flattened order: TN, FP, FN, TP
        n_sigmas : double, default=6.0
            number std deviations of the marginal distributions to use as
            grid boundaries
        epsilon : double, default=1e-4
            epsilon used to clip recall or precision at the 0, 1 boundaries

        Returns
        -------
        scores : np.ndarray[float64]
            array with the minimum profile log likelihood scores over the thresholds.

        )pbdoc",
        py::arg("n_conf_mats"),
        py::arg("precs_grid"),
        py::arg("recs_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
void bind_multinomial_uncertainty_over_grid_thresholds_mt(py::module& m) {
    m.def(
        "multinomial_uncertainty_over_grid_thresholds_mt",
        &api::multn_uncertainty_over_grid_thresholds_mt,
        R"pbdoc(Compute minimum profile loglike scores for each confusion matrix and grid.

        Parameters
        ----------
        n_conf_mats : int
            number of confusion matrices to evaluate
        precs_grid : np.ndarray[float64]
            the precision space over which to evaluate the uncertainty give the
            confusion matrix
        recs_grid : np.ndarray[float64]
            the recall space over which to evaluate the uncertainty give the
            confusion matrix
        conf_mat : np.ndarray[int64]
            the confusion matrices with flattened order: TN, FP, FN, TP
        n_sigmas : double, default=6.0
            number std deviations of the marginal distributions to use as
            grid boundaries
        epsilon : double, default=1e-4
            epsilon used to clip recall or precision at the 0, 1 boundaries
        n_threads : int, default=4
            number of threads to use in the computation. Keep in mind that
            each thread requires it's own block of size
            (precs_grid.size * recs_grid.size)

        Returns
        -------
        scores : np.ndarray[float64]
            array with the minimum profile log likelihood scores over the thresholds.

        )pbdoc",
        py::arg("n_conf_mats"),
        py::arg("precs_grid"),
        py::arg("recs_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4,
        py::arg("n_threads") = 4);
}
#endif  // MMU_HAS_OPENMP_SUPPORT

void bind_simulated_multinomial_uncertainty(py::module& m) {
    m.def(
        "simulated_multinomial_uncertainty",
        &api::simulated_multinomial_uncertainty,
        R"pbdoc(Compute multinomial uncertainty.

        Parameters
        ----------
        n_sims : int,
            number of confusion matrices to simulate for each point in the grid
        n_bins : int,
            number of bins in one axis for the precision and recall grid
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        n_sigmas : double, default=6.0
            number std deviations of the marginal distributions to use as
            grid boundaries
        epsilon : double, default=1e-4
            epsilon used to clip recall or precision at the 0, 1 boundaries
        seed : int, default=0
            seed for the random number generator, if equal to zero
            the random device is used to generate a seed
        stream : int, default=0
            which of the 2^127 independent streams of the pcg64_dxsm to be used.
            Parameter is ignored when seed == 0

        Returns
        -------
        chi2 : np.array[np.float64]
            chi2 score of the profile log-likelihoods over the Precision-Recall grid
        bounds : np.array[np.float64]
            the lower and upper bound of the grids for precision and recall respectively
        )pbdoc",
        py::arg("n_sims"),
        py::arg("n_bins"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4,
        py::arg("seed") = 0,
        py::arg("stream") = 0);
}
}  // namespace bindings
}  // namespace mmu
