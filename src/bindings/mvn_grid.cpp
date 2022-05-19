/* multn_loglike.cpp -- Python bindings of multinomial log-likelihood uncertainty
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/bindings/mvn_grid.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_mvn_uncertainty_over_grid(py::module &m) {
    m.def(
        "mvn_uncertainty_over_grid",
        &api::mvn_uncertainty_over_grid,
        R"pbdoc(Compute minimum 2d-Z score for a confusion matrix and PR grid.

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
            array with the minimum 2d Z-scores.
        )pbdoc",
        py::arg("precs_grid"),
        py::arg("recs_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4
    );
}

void bind_mvn_uncertainty_over_grid_thresholds(py::module &m) {
    m.def(
        "mvn_uncertainty_over_grid_thresholds",
        &api::mvn_uncertainty_over_grid_thresholds,
        R"pbdoc(Compute minimum 2d-Z scores for each confusion matrix and grid.

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
            array with the minimum 2d Z-scores over the thresholds.

        )pbdoc",
        py::arg("n_conf_mats"),
        py::arg("precs_grid"),
        py::arg("recs_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4
    );
}

#ifdef MMU_HAS_OPENMP_SUPPORT
void bind_mvn_uncertainty_over_grid_thresholds_mt(py::module &m) {
    m.def(
        "mvn_uncertainty_over_grid_thresholds_mt",
        &api::mvn_uncertainty_over_grid_thresholds_mt,
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
            array with the minimum 2d Z-scores over the thresholds.

        )pbdoc",
        py::arg("n_conf_mats"),
        py::arg("precs_grid"),
        py::arg("recs_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4,
        py::arg("n_threads") = 4
    );
}
#endif  // MMU_HAS_OPENMP_SUPPORT
}  // namespace bindings
}  // namespace mmu
