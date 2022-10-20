/* roc_bvn_grid.cpp -- Implementation of Python API of bvn uncertainty over grid
 * precision recall grid Copyright 2022 Ralph Urlus
 */
#include <mmu/bindings/roc_bvn_grid.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace roc {

void bind_bvn_grid_error(py::module& m) {
    m.def(
        "roc_bvn_grid_error",
        &api::roc::bvn_grid_error,
        R"pbdoc(Compute sum of squared Z scores for a confusion matrix and PR grid.

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
        prec_rec_cov : np.ndarray[float64]
            the precision, recall and their covariance matrix
        scores : np.ndarray[float64]
            array with the minimum 2d Z-scores.
        )pbdoc",
        py::arg("precs_grid"),
        py::arg("recs_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

void bind_bvn_grid_curve_error(py::module& m) {
    m.def(
        "roc_bvn_grid_curve_error",
        &api::roc::bvn_grid_curve_error,
        R"pbdoc(Compute minimum sum of squared Z scores for each confusion matrix and grid.

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
        prec_rec_cov : np.ndarray[float64]
            the precision, recall and their covariance matrix over the thresholds
        scores : np.ndarray[float64]
            array with the minimum 2d Z-scores over the thresholds.

        )pbdoc",
        py::arg("n_conf_mats"),
        py::arg("precs_grid"),
        py::arg("recs_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
void bind_bvn_grid_curve_error_mt(py::module& m) {
    m.def(
        "roc_bvn_grid_curve_error_mt",
        &api::roc::bvn_grid_curve_error_mt,
        R"pbdoc(Compute minimum sum of squared Z scores for each confusion matrix and grid.

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
        prec_rec_cov : np.ndarray[float64]
            the precision, recall and their covariance matrix over the thresholds
        scores : np.ndarray[float64]
            array with the minimum 2d Z-scores over the thresholds.

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

void bind_bvn_chi2_score(py::module& m) {
    m.def(
        "roc_bvn_chi2_score",
        &api::roc::bvn_chi2_score,
        R"pbdoc(
            Compute sum of squared Z score for a confusion matrix and precision and recall.

        Parameters
        ----------
        prec : float
            positive precision
        rec : float
            positive recall
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        epsilon : double, default=1e-4
            epsilon used to clip recall or precision at the 0, 1 boundaries

        Returns
        -------
        score : float
            profile log likelihood score
        )pbdoc",
        py::arg("prec"),
        py::arg("rec"),
        py::arg("conf_mat"),
        py::arg("epsilon") = 1e-4);
}

void bind_bvn_chi2_scores(py::module& m) {
    m.def(
        "roc_bvn_chi2_scores",
        &api::roc::bvn_chi2_scores,
        R"pbdoc(Compute sum of squared Z scores for a confusion matrix and given precision and recalls.

        Parameters
        ----------
        precs : np.ndarray[float64]
            the positive precisions to evaluate
        precs : np.ndarray[float64]
            the positive recall to evaluate
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        epsilon : double, default=1e-4
            epsilon used to clip recall or precision at the 0, 1 boundaries

        Returns
        -------
        scores : np.ndarray[float64]
            array with the profile log likelihood scores for the given precisions and recalls.
        )pbdoc",
        py::arg("precs"),
        py::arg("recs"),
        py::arg("conf_mat"),
        py::arg("epsilon") = 1e-4);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
void bind_bvn_chi2_scores_mt(py::module& m) {
    m.def(
        "roc_bvn_chi2_scores_mt",
        &api::roc::bvn_chi2_scores_mt,
        R"pbdoc(Compute sum of squared Z scores for a confusion matrix and given precision and recalls.

        Parameters
        ----------
        precs : np.ndarray[float64]
            the positive precisions to evaluate
        precs : np.ndarray[float64]
            the positive recall to evaluate
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        epsilon : double, default=1e-4
            epsilon used to clip recall or precision at the 0, 1 boundaries

        Returns
        -------
        scores : np.ndarray[float64]
            array with the profile log likelihood scores for the given precisions and recalls.
        )pbdoc",
        py::arg("precs"),
        py::arg("recs"),
        py::arg("conf_mat"),
        py::arg("epsilon") = 1e-4);
}
#endif  // MMU_HAS_OPENMP_SUPPORT
}  // namespace roc
}  // namespace bindings
}  // namespace mmu
