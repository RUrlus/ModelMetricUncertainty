/* recall_ppn_multn_loglike.cpp -- Python bindings of multinomial log-likelihood
 * uncertainty for Recall-PPN
 * Copyright 2022 Ralph Urlus
 */
#include <mmu/bindings/recall_ppn_multn_loglike.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace recall_ppn {

void bind_multn_error(py::module& m) {
    m.def(
        "recall_ppn_multn_error",
        &api::recall_ppn::multn_error,
        R"pbdoc(
        Compute chi2 scores on a uniform grid for Recall-PPN uncertainty.

        Parameters
        ----------
        n_bins : int
            Number of bins in each dimension.
        conf_mat : np.ndarray[int64]
            Confusion matrix with layout [TN, FP, FN, TP].
        n_sigmas : float
            Number of sigmas for grid bounds.
        epsilon : float
            Clipping value to avoid boundary issues.

        Returns
        -------
        Tuple[np.ndarray[float64], np.ndarray[float64]]
            (scores, bounds) where scores has shape (n_bins, n_bins)
            and bounds has shape (2, 2) with [[ppn_min, ppn_max], [recall_min, recall_max]].
        )pbdoc",
        py::arg("n_bins"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
void bind_multn_error_mt(py::module& m) {
    m.def(
        "recall_ppn_multn_error_mt",
        &api::recall_ppn::multn_error_mt,
        R"pbdoc(
        Compute chi2 scores on a uniform grid for Recall-PPN uncertainty (multi-threaded).

        Parameters
        ----------
        n_bins : int
            Number of bins in each dimension.
        conf_mat : np.ndarray[int64]
            Confusion matrix with layout [TN, FP, FN, TP].
        n_sigmas : float
            Number of sigmas for grid bounds.
        epsilon : float
            Clipping value to avoid boundary issues.
        n_threads : int
            Number of threads to use.

        Returns
        -------
        Tuple[np.ndarray[float64], np.ndarray[float64]]
            (scores, bounds) where scores has shape (n_bins, n_bins)
            and bounds has shape (2, 2) with [[ppn_min, ppn_max], [recall_min, recall_max]].
        )pbdoc",
        py::arg("n_bins"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4,
        py::arg("n_threads") = 4);
}
#endif  // MMU_HAS_OPENMP_SUPPORT

void bind_multn_chi2_score(py::module& m) {
    m.def(
        "recall_ppn_multn_chi2_score",
        &api::recall_ppn::multn_chi2_score,
        R"pbdoc(
        Compute chi2 score for a single (PPN, Recall) point.

        Parameters
        ----------
        ppn : float
            Proportion Predicted Negative value.
        recall : float
            Recall value.
        conf_mat : np.ndarray[int64]
            Confusion matrix with layout [TN, FP, FN, TP].
        epsilon : float
            Clipping value to avoid boundary issues.

        Returns
        -------
        float
            The chi2 score.
        )pbdoc",
        py::arg("ppn"),
        py::arg("recall"),
        py::arg("conf_mat"),
        py::arg("epsilon") = 1e-4);
}

void bind_multn_chi2_scores(py::module& m) {
    m.def(
        "recall_ppn_multn_chi2_scores",
        &api::recall_ppn::multn_chi2_scores,
        R"pbdoc(
        Compute chi2 scores for multiple (PPN, Recall) points.

        Parameters
        ----------
        ppns : np.ndarray[float64]
            Array of PPN values.
        recalls : np.ndarray[float64]
            Array of Recall values.
        conf_mat : np.ndarray[int64]
            Confusion matrix with layout [TN, FP, FN, TP].
        epsilon : float
            Clipping value to avoid boundary issues.

        Returns
        -------
        np.ndarray[float64]
            Array of chi2 scores.
        )pbdoc",
        py::arg("ppns"),
        py::arg("recalls"),
        py::arg("conf_mat"),
        py::arg("epsilon") = 1e-4);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
void bind_multn_chi2_scores_mt(py::module& m) {
    m.def(
        "recall_ppn_multn_chi2_scores_mt",
        &api::recall_ppn::multn_chi2_scores_mt,
        R"pbdoc(
        Compute chi2 scores for multiple (PPN, Recall) points (multi-threaded).

        Parameters
        ----------
        ppns : np.ndarray[float64]
            Array of PPN values.
        recalls : np.ndarray[float64]
            Array of Recall values.
        conf_mat : np.ndarray[int64]
            Confusion matrix with layout [TN, FP, FN, TP].
        epsilon : float
            Clipping value to avoid boundary issues.

        Returns
        -------
        np.ndarray[float64]
            Array of chi2 scores.
        )pbdoc",
        py::arg("ppns"),
        py::arg("recalls"),
        py::arg("conf_mat"),
        py::arg("epsilon") = 1e-4);
}
#endif  // MMU_HAS_OPENMP_SUPPORT

void bind_multn_grid_error(py::module& m) {
    m.def(
        "recall_ppn_multn_grid_error",
        &api::recall_ppn::multn_grid_error,
        R"pbdoc(
        Compute chi2 scores on a user-provided grid.

        Parameters
        ----------
        ppn_grid : np.ndarray[float64]
            Array of PPN values.
        recall_grid : np.ndarray[float64]
            Array of Recall values.
        conf_mat : np.ndarray[int64]
            Confusion matrix with layout [TN, FP, FN, TP].
        n_sigmas : float
            Number of sigmas for grid bounds.
        epsilon : float
            Clipping value to avoid boundary issues.

        Returns
        -------
        np.ndarray[float64]
            Chi2 scores with shape (len(ppn_grid), len(recall_grid)).
        )pbdoc",
        py::arg("ppn_grid"),
        py::arg("recall_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

void bind_multn_grid_curve_error(py::module& m) {
    m.def(
        "recall_ppn_multn_grid_curve_error",
        &api::recall_ppn::multn_grid_curve_error,
        R"pbdoc(
        Compute chi2 scores on a user-provided grid for multiple confusion matrices.

        Parameters
        ----------
        n_conf_mats : int
            Number of confusion matrices.
        ppn_grid : np.ndarray[float64]
            Array of PPN values.
        recall_grid : np.ndarray[float64]
            Array of Recall values.
        conf_mat : np.ndarray[int64]
            Array of confusion matrices with shape (n_conf_mats, 4).
        n_sigmas : float
            Number of sigmas for grid bounds.
        epsilon : float
            Clipping value to avoid boundary issues.

        Returns
        -------
        np.ndarray[float64]
            Chi2 scores with shape (len(ppn_grid), len(recall_grid)).
        )pbdoc",
        py::arg("n_conf_mats"),
        py::arg("ppn_grid"),
        py::arg("recall_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

#ifdef MMU_HAS_OPENMP_SUPPORT
void bind_multn_grid_curve_error_mt(py::module& m) {
    m.def(
        "recall_ppn_multn_grid_curve_error_mt",
        &api::recall_ppn::multn_grid_curve_error_mt,
        R"pbdoc(
        Compute chi2 scores on a user-provided grid for multiple confusion matrices (multi-threaded).

        Parameters
        ----------
        n_conf_mats : int
            Number of confusion matrices.
        ppn_grid : np.ndarray[float64]
            Array of PPN values.
        recall_grid : np.ndarray[float64]
            Array of Recall values.
        conf_mat : np.ndarray[int64]
            Array of confusion matrices with shape (n_conf_mats, 4).
        n_sigmas : float
            Number of sigmas for grid bounds.
        epsilon : float
            Clipping value to avoid boundary issues.
        n_threads : int
            Number of threads to use.

        Returns
        -------
        np.ndarray[float64]
            Chi2 scores with shape (len(ppn_grid), len(recall_grid)).
        )pbdoc",
        py::arg("n_conf_mats"),
        py::arg("ppn_grid"),
        py::arg("recall_grid"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4,
        py::arg("n_threads") = 4);
}
#endif  // MMU_HAS_OPENMP_SUPPORT

void bind_recall_ppn(py::module& m) {
    m.def(
        "recall_ppn",
        &api::recall_ppn::recall_ppn,
        R"pbdoc(
        Compute (PPN, Recall) from a confusion matrix.

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            Confusion matrix with layout [TN, FP, FN, TP].

        Returns
        -------
        Tuple[float, float]
            (ppn, recall) tuple.
        )pbdoc",
        py::arg("conf_mat"));
}

void bind_recall_ppn_2d(py::module& m) {
    m.def(
        "recall_ppn_2d",
        &api::recall_ppn::recall_ppn_2d,
        R"pbdoc(
        Compute (PPN, Recall) for multiple confusion matrices.

        Parameters
        ----------
        conf_mats : np.ndarray[int64]
            Array of confusion matrices with shape (n, 4) and columns [TN, FP, FN, TP].

        Returns
        -------
        np.ndarray[float64]
            Array of shape (n, 2) with columns [ppn, recall].
        )pbdoc",
        py::arg("conf_mats"));
}

}  // namespace recall_ppn
}  // namespace bindings
}  // namespace mmu

