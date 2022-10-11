/* pr_dirich_multn.hpp -- Python bindings for Bayesian Precision-Recall
posterior PDF with Dirichlet-Multinomial prior. Copyright 2022 Max Baak, Ralph
Urlus
 */
#include <mmu/bindings/pr_dirich_multn.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace pr {

void bind_neg_log_dirich_multn_pdf(py::module& m) {
    m.def(
        "neg_log_dirich_multn_pdf",
        &api::pr::neg_log_dirich_multn_pdf,
        R"pbdoc(Compute sum of squared Z scores for a confusion matrix and PR grid.

        Parameters
        ----------
        probas : np.ndarray[float64]
            the simulated confusion matrix probabilities
        alphas : np.ndarray[float64]
            the alpha paramters of the posterior with column order TN, FP, FN, TP

        Returns
        -------
        neg_log_likes : np.ndarray[float64]
            the negative loglikelihood scores for P-R given alpha
        )pbdoc",
        py::arg("probas"),
        py::arg("alphas"));
}

void bind_neg_log_dirich_multn_pdf_mt(py::module& m) {
    m.def(
        "neg_log_dirich_multn_pdf_mt",
        &api::pr::neg_log_dirich_multn_pdf_mt,
        R"pbdoc(Compute sum of squared Z scores for a confusion matrix and PR grid.

        Parameters
        ----------
        probas : np.ndarray[float64]
            the simulated confusion matrix probabilities
        alphas : np.ndarray[float64]
            the alpha paramters of the posterior with column order TN, FP, FN, TP
        n_threads : int
            number of threads to use in computation

        Returns
        -------
        neg_log_likes : np.ndarray[float64]
            the negative loglikelihood scores for P-R given alpha
        )pbdoc",
        py::arg("probas"),
        py::arg("alphas"),
        py::arg("n_threads"));
}

void bind_dirich_multn_error(py::module& m) {
    m.def(
        "pr_dirich_multn_error",
        &api::pr::dirich_multn_error,
        R"pbdoc(Compute joint Dirichlet-multinomial uncertainty on precision-recall.

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
            chi2 score of the NLL of the PR posterior pdf
        bounds : np.array[np.float64]
            the lower and upper bound of the grids for precision and recall respectively
        )pbdoc",
        py::arg("n_bins"),
        py::arg("conf_mat"),
        py::arg("n_sigmas") = 6.0,
        py::arg("epsilon") = 1e-4);
}

// void bind_dirich_multn_error(py::module& m) {
//     m.def(
//         "pr_dirich_multn_error",
//         &api::pr::dirich_multn_error,
//         R"pbdoc(Compute joint Dirichlet-multinomial uncertainty on
//         precision-recall.
//
//         Parameters
//         ----------
//         n_bins : int,
//             number of bins in one axis for the precision and recall grid
//         conf_mat : np.ndarray[int64]
//             the confusion matrix with flattened order: TN, FP, FN, TP
//         ref_samples : np.ndarray[float64]
//             samples from a Dirichlet with paramters `conf_mat + prior`
//         n_sigmas : double, default=6.0
//             number std deviations of the marginal distributions to use as
//             grid boundaries
//         epsilon : double, default=1e-4
//             epsilon used to clip recall or precision at the 0, 1 boundaries
//
//         Returns
//         -------
//         chi2 : np.array[np.float64]
//             chi2 score of the NLL of the PR posterior pdf
//         bounds : np.array[np.float64]
//             the lower and upper bound of the grids for precision and recall
//             respectively
//         )pbdoc",
//         py::arg("n_bins"),
//         py::arg("conf_mat"),
//         py::arg("ref_samples"),
//         py::arg("n_sigmas") = 6.0,
//         py::arg("epsilon") = 1e-4);
// }

}  // namespace pr
}  // namespace bindings
}  // namespace mmu
