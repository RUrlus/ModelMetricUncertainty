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

}  // namespace pr
}  // namespace bindings
}  // namespace mmu
