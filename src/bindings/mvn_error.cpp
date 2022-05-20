/* lep_mvn.cpp -- Python bindings for lep_mvn
 * Copyright 2021 Ralph Urlus
 */
#include <mmu/api/mvn_error.hpp>
#include <mmu/bindings/mvn_error.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_pr_mvn_error(py::module& m) {
    m.def(
        "pr_mvn_error",
        &api::pr_mvn_error,
        R"pbdoc(Compute Precision, Recall and their, joint, uncertainty.

        --- Parameters ---
        conf_mat : np.ndarray[int64]
            confusion_matrix with shape (2, 2) or (4, )
        - alpha : float
            the density inside the confidence interval

        --- Returns ---
        metrics : np.ndarray
            with columns
                0. precision
                1. LB CI precision
                2. UB CI precision
                3. recall
                4. LB CI recall
                5. UB CI recall
                6. V[precision]
                7. COV[precision, recall]
                8. COV[precision, recall]
                9. V[recall]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("alpha"));
}

void bind_pr_mvn_error_runs(py::module& m) {
    m.def(
        "pr_mvn_error_runs",
        &api::pr_mvn_error_runs,
        R"pbdoc(Compute Precision, Recall and their, joint, uncertainty.

        --- Parameters ---
        conf_mat : np.ndarray[int64]
            confusion_matrix with shape (2, 2) or (4, )
        - alpha : float
            the density inside the confidence interval

        --- Returns ---
        metrics : np.ndarray
            with columns
                0. precision
                1. LB CI precision
                2. UB CI precision
                3. recall
                4. LB CI recall
                5. UB CI recall
                6. V[precision]
                7. COV[precision, recall]
                8. COV[precision, recall]
                9. V[recall]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("alpha"));
}

void bind_pr_curve_mvn_error(py::module& m) {
    m.def(
        "pr_curve_mvn_error",
        &api::pr_curve_mvn_error,
        R"pbdoc(Compute Precision-Recall curve and their, joint, uncertainty.

        --- Parameters ---
        conf_mat : np.ndarray[int64]
            confusion_matrix with shape (N, 4)
        - alpha : float
            the density inside the confidence interval

        --- Returns ---
        metrics : np.ndarray
            with columns
                0. precision
                1. LB CI precision
                2. UB CI precision
                3. recall
                4. LB CI recall
                5. UB CI recall
                6. V[precision]
                7. COV[precision, recall]
                8. COV[precision, recall]
                9. V[recall]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("alpha"));
}

void bind_pr_mvn_cov(py::module& m) {
    m.def(
        "pr_mvn_cov",
        &api::pr_mvn_cov,
        R"pbdoc(Compute Precision, Recall and their covariance matrix.

        --- Parameters ---
        conf_mat : np.ndarray[int64]
            confusion_matrix with shape (2, 2) or (4, )

        --- Returns ---
        metrics : np.ndarray
            with columns
                0. precision
                1. recall
                2. V[precision]
                3. COV[precision, recall]
                4. COV[precision, recall]
                5. V[recall]
        )pbdoc",
        py::arg("conf_mat"));
}

void bind_pr_mvn_cov_runs(py::module& m) {
    m.def(
        "pr_mvn_cov_runs",
        &api::pr_mvn_cov_runs,
        R"pbdoc(Compute Precision, Recall and their covariance.

        --- Parameters ---
        conf_mat : np.ndarray[int64]
            confusion_matrix with shape (2, 2) or (4, )

        --- Returns ---
        metrics : np.ndarray
            with columns
                0. precision
                1. recall
                2. V[precision]
                3. COV[precision, recall]
                4. COV[precision, recall]
                5. V[recall]
        )pbdoc",
        py::arg("conf_mat"));
}

void bind_pr_curve_mvn_cov(py::module& m) {
    m.def(
        "pr_curve_mvn_cov",
        &api::pr_curve_mvn_cov,
        R"pbdoc(Compute Precision-Recall curve and their covariance.

        --- Parameters ---
        conf_mat : np.ndarray[int64]
            confusion_matrix with shape (N, 4)

        --- Returns ---
        metrics : np.ndarray
            with columns
                0. precision
                1. recall
                2. V[precision]
                3. COV[precision, recall]
                4. COV[precision, recall]
                5. V[recall]
        )pbdoc",
        py::arg("conf_mat"));
}

}  // namespace bindings
}  // namespace mmu
