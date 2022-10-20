/* bvn_error.cpp -- Python bindings for core/bvn_error
 * Copyright 2022 Ralph Urlus
 */
#include <mmu/api/bvn_error.hpp>
#include <mmu/bindings/bvn_error.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {
namespace pr {

void bind_bvn_error(py::module& m) {
    m.def(
        "pr_bvn_error",
        &api::pr::bvn_error,
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

void bind_bvn_error_runs(py::module& m) {
    m.def(
        "pr_bvn_error_runs",
        &api::pr::bvn_error_runs,
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

void bind_curve_bvn_error(py::module& m) {
    m.def(
        "pr_curve_bvn_error",
        &api::pr::curve_bvn_error,
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

void bind_bvn_cov(py::module& m) {
    m.def(
        "pr_bvn_cov",
        &api::pr::bvn_cov,
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

void bind_bvn_cov_runs(py::module& m) {
    m.def(
        "pr_bvn_cov_runs",
        &api::pr::bvn_cov_runs,
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

void bind_curve_bvn_cov(py::module& m) {
    m.def(
        "pr_curve_bvn_cov",
        &api::pr::curve_bvn_cov,
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

}  // namespace pr

namespace roc {

void bind_bvn_error(py::module& m) {
    m.def(
        "roc_bvn_error",
        &api::roc::bvn_error,
        R"pbdoc(Compute TPR, FPR and their, joint, uncertainty.

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

void bind_bvn_error_runs(py::module& m) {
    m.def(
        "roc_bvn_error_runs",
        &api::roc::bvn_error_runs,
        R"pbdoc(Compute TPR, FPR and their, joint, uncertainty.

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

void bind_curve_bvn_error(py::module& m) {
    m.def(
        "roc_curve_bvn_error",
        &api::roc::curve_bvn_error,
        R"pbdoc(Compute TPR, FPR curve and their, joint, uncertainty.

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

void bind_bvn_cov(py::module& m) {
    m.def(
        "roc_bvn_cov",
        &api::roc::bvn_cov,
        R"pbdoc(Compute TPR, FPR and their covariance matrix.

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

void bind_bvn_cov_runs(py::module& m) {
    m.def(
        "roc_bvn_cov_runs",
        &api::roc::bvn_cov_runs,
        R"pbdoc(Compute TPR, FPR and their covariance.

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

void bind_curve_bvn_cov(py::module& m) {
    m.def(
        "roc_curve_bvn_cov",
        &api::roc::curve_bvn_cov,
        R"pbdoc(Compute TPR, FPR curve and their covariance.

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

}  // namespace roc


}  // namespace bindings
}  // namespace mmu
