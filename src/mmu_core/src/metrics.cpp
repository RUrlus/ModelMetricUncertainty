/* metrics_bindings.cpp -- Python bindings for metrics.hpp
 *
 * Copyright 2022 Ralph Urlus
 */
#include <mmu/api/metrics.hpp>
#include <mmu/bindings/metrics.hpp>

namespace py = pybind11;

namespace mmu {
namespace bindings {

void bind_precision_recall(py::module& m) {
    m.def(
        "precision_recall",
        &api::precision_recall,
        R"pbdoc(Compute Precision and Recall.

            0 - pos.precision aka Positive Predictive Value (PPV)
            1 - pos.recall aka True Positive Rate (TPR) aka Sensitivity

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}

void bind_precision_recall_2d(py::module& m) {
    m.def(
        "precision_recall_2d",
        &api::precision_recall_2d,
        R"pbdoc(Compute precision and recall.

        Computes the following metrics:
            0 - pos.precision aka Positive Predictive Value (PPV)
            1 - pos.recall aka True Positive Rate (TPR) aka Sensitivity

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with with column order: TN, FP, FN, TP.
            The array should be C-contiguous and have shape (N, 4)
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}

void bind_precision_recall_flattened(py::module& m) {
    m.def(
        "precision_recall_flattened",
        &api::precision_recall_flattened,
        R"pbdoc(Compute precision and recall.

        Computes the following metrics:
            0 - pos.precision aka Positive Predictive Value (PPV)
            1 - pos.recall aka True Positive Rate (TPR) aka Sensitivity

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with with column order: TN, FP, FN, TP.
            The array should be C-contiguous and should be 1D with size (N * 4)
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}

void bind_binary_metrics(py::module& m) {
    m.def(
        "binary_metrics",
        &api::binary_metrics,
        R"pbdoc(Compute binary classification metrics.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}

void bind_binary_metrics_2d(py::module& m) {
    m.def(
        "binary_metrics_2d",
        &api::binary_metrics_2d,
        R"pbdoc(Compute binary classification metrics.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with with column order: TN, FP, FN, TP.
            The array should be C-contiguous and have shape (N, 4)
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}

void bind_binary_metrics_flattened(py::module& m) {
    m.def(
        "binary_metrics_flattened",
        &api::binary_metrics_flattened,
        R"pbdoc(Compute binary classification metrics.

        Computes the following metrics:
            0 - neg.precision aka Negative Predictive Value (NPV)
            1 - pos.precision aka Positive Predictive Value (PPV)
            2 - neg.recall aka True Negative Rate (TNR) aka Specificity
            3 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            4 - neg.f1 score
            5 - pos.f1 score
            6 - False Positive Rate (FPR)
            7 - False Negative Rate (FNR)
            8 - Accuracy
            9 - MCC

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with with column order: TN, FP, FN, TP.
            The array should be C-contiguous and should be 1D with size (N * 4)
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}


void bind_ROC(py::module& m) {
    m.def(
        "ROC",
        &api::ROC,
        R"pbdoc(Compute ROC classification metrics.

        Computes the following metrics:
            0 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            1 - False Positive Rate (FPR)

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with flattened order: TN, FP, FN, TP
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}

void bind_ROC_2d(py::module& m) {
    m.def(
        "ROC_2d",
        &api::ROC_2d,
        R"pbdoc(Compute ROC classification metrics.

        Computes the following metrics:
            0 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            1 - False Positive Rate (FPR)

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with with column order: TN, FP, FN, TP.
            The array should be C-contiguous and have shape (N, 4)
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}

void bind_ROC_flattened(py::module& m) {
    m.def(
        "ROC_flattened",
        &api::ROC_flattened,
        R"pbdoc(Compute ROC classification metrics.

        Computes the following metrics:
            0 - pos.recall aka True Positive Rate (TPR) aka Sensitivity
            1 - False Positive Rate (FPR)

        Parameters
        ----------
        conf_mat : np.ndarray[int64]
            the confusion matrix with with column order: TN, FP, FN, TP.
            The array should be C-contiguous and should be 1D with size (N * 4)
        fill : double, optional
            value to fill when a metric is not defined, e.g. divide by zero.
            Default is 0.

        Returns
        -------
        metrics : np.array[np.float64]
        )pbdoc",
        py::arg("conf_mat"),
        py::arg("fill") = 1.);
}

}  // namespace bindings
}  // namespace mmu
