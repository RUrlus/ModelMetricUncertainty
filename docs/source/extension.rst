.. role:: bash(code)
   :language: bash

Extension API
-------------

The core functions are wrapped using Pybind11_ and are callable from
``mmu.lib._mmu_core``.

.. note::
    This section is intended mainly intended for contributors of MMU or those
    that know what they are doing.
    ``_mmu_core`` is not strictly speaking part of the public API although we
    intend to keep it stable.

.. warning::

    The functions in ``_mmu_core`` perform the bare minimum of condition checks
    on the input arrays to prevent buffer overruns and segfaults.
    The function docstrings indicate what conditions are assumed, violating
    these is likely to result in incorrect results and or poor performance.

Confusion Matrix
++++++++++++++++

.. function:: template <typename T1, typename T2> py::array_t<int64_t> confusion_matrix(const py::array_t<T1>& y, const py::array_t<T2>& yhat)

    Where ``T1`` and ``T2`` are one of ``bool``, ``int``, ``int64_t``,
    ``float`` or ``double``.

    Compute the confusion matrix given true labels y and estimated labels yhat.

    :param y: true labels
    :param yhat: predicted labels
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    The arrays are assumed to be one-dimensional, contiguous in
    memory and have equal size. The size of the smallest array is used.

.. function:: template <typename T1, typename T2> py::array_t<int64_t> confusion_matrix_score(const py::array_t<T1>& y, const py::array_t<T2>& score, const T2 threshold)

    Where ``T1`` is one of ``bool``, ``int``, ``int64_t``, ``float``, ``double``
    and ``T2`` is ``float`` or ``double``.

    Compute the confusion matrix given true labels ``y`` and classifier scores
    ``score``.
    
    :param y: true labels
    :param score: classifier scores
    :param threshold: inclusive classification threshold
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    The arrays are assumed to be one-dimensional, contiguous in
    memory and have equal size. The size of the smallest array is used.

.. function:: template <typename T1, typename T2> py::array_t<int64_t> confusion_matrix_runs(const py::array_t<T1>& y, const py::array_t<T2>& yhat)

    Where ``T1`` and ``T2`` are one of ``bool``, ``int``, ``int64_t``,
    ``float`` or ``double``.

    Compute the confusion matrix given true labels y and estimated labels yhat.

    :param y: true labels
    :param yhat: predicted labels
    :param obs_axis: the axis containing the observations beloning to the same run, e.g. 0 when a column contains the scores/labels for a single run.
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    The arrays are assumed to be two-dimensional, contiguous in
    memory and have equal size. The size of the smallest array is used.

.. function:: template <typename T1, typename T2> py::array_t<int64_t> confusion_matrix_score_runs(const py::array_t<T1>& y, const py::array_t<T2>& score, const T2 threshold, const int obs_axis)

    Where ``T1`` is one of ``bool``, ``int``, ``int64_t``, ``float``, ``double``
    and ``T2`` is ``float`` or ``double``.

    Compute the confusion matrix given true labels ``y`` and classifier scores
    ``score``.
    
    :param y: true labels
    :param score: classifier scores
    :param threshold: inclusive classification threshold
    :param obs_axis: the axis containing the observations beloning to the same run, e.g. 0 when a column contains the scores/labels for a single run.
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    The arrays are assumed to be two-dimensional, contiguous in
    memory and have equal size. The size of the smallest array is used.


Binary Metrics
++++++++++++++

The binary metrics functions only operate on confusion matrices.

.. function:: py::array_t<double> binary_metrics(const py::array_t<int64_t>& conf_mat, const double fill)

    Computes the following metrics where [i] indicates the i'th value in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC
    
    :param conf_mat: confusion matrix
    :param fill: value to set when computed metric will be undefined
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    `conf_mat` should be aligned and contiguous.

.. function:: py::array_t<double> binary_metrics_2d(const py::array_t<int64_t>& conf_mat, const double fill)

    Computes the following metrics where [i] indicates the i'th column in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC
    
    :param conf_mat: confusion matrix
    :param fill: value to set when computed metric will be undefined
    :exception ``runtime_error``: if an array is not aligned or not C-contiguous.

    `conf_mat` should be aligned and C-contiguous and have shape (N, 4).

.. function:: py::array_t<double> binary_metrics_flattened(const py::array_t<int64_t>& conf_mat, const double fill)

    Computes the following metrics where [i] indicates the i'th column in the
    array.

        * [0] neg.precision aka Negative Predictive Value (NPV)
        * [1] pos.precision aka Positive Predictive Value (PPV)
        * [2] neg.recall aka True Negative Rate (TNR) aka Specificity
        * [3] pos.recall aka True Positive Rate (TPR) aka Sensitivity
        * [4] neg.f1 score
        * [5] pos.f1 score
        * [6] False Positive Rate (FPR)
        * [7] False Negative Rate (FNR)
        * [8] Accuracy
        * [9] MCC
    
    :param conf_mat: confusion matrix
    :param fill: value to set when computed metric will be undefined
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    `conf_mat` should be aligned and contiguous and have shape (N * 4).

.. _pybind11: https://pybind11.readthedocs.io/en/stable/#
