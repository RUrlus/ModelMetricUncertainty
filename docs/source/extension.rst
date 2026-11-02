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

PPN-Recall bindings
+++++++++++++++++++

.. function:: py::array_t<double> ppn_recall(const py::array_t<int64_t>& conf_mat)

    Compute PPN (Proportion Predicted Negative) and Recall from a confusion
    matrix. Used by :class:`mmu.PPNRecallUncertainty`.

.. function:: py::array_t<double> ppn_recall_2d(const py::array_t<int64_t>& conf_mat)

    Compute PPN and Recall for a two-dimensional array of confusion matrices,
    e.g. over multiple thresholds. Used by
    :class:`mmu.PPNRecallCurveUncertainty`.

.. function:: tuple ppn_recall_multn_error(const int64_t n_bins, const py::array_t<int64_t>& conf_mat, const double n_sigmas, const double epsilon)

    Compute the Multinomial profile log-likelihood grid of chi2 scores
    around the PPN/Recall point of a single confusion matrix on a uniform
    grid of ``n_bins`` x ``n_bins``.

.. function:: tuple ppn_recall_multn_error_mt(const int64_t n_bins, const py::array_t<int64_t>& conf_mat, const double n_sigmas, const double epsilon, const int n_threads)

    Multithreaded variant of ``ppn_recall_multn_error``. Only available when
    ``mmu.lib._MMU_MT_SUPPORT`` is ``True``.

.. function:: double ppn_recall_multn_chi2_score(const double ppn, const double recall, const py::array_t<int64_t>& conf_mat, const double epsilon)

    Compute the chi2 score of a given PPN/Recall pair relative to the
    observed confusion matrix.

.. function:: py::array_t<double> ppn_recall_multn_chi2_scores(const py::array_t<double>& ppns, const py::array_t<double>& recalls, const py::array_t<int64_t>& conf_mat, const double epsilon)

    Compute the chi2 scores for arrays of PPN/Recall pairs relative to the
    observed confusion matrix.

.. function:: py::array_t<double> ppn_recall_multn_chi2_scores_mt(const py::array_t<double>& ppns, const py::array_t<double>& recalls, const py::array_t<int64_t>& conf_mat, const double epsilon)

    Multithreaded variant of ``ppn_recall_multn_chi2_scores``. Only available
    when ``mmu.lib._MMU_MT_SUPPORT`` is ``True``.

.. function:: py::array_t<double> ppn_recall_multn_grid_curve_error(const int64_t n_conf_mats, const py::array_t<double>& ppn_grid, const py::array_t<double>& recall_grid, const py::array_t<int64_t>& conf_mat, const double n_sigmas, const double epsilon)

    Compute the conservative (minimum chi2 score per grid bin, over
    thresholds) Multinomial profile log-likelihood on a user-provided
    PPN/Recall grid for multiple confusion matrices. Used by
    :class:`mmu.PPNRecallCurveUncertainty`.

.. function:: py::array_t<double> ppn_recall_multn_grid_curve_error_mt(const int64_t n_conf_mats, const py::array_t<double>& ppn_grid, const py::array_t<double>& recall_grid, const py::array_t<int64_t>& conf_mat, const double n_sigmas, const double epsilon, const int n_threads)

    Multithreaded variant of ``ppn_recall_multn_grid_curve_error``. Only
    available when ``mmu.lib._MMU_MT_SUPPORT`` is ``True``.

.. note::

    ``mmu.lib._MMU_MT_SUPPORT`` indicates whether the installed extension was
    built with multithreading support (OpenMP). The ``_mt`` suffixed variants
    of the functions above (and their Precision-Recall/ROC equivalents,
    ``pr_multn_*_mt``/``roc_multn_*_mt``) are only present on the extension
    module when this flag is ``True``. The PyPi wheels vendor OpenMP so this
    is ``True`` out-of-the-box; a source install may need
    ``CMAKE_ARGS="-DMMU_ENABLE_OPENMP=ON"`` to guarantee it, see
    :doc:`installation`.

.. _pybind11: https://pybind11.readthedocs.io/en/stable/#
