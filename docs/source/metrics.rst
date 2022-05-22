.. _confusion_api:

Confusion Matrix
================
.. autofunction:: mmu.confusion_matrix
.. autofunction:: mmu.confusion_matrices
.. autofunction:: mmu.confusion_matrices_thresholds
.. autofunction:: mmu.confusion_matrices_runs_thresholds


.. _metrics_api:

Metrics
=======

Precision-Recall
****************

.. autofunction:: mmu.precision_recall
.. autofunction:: mmu.precision_recall_curve

Binary Metrics
**************

The ``binary_metrics`` functions compute the 10 most commonly used metrics:

- Negative precision aka Negative Predictive Value (NPV)
- Positive recision aka Positive Predictive Value (PPV)
- Negative recall aka True Negative Rate (TNR) aka Specificity
- Positive recall aka True Positive Rate (TPR) aka Sensitivity
- Negative f1 score
- Positive f1 score
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Accuracy
- Mathew's Correlation Coefficient (MCC)

Most other metrics should be computable from these.

.. autofunction:: mmu.binary_metrics
.. autofunction:: mmu.binary_metrics_thresholds
.. autofunction:: mmu.binary_metrics_confusion_matrix
.. autofunction:: mmu.binary_metrics_confusion_matrices
.. autofunction:: mmu.binary_metrics_runs
.. autofunction:: mmu.binary_metrics_runs_thresholds

Utilities
=========

.. autofunction:: mmu.auto_thresholds
.. autofunction:: mmu.metrics.confusion_matrix_to_dataframe
.. autofunction:: mmu.metrics.confusion_matrices_to_dataframe
.. autofunction:: mmu.metrics.metrics_to_dataframe
