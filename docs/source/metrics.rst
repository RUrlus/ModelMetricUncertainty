Metrics
=======

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

``threshold`` accepts any finite real scalar -- it is not restricted to
``[0, 1]`` -- so raw ``decision_function``/logit-style scores can be used
directly. Passing a non-numeric value raises a ``TypeError``, and a
non-finite (``NaN``/``inf``) value raises a ``ValueError``.

.. autoapifunction:: mmu.auto_thresholds
.. autoapifunction:: mmu.binary_metrics
.. autoapifunction:: mmu.binary_metrics_thresholds
.. autoapifunction:: mmu.binary_metrics_confusion_matrix
.. autoapifunction:: mmu.binary_metrics_confusion_matrices
.. autoapifunction:: mmu.binary_metrics_runs
.. autoapifunction:: mmu.binary_metrics_runs_thresholds
.. autoapifunction:: mmu.confusion_matrix
.. autoapifunction:: mmu.confusion_matrices
.. autoapifunction:: mmu.confusion_matrices_thresholds
.. autoapifunction:: mmu.confusion_matrices_runs_thresholds
.. autoapifunction:: mmu.precision_recall
.. autoapifunction:: mmu.precision_recall_curve
.. autoapifunction:: mmu.metrics.col_index
.. autoapifunction:: mmu.metrics.col_names
.. autoapifunction:: mmu.metrics.confusion_matrix_to_dataframe
.. autoapifunction:: mmu.metrics.confusion_matrices_to_dataframe
.. autoapifunction:: mmu.metrics.metrics_to_dataframe
