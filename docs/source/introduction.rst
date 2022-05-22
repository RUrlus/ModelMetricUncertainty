Introduction
------------

**Model-Metric-Uncertainty (MMU) is a library for the evaluation of model performance and estimation of the uncertainty on these metrics.**

We currently focus on binary classification models but aim to include support for other models and their metrics in the future.

We provide functions to compute the confusion matrix and `binary_metrics` which returns the most common binary classification metrics:

1. - Negative Precision aka Negative Predictive Value
2. - Positive Precision aka Positive Predictive Value
3. - Negative Recall aka True Negative Rate aka Specificity
4. - Positive Recall aka True Positive Rate aka Sensitivity
5. - Negative F1 score
6. - Positive F1 score
7. - False Positive Rate
8. - False Negative Rate
9. - Accuracy
10. - Matthew's Correlation Coefficient

We support computing the confusion matrix and metrics over:
* the predicted labels
* probabilities with a single threshold
* probabilities with multiple thresholds
* probabilities with a single threshold and multiple runs
* probabilities with multiple thresholds and multiple runs
