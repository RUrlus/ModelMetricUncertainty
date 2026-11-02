Uncertainty Estimators
======================

MMU can compute uncertainty for three difference curves:

- ROC curve where points are described by:

  - y-axis: True Positive Rate (TPR) i.e. recall
  - x-axis: False Positive Rate (FPR)

- Precision-Recall curve where points are described by:

  - y-axis: Precision
  - x-axis: Recall i.e. TPR

- PPN-Recall curve where points are described by:

  - y-axis: PPN (Proportion Predicted Negative)
  - x-axis: Recall i.e. TPR


MMU models the joint uncertainty on a metric point of a curve using the Multinomial approach:

**Multinomial approach**

The Multinomial approach computes the profile log-likelihoods scores for a grid around a point (e.g. precision and recall). The bounds of the grid are determined by ``n_sigmas`` times the standard deviation of the marginals. The scores are chi2 distributed with 2 degrees of freedom.

The Multinomial approach is usually robust for relatively low statistics tests.
Additionally it is valid for the extreme values (e.g. of precision and recall).

**Threshold vs curve uncertainty**

Both methods can be applied for a specific threshold or over curve. The curve uncertainty represents a conservative view on the uncertainty.
The grid is divided into, by default, 1000 bins per axis.
For each bin in the two dimensional grid we retain the minimum score, highest probability, over the thresholds. This means that for any threshold, the curve's CI will never be smaller than the CI of the corresponding threshold.
Hence, the curve uncertainty can slightly over-cover the true confidence interval.
To help with this the curve methods provide functionality to overlay threshold uncertainty(ies) on the curve. 


.. toctree::
    :maxdepth: 1

    pr
    roc
    ppn_recall