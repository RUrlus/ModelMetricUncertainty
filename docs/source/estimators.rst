Uncertainty Estimators
======================

MMU can compute uncertainty for two difference curves:

- ROC curve where points are described by:

  - y-axis: True Positive Rate (TPR) i.e. recall
  - x-axis: False Positive Rate (FPR)

- Precision-Recall curve where points are described by:

  - y-axis: Precision
  - x-axis: Recall i.e. TPR


MMU provides two approaches for modelling the joint uncertainty on a metric point of a curve:

**Multinomial approach**

The Multinomial approach computes the profile log-likelihoods scores for a grid around a point (e.g. precision and recall). The bounds of the grid are determined by ``n_sigmas`` times the standard deviation of the marginals. The scores are chi2 distributed with 2 degrees of freedom.

The Multinomial approach is usually robust for relatively low statistics tests.
Additionally it is valid for the extreme values (e.g. of precision and recall), which the Bivariate-Normal approach is not. However, the Multinomial approach does not allow the statistical uncertainty of the train set to be incorporated which the Bivariate-Normal does.

**Bivariate-Normal approach**

The statistical/sampling uncertainty (e.g. over the precision and recall) are modelled
as a Bivariate-Normal over the linearly propagated errors of the confusion
matrix. For a threshold/confusion matrix a covariance matrix is computed which is used to determine the elliptical uncertainty.
The curve uncertainty computes chi2 scores in a similar manner to Multinomial approach.

Note that the Bivariate-Normal (Elliptical) approach is only valid for medium to high statistic datasets. A warning is raised when the Normal approximation to the Binomial may not be valid. Additionally, the estimation is not valid for the extremes (e.g. of precision and recall). However, the train set uncertainty can be added to the test uncertainty.

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