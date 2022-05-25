Uncertainty Estimators
======================

Precision-Recall
----------------

MMU provides two methods for modelling the joint uncertainty on precision and recall.

**Multinomial approach**

The Multinomial approach computes the profile log-likelihoods scores for a grid around the precision and recall. The bounds of the grid are determined by ``n_sigmas`` times the standard deviation of the marginals. The scores are chi2 distributed with 2 degrees of freedom.

The Multinomial approach is usually robust for relatively low statistics tests.
Additionally it is valid for the extreme values of precision and recall, which the Bivariate-Normal approach is not. However, the Multinomial approach does not allow the statistical uncertainty of the train set to be incorporated which the Bivariate-Normal does.

**Bivariate-Normal approach**

The statistical/sampling uncertainty over the Precision and Recall are modelled
as a Bivariate-Normal over the linearly propagated errors of the confusion
matrix. For a threshold/confusion matrix a covariance matrix is computed which is used to determine the elliptical uncertainty.
The curve uncertainty computes chi2 scores in a similar manner to Multinomial approach.

Note that the Bivariate-Normal (Elliptical) approach is only valid for medium to high statistic datasets. A warning is raised when the Normal approximation to the Binomial may not be valid. Additionally, the estimation is not valid for the extremes of precision/recall. However, the train set uncertainty can be added to the test uncertainty.

**Threshold vs curve uncertainty**

Both methods can be applied for a specific threshold or over the precision recall curve. The curve uncertainty represents a conservative view on the uncertainty.
The precision-recall grid is divided into, by default, 1000 bins per axis.
For each bin in the two dimensional grid we retain the minimum score, highest probability, over the thresholds. This means that for any threshold, the curve's CI will never be smaller than the CI of the corresponding threshold.
Hence, the curve uncertainty can slightly over-cover the true confidence interval.
To help with this the curve methods provide functionality to overlay threshold uncertainty(ies) on the curve. 

.. autoapiclass:: mmu.PRU

.. autoapiclass:: mmu.PrecisionRecallUncertainty
    :members:
    :inherited-members:

.. autoapiclass:: mmu.PRCU

.. autoapiclass:: mmu.PrecisionRecallCurveUncertainty
    :members:
    :inherited-members:

In some very specific cases you may want to compute the uncertainty through simulation of the profile likelihoods rather than through the Chi2 distribution.
Note though that the simulation is very compute intensive, each grid point is simulated ``n_simulations`` times.
Hence, you will perform ``n_bins`` * ``n_bins`` * ``n_simulations`` simulations in total.

.. autoapiclass:: mmu.PrecisionRecallSimulatedUncertainty
    :members:
    :inherited-members:
