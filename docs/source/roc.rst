ROC
---

- y-axis: True Positive Rate (TPR) i.e. Recall
- x-axis: False Positive Rate (FPR)

.. autoapiclass:: mmu.ROCU

.. autoapiclass:: mmu.ROCUncertainty
    :show-inheritance:
    :members:
    :inherited-members:

.. autoapiclass:: mmu.ROCCU

.. autoapiclass:: mmu.ROCCurveUncertainty
    :show-inheritance:
    :members:
    :inherited-members:

In some very specific cases you may want to compute the uncertainty through simulation of the profile likelihoods rather than through the Chi2 distribution.
Note though that the simulation is very compute intensive, each grid point is simulated ``n_simulations`` times.
Hence, you will perform ``n_bins`` * ``n_bins`` * ``n_simulations`` simulations in total.

.. autoapiclass:: mmu.ROCSimulatedUncertainty
    :show-inheritance:
    :members:
    :inherited-members:
