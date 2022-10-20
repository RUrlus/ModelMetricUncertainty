Precision-Recall
----------------

- y-axis: Precision
- x-axis: Recall i.e. TPR

.. autoapiclass:: mmu.PRU

.. autoapiclass:: mmu.PrecisionRecallUncertainty
    :show-inheritance:
    :members:
    :inherited-members:

.. autoapiclass:: mmu.PRCU

.. autoapiclass:: mmu.PrecisionRecallCurveUncertainty
    :show-inheritance:
    :members:
    :inherited-members:

In some very specific cases you may want to compute the uncertainty through simulation of the profile likelihoods rather than through the Chi2 distribution.
Note though that the simulation is very compute intensive, each grid point is simulated ``n_simulations`` times.
Hence, you will perform ``n_bins`` * ``n_bins`` * ``n_simulations`` simulations in total.

.. autoapiclass:: mmu.PrecisionRecallSimulatedUncertainty
    :show-inheritance:
    :members:
    :inherited-members:
