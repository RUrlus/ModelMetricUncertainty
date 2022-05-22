==============================
Model-Metric-Uncertainty (MMU)
==============================

.. figure:: https://github.com/RUrlus/ModelMetricUncertainty/tree/stable/docs/source/figs/pr_curve_mult_w_points.png
  :alt: Uncertainty on the Precision-Recall curve

**Model-Metric-Uncertainty (MMU) is a library for the evaluation of model performance and estimation of the uncertainty on these metrics.**

.. start


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

**Performance**

We believe performance is important as you are likely to compute the metrics over many runs, bootstraps or simulations.
To ensure high performance the core computations are performed in a C++ extension.

This gives significant speed-ups over Scikit-learn (v1.0.1):
* `confusion matrix`: ~100 times faster than `sklearn.metrics.confusion_matrix`
* `binary_metrics`: ~600 times faster than Scikit's equivalent

On a 2,3 GHz 8-Core Intel Core i9 with sample size of 1e6

Installation
------------

``mmu`` can be installed from PyPi.

.. code-block:: bash

    pip install mmu

We provide wheels for:

* MacOS [x86, ARM]
* Linux
* Windows 

Installing the package from source requires a C++ compiler with support for C++14.
If you have a compiler available it is advised to install without
the wheel as this enables architecture specific optimisations.

.. code-block:: bash

    pip install mmu --no-binary mmu

Multithreading
++++++++++++++

The extension of ``mmu`` has extensive multithreading support through OpenMP.
If you install the package from source multithreading will be enabled automatically if OpenMP is found.

You can either explicitly enable OpenMP, which will now raise an exception if it cannot be found:

.. code-block:: bash

    CMAKE_ARGS="-DMMU_ENABLE_OPENMP=ON" pip install mmu --no-binary mmu

or explicitly disable it:

.. code-block:: bash

    CMAKE_ARGS="-DMMU_DISABLE_OPEMP=ON" pip install mmu --no-binary mmu

Other build time options exist, see the `Installation section <https://mmu.readthedocs.io/en/latest/installation.html>`_ of the docs.

Usage
-----

.. code-block:: python3

    import mmu

    # Create some example data
    scores, y, yhat = mmu.generate_data(n_samples=1000)

    # Compute the joint uncertainty on precision and recall
    pr_err = mmu.PrecisionRecallMultinomialUncertainty.from_scores(y, scores, 0.85)
    
    # Plot the uncertainty
    pr_err.plot()


See the `tutorials <https://github.com/RUrlus/ModelMetricUncertainty/blob/main/notebooks>`_ for more examples.

Contributing
------------

We very much welcome contributions, please see the `contributing section <https://mmu.readthedocs.io/en/latest/contributing.html>`_ for details.


