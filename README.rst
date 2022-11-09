==============================
MMU - Model-Metric-Uncertainty
==============================

**A library for the evaluation of model performance and estimation of the uncertainty on these metrics.**

.. figure:: docs/source/figs/pr_curve_mult_w_points.png
    :alt: Uncertainty on the precision-recall curve
    :align: center

.. image:: https://github.com/RUrlus/ModelMetricUncertainty/actions/workflows/macos.yml/badge.svg?branch=stable
    :target: https://github.com/RUrlus/ModelMetricUncertainty/actions/workflows/macos.yml
    :alt: MacOS build
.. image:: https://github.com/RUrlus/ModelMetricUncertainty/actions/workflows/linux.yml/badge.svg?branch=stable
    :target: https://github.com/RUrlus/ModelMetricUncertainty/actions/workflows/linux.yml
    :alt: Linux build
.. image:: https://github.com/RUrlus/ModelMetricUncertainty/actions/workflows/windows.yml/badge.svg?branch=stable
    :target: https://github.com/RUrlus/ModelMetricUncertainty/actions/workflows/windows.yml
    :alt: Windows build
.. image:: https://readthedocs.org/projects/mmu/badge/?version=latest
    :target: https://mmu.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation
.. image:: https://img.shields.io/github/license/RUrlus/modelmetricuncertainty
    :target: https://github.com/RUrlus/carma/blob/stable/LICENSE
    :alt: License
.. image:: http://img.shields.io/pypi/v/mmu.svg
    :target: https://pypi.org/project/mmu/
    :alt: PyPi

`Documentation <https://mmu.readthedocs.io/en/latest/>`_
--------------------------------------------------------

Functionality
-------------

On a high level ``MMU`` provides two types of functionality:

* **Metrics** - functions to compute confusion matrix(ces) and binary classification metrics over classifier scores or predictions.
* **Uncertainty estimators** - functionality to compute the joint uncertainty over classification metrics.

We currently focus on binary classification models but aim to include support for other types of models and their metrics in the future.

Confusion Matrix & Metrics
**************************

Metrics consist mainly of high-performance functions to compute the confusion matrix and metrics over a single test set, multiple classification thresholds and or multiple runs.

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

Uncertainty estimators
**********************

MMU provides two methods for modelling the joint uncertainty on precision and recall: Multinomial uncertainty and Bivariate-Normal.

The Multinomial approach estimates the uncertainty by computing the profile log-likelihoods scores for a grid around the precision and recall. The scores are chi2 distributed with 2 degrees of freedom which can be used to determine the confidence interval.

The Bivariate-Normal approach models the statistical uncertainty over the linearly propagated errors of the confusion matrix and the analytical covariance matrix. The resulting joint uncertainty is elliptical in nature.

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

Other build options exist, see the `Installation section <https://mmu.readthedocs.io/en/latest/installation.html>`_ of the docs.

Usage
-----

.. code-block:: python3

    import mmu

    # Create some example data
    scores, yhat, y = mmu.generate_data(n_samples=1000)

    # Compute the joint uncertainty on precision-recall curve
    pr_err = mmu.PrecisionRecallCurveUncertainty.from_scores(y, scores)
    
    # Plot the uncertainty
    pr_err.plot()

See `Basics section <https://mmu.readthedocs.io/en/latest/basics.html>`_ of the docs or the `tutorial notebooks <https://github.com/RUrlus/ModelMetricUncertainty/blob/stable/notebooks>`_ for more examples.

Contributing
------------

We very much welcome contributions, please see the `contributing section <https://mmu.readthedocs.io/en/latest/contributing.html>`_ for details.
