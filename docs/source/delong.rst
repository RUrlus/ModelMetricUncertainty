DeLong's test
=============

MMU implements DeLong's non-parametric method [1]_ for comparing the areas under
correlated ROC curves, i.e. ROC AUCs of multiple models evaluated on the same
observations, using the fast algorithm of Sun & Xu [2]_.

- :func:`mmu.delong_test` -- paired test of two AUCs (two-sided or one-sided)
- :func:`mmu.delong_contrast_test` -- test of one or more linear contrasts of
  :math:`k` AUCs, by default whether all AUCs are equal
- :func:`mmu.auc_confidence_interval` -- confidence interval of each AUC
- :func:`mmu.delong_covariance` -- the AUC estimates and their covariance matrix

Method
------

Let there be :math:`m` positive and :math:`n` negative observations, and
:math:`k` models. For model :math:`r`, a positive :math:`x_i` and a negative
:math:`y_j` the structural components are

.. math::

    V^r_{10}(x_i) = \frac{1}{n}\sum_{j=1}^{n} \psi(x_i, y_j), \qquad
    V^r_{01}(y_j) = \frac{1}{m}\sum_{i=1}^{m} \psi(x_i, y_j), \qquad
    \psi(x, y) = \begin{cases} 1 & x > y \\ \tfrac{1}{2} & x = y \\ 0 & x < y \end{cases}

The AUC estimate is :math:`\hat\theta^r = \frac{1}{m}\sum_i V^r_{10}(x_i)`, the
Mann-Whitney statistic, and its covariance matrix is estimated by

.. math::

    S = \frac{1}{m} S_{10} + \frac{1}{n} S_{01},

where :math:`S_{10}` and :math:`S_{01}` are the sample covariance matrices of the
:math:`V_{10}` and :math:`V_{01}` components across the :math:`k` models.

Sun & Xu showed that the components follow from midranks, so they can be
computed in :math:`O(N \log N)` per model instead of :math:`O(mn)`. MMU uses a
single sort per model and counts the negatives below and tied with each positive,
which is equivalent to the midrank formulation. The covariance is
:math:`O(k^2 N)`.

**Tests.** For a contrast :math:`L` (shape :math:`c \times k`) the hypothesis
:math:`H_0: L\theta = 0` is tested with

.. math::

    z = \frac{L\hat\theta}{\sqrt{L S L^\top}} \quad (c = 1), \qquad
    \chi^2 = (L\hat\theta)^\top (L S L^\top)^{+} (L\hat\theta) \quad (c > 1),

where the :math:`\chi^2` statistic has degrees of freedom equal to the rank of
:math:`L S L^\top`. The paired test of two models is :math:`L = [1, -1]`.
One-sided alternatives are available when :math:`c = 1`.

**Confidence intervals.** The ``"wald"`` interval is
:math:`\hat\theta \pm z_{1-\alpha/2}\sqrt{S_{rr}}`, clipped to :math:`[0, 1]`, and
matches ``pROC::ci.auc(method="delong")``. The ``"logit"`` interval is computed
on the logit scale with delta-method standard error
:math:`\sqrt{S_{rr}} / (\hat\theta(1-\hat\theta))` and transformed back. It stays
within :math:`(0, 1)` and has better coverage for small samples or AUCs close to
0 or 1.

Implementation notes
--------------------

- Variances of differences and contrasts are computed from the contrasted
  components :math:`L V` rather than as :math:`L S L^\top`, which avoids
  catastrophic cancellation for highly correlated models.
- If the estimated variance is zero but the AUCs differ (e.g. perfect
  separation), the statistic and p-value are ``NaN`` and a ``RuntimeWarning`` is
  raised. Identical models give :math:`z = 0` and :math:`p = 1`.
- Scores must be finite and each class needs at least two observations.
- ``n_threads`` parallelises over models and covariance entries if MMU was
  compiled with OpenMP. Results do not depend on the number of threads. For
  :func:`mmu.delong_test` only two models are processed, so more than two
  threads has no effect.
- Results were validated against R's ``pROC`` [3]_.

Example
-------

.. code-block:: python

    import numpy as np
    import mmu

    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.3, 1000)
    noise = rng.normal(size=1000)
    scores = np.column_stack([
        1.0 * y + noise + rng.normal(0, 0.5, 1000),
        0.8 * y + noise + rng.normal(0, 0.5, 1000),
        0.6 * y + rng.normal(size=1000),
    ])

    # paired test of the first two models
    res = mmu.delong_test(y, scores[:, 0], scores[:, 1])
    res.auc1, res.auc2, res.z, res.p_value
    # (0.7639, 0.7090, 5.009, 5.48e-07)

    # is model 1 better than model 2?
    mmu.delong_test(y, scores[:, 0], scores[:, 1], alternative="greater").p_value
    # 2.74e-07

    # are all three AUCs equal? chi2 with 2 degrees of freedom
    res = mmu.delong_contrast_test(y, scores)
    res.statistic, res.df, res.p_value
    # (35.14, 2, 2.34e-08)

    # is model 1 better than the average of models 2 and 3?
    res = mmu.delong_contrast_test(y, scores, contrast=[1.0, -0.5, -0.5], alternative="greater")
    res.statistic, res.p_value
    # (5.284, 6.32e-08)

    # 95% confidence interval per model
    ci = mmu.auc_confidence_interval(y, scores, method="logit")
    ci.lower, ci.upper
    # ([0.7314, 0.6738, 0.6375], [0.7936, 0.7418, 0.7084])

References
----------

.. [1] E. R. DeLong, D. M. DeLong and D. L. Clarke-Pearson, "Comparing the Areas
   under Two or More Correlated Receiver Operating Characteristic Curves: A
   Nonparametric Approach", *Biometrics*, 44(3), 837-845, 1988.
.. [2] X. Sun and W. Xu, "Fast Implementation of DeLong's Algorithm for Comparing
   the Areas Under Correlated Receiver Operating Characteristic Curves", *IEEE
   Signal Processing Letters*, 21(11), 1389-1393, 2014.
.. [3] X. Robin et al., "pROC: an open-source package for R and S+ to analyze and
   compare ROC curves", *BMC Bioinformatics*, 12, 77, 2011.

API
---

.. autoapifunction:: mmu.delong_test
.. autoapifunction:: mmu.delong_contrast_test
.. autoapifunction:: mmu.auc_confidence_interval
.. autoapifunction:: mmu.delong_covariance

.. autoapiclass:: mmu.metrics.delong.DeLongTestResult
.. autoapiclass:: mmu.metrics.delong.DeLongContrastResult
.. autoapiclass:: mmu.metrics.delong.AUCConfidenceInterval
