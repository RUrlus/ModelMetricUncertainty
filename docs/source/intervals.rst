Marginal intervals
==================

``mmu.methods`` provides marginal binomial intervals alongside the
:doc:`joint uncertainty estimators <estimators>`. These functions are also
available directly from ``mmu``.

.. code-block:: python

    import mmu

    # Exact interval for 8 successes in 10 trials.
    lower, upper = mmu.clopper_pearson(8, 10)
    lower, upper = mmu.wilson(8, 10, continuity=True)
    lower, upper = mmu.jeffreys(8, 10)

    # Confusion matrix order: TN, FP, FN, TP.
    intervals = mmu.bmetrics_ci(
        [50, 5, 10, 35], metrics=["precision", "recall"], levels=0.95
    )  # shape (2, 2): metrics, then lower/upper

Methods and interpretation
--------------------------

Clopper-Pearson is the default: under a binomial sampling model it has at
least nominal coverage, often conservatively so. Wilson is an approximate
score interval, optionally continuity-corrected. The driver's ``continuity``
option is ignored for other methods. Jeffreys uses a
Beta(x + 1/2, n - x + 1/2) posterior, modified to include zero at x=0 and
one at x=n; it is a credible interval, not an exact frequentist interval.
No hybrid method or ``min_count`` selection is provided.

.. important::

    Recall, specificity, FPR and FNR condition on the observed class counts.
    Precision (PPV) and NPV instead condition on the predicted class counts,
    which are themselves random. A binomial interval on precision therefore
    quantifies **conditional** uncertainty only: interpreting it as all
    precision-recall uncertainty can understate uncertainty. Use
    :class:`mmu.methods.PrecisionRecallUncertainty` (``PRU``) for joint
    precision-recall uncertainty.

Marginal intervals are not a simultaneous confidence region. In particular,
the rectangle formed from two 95% marginal intervals is not generally a 95%
joint region.

Levels, shapes and boundaries
-----------------------------

The ``levels`` convention matches the joint estimators' ``plot`` methods:
integers specify normal standard deviations, while floats specify probability
mass. Thus ``levels=2`` means approximately 95.45%, whereas ``levels=0.95``
means 95%. Floats must be finite and strictly between zero and one; positive
integer sigmas must be small enough that their probability does not round to
one. Arrays must be nonempty and one-dimensional, with their dtype determining
the convention; mixing integers and floats in a list raises a ``TypeError``.

Primitive functions broadcast ``x`` and ``n``. Array levels add a separate,
trailing axis, preserving order and duplicates. Two-sided primitive calls
return a ``(lower, upper)`` tuple; ``side="lower"`` or ``side="upper"`` returns
only that one-sided bound.

The confusion-matrix driver accepts ``(4,)``, ``(2, 2)``, ``(..., 2, 2)`` or
``(..., 4)`` counts, as well as the ``(n_thresholds, 4, n_runs)`` output of
``mmu.confusion_matrices_runs_thresholds``, and returns ``(..., n_metrics, 2)`` for a scalar level or
``(..., n_levels, n_metrics, 2)`` for array levels, where ``...`` are the
batch axes, e.g. ``(n_thresholds, n_runs)``. One-sided calls omit the
last bounds axis. A ``(n, 4, 4)`` array is ambiguous and requires an explicit
``axis`` indicating which axis holds the counts. ``metrics`` accepts the names and aliases in
``mmu.metrics.col_index``, preserving order and duplicates; it must not be a
bare string. With no selection, all ten metric slots are returned; F1 and
MCC bounds are NaN, and requesting either explicitly raises an error.

Zero-denominator metrics have NaN bounds, including an all-zero confusion
matrix. Otherwise, lower bounds at zero successes are exactly zero and upper
bounds at all successes are exactly one. Invalid or fractional counts raise
an error. ``confidence=`` is not an alias for ``levels=``.

DataFrame output
----------------

``return_df=True`` gives canonical metric names from ``mmu.metrics.col_names``
and a column MultiIndex ``(metric, bound)``. Batch dimensions are flattened
in row-major order. Array levels add a row MultiIndex ``(sample, level)``,
retaining their input order, including duplicates.

.. code-block:: python

    frame = mmu.bmetrics_ci(
        [50, 5, 10, 35],
        metrics=["recall", "fpr"],
        levels=[0.90, 0.95],
        return_df=True,
    )
    recall_lower = frame[("pos.recall", "lower")]

API
---

.. autoapifunction:: mmu.methods.clopper_pearson
.. autoapifunction:: mmu.methods.wilson
.. autoapifunction:: mmu.methods.jeffreys
.. autoapifunction:: mmu.methods.binary_metrics_confidence_intervals

``bmetrics_ci`` is an alias for ``binary_metrics_confidence_intervals``.
