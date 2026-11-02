# MMU v0.2.0rc5

## Breaking changes

* BREAK: Remove the Bivariate-Normal (elliptical) uncertainty method and the
  `method=` keyword on the point/curve uncertainty estimators. The
  Multinomial profile-likelihood method (Wilks' theorem) is now the only
  supported approach for `PrecisionRecallUncertainty`, `ROCUncertainty` and
  their curve counterparts.
* BREAK: Remove `PrecisionRecallSimulatedUncertainty` and
  `ROCSimulatedUncertainty` and the underlying sampling-based error
  computation. Use the analytical Multinomial profile-likelihood classes
  instead.
* DEP: Bump minimum supported CPython version to 3.10 (drops 3.9 support).
* DEP: Bump minimum dependency versions: `numpy>=2.0.2`, `pandas>=2.3.3`,
  `scikit-learn>=1.6.1`, `pytest>=8.4.2`, `scikit-build-core>=0.12.2`.
* BLD: Restructure the package to build with `scikit-build-core` instead of
  the legacy `setup.py` + `scikit-build` combination; the C++ standard used
  to build the extension is now C++17.
* CHANGE: Use acronyms (TPR/FPR) for the ROC axis labels in plots instead of
  the full metric names.

## Features

* FEAT: Add joint uncertainty estimators for Recall vs. Proportion
  Predicted Negative (PPN): `PPNRecallUncertainty`/`PPNRU` for a single
  threshold/confusion matrix and `PPNRecallCurveUncertainty`/`PPNRCU` for
  the curve over multiple thresholds, both backed by a new C++ Multinomial
  profile-likelihood implementation.
* BLD: Modernise the CMake build, vendor OpenMP for wheels so multithreading
  works out-of-the-box without a system OpenMP install, and enable
  interprocedural/link-time optimisation (ITO/LTO) for Release builds.

## Fixes

* CHG: Implement more robust and consistent threshold validation/parsing in
  the confusion matrix and metrics functions as well as the uncertainty
  estimators; thresholds are no longer restricted to `[0, 1]` floats, so
  `decision_function`/logit-style scores are now supported.
* FIX: Prevent incorrect transposition of confusion matrices when a small
  number of observations is used.
* FIX: [C++] Avoid assuming contiguous strides when computing grid bounds
  for non-contiguous arrays.
* CHG: [C++] Fix and harden edge-case handling in the grid computation and
  around `fill` in the metrics functions.
* FIX: [C++] Fix a buffer overrun in the multi-threaded bounds computation.
* FIX: [C++] Fix a conflicting definition of `size_t` on 32-bit Windows.
* CHG: [C++] Centralise the default values of `n_sigmas` and `epsilon` used
  by the profile-likelihood grid so every estimator agrees on them.
* CHG: [C++] Consolidate and simplify the Multinomial profile-likelihood
  implementations shared across Precision-Recall, ROC and PPN-Recall.

## Performance

* PERF: [C++] Improve OpenMP scheduling/parallelisation of the profile
  log-likelihood grid computation and the accumulation of confusion
  matrices.
* PERF: [C++] Minor performance improvements in the Multinomial
  log-likelihood kernels and the confusion matrix computation.

## Chores

* CICD: Update the CI/CD suite for the new build system; vendor OpenMP and
  skip wheel builds for obscure architectures.
* CICD: Various `pypa/cibuildwheel`, `pypa/gh-action-pypi-publish`,
  `actions/checkout`, `actions/cache` and `actions/setup-python` version
  bumps.
* DOC: Add the paper URL and a `CITATION` file.

# MMU v0.1.2

## Fixes

* FIX: Pass non-numpy arrays onto sklearn's check arrray without assuming attributes @RUrlus in https://github.com/RUrlus/ModelMetricUncertainty/pull/56

## Chores

* CICD: Bump pypa/gh-action-pypi-publish from 1.6.1 to 1.6.4 by @dependabot in https://github.com/RUrlus/ModelMetricUncertainty/pull/55

# MMU v0.1.1

## Enhancements

* Adds support for Python 3.11
