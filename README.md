# Model-Metric-Uncertainty

**Note this library is under heavy development and not yet ready for production use.**

Model-Metric-Uncertainty (MMU) is a library for the evaluation of model performance and estimation of the uncertainty on these metrics.
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

## Installation

We currently do not provide wheels for the package.
Installation requires a C++ compiler with support for C++14 to be present.
Please refer to the instructions for your system on how to install a compiler.

```bash
cd ModelMetricUncertainty
pip install .

```

## Usage

```python3
import numpy as np
import mmu


# Create some example data
proba, y, yhat = mmu.generate_data(n_samples=1000)

# compute the confusion_matrix
conf_mat = mmu.confusion_matrix(y, yhat)
cm_dm = mmu.confusion_matrix_to_dataframe(conf_mat)

# alternatively compute the confusion matrix and metrics
conf_mat, metrics = mmu.binary_metrics(y, yhat)
cm_dm = mmu.confusion_matrix_to_dataframe(conf_mat)
mtr_dm = mmu.metrics_to_dataframe(metrics)

```
See [metrics_tutorial](https://github.com/RUrlus/ModelMetricUncertainty/blob/main/notebooks/metrics_tutorial.ipynb) for more examples.

## Contributing

We very much welcome contributions. When contributing please make sure to follow the below conventions.

### Installation

The package can be installed using `pip install -e .` for local development.
Note that the C++ extension is (re-)compiled during local installations.

If you only want to recompile the extension you can do so with:

```bash
cmake -G Ninja -S . -B mbuild \
 -DCMAKE_BUILD_TYPE=Release \
 -Dpybind11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())') \
 -DPython3_EXECUTABLE=$(python3 -c 'import sys; print(sys.executable)')

cmake --build mbuild --target install --config Release -j 4

```

### Testing

To run the test we need to install the test package:

```bash
cd tests
pip install -e .
```
Test can than be run with `pytest`.

### Conventions

#### Pull requests

Please open pull-requests to the `unstable` branch.
Merges to `main` are largely reserved for new releases.

We use a semi-linear merge strategy.
Hence you have to make sure that your branch/fork contains the latest state of `unstable`, otherwise the merge cannot be done.

#### Commits

Please prepend your commit messages with:

* `ENH: ` -- the commit introduces new functionality
* `CHG: ` -- the commit changes existing functionality
* `FIX: ` -- the commit fixes incorrect behaviour of existing functionality
* `STY: ` -- the commit improves the readability of the code but not the functioning of the code
* `DOC: ` -- the commit only relates to the documentation
* `TST: ` -- the commit only relates to the tests
* `CICD: ` -- the commit only relates to the CI/CD templates
* `BLD: ` -- changes related to setup files or build instructions (`BLD: [CPP] ...`)

Additionally use `[CPP]` after the prefix when your commit touches C++ code or build instructions.

For example, `FIX: [CPP] Fix stride offset in foo`.
