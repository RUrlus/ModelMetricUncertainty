.. role:: bash(code)
   :language: bash

=========
Extension
=========

`MMU` relies on a C++ extension for the computationally intensive routines or
those that will be called over many items.

Building
********

The extension is wrapped using Pybind11_ and requires a compiler with support
for C++14.
The compilation of the extension is triggered by the ``setup.py`` using
Scikit-Build_.
We provide wheels for all major OS's, architectures and CPython versions, hence
building from source should be uncommon for users.
If you encounter an issue please rerun the installation with verbose mode on,
``pip install mmu -v``, and create an `issue <https://github.com/RUrlus/ModelMetricUncertainty/issues>`_ with the output.

Debug build
+++++++++++

By default we create a CMake ``Release`` build, if you need to create a
``Debug`` build use:

.. code-block:: bash

   CMAKE_ARGS="CMAKE_BUILD_TYPE=Debug" pip install -e .

Independent build
+++++++++++++++++

When you are working on the extension, it can be beneficial to compile the
extension directly rather than through ``setup.py``.
Unfortunately building through the ``setup.py`` does not allow for incremental
builds.

If you have not installed the package in develop/editable mode, you should do so
first. Also make sure you have all the build requirements installed in your virtual
environment, these can be found in ``pyproject.toml`` file.

.. code-block:: bash

    cmake -S . -G Ninja -B mbuild \
        -DCMAKE_BUILD_TYPE=Release \
        -DMMU_ENABLE_INTERNAL_TESTS=ON \
        -DMMU_DEV_MODE=ON \
        -DPython3_EXECUTABLE=$(python3 -c 'import sys; print(sys.executable)') \
        -Dpybind11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())') \
        -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
    
    cmake --build mbuild --target install --config Release -j 4

Extension specific tests
++++++++++++++++++++++++

The library contains unit tests for the internals of the C++ extension.
For example, unit tests of the functions in ``include/mmu/numpy.hpp``.
These are specific tests that have their own extension which can be build using:

.. code-block:: bash

   CMAKE_ARGS="-DMMU_ENABLE_INTERNAL_TESTS=ON" pip install -e .

This will place ``_mmu_core_tests`` extension in ``mmu/lib``.
You can run the tests with

.. code-block:: bash

   pytest tests/internal

Extension API
*************

The core functions are wrapped using Pybind11_ and are callable from
``mmu.lib._mmu_core``.

.. note::
    This section is intended mainly intended for contributors of MMU or those
    that know what they are doing.
    ``_mmu_core`` is not strictly speaking part of the public API although we
    intend to keep it stable.

.. warning::

    The functions in ``_mmu_core`` perform the bare minimum of condition checks
    on the input arrays to prevent buffer overruns and segfaults.
    The function docstrings indicate what conditions are assumed, violating
    these is likely to result in incorrect results and or poor performance.

Confusion Matrix
++++++++++++++++

.. function:: template <typename T1, typename T2> py::array_t<int64_t> confusion_matrix(const py::array_t<T1>& y, const py::array_t<T2>& yhat)

    Where ``T1`` and ``T2`` are one of ``bool``, ``int``, ``int64_t``,
    ``float`` or ``double``.

    Compute the confusion matrix given true labels y and estimated labels yhat.

    :param y: true labels
    :param yhat: predicted labels
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    The arrays are assumed to be one-dimensional, contiguous in
    memory and have equal size. The size of the smallest array is used.

.. function:: template <typename T1, typename T2> py::array_t<int64_t> confusion_matrix_score(const py::array_t<T1>& y, const py::array_t<T2>& score, const T2 threshold)

    Where ``T1`` is one of ``bool``, ``int``, ``int64_t``, ``float``, ``double``
    and ``T2`` is ``float`` or ``double``.

    Compute the confusion matrix given true labels ``y`` and classifier scores
    ``score``.
    
    :param y: true labels
    :param score: classifier scores
    :param threshold: inclusive classification threshold
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    The arrays are assumed to be one-dimensional, contiguous in
    memory and have equal size. The size of the smallest array is used.

.. function:: template <typename T1, typename T2> py::array_t<int64_t> confusion_matrix_runs(const py::array_t<T1>& y, const py::array_t<T2>& yhat)

    Where ``T1`` and ``T2`` are one of ``bool``, ``int``, ``int64_t``,
    ``float`` or ``double``.

    Compute the confusion matrix given true labels y and estimated labels yhat.

    :param y: true labels
    :param yhat: predicted labels
    :param obs_axis: the axis containing the observations beloning to the same run, e.g. 0 when a column contains the scores/labels for a single run.
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    The arrays are assumed to be two-dimensional, contiguous in
    memory and have equal size. The size of the smallest array is used.

.. function:: template <typename T1, typename T2> py::array_t<int64_t> confusion_matrix_score_runs(const py::array_t<T1>& y, const py::array_t<T2>& score, const T2 threshold, const int obs_axis)

    Where ``T1`` is one of ``bool``, ``int``, ``int64_t``, ``float``, ``double``
    and ``T2`` is ``float`` or ``double``.

    Compute the confusion matrix given true labels ``y`` and classifier scores
    ``score``.
    
    :param y: true labels
    :param score: classifier scores
    :param threshold: inclusive classification threshold
    :param obs_axis: the axis containing the observations beloning to the same run, e.g. 0 when a column contains the scores/labels for a single run.
    :exception ``runtime_error``: if an array is not aligned or not contiguous.

    The arrays are assumed to be two-dimensional, contiguous in
    memory and have equal size. The size of the smallest array is used.


.. _pybind11: https://pybind11.readthedocs.io/en/stable/#
.. _scikit-build: https://scikit-build.readthedocs.io/en/latest/index.html
