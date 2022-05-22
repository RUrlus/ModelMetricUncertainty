.. role:: bash(code)
   :language: bash

`MMU` relies on a C++ extension for the computationally intensive routines or
those that will be called over many items.

Build configuration
-------------------

The extension is wrapped using Pybind11_ and requires a compiler with support for C++14.
The compilation of the extension is triggered by the ``setup.py`` using Scikit-Build_.
We provide wheels for all major OS's, architectures and CPython versions, hence building from source should be uncommon for users.
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

.. _pybind11: https://pybind11.readthedocs.io/en/stable/#
.. _scikit-build: https://scikit-build.readthedocs.io/en/latest/index.html
