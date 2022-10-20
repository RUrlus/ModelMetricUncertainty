Contributing
------------

We very much welcome contributions.
Before starting work on a big PR it is always good to check if there is interest
and or how this would fit with existing development activities.

Installation
************

In order to contribute to ``mmu`` you have to install the package from source.
The package can be installed for local development with:

.. code-block:: bash
    
    git clone https://github.com/RUrlus/ModelMetricUncertainty.git
    cd modelmetricuncertainty
    pip install -e .

Note that the C++ extension is (re-)compiled during local installation.
If you don't have a compiler installed and are not working on the extension the best solution is to download the wheels from PyPi.
Unzip the version for you system and Python version and copy it to ``mmu/lib/``.

Debug build
+++++++++++

By default we create a CMake ``Release`` build, if you need to create a
``Debug`` build use:

.. code-block:: bash

   CMAKE_ARGS="CMAKE_BUILD_TYPE=Debug" pip install -e .

Manual compilation
++++++++++++++++++

If you only want to recompile the extension you can do so with:

.. code-block:: bash

    cmake -S . -B mbuild \
        -DCMAKE_BUILD_TYPE=Release \
        -DMMU_DEV_MODE=ON \
        -DMMU_ENABLE_INTERNAL_TESTS=ON \
        -Dpybind11_DIR=$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())') \
        -DPython3_EXECUTABLE=$(python3 -c 'import sys; print(sys.executable)')
    
    cmake --build mbuild --target install --config Release -j 4

This requires that you have the following pip packages installed:

* cmake
* scikit-build
* pybind11
* numpy

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

Testing
*******

MMU's test suite can than be run with:
 
.. code-block:: bash

   cd tests
   pytest 

This will run both the API tests and the internal tests.
In order to run the internal tests you need to have compiled the test extension.
Please see :ref:`Extension Specific Tests` for details.

If you have not build the extension tests, you can test the API tests with:

.. code-block:: bash

   cd tests
   pytest api

CICD
++++

There are a number of Github Actions that are triggered on pull requests:

- Windows - Python 3.7 - 3.10
- Linux - Python 3.7 - 3.10
- MacOS - Python 3.7 - 3.10
- Valgrind
- Asan

A lighter Build & Test pipeline is run on each push, this should help you check
compatibility across the platforms.
Valgrind and Asan are only relevant if you have touched the C++ extension and
can be largely ignored otherwise.

Conventions
***********

We have a few conventions that we would like you to adhere to.

Pull requests
+++++++++++++

Please open pull-requests to the ``stable`` branch.

We use a linear merge strategy.
Please make sure that your branch/fork contains the latest state of ``stable``, otherwise the merge cannot be completed.

Commits
+++++++

Please prepend your commit messages with:

* ``ENH:`` the commit introduces new functionality
* ``CHG:`` the commit changes existing functionality
* ``FIX:`` the commit fixes incorrect behaviour of existing functionality
* ``STY:`` the commit improves the readability of the code but not the functioning of the code
* ``DOC:`` the commit only relates to the documentation
* ``TST:`` the commit only relates to the tests
* ``BLD:`` changes related to setup files or build instructions (``BLD: [CPP] ...``)
* ``CICD:`` the commit only relates to the CI/CD templates

Additionally use ``[CPP]`` after the prefix when your commit touches C++ code or build instructions.

For example, ``FIX: [CPP] Fix stride offset in foo``.

Clang format
++++++++++++

If your contribution touches C++ code, please run clang-format.
The format is specified by the ``.clang-format`` file at the root of the repo, it should be picked up automatically.
For example:

.. code-block::

   clang-format -i include/mmu/core/*.hpp

.. _pybind11: https://pybind11.readthedocs.io/en/stable/#
.. _scikit-build: https://scikit-build.readthedocs.io/en/latest/index.html

Building documentation
++++++++++++++++++++++

You can build documentation like this:

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   sphinx-build source build
   open build/index.html
