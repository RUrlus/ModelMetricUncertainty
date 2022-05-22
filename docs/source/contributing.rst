Contributing
------------

We very much welcome contributions, contributing please make sure to follow the below conventions.

Installation
************

In order to contribute to ``mmu`` you
The package can be installed for local development with:

.. code-block:: bash

    pip install -e ..

Note that the C++ extension is (re-)compiled during local installation.
If you don't have a compiler installed and are not working on the extension the best solution is to download the wheels from PyPi.
Unzip the version for you system and Python version and copy it to ``mmu/lib/``.

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


Testing
*******

MMU's test suite can than be run with:
 
.. code-block:: bash

   cd tests
   pytest 

This will run both the API tests and the internal tests.
The internal tests are 


Conventions
***********

Carma

Pull requests
+++++++++++++

Please open pull-requests to the ``unstable`` branch.
Merges to ``stable`` are largely reserved for new releases.

We use a semi-linear merge strategy.
Please make sure that your branch/fork contains the latest state of ``unstable``, otherwise the merge cannot be completed.

Commits
+++++++

Please prepend your commit messages with:

* ``ENH: `` the commit introduces new functionality
* ``CHG: `` the commit changes existing functionality
* ``FIX: `` the commit fixes incorrect behaviour of existing functionality
* ``STY: `` the commit improves the readability of the code but not the functioning of the code
* ``DOC: `` the commit only relates to the documentation
* ``TST: `` the commit only relates to the tests
* ``BLD: `` changes related to setup files or build instructions (``BLD: [CPP] ...``)
* ``CICD: `` the commit only relates to the CI/CD templates

Additionally use ``[CPP]`` after the prefix when your commit touches C++ code or build instructions.

For example, ``FIX: [CPP] Fix stride offset in foo``.

Clang format
++++++++++++

If your contribution touches C++ code, please run clang-format.
The format is specified by the ``.clang-format`` file at the root of the repo.
It should be picked up automatically.
