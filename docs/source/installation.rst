Installation
------------

``mmu`` can be installed from PyPi.

.. code-block:: bash

    pip install mmu

We provide wheels with pre-compiled binaries for:

* MacOS [x86, ARM]
* Linux
* Windows 

``MMU`` relies on a C++ extension for the computationally intensive routines or those that will be called over many items.

The extension is wrapped using Pybind11 and requires a compiler with support for C++14.
The compilation of the extension is triggered by the ``setup.py`` using Scikit-Build.

If you encounter an issue please rerun the installation with verbose mode on,
``pip install mmu -v``, and create an `issue <https://github.com/RUrlus/ModelMetricUncertainty/issues>`_ with the output.

Optimised build
+++++++++++++++

Installing the package from source requires a C++ compiler with support for C++14.
If you have a compiler available it is advised to install without the wheel as this enables architecture specific optimisations.

When building from source the extension is optimised for your architecture, e.g. .
Native architecture flags are enabled, ``-march=native``, ``-mtune=native`` and the CPU is checked for support of SSE3, SSE4, AVX and AX2.
You can install from PyPi with these optimisations enabled using:

.. code-block:: bash

    pip install mmu --no-binary mmu

If you are installing from source and want to disable these, e.g. when compiling for usage on a different architecture use:

.. code-block:: bash

   CMAKE_ARGS="-DMMU_ENABLE_ARCH_FLAGS=OFF" pip install mmu --no-binary mmu

Multithreading
++++++++++++++

The extension of ``mmu`` has extensive multithreading support through OpenMP.
If you install the package from source multithreading will be enabled automatically if OpenMP is found.

You can either explicitly enable OpenMP, which will now raise an exception if it cannot be found:

.. code-block:: bash

    CMAKE_ARGS="-DMMU_ENABLE_OPENMP=ON" pip install mmu --no-binary mmu
    # or if you want en

or explicitly disable it:

.. code-block:: bash

    CMAKE_ARGS="-DMMU_DISABLE_OPEMP=ON" pip install mmu --no-binary mmu
