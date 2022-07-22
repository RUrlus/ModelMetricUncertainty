"""Project: ModelMetricUncertainty

Created: 2021/09/24

Description:
    setup script to install ModelMetricUncertainty package.

Authors:
    Ralph Urlus [rurlus.dev@gmail.com]

Redistribution and use in source and binary forms, with or without
modification, are permitted according to the terms listed in the file
LICENSE.
"""

import sys
import pybind11
from setuptools import find_packages
from skbuild import setup

# NOTE: also update:
# CMakeLists.txt
# docs/source/conf.py

MAJOR = 0
REVISION = 1
PATCH = 0
# dev should be the dev version number (typically 0) or None if not dev version
DEV = None
# rc should be the release candidate version (typically 0) or None if not rc
RC = 4

FULL_VERSION = VERSION = f'{MAJOR}.{REVISION}.{PATCH}'
IS_RELEASE = False
if DEV is not None:
    FULL_VERSION += f'.dev{int(DEV)}'
elif RC is not None:
    FULL_VERSION += f'.rc{int(RC)}'
else:
    IS_RELEASE = True

def write_version_py(filename: str = 'mmu/version.py') -> None:
    """Write package version to version.py.

    This will ensure that the version in version.py is in sync with us.

    Parameters
    ----------
    filename : str
        the path the file to write the version.py

    """
    # Do not modify the indentation of version_str!
    version_str = (
        '"""THIS FILE IS AUTO-GENERATED BY ModelMetricUncertainty SETUP.PY."""\n\n'
        "name = 'mmu'\n"
        f"version = '{VERSION}'\n"
        f"full_version = '{FULL_VERSION}'\n"
        f"release = {IS_RELEASE}"

    )
    with open(filename, 'w') as version_file:
        version_file.write(version_str)

if __name__ == '__main__':
    write_version_py()
    cmake_args=[
        f"-DMMU_VERSION_INFO:STRING={VERSION}",
        f"-DPython3_EXECUTABLE:STRING={sys.executable}",
        f"-Dpybind11_DIR:STRING={pybind11.get_cmake_dir()}",
        "-DMMU_ENABLE_ARCH_FLAGS:BOOL=ON",
    ]
    if DEV is not None:
        cmake_args += [
            "-DMMU_DEV_MODE=ON",
            "-DMMU_ENABLE_INTERNAL_TESTS=ON",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON",
        ]

    setup(
        name="mmu",
        packages=find_packages(),
        version=FULL_VERSION,
        cmake_args=cmake_args
    )
