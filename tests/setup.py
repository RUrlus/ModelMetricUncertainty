import re
import sys
from pathlib import Path
from setuptools import setup
from cmake_build_extension import BuildExtension, CMakeExtension

NAME = 'mmu_tests'


if __name__ == '__main__':
    setup(
        name=NAME,
        version='0.1.0',
        author='Ralph Urlus',
        author_email='rurlus.dev@gmail.com',
        description='MMU test package',
        python_requires='>=3.5',
        packages=['mmu_tests'],
        install_requires=[
            'scikit-learn>=1.0',
        ],
        ext_modules=[
            CMakeExtension(
                name="CMakeProject",
                install_prefix="tests/mmu_tests/lib",
                cmake_depends_on=["pybind11"],
                disable_editable=False,
                cmake_configure_options=[
                    f"-DPython3_EXECUTABLE:STRING={sys.executable}",
                ],
            )
        ],
        cmdclass=dict(build_ext=BuildExtension),
    )
    # Fix CMake cache issue with in-place builds
    cmake_cache_path = (Path(__file__).resolve().parent / "build")
    pip_env_re = "^//.*$\n^[^#].*pip-build-env.*$"
    for i in cmake_cache_path.rglob("CMakeCache.txt"):
        i.write_text(re.sub(pip_env_re, "", i.read_text(), flags=re.M))
