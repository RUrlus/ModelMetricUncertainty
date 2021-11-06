from setuptools import setup

NAME = 'mmu_tests'


def setup_package() -> None:
    """
    The main setup method. It is responsible for setting up and installing the package.
    """
    setup(name=NAME,
        version='0.1.0',
        url='http://phik.kave.io',
        license='',
        author='Ralph Urlus',
        author_email='rurlus.dev@gmail.com',
        description='MMU test package',
        python_requires='>=3.5',
        packages=['mmu_tests'],
        install_requires=[]
    )


if __name__ == '__main__':
    setup_package()
