import pathlib
import setuptools
from setuptools import Extension
from Cython.Build import cythonize
import numpy as np

NAME = "fairpyx"
URL = "https://github.com/ariel-research/" + NAME
HERE = pathlib.Path(__file__).parent
print(f"\nHERE = {HERE.absolute()}\n")
README = (HERE / "README.md").read_text()
REQUIRES = (HERE / "requirements.txt").read_text().strip().split("\n")
REQUIRES = [lin.strip() for lin in REQUIRES]
print(f'\nVERSION = {(HERE / NAME / "VERSION").absolute()}\n')
VERSION = (HERE / NAME / "VERSION").read_text().strip()
# See https://packaging.python.org/en/latest/guides/single-sourcing-package-version/

packages = setuptools.find_packages()
packages.append('fairpyx.zalternatives.yekta_day_impl')
print ("packages: ", packages)

# Define Cython extensions
extensions = [
    Extension(
        name="fairpyx.algorithms.second_improved_high_multiplicity",
        sources=["fairpyx/algorithms/second_improved_high_multiplicity.pyx"],
        include_dirs=[np.get_include()]
    )
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    packages=packages,
    install_requires=REQUIRES,
    author="Erel Segal-Halevi",
    author_email="erelsgl@gmail.com",
    description="Fair division algorithms in Python",
    keywords="fair division algorithms",
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={
        "Documentation": URL,
        "Source Code": URL,
        "Bug Reports": f"{URL}/issues",
    },
    python_requires=">=3.9",
    include_package_data=True,
    classifiers=[
        # see https://pypi.org/classifiers/
        "Development Status :: 1 - Planning",
        # "Development Status :: 2 - Pre-Alpha",
        # "Development Status :: 3 - Alpha",
    ],
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
)

# Build:
#   Delete old folders: build, dist, *.egg_info, .venv_test.
#   Then run:
#        pip install build
#        python -m build
#   Or (old version):
#        python setup.py sdist bdist_wheel

# Publish to test PyPI:
#   twine upload --repository testpypi dist/*

# Publish to real PyPI (make sure you set the environment variables TWINE_USERNAME and TWINE_PASSWORD):
#   twine upload --repository pypi dist/*
