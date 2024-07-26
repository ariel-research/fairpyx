import pathlib
import setuptools

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
print ("packages: ", packages)

setuptools.setup(
    name=NAME,
    version=VERSION,
    packages=setuptools.find_packages(),
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