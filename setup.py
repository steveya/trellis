import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rate-model",
    version="0.1.0",
    author="Steve Yang",
    author_email="steveya@gmail.com",
    description="Rate Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://steveya.github.io/rate-model/",
    license="http://www.apache.org/licenses/LICENSE-2.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "aenum",
        "numpy",
    ],
    extras_require={
        "test": ["pytest", "pytest-cov"],
        "develop": ["wheel", "pytest", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Operating System :: OS Independent",
    ],
)
