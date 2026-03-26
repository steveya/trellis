import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="trellis",
    version="0.1.0",
    author="Steve Yang",
    author_email="steveya@gmail.com",
    description="AI-augmented quantitative pricing library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/steveya/trellis",
    license="http://www.apache.org/licenses/LICENSE-2.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "autograd>=1.7",
        "scipy>=1.10",
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
