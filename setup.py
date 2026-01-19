import os
import shutil
import subprocess
import sys
from pathlib import Path

import setuptools
import versioneer

if "sdist" in sys.argv:
    reference = os.path.dirname(__file__)
    doc_dir = os.path.join(reference, "docs")
    p = subprocess.Popen("sphinx-build -M help . _build", cwd=doc_dir, shell=True)
    p.wait(30)
    if p.returncode != 0:
        raise RuntimeError("unable to make docs")

    generated_dir = Path(os.path.join(doc_dir, "_build", "html", "functions"))
    generated_dir.mkdir(parents=True, exist_ok=True)
    target_dir = os.path.join(reference, "rate_model", "docs")
    shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(generated_dir, target_dir)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gs_quant",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Steve Yang",
    author_email="steveya@gmail.com",
    description="Rate Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://steveya.github.io/rate-model/",
    license="http://www.apache.org/licenses/LICENSE-2.0",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "aenum",
        "backoff",
        "cachetools",
        "certifi",
        "dataclasses;python_version<'3.7'",
        "contextvars;python_version<'3.7'",
        "dataclasses_json",
        "deprecation",
        "funcsigs",
        "inflection",
        "lmfit",
        "more_itertools",
        "msgpack",
        "nest-asyncio",
        "opentracing",
        "pandas>1.0.0,<2.0.0;python_version<'3.7'",
        "pandas>=1.4,<2.2;python_version>'3.7'",
        "pydash<7.0.0",
        "python-dateutil>=2.7.0",
        "requests",
        "httpx>=0.23.3;python_version>'3.6'",
        "scipy>=1.2.0;python_version>'3.8'",
        "scipy>=1.2.0,<1.6.0;python_version<'3.7'",
        "scipy>=1.2.0,<1.8.0;python_version<'3.8'",
        "statsmodels<=0.12.2;python_version<'3.7'",
        "statsmodels>=0.13.0;python_version>'3.6'",
        "tqdm",
        "typing;python_version<'3.7'",
        "websockets"
    ],
    extras_require={
        "notebook": ["jupyter", "matplotlib", "seaborn", "treelib"],
        "test": ["pytest", "pytest-cov", "pytest-mock", "pytest-ordering", "testfixtures", "nbconvert", "nbformat",
                 "jupyter_client", "pipreqs"],
        "develop": ["wheel", "sphinx", "sphinx_rtd_theme", "sphinx_autodoc_typehints", "pytest", "pytest-cov",
                    "pytest-mock", "pytest-ordering", "testfixtures"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)