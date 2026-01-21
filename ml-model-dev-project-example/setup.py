#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = ""
__copyright__ = ""
__credits__ = [""]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = ""
__email__ = ""
__status__ = ""
__date__ = ""


# ======================================================================================================================
#   Libraries
# ======================================================================================================================
# ------- standard modules -------
import os
from setuptools import setup, find_packages

from pathlib import Path


# ======================================================================================================================
#   Functions
# ======================================================================================================================
# Function to retrieve resources files
def _get_resources(package_name: str) -> list:
    # Get all the resources (also on nested levels)
    res_paths: str = os.path.join(package_name, "resources")
    all_resources: list = [os.path.join(folder, file) for folder, _, files in os.walk(res_paths) for file in files]
    # Remove the prefix: start just from "resources"
    return [resource[resource.index("resources"):] for resource in all_resources]


# Read requirements
def _read_requirements() -> list:
    requirements_file: Path = Path(__file__).parent / "requirements.txt"
    with open(requirements_file, "r", encoding='utf-16') as f:
        requirements: [str] = f.read().splitlines()
    return requirements


# ======================================================================================================================
#   Package Creation
# ======================================================================================================================
# Package configuration
setup(name="package",
      version="0.0.0",
      description="FILLME",
      author="",
      author_email="",
      packages=find_packages(),
      package_data={"package": _get_resources("package"), "tests": ["stubs/*"]},
      install_requires=_read_requirements())
