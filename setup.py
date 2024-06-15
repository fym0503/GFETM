import os
from setuptools import setup, find_packages

BUILD_ID = os.environ.get("BUILD_BUILDID", "0")

setup(
    name="GFETM",
    version="0.1" + "." + BUILD_ID,
    # Author details
    author="Yimin Fan",
    packages=find_packages(),
    author_email="fanyimin@link.cuhk.edu.hk",
)

