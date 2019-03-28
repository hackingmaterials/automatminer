#!/usr/bin/env python

from setuptools import setup, find_packages
import os

from benchdev import __version__

"""Bench-dev (HT-benchmarking) package. If you're looking to install 
automatminer (the regular package), just use setup.py."""

module_dir = os.path.dirname(os.path.abspath(__file__))
reqs_raw = open(os.path.join(module_dir, "requirements_dev.txt")).read()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

if __name__ == "__main__":
    setup(
        name='benchdev',
        version=__version__,
        description='benchmarking infrastructure for automatminer',
        long_description="",
        url='https://github.com/hackingmaterials/automatminer',
        author=['Alex Dunn'],
        author_email='ardunn@lbl.gov',
        license='modified BSD',
        packages=find_packages(include="./dev"),
        package_data={},
        zip_safe=False,
        install_requires=reqs_list,
        extras_require={},
        classifiers=[])
