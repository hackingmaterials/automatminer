#!/usr/bin/env python

from setuptools import setup, find_packages
import os

from automatminer import __version__

module_dir = os.path.dirname(os.path.abspath(__file__))
reqs_raw = open(os.path.join(module_dir, "requirements.txt")).read()
# reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]
reqs_list = [r for r in reqs_raw.split("\n")]


if __name__ == "__main__":
    setup(
        name='automatminer',
        version=__version__,
        description='automated machine learning for materials science',
        long_description="",
        url='https://github.com/hackingmaterials/automatminer',
        author=['Alex Dunn', 'Alex Ganose', 'Alireza Faghaninia', 'Qi Wang',
                'Anubhav Jain'],
        author_email='ardunn@lbl.gov',
        license='modified BSD',
        packages=find_packages(where=".", exclude=("benchdev", "benchdev.*")),
        package_data={},
        zip_safe=False,
        install_requires=reqs_list,
        extras_require={},
        classifiers=['Programming Language :: Python :: 3.6',
                     'Development Status :: 4 - Beta',
                     'Intended Audience :: Science/Research',
                     'Intended Audience :: System Administrators',
                     'Intended Audience :: Information Technology',
                     'Operating System :: OS Independent',
                     'Topic :: Other/Nonlisted Topic',
                     'Topic :: Scientific/Engineering'],
        test_suite='automatminer',
        tests_require='tests'
    )
