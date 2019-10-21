"""Automatminer web edition. If you're looking to install
automatminer (the regular package), just use setup.py."""

from setuptools import setup, find_packages
import os

from automatminer_web import __version__

module_dir = os.path.dirname(os.path.abspath(__file__))
reqs_raw = open(os.path.join(module_dir, "requirements_web.txt")).read()
reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]

if __name__ == "__main__":
    setup(
        name='automatminer_web',
        description='a web app for automatminer',
        long_description="",
        url='https://github.com/hackingmaterials/automatminer',
        author=['Alex Dunn'],
        author_email='ardunn@lbl.gov',
        license='modified BSD',
        packages=find_packages(include="./automatminer_web"),
        package_data={},
        version=__version__,
        zip_safe=False,
        install_requires=reqs_list,
        extras_require={},
        classifiers=[]
    )
