from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))
reqs_raw = open(os.path.join(module_dir, "requirements.txt")).read()
# reqs_list = [r.replace("==", ">=") for r in reqs_raw.split("\n")]
reqs_list = [r for r in reqs_raw.split("\n")]

# Version is MAJOR.MINOR.PATCH.YYYYMMDD
version = "1.0.3.20200727"

if __name__ == "__main__":
    setup(
        name='automatminer',
        version=version,
        description='automated machine learning for materials science',
        long_description="Automated machine learning for materials science. "
                         "https://github.com/hackingmaterials/automatminer",
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
        tests_require='tests',
        include_package_data=True
    )
