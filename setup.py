#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# with open('README.md') as f:
#     readme = f.read()

def read_requirements(filename):
    with open(filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

PACKAGE_NAME = 'wavebuoys'

PACKAGE_EXCLUDES = [
    "config",
    "dep_files",
    "incoming_path",
    "notebooks",
    "output_path",
    "pipeline_design",
    "wavebuoy/sofar",
    "netcdf"
]

PACKAGE_DATA = {
    'wavebuoy.': ['*.csv'],
}

SCRIPTS = []
    # 'scripts/cwb-spotter-netcdf.py',
    # 'scripts/cwb-netcdf-aodn.py']

setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    description='Python package to process SD card data to AODN',
    # long_description=readme,
    author='Thiago Caminha, Michael Cuttler',
    author_email='thiago.caminha@uwa.edu.au, michael.cuttler@uwa.edu.au',
    url='',
    install_requires=read_requirements("requirements.txt"),
    packages=find_packages(exclude=PACKAGE_EXCLUDES),
    scripts=SCRIPTS,
    # entry_points={
    #     'console_scripts': [
    #         'cwb-spotter-netcdf=scripts.cwb_spotter_netcdf:main',
    #     ]
    # },
    package_data=PACKAGE_DATA,
    # test_suite='test_ardc_nrt',
    # tests_require=TESTS_REQUIRE,
    # extras_require=EXTRAS_REQUIRE,
    zip_safe=False,
    python_requires='>3.8'
    # classifiers=[
    #     'Development Status :: 5 - Production/Stable',
    #     'Intended Audience :: Developers',
    #     'Natural Language :: English',
    #     'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    #     'Programming Language :: Python',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.8',
    #     'Programming Language :: Python :: Implementation :: CPython',
    # ]
)
