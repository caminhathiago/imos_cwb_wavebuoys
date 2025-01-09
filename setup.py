#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

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

setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    description='Python package to streamline wavebuoy data to AODN',
    long_description=readme,
    author='Thiago Caminha',
    author_email='thiago.caminha@uwa.edu.au',
    url='',
    # install_requires=INSTALL_REQUIRES,
    packages=find_packages(exclude=PACKAGE_EXCLUDES),
    scripts=['wavebuoy/scripts/cwb-aodn-nrt.py'],
    #  entry_points={
    #     'console_scripts': [
    #         'cwb-aodn-nrt = wavebuoys.scripts.cwb-aodn-.py:main',  # Replace with actual module and function
    #     ]},
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
