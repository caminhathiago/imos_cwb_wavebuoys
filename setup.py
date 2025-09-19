#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# with open('README.md') as f:
#     readme = f.read()

def read_requirements(filename):
    with open(filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

PACKAGE_NAME = 'wavebuoys_dm'

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


setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    description='Python package to process SD card data to AODN',
    author='Thiago Caminha, Michael Cuttler',
    author_email='thiago.caminha@uwa.edu.au, michael.cuttler@uwa.edu.au',
    url='',
    install_requires=read_requirements("requirements.txt"),
    packages=find_packages(exclude=PACKAGE_EXCLUDES),
    scripts=SCRIPTS,
    package_data=PACKAGE_DATA,
    zip_safe=False,
    python_requires='>3.8'
)
