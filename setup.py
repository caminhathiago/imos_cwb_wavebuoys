#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

common_reqs = read_requirements("requirements.txt")
nrt_reqs = common_reqs + read_requirements("requirements-nrt.txt")
dm_reqs = common_reqs + read_requirements("requirements-dm.txt")

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

PACKAGE_INCLUDES = [
    "wavebuoy_nrt*",
    "wavebuoy_dm*"
]

PACKAGE_DATA = {
    'wavebuoy.': ['*.csv'],
}

setup(
    name='imos-cwb-wavebuoys',
    version='0.1.0',
    description='Toolboxes for near real-time and delayed-mode wave buoy data processing',
    author='Thiago Caminha, Michael Cuttler',
    author_email='thiago.caminha@uwa.edu.au, michael.cuttler@uwa.edu.au',
    url='',
    install_requires=common_reqs,
    extras_require={"wavebuoy_nrt": nrt_reqs, "wavebuoy_dm": dm_reqs},
    packages=find_packages(include=PACKAGE_INCLUDES, exclude=PACKAGE_EXCLUDES),
    package_data=PACKAGE_DATA,
    zip_safe=False,
    python_requires='>3.8'
)
