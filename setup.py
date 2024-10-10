#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from setuptools import setup, find_packages
from codecs import open
from os import path, system
from re import compile as re_compile

# For convenience.
if sys.argv[-1] == "publish":
    system("python setup.py sdist upload")
    sys.exit()

def read(filename):
    kwds = {"encoding": "utf-8"} if sys.version_info[0] >= 3 else {}
    with open(filename, **kwds) as fp:
        contents = fp.read()
    return contents

# Get the version information.
here = path.abspath(path.dirname(__file__))
vre = re_compile("__version__ = \"(.*?)\"")
version = vre.findall(read(path.join(here, "__init__.py")))[0]

setup(
    name="spag",
    version=version,
    author="Alexander Yelland",
    author_email="ayelland@mit.edu",
    description="Stellar Patterns and Abundances in Galaxies (SPAG). \
        A science package for researching and analyzing stellar abundances in \
        galaxies created by Alexander Yelland.",
    long_description=read(path.join(here, "README.md")),
    long_description_content_type='text/markdown',  # Format of long description
    url="https://github.com/AYelland/spag.git",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    keywords="astronomy",
    packages=find_packages(exclude=["test"]),
    install_requires=[
        "numpy","pandas","astropy","scipy",
        "six","seaborn","pyyaml"#,"astro-gala"
        ],
    extras_require={
        #"test": ["coverage"]
    },
    package_data={
        "": ["LICENSE"]
    },
    include_package_data=True,
    data_files=None,
    entry_points=None
)