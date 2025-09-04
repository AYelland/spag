#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)
import  sys, os, glob, time, IPython

import astropy.constants as const
import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
from astropy.table import Table
# from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import SkyCoord, EarthLocation

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import pandas as pd

import seaborn as sns
sns.set_palette("colorblind")
colors = sns.color_palette("colorblind", 20)

from spag.read_data import *
from spag.convert import *
from spag.utils import *
from spag.calculate import *
import spag.read_data as rd
import spag.coordinates as coord

script_dir = "/".join(IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]) + "/"
# script_dir = os.path.dirname(os.path.realpath(__file__))+"/"
data_dir = '/Users/ayelland/Research/metal-poor-stars/spag/spag/data/'




mcconnachie2012_ref_df = pd.read_csv(data_dir+'galaxy_properties/mcconnachie2012/refs.csv')

mcconnachie2012_ref_df['year'] = ''
mcconnachie2012_ref_df['citation_key'] = ''
for i, row in mcconnachie2012_ref_df.iterrows():
    year = row['bibcode'][0:4] if pd.notna(row['bibcode']) else ''
    if pd.notna(row['author']):
        author = row['author']
        author = author.replace(' et al.', '')
        author = author.replace('de ', '')
        author = author.replace('da ', '')
        author = author.replace('van de ', '')
        author = author.replace('van der ', '')
        author = author.replace('van den ', '')
        if ' & ' in author:
            author = author.split(' & ')[0]
    if year != '' and pd.notna(row['author']):
        citation_key = author + '+' + year
    else:
        citation_key = author

    mcconnachie2012_ref_df.at[i, 'year'] = year
    mcconnachie2012_ref_df.at[i, 'citation_key'] = citation_key

mcconnachie2012_ref_df = mcconnachie2012_ref_df[['ref_number', 'bibcode', 'author', 'year', 'citation_key', 'comments']]
display(mcconnachie2012_ref_df)

mcconnachie2012_ref_df.to_csv(data_dir+'galaxy_properties/mcconnachie2012/refs_new.csv', index=False)

