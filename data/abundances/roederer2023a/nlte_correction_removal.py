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

# from PyAstronomy import pyasl

import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

import numpy as np
import scipy as sp
import pandas as pd
import re

import seaborn as sns
sns.set_palette("colorblind")
colors = sns.color_palette("colorblind", 20)

# from smh import Session

from spag.read_data import *
from spag.convert import *
from spag.utils import *
import spag.utils as spagu
from spag.calculate import *
import spag.read_data as rd
import spag.coordinates as coord

# import alexmods.read_data as rd

# script_dir = "/".join(IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]) + "/"
script_dir = os.path.dirname(os.path.realpath(__file__))+"/"
data_dir = '/Users/ayelland/Research/metal-poor-stars/spag/data/abundances/'

with open(os.path.join(script_dir, 'create-tables-0-date.txt'), 'r') as f:
    date = f.readline().strip()

plotting_dir = script_dir+f"plots/plots-{date}/"
table_dir = script_dir+f"tables/tables-{date}/"

#%%
abunds = pd.read_csv(data_dir + 'roederer2023a/table567_nlte.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
lines = pd.read_csv(data_dir + 'roederer2023a/table4.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

new_abunds = pd.DataFrame(columns=abunds.columns)
for star_name in abunds['Name'].unique():
    # star_name = 'J1010-0220'
    print(f"Processing {star_name} ...")
    star_abunds = abunds[abunds['Name'] == star_name]
    star_lines = lines[lines['Name'] == star_name]

    NLTE_corr_abunds = ['Li I', 'Na I', 'Mg I', 'Al I', 'Si I', 'K I', 'Fe I', 'Pb I']
    for ion in NLTE_corr_abunds:
        print("-------------------")
        
        # Current Abundance
        logepsX_curr = star_abunds[star_abunds['Ion'] == ion]['logepsX'].values[0]
        print(f'Curr {ion:<5}: ', normal_round(logepsX_curr, 2))

        # NLTE & LTE Abundance
        line_logepsX = np.array(star_lines[star_lines['Ion'] == ion]['logepsX'].values)
        line_corr = np.array(star_lines[star_lines['Ion'] == ion]['NLTEcorr'].values)
        lines_used = len(line_logepsX[~np.isnan(line_logepsX)])
        
        line_logepsX_NLTE = np.append(line_logepsX, line_corr)
        logepsX_NLTE = np.nansum(line_logepsX_NLTE) / lines_used
        print(f'NLTE {ion:<5}: ', normal_round(logepsX_NLTE, 2))
        
        logepsX_LTE = np.nansum(line_logepsX) / lines_used
        print(f'LTE  {ion:<5}: ', normal_round(logepsX_LTE, 2))
        
        # Set New Abundance as LTE value
        star_abunds.loc[star_abunds['Ion'] == ion, 'logepsX'] = normal_round(logepsX_LTE, 2)
        print(f'New  {ion:<5}: ', star_abunds[star_abunds['Ion'] == ion]['logepsX'].values[0])

    print("=========================================")
    new_abunds = pd.concat([new_abunds, star_abunds], ignore_index=True)

new_abunds = new_abunds[['Name', 'Ion', 'Z', 'logepsX_sun', 'N', 'l_logepsX', 'logepsX', 'e_logepsX']]
new_abunds.to_csv(data_dir + 'roederer2023a/table567_lte.csv', index=False)
    