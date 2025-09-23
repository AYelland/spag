#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import pkg_resources
import  sys, os, glob, time
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits

from spag.convert import *
from spag.utils import *
import spag.coordinates as coord

sns.set_palette("colorblind")
sns_palette = sns.color_palette()

################################################################################
## Directory Variables

# script_dir = "/".join(IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]) + "/" # use this if in ipython
script_dir = os.path.dirname(os.path.realpath(__file__))+"/" # use this if not in ipython (i.e. terminal script)
# script_dir = pkg_resources.resource_filename(__name__)
data_dir = script_dir+"data/"
plots_dir = script_dir+"plots/"
linelist_dir = script_dir+"linelists/"

################################################################################
## Calculating the Carbon Corrections=, using Placco et al. 2014 website
# https://vplacco.pythonanywhere.com/

import requests
from bs4 import BeautifulSoup

def calc_carbon_correction(logg, feh, cfe):
    payload = {
        'lgg': str(logg),
        'feh': str(feh),
        'cfe': str(cfe)
    }

    URL = 'https://vplacco.pythonanywhere.com'  # example: 'http://vmplacco.pythonanywhere.com'
    session = requests.Session()
    response = session.post(URL, data=payload)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    correction_tag = soup.find('pre', style=lambda s: s and 'font-size: 30px' in s)
    if correction_tag:
        correction = str(correction_tag.text.strip())
        correction = float(correction.split(' ')[4])
    else:
        correction = None
    return correction


################################################################################
## Calculating the CEMP fraction

def calc_cemp_fraction(df, feh_limit=-2.0, cfe_limit=0.7):
    """
    Calculate the carbon fraction for a given DataFrame and [Fe/H] limit.

    Returns: (cemp_fraction, n_CEMP, n_tot)
    """
    df_filtered = df[(df['[Fe/H]'] <= feh_limit) | (df['ul[Fe/H]'] <= feh_limit)]
    
    ## n_CEMP = (all measured values) + (lower limits above the cfe threshold)
    n_CEMP = len(df_filtered[df_filtered['[C/Fe]f'] > cfe_limit])
    n_CEMP += len(df_filtered[(df_filtered['ll[C/Fe]f'].notna()) & (df_filtered['ll[C/Fe]f'] >= cfe_limit-0.2)]) # lower limits
    
    ## n_tot = (all measured values) + (lower limits above the cfe threshold) + (upper limits below the cfe threshold)
    n_tot = len(df_filtered[df_filtered['[C/Fe]f'].notna()]) # real data values
    n_tot += len(df_filtered[(df_filtered['ll[C/Fe]f'].notna()) & (df_filtered['ll[C/Fe]f'] >= cfe_limit-0.2)]) # lower limits
    n_tot += len(df_filtered[(df_filtered['ul[C/Fe]f'].notna()) & (df_filtered['ul[C/Fe]f'] <= cfe_limit+0.2)]) # upper limits
    
    if n_tot == 0 and n_CEMP == 0:
        cemp_fraction = -1
    elif n_tot == 0 and n_CEMP != 0:
        raise ValueError("n_tot is 0 but n_CEMP is not 0, which should not be possible.")
    else:
        cemp_fraction = n_CEMP / (n_tot)
        if cemp_fraction > 1.0: 
            raise ValueError("CEMP fraction is greater than 1.0, which should not be possible.")

    # print(n_CEMP, n_tot, feh_limit, cfe_limit)
    # print(f"[Fe/H] <= {feh_limit}, [C/Fe] >= {cfe_limit}: {n_CEMP}/{n_tot} = {cemp_fraction:.2f}")

    if not np.isnan(cemp_fraction):
        cemp_fraction = int(normal_round(cemp_fraction * 100, 0))
    else:
        cemp_fraction = -1
        
    return cemp_fraction, n_CEMP, n_tot

################################################################################
## Calculating Dtrans 

def calc_dtrans(ch):
    """
    Calculate Dtrans values for the given [C/H] value.
    """
    
    assert isinstance(ch, (float, int)), "Input ch must be a float or int"

    # [C/O] values, representing the delta (dex) between C and O abundances
    co_lower = 0.0 
    co_upper = -0.6

    oh_l = ch - co_lower
    oh_u = ch - co_upper

    Dtrans_l = np.log10(10**ch + (0.9 * 10**oh_l))
    Dtrans_u = np.log10(10**ch + (0.9 * 10**oh_u))

    return (Dtrans_l, Dtrans_u)

def calc_dtrans_line(feh):
    """
    Calculate Dtrans values for the solar abundances, scaled by metallicity 
    from a given [Fe/H] value (or array of values).
    This function creates the diagonal line in the Dtrans vs [Fe/H] plot.
    """
    ch = 0.0 # [C/H] = 0.0, the solar ratio for carbon

    # [C/O] values, representing the delta (dex) between C and O abundances
    co_lower = 0.0 
    co_upper = -0.6

    oh_l = ch - co_lower # [C/H] = 0.0
    oh_u = ch - co_upper # [C/H] = 0.6

    Dtrans_l = np.log10(10**(ch+feh) + (0.9 * 10**(oh_l+feh)))
    Dtrans_u = np.log10(10**(ch+feh) + (0.9 * 10**(oh_u+feh)))

    return Dtrans_l , Dtrans_u
    
def calc_dtrans_columns(df, precision=2):
    """
    Calculate Dtrans values for the given dataframe. The dataframe must have
    either a [C/H]f or ul[C/H]f column. 
    
    The function will add four new columns to the dataframe:
        Dtrans_l: lower [O/H] value used when calculating Dtrans (using co_lower)
        Dtrans_llim: lower [O/H] value used when calculating Dtrans (using co_lower) for upper limits
        Dtrans_u: upper [O/H] value used when calculating Dtrans (using co_upper)
        Dtrans_ulim: upper [O/H] value used when calculating Dtrans (using co_upper) for upper limits
    """
    
    # [C/O] values, representing the delta (dex) between C and O abundances
    co_lower = 0.0 
    co_upper = -0.6

    def dtrans(ch, oh, precision=precision):
        return normal_round(np.log10(10**ch + (0.9 * 10**oh)), precision)
    
    for i, row in df.iterrows():
        if not pd.isna(row['[C/H]f']):
            ch = float(row['[C/H]f'])

            oh_l = ch - co_lower
            Dtrans_l = dtrans(ch, oh_l)
            df.loc[i, 'Dtrans_l'] = Dtrans_l
            df.loc[i, 'Dtrans_llim'] = np.nan

            oh_u = ch - co_upper
            Dtrans_u = dtrans(ch, oh_u)
            df.loc[i, 'Dtrans_u'] = Dtrans_u
            df.loc[i, 'Dtrans_ulim'] = np.nan

        elif not pd.isna(row['ul[C/H]f']):
            ch = float(row['ul[C/H]f'])

            oh_l = ch - co_lower
            Dtrans_l = dtrans(ch, oh_l)
            df.loc[i, 'Dtrans_l'] = np.nan
            df.loc[i, 'Dtrans_llim'] = Dtrans_l
            
            oh_u = ch - co_upper
            Dtrans_u = dtrans(ch, oh_u)
            df.loc[i, 'Dtrans_u'] = np.nan
            df.loc[i, 'Dtrans_ulim'] = Dtrans_u

        else:
            print(f"Row {i} does not have [C/H]f or ul[C/H]f: {row['Name']}")

    ## Add the columns to the dataframe, if they do not exist already
    for col in ['Dtrans_u', 'Dtrans_ulim', 'Dtrans_l', 'Dtrans_llim']:
        if col not in df.columns:
            df[col] = np.nan

    return df