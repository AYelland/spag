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
## Calculating the CEMP fraction

def calc_cemp_fraction(df, feh_limit=-2.0, cfe_limit=0.7):
    """
    Calculate the carbon fraction for a given DataFrame and [Fe/H] limit.
    """
    df_filtered = df[df['[Fe/H]'] <= feh_limit]
    n_CEMP = len(df_filtered[df_filtered['[C/Fe]f'] >= cfe_limit])
    n_tot = len(df_filtered)
    # print(n_CEMP, n_tot, feh_limit, cfe_limit)

    if n_tot == 0 and n_CEMP != 0:
        cemp_fraction = 1.0
    elif n_tot == 0 and n_CEMP == 0:
        cemp_fraction = -1
    else:
        cemp_fraction = n_CEMP / (n_tot)
        if cemp_fraction > 1.0: 
            cemp_fraction = 1.0
    # print(f"[Fe/H] <= {feh_limit}, [C/Fe] >= {cfe_limit}: {n_CEMP}/{n_tot} = {cemp_fraction:.2f}")

    if not np.isnan(cemp_fraction):
        cemp_fraction = int(normal_round(cemp_fraction * 100, 0))
    else:
        cemp_fraction = -1
        
    return cemp_fraction