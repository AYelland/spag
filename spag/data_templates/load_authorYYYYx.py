#!/usr/bin/env python
# -*- coding: utf-8 -*-

###############################################################################
# This file is a example/template for the `read_data.py` script, that loads the
# data from a specific paper and reformats it into the standard format used by
# SPAG. To use this, copy the following function into a testing script or
# Jupyter notebook, and modify the function name and contents as needed.


# Specifically, change all instances of...
#
# - `authorYYYYx` to the first author's last name (maybe with their first 
# initial of there is a duplicate last name) and year of the paper. When there
# are instances of an author releasing more than one paper in a year, append a
# letter to the year in `x` position.
#   (e.g., `Ji et al. 2016a` --> `ji2016a`, if there is also a 
#       `Ji et al. 2016b`)
#   (e.g., `Hansen et al. 2018` --> `hansent2018c`, if there are multiple
#       authors with the same last name and this was T. Hansen's 3rd paper in 
#       2018)
#
# - the path to the data files to point to the correct location of the data 
#   tables.
#   (e.g., `data_dir + "abundance_tables/authorYYYYx/table1.csv"`)
#
# - The `Reference` and `Ref` columns to match the citation for the paper.
#   --> `Reference` column should be in the form 'Author+Year'.
#       (e.g., 'Ji+2016a', 'HansenT+2018c')
#   --> `Ref` column should shortened form of the reference, typically the 
#       first 3 letters of the author's last name (all uppercase) and the last
#       two digits of the year, with a letter appended if there are multiple
#       papers by the same author(s) in that year. (e.g.'JI16a', 'HANt18c')
#
# - The `Loc` column to match the type of object the stars are in. Use:
#   --> 'HA' for halo stars
#   --> 'BU' for bulge stars
#   --> 'DS' for disk stars
#   --> 'DW' for dwarf galaxy stars
#   --> 'UF' for ultra-faint dwarf galaxy stars
#   --> 'GC' for globular cluster stars
#
# - The `System` column to match the name of the system the stars are in, if 
#   applicable and not already in the data table.


# Beyond this, confirm that the datafiles are reading in correctly, and that
# the columns are being filled in properly. You may need to modify how the data 
# is read in or the structure of the datafile themselves, depending on the 
# format of the data tables.
# 
# Things to be wary of...
#
# - Some papers use different formats for the species names, such as "Fe I" vs
#   "FeI" vs "Fe1". The `ion_to_species()` function should be able to handle 
#   most of these, but you may need to modify the datafile if there are any 
#   issues, or contact A. Yelland.
#
# - Some papers may not provide all the necessary columns, such as separate
#   upper/lower limit flag column. You may need to modify the datafile to add
#   the flag columns.
#
# - Additionally, different authors at different times may use different solar
#   abundance scales. The `get_solar()` function can be used to retrieve the
#   solar abundances of the most commonly used scales, but if one is not
#   available, please let A. Yelland know.


############################################################################################

from __future__ import (division, print_function, absolute_import, unicode_literals)

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

# script_dir = "/".join(IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]) + "/" # use this if in ipython
script_dir = os.path.dirname(os.path.realpath(__file__))+"/" # use this if not in ipython (i.e. terminal script)
data_dir = script_dir+"../data/"
plots_dir = script_dir+"../plots/"
linelist_dir = script_dir+"../linelists/"


def load_authorYYYYx(io=None):
    """
    Load the Author et al. YYYYx data for the <grouping or classification of stars>.

    Table 1 - Observations
    Table 2 - Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/authorYYYYx/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/authorYYYYx/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/authorYYYYx/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    ## Make the new column names
    species = []
    for ion in abund_df["Species"].unique():
        species_i = ion_to_species(ion)
        elem_i = ion_to_element(ion)
        if species_i not in species:
            species.append(species_i)

    epscols = [make_epscol(s) for s in species]
    ulcols = [make_ulcol(s) for s in species]
    XHcols = [make_XHcol(s).replace(" ", "") for s in species]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(" ", "") for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    authorYYYYx_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        authorYYYYx_df.loc[i,'Name'] = name
        authorYYYYx_df.loc[i,'Simbad_Identifier'] = name
        authorYYYYx_df.loc[i,'Reference'] = 'Author+YYYYx'
        authorYYYYx_df.loc[i,'Ref'] = 'AUTYYx'
        authorYYYYx_df.loc[i,'I/O'] = 1
        authorYYYYx_df.loc[i,'Loc'] = 'UF' # [HA, BU, DS, DW, UF, GC]
        authorYYYYx_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        authorYYYYx_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        authorYYYYx_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(authorYYYYx_df.loc[i,'RA_hms'], precision=6)
        authorYYYYx_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        authorYYYYx_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(authorYYYYx_df.loc[i,'DEC_dms'], precision=2)
        authorYYYYx_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        authorYYYYx_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        authorYYYYx_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        authorYYYYx_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                authorYYYYx_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                authorYYYYx_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    authorYYYYx_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    authorYYYYx_df.loc[i, 'ul'+col] = np.nan
                else:
                    authorYYYYx_df.loc[i, col] = np.nan
                    authorYYYYx_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    authorYYYYx_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    authorYYYYx_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    authorYYYYx_df.loc[i, 'ul'+col] = np.nan
                else:
                    authorYYYYx_df.loc[i, col] = np.nan
                    authorYYYYx_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    authorYYYYx_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    authorYYYYx_df.loc[i, col] = e_logepsX
                else:
                    authorYYYYx_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    authorYYYYx_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return authorYYYYx_df