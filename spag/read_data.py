#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import pkg_resources
import  sys, os, glob, time
import numpy as np
import pandas as pd
import seaborn as sns

from spag.convert import *

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
## Solar Composition Functions

def solar_logepsX_asplund09(Z=None, elem=None, return_error=False):
    """
    Z: int or None
        Atomic number of the element.
    elem: str or None
        Element symbol of the element.
    return_error: bool
        If True, the function will return the abundance and error of the solar
        composition in a tuple.
        
    Reads the datafile for Table 1 in "Asplund et al. 2009".
    (https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract)
    
    Returns a pandas table of the solar composition. If 'Z' or 'elem' is 
    provided, it will return the photospheric abundances if available, 
    otherwise it will use the meteoritic abundances. 
    """
    
    datafile_path = data_dir+"solar/asplund2009_solarabundance_table1.csv"
    asplund09 = pd.read_csv(datafile_path, skipinitialspace=True)
    print("Loading Datafile: ", datafile_path)
    
    if (elem is None) and (Z is None):
        return asplund09
    
    elif (elem is not None) and (Z is None):
        if elem not in asplund09['elem'].values:
            raise ValueError("Element symbol (elem) is invalid.")
        abund = asplund09[asplund09['elem'] == elem]['photosphere_logeps'].values[0]
        error = asplund09[asplund09['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund09[asplund09['elem'] == elem]['meteorite_logeps'].values[0]
            error = asplund09[asplund09['elem'] == elem]['meteorite_logeps_err'].values[0]
        
    elif (elem is None) and (Z is not None):
        if Z not in asplund09['Z'].values:
            raise ValueError("Atomic number (Z) is invalid.")
        abund = asplund09[asplund09['Z'] == Z]['photosphere_logeps'].values[0]
        error = asplund09[asplund09['Z'] == Z]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund09[asplund09['Z'] == Z]['meteorite_logeps'].values[0]
            error = asplund09[asplund09['Z'] == Z]['meteorite_logeps_err'].values[0]
        
    elif (elem is not None) and (Z is not None):
        if not element_matches_atomic_number(elem, Z):
            raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
        abund = asplund09[asplund09['elem'] == elem]['photosphere_logeps'].values[0]
        error = asplund09[asplund09['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund09[asplund09['elem'] == elem]['meteorite_logeps'].values[0]
            error = asplund09[asplund09['elem'] == elem]['meteorite_logeps_err'].values[0]
    else:
        raise AttributeError("Please check the function's input parameters.")
    
    if return_error:
        return (abund, error)
    else:
        return abund

def solar_logepsX_asplund21(Z=None, elem=None, return_error=False, isotopes=False, A=None):
    """
    Z: int or None (default=None)
        Atomic number of the element.
    elem: str or None (default=None)
        Element symbol of the element.
    return_error: bool (default=False)
        If True and isotope=False (default), the function will return the 
        abundance and error of the solar composition in a tuple (logeps, err).
    isotopes: bool (default=False)
        If True, the function will return the pandas dataframe of the pre-solar
        isotopic abundances (logeps). If elem, Z, or A is provided, the 
        function will return a reduced dataframe. If A is provided with elem or 
        Z, the function will return the specific isotope's abundance. (logeps)
    A: int or None
        Mass number of the isotope.
        
    Reads the datafile for Table 2 or Table B.1 in "Asplund et al. 2021".
    (https://ui.adsabs.harvard.edu/abs/2021A%26A...653A.141A/abstract)
    
    Returns a pandas table of the solar logarithmic abundances (logeps) when 
    'isotopes' is False. Returns the elemental isotopic abundances fractions and
    their pre-solar logarithmic abundances (logeps) when 'isotopes' is True.
    
    The abundance tables are composed of photospheric abundances for most 
    elements and meteoritic abundances for the remaining elements.
    
    If 'Z', 'elem', or 'A' is provided, the function will return the solar/
    pre-solar logrithmic abundance (logeps) or the associated reduced dataframe.
    Combinations of any of the three parameters are allowed for more specific
    queries. If 'return_error' is True, the function will return the error of
    the abundance as well -- only when 'isotopes' is False. 
    """
    
    if isotopes == False:
        datafile_path = data_dir+"solar/asplund2021_solarabundance_table2.csv"
        asplund21 = pd.read_csv(datafile_path, skipinitialspace=True)
        # print("Loading Datafile: ", datafile_path)
        
        if (elem is None) and (Z is None):
            return asplund21
        
        elif (elem is not None) and (Z is None):
            if elem not in asplund21['elem'].values:
                raise ValueError("Element symbol (elem) is invalid.")
            abund = asplund21[asplund21['elem'] == elem]['photosphere_logeps'].values[0]
            error = asplund21[asplund21['elem'] == elem]['photosphere_logeps_err'].values[0]
            if np.isnan(abund):
                abund = asplund21[asplund21['elem'] == elem]['meteorite_logeps'].values[0]
                error = asplund21[asplund21['elem'] == elem]['meteorite_logeps_err'].values[0]
            
        elif (elem is None) and (Z is not None):
            if Z not in asplund21['Z'].values:
                raise ValueError("Atomic number (Z) is invalid.")
            abund = asplund21[asplund21['Z'] == Z]['photosphere_logeps'].values[0]
            error = asplund21[asplund21['Z'] == Z]['photosphere_logeps_err'].values[0]
            if np.isnan(abund):
                abund = asplund21[asplund21['Z'] == Z]['meteorite_logeps'].values[0]
                error = asplund21[asplund21['Z'] == Z]['meteorite_logeps_err'].values[0]
            
        elif (elem is not None) and (Z is not None):
            if not element_matches_atomic_number(elem, Z):
                raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
            abund = asplund21[asplund21['elem'] == elem]['photosphere_logeps'].values[0]
            error = asplund21[asplund21['elem'] == elem]['photosphere_logeps_err'].values[0]
            if np.isnan(abund):
                abund = asplund21[asplund21['elem'] == elem]['meteorite_logeps'].values[0]
                error = asplund21[asplund21['elem'] == elem]['meteorite_logeps_err'].values[0]
        else:
            raise AttributeError("Please check the function's input parameters.")
        
        if return_error:
            return (abund, error)
        else:
            return abund
    
    elif isotopes == True:
        if return_error == True:
            raise ValueError("The 'return_error' parameter is not valid when isotopes=True.")
        
        datafile_path = data_dir+"solar/asplund2021_presolarabundance_isotopic_tableB1.csv"
        asplund21 = pd.read_csv(datafile_path, skipinitialspace=True)
        # print("Loading Datafile: ", datafile_path)      

        if (elem is None) and (Z is None) and (A is None):
            return asplund21
        
        elif (elem is not None) and (Z is None) and (A is None):
            if elem not in asplund21['elem'].values:
                raise ValueError("Element symbol (elem) is invalid.")
            return asplund21[asplund21['elem'] == elem]
            
        elif (elem is None) and (Z is not None) and (A is None):
            if Z not in asplund21['Z'].values:
                raise ValueError("Atomic number (Z) is invalid.")
            return asplund21[asplund21['Z'] == Z]
        
        elif (elem is None) and (Z is None) and (A is not None):
            if A not in asplund21['A'].values:
                raise ValueError("Mass number (A) is invalid.")
            return asplund21[asplund21['A'] == A]
        
        elif (elem is not None) and (Z is not None) and (A is None):
            if not element_matches_atomic_number(elem, Z):
                raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
            return asplund21[asplund21['elem'] == elem]
            
        elif ((elem is not None) or (Z is not None)) and (A is not None):
            Z = asplund21[asplund21['elem'] == elem]['Z'].values[0] if Z is None else Z
            elem = asplund21[asplund21['Z'] == Z]['elem'].values[0] if elem is None else elem
            if not element_matches_atomic_number(elem, Z):
                raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
            
            if A not in asplund21[asplund21['Z'] == Z]['A'].values:
                raise ValueError("Mass number (A) is invalid for this specific element (elem, Z).")
            return asplund21[(asplund21['elem'] == elem) & (asplund21['A'] == A)]['logeps'].values[0]
            
        else:
            raise AttributeError("Please check the function's input parameters.")
    
    else:
        raise ValueError("'isotopes' must be a boolean (i.e. either True or False).")


################################################################################
## Solar r-process and s-process Abundance Patterns

## Read-in the solar abundnace table for r- and s-process elements

def solar_r_s_abundances():
    """
    Reads the solar r- and s-process abundance fractions from the datafile.
    """
    
    solar_rs_abund = pd.read_csv(data_dir+"solar_r_s_fractions.csv", skipinitialspace=True)

    solar_rs_abund['logeps_r'] = np.log10(solar_rs_abund['rproc'])
    solar_rs_abund['logeps_s'] = np.log10(solar_rs_abund['sproc'])

    solar_Z = np.array(solar_rs_abund['Z'])
    solar_logeps_r = np.array(solar_rs_abund['logeps_r'])
    solar_logeps_s = np.array(solar_rs_abund['logeps_s'])

    ## Replace any NaN or Inf values with np.nan
    for i in range(len(solar_Z)):
        if np.isnan(solar_logeps_r[i]) or np.isinf(solar_logeps_r[i]):
            solar_logeps_r[i] = np.nan #does not change the pandas dataframe
        if np.isnan(solar_logeps_s[i]) or np.isinf(solar_logeps_s[i]):
            solar_logeps_s[i] = np.nan #does not change the pandas dataframe

    solar_logeps_r
