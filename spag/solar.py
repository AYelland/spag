#!/usr/bin/env python
# -*- coding: utf-8 -*-
# add to top of read_data.py temporarily

from __future__ import (division, print_function, absolute_import, unicode_literals)


import  sys, os, glob, time

import numpy as np
import pandas as pd
import re

from spag.utils import identify_prefix, element_matches_atomic_number

################################################################################
## Directory Variables

# script_dir = "/".join(IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]) + "/" # use this if in ipython
script_dir = os.path.dirname(os.path.realpath(__file__))+"/" # use this if not in ipython (i.e. terminal script)
data_dir = script_dir+"data/"
plots_dir = script_dir+"plots/"
linelist_dir = script_dir+"linelists/"

################################################################################
## Solar Composition Functions

def solar_anders1989(Z=None, elem=None, return_error=False):
    """
    Z: int or None
        Atomic number of the element.
    elem: str or None
        Element symbol of the element.
    return_error: bool
        If True, the function will return the abundance and error of the solar
        composition in a tuple.
        
    Returns a pandas table of the solar composition. If 'Z' or 'elem' is 
    provided, it will return the photospheric abundances if available, 
    otherwise it will use the meteoritic abundances. 
    """
    
    datafile_path = data_dir+"solar/anders1989_table2.csv"
    # load the datafile into a pandas dataframe and strip all whitespaces in the columns names and values
    anders1989 = pd.read_csv(datafile_path, skipinitialspace=True)
    for col in anders1989.columns:
        if anders1989[col].dtype == "object":
            anders1989[col] = anders1989[col].str.strip()
        anders1989.rename(columns={col:col.strip()}, inplace=True)
    # print("Loading Datafile: ", datafile_path)
    
    if (elem is None) and (Z is None):
        return anders1989
    
    elif (elem is not None) and (Z is None):
        if elem not in anders1989['elem'].values:
            raise ValueError("Element symbol (elem) is invalid.")
        abund = anders1989[anders1989['elem'] == elem]['photosphere_logeps'].values[0]
        error = anders1989[anders1989['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = anders1989[anders1989['elem'] == elem]['meteorite_logeps'].values[0]
            error = anders1989[anders1989['elem'] == elem]['meteorite_logeps_err'].values[0]
        
    elif (elem is None) and (Z is not None):
        if Z not in anders1989['Z'].values:
            raise ValueError("Atomic number (Z) is invalid.")
        abund = anders1989[anders1989['Z'] == Z]['photosphere_logeps'].values[0]
        error = anders1989[anders1989['Z'] == Z]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = anders1989[anders1989['Z'] == Z]['meteorite_logeps'].values[0]
            error = anders1989[anders1989['Z'] == Z]['meteorite_logeps_err'].values[0]
        
    elif (elem is not None) and (Z is not None):
        if not element_matches_atomic_number(elem, Z):
            raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
        abund = anders1989[anders1989['elem'] == elem]['photosphere_logeps'].values[0]
        error = anders1989[anders1989['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = anders1989[anders1989['elem'] == elem]['meteorite_logeps'].values[0]
            error = anders1989[anders1989['elem'] == elem]['meteorite_logeps_err'].values[0]
    else:
        raise AttributeError("Please check the function's input parameters.")
    
    if return_error:
        return (abund, error)
    else:
        return abund

def solar_asplund2005(Z=None, elem=None, return_error=False):
    """
    Z: int or None
        Atomic number of the element.
    elem: str or None
        Element symbol of the element.
    return_error: bool
        If True, the function will return the abundance and error of the solar
        composition in a tuple.
        
    Reads the datafile for Table 1 in "Asplund et al. 2005".
    (https://ui.adsabs.harvard.edu/abs/2005ASPC..336...25A/abstract)
    
    Returns a pandas table of the solar composition. If 'Z' or 'elem' is 
    provided, it will return the photospheric abundances if available, 
    otherwise it will use the meteoritic abundances. 
    """
    
    datafile_path = data_dir+"solar/asplund2005_table1.csv"
    # load the datafile into a pandas dataframe and strip all whitespaces in the columns names and values
    asplund2005 = pd.read_csv(datafile_path, skipinitialspace=True)
    for col in asplund2005.columns:
        if asplund2005[col].dtype == "object":
            asplund2005[col] = asplund2005[col].str.strip()
        asplund2005.rename(columns={col:col.strip()}, inplace=True)
    # print("Loading Datafile: ", datafile_path)
    
    if (elem is None) and (Z is None):
        return asplund2005
    
    elif (elem is not None) and (Z is None):
        if elem not in asplund2005['elem'].values:
            raise ValueError("Element symbol (elem) is invalid.")
        abund = asplund2005[asplund2005['elem'] == elem]['photosphere_logeps'].values[0]
        error = asplund2005[asplund2005['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund2005[asplund2005['elem'] == elem]['meteorite_logeps'].values[0]
            error = asplund2005[asplund2005['elem'] == elem]['meteorite_logeps_err'].values[0]
        
    elif (elem is None) and (Z is not None):
        if Z not in asplund2005['Z'].values:
            raise ValueError("Atomic number (Z) is invalid.")
        abund = asplund2005[asplund2005['Z'] == Z]['photosphere_logeps'].values[0]
        error = asplund2005[asplund2005['Z'] == Z]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund2005[asplund2005['Z'] == Z]['meteorite_logeps'].values[0]
            error = asplund2005[asplund2005['Z'] == Z]['meteorite_logeps_err'].values[0]
        
    elif (elem is not None) and (Z is not None):
        if not element_matches_atomic_number(elem, Z):
            raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
        abund = asplund2005[asplund2005['elem'] == elem]['photosphere_logeps'].values[0]
        error = asplund2005[asplund2005['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund2005[asplund2005['elem'] == elem]['meteorite_logeps'].values[0]
            error = asplund2005[asplund2005['elem'] == elem]['meteorite_logeps_err'].values[0]
    else:
        raise AttributeError("Please check the function's input parameters.")
    
    if return_error:
        return (abund, error)
    else:
        return abund
    
def solar_asplund2009(Z=None, elem=None, return_error=False):
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
    
    datafile_path = data_dir+"solar/asplund2009_table1.csv"
    # load the datafile into a pandas dataframe and strip all whitespaces in the columns names and values
    asplund2009 = pd.read_csv(datafile_path, skipinitialspace=True)
    for col in asplund2009.columns:
        if asplund2009[col].dtype == "object":
            asplund2009[col] = asplund2009[col].str.strip()
        asplund2009.rename(columns={col:col.strip()}, inplace=True)
    # print("Loading Datafile: ", datafile_path)
    
    if (elem is None) and (Z is None):
        return asplund2009
    
    elif (elem is not None) and (Z is None):
        if elem not in asplund2009['elem'].values:
            raise ValueError("Element symbol (elem) is invalid.")
        abund = asplund2009[asplund2009['elem'] == elem]['photosphere_logeps'].values[0]
        error = asplund2009[asplund2009['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund2009[asplund2009['elem'] == elem]['meteorite_logeps'].values[0]
            error = asplund2009[asplund2009['elem'] == elem]['meteorite_logeps_err'].values[0]
        
    elif (elem is None) and (Z is not None):
        if Z not in asplund2009['Z'].values:
            raise ValueError("Atomic number (Z) is invalid.")
        abund = asplund2009[asplund2009['Z'] == Z]['photosphere_logeps'].values[0]
        error = asplund2009[asplund2009['Z'] == Z]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund2009[asplund2009['Z'] == Z]['meteorite_logeps'].values[0]
            error = asplund2009[asplund2009['Z'] == Z]['meteorite_logeps_err'].values[0]
        
    elif (elem is not None) and (Z is not None):
        if not element_matches_atomic_number(elem, Z):
            raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
        abund = asplund2009[asplund2009['elem'] == elem]['photosphere_logeps'].values[0]
        error = asplund2009[asplund2009['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = asplund2009[asplund2009['elem'] == elem]['meteorite_logeps'].values[0]
            error = asplund2009[asplund2009['elem'] == elem]['meteorite_logeps_err'].values[0]
    else:
        raise AttributeError("Please check the function's input parameters.")
    
    if return_error:
        return (abund, error)
    else:
        return abund

def solar_asplund2021(Z=None, elem=None, return_error=False, isotopes=False, A=None):
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
        datafile_path = data_dir+"solar/asplund2021_table2.csv"
        asplund2021 = pd.read_csv(datafile_path, skipinitialspace=True)
        # print("Loading Datafile: ", datafile_path)
        
        if (elem is None) and (Z is None):
            return asplund2021
        
        elif (elem is not None) and (Z is None):
            if elem not in asplund2021['elem'].values:
                raise ValueError("Element symbol (elem) is invalid.")
            abund = asplund2021[asplund2021['elem'] == elem]['photosphere_logeps'].values[0]
            error = asplund2021[asplund2021['elem'] == elem]['photosphere_logeps_err'].values[0]
            if np.isnan(abund):
                abund = asplund2021[asplund2021['elem'] == elem]['meteorite_logeps'].values[0]
                error = asplund2021[asplund2021['elem'] == elem]['meteorite_logeps_err'].values[0]
            
        elif (elem is None) and (Z is not None):
            if Z not in asplund2021['Z'].values:
                raise ValueError("Atomic number (Z) is invalid.")
            abund = asplund2021[asplund2021['Z'] == Z]['photosphere_logeps'].values[0]
            error = asplund2021[asplund2021['Z'] == Z]['photosphere_logeps_err'].values[0]
            if np.isnan(abund):
                abund = asplund2021[asplund2021['Z'] == Z]['meteorite_logeps'].values[0]
                error = asplund2021[asplund2021['Z'] == Z]['meteorite_logeps_err'].values[0]
            
        elif (elem is not None) and (Z is not None):
            if not element_matches_atomic_number(elem, Z):
                raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
            abund = asplund2021[asplund2021['elem'] == elem]['photosphere_logeps'].values[0]
            error = asplund2021[asplund2021['elem'] == elem]['photosphere_logeps_err'].values[0]
            if np.isnan(abund):
                abund = asplund2021[asplund2021['elem'] == elem]['meteorite_logeps'].values[0]
                error = asplund2021[asplund2021['elem'] == elem]['meteorite_logeps_err'].values[0]
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
        asplund2021 = pd.read_csv(datafile_path, skipinitialspace=True)
        # print("Loading Datafile: ", datafile_path)      

        if (elem is None) and (Z is None) and (A is None):
            return asplund2021
        
        elif (elem is not None) and (Z is None) and (A is None):
            if elem not in asplund2021['elem'].values:
                raise ValueError("Element symbol (elem) is invalid.")
            return asplund2021[asplund2021['elem'] == elem]
            
        elif (elem is None) and (Z is not None) and (A is None):
            if Z not in asplund2021['Z'].values:
                raise ValueError("Atomic number (Z) is invalid.")
            return asplund2021[asplund2021['Z'] == Z]
        
        elif (elem is None) and (Z is None) and (A is not None):
            if A not in asplund2021['A'].values:
                raise ValueError("Mass number (A) is invalid.")
            return asplund2021[asplund2021['A'] == A]
        
        elif (elem is not None) and (Z is not None) and (A is None):
            if not element_matches_atomic_number(elem, Z):
                raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
            return asplund2021[asplund2021['elem'] == elem]
            
        elif ((elem is not None) or (Z is not None)) and (A is not None):
            Z = asplund2021[asplund2021['elem'] == elem]['Z'].values[0] if Z is None else Z
            elem = asplund2021[asplund2021['Z'] == Z]['elem'].values[0] if elem is None else elem
            if not element_matches_atomic_number(elem, Z):
                raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
            
            if A not in asplund2021[asplund2021['Z'] == Z]['A'].values:
                raise ValueError("Mass number (A) is invalid for this specific element (elem, Z).")
            return asplund2021[(asplund2021['elem'] == elem) & (asplund2021['A'] == A)]['logeps'].values[0]
            
        else:
            raise AttributeError("Please check the function's input parameters.")
    
    else:
        raise ValueError("'isotopes' must be a boolean (i.e. either True or False).")

def solar_grevesse1998(Z=None, elem=None, return_error=False):
    """
    Z: int or None
        Atomic number of the element.
    elem: str or None
        Element symbol of the element.
    return_error: bool
        If True, the function will return the abundance and error of the solar
        composition in a tuple.
        
    Returns a pandas table of the solar composition. If 'Z' or 'elem' is 
    provided, it will return the photospheric abundances if available, 
    otherwise it will use the meteoritic abundances. 
    """
    
    datafile_path = data_dir+"solar/grevesse1998_table1.csv"
    # load the datafile into a pandas dataframe and strip all whitespaces in the columns names and values
    grevesse1998 = pd.read_csv(datafile_path, skipinitialspace=True)
    for col in grevesse1998.columns:
        if grevesse1998[col].dtype == "object":
            grevesse1998[col] = grevesse1998[col].str.strip()
        grevesse1998.rename(columns={col:col.strip()}, inplace=True)
    # print("Loading Datafile: ", datafile_path)
    
    if (elem is None) and (Z is None):
        return grevesse1998
    
    elif (elem is not None) and (Z is None):
        if elem not in grevesse1998['elem'].values:
            raise ValueError("Element symbol (elem) is invalid.")
        abund = grevesse1998[grevesse1998['elem'] == elem]['photosphere_logeps'].values[0]
        error = grevesse1998[grevesse1998['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = grevesse1998[grevesse1998['elem'] == elem]['meteorite_logeps'].values[0]
            error = grevesse1998[grevesse1998['elem'] == elem]['meteorite_logeps_err'].values[0]
        
    elif (elem is None) and (Z is not None):
        if Z not in grevesse1998['Z'].values:
            raise ValueError("Atomic number (Z) is invalid.")
        abund = grevesse1998[grevesse1998['Z'] == Z]['photosphere_logeps'].values[0]
        error = grevesse1998[grevesse1998['Z'] == Z]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = grevesse1998[grevesse1998['Z'] == Z]['meteorite_logeps'].values[0]
            error = grevesse1998[grevesse1998['Z'] == Z]['meteorite_logeps_err'].values[0]
        
    elif (elem is not None) and (Z is not None):
        if not element_matches_atomic_number(elem, Z):
            raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
        abund = grevesse1998[grevesse1998['elem'] == elem]['photosphere_logeps'].values[0]
        error = grevesse1998[grevesse1998['elem'] == elem]['photosphere_logeps_err'].values[0]
        if np.isnan(abund):
            abund = grevesse1998[grevesse1998['elem'] == elem]['meteorite_logeps'].values[0]
            error = grevesse1998[grevesse1998['elem'] == elem]['meteorite_logeps_err'].values[0]
    else:
        raise AttributeError("Please check the function's input parameters.")
    
    if return_error:
        return (abund, error)
    else:
        return abund
    
# ------------------------------------------------------------------------------
   
def get_solar_abund_dict(version='asplund2009'):
    """
    Returns the solar abundances from Asplund et al. 2009 as a dictionary.
    """

    if version == 'anders1989':
        solar_abund = solar_anders1989()
    elif version == 'asplund2005':
        solar_abund = solar_asplund2005()
    elif version == 'asplund2009':
        solar_abund = solar_asplund2009()
    elif version == 'asplund2021':
        solar_abund = solar_asplund2021()
    elif version == 'grevesse1998':
        solar_abund = solar_grevesse1998()
    else:
        raise ValueError("Invalid version. Choose from ('anders1989','asplund2005', 'asplund2009', 'asplund2021', 'grevesse1998').")

    solar_abund_dict = dict(zip(solar_abund['elem'], solar_abund['photosphere_logeps']))

    # if the value is NaN, use the meteoritic value
    for elem in solar_abund_dict.keys():
        if pd.isna(solar_abund_dict[elem]):
            solar_abund_dict[elem] = float(solar_abund[solar_abund['elem'] == elem]['meteorite_logeps'].values[0])

    # appends a np.nan value for Tc
    solar_abund_dict['Tc'] = np.nan

    return solar_abund_dict

def get_solar(elems, version='asplund2009'):
    """
    Returns the solar abundance of the elements in the list 'elems'.
    Keeps original input names in the index.
    """

    elems = np.ravel(elems)
    solar_abund = get_solar_abund_dict(version=version)

    def get_clean_element(elem):
        try:
            _, elem_ = identify_prefix(elem)
        except ValueError:
            elem_ = elem
        # Strip ionization
        match = re.match(r'^([A-Z][a-z]?)', elem_.title())
        return match.group(1) if match else elem_.title()

    good_elems = [get_clean_element(str(e)) for e in elems]

    return pd.Series([solar_abund[elem] for elem in good_elems], index=elems, name=version)

################################################################################
## Solar r-process and s-process Abundance Patterns

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
