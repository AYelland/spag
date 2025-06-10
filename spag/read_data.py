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
## Solar Composition Functions

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

    if version == 'asplund2005':
        solar_abund = solar_asplund2005()
    elif version == 'asplund2009':
        solar_abund = solar_asplund2009()
    elif version == 'asplund2021':
        solar_abund = solar_asplund2021()
    elif version == 'grevesse1998':
        solar_abund = solar_grevesse1998()
    else:
        raise ValueError("Invalid version. Choose from ('asplund2005', 'asplund2009', 'asplund2021', 'grevesse1998').")

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
    """
    
    elems = np.ravel(elems)
    good_elems = [getelem(elem) for elem in elems]
    solar_abund = get_solar_abund_dict(version=version)

    return pd.Series([solar_abund[elem] for elem in good_elems], index=elems, name=version)

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

################################################################################
## JINAbase Data Read-in

def load_jinabase(sci_key=None, priority=1, load_eps=True, load_ul=True, load_XH=True, load_XFe=True, load_aux=True, name_as_index=False, feh_ulim=None, version="yelland"):
    """
    sci_key: str or None
        A label used for interesting stars in the JINAbase database. There are four different types of keys.
        ('Ncap_key', 'C_key', 'MP_key', 'alpha_key') Use the value from one of the following keys to filter the data:
        - CE: Carbon-enhanced stars (C_key)
        - NO: Carbon-enhanced stars, neutron-capture-normal (C_key)
        - R1: rI-rich (Ncap_key)
        - R2: rII-rich (Ncap_key)
        - S: s-rich (Ncap_key)
        - RS: r/s-rich (Ncap_key)
        - I: i-process rich (Ncap_key)
        - alpha: Alpha-enhanced stars (alpha_key)
        - _MP: ________ metal-poor stars (MP_key) (e.g MP, VMP, EMP, UMP, etc.)
    priority: int
        The priority level of the sources in the JINAbase database for duplicate entries. (1, 2)
    load_eps: bool
        Load the log(eps) columns from the JINAbase database.
    load_ul: bool
        Load the upper limit value columns from the JINAbase database.
    load_XH: bool
        Calculate the [X/H] columns from the log(eps) columns, using Asplund et al. (2009) solar abundances.
    load_XFe: bool
        Calculate the [X/Fe] columns from the log(eps) columns, using Asplund et al. (2009) solar abundances.
    load_aux: bool
        Load the auxiliary columns from the JINAbase database. (e.g. JINA_ID, Name, Ref, Priority, stellar parameters, etc.)
    name_as_index: bool
        Set the "Name" column as the index of the DataFrame.
    version: str
        The version of the JINAbase data to load. Options are "abohalima", "ji", "mardini", or "yelland".

    Load the JINAbase data from the local copy of the JINAbase-updated repository. 
    Speak with Mohammad Mardini for more details.
    https://github.com/Mohammad-Mardini/JINAbase-updated
    """

    ## Read data
    data = pd.read_csv(data_dir+"abundance_tables/JINAbase-yelland/JINAbase-yelland25.csv", header=0, na_values=["*"]) #index_col=0
    uls  = pd.read_csv(data_dir+"abundance_tables/JINAbase-yelland/JINAbase-yelland25-ulimits.csv", header=0, na_values=["*"]) #index_col=0
    Nstars = len(data)

    ## Get the list of elements & create the corresponding column names for the element abundance columns
    ## NOTE: The ionization state columns are dropped and not used in this analysis. Hence, the -7 
    ##       in the column slicing. Every application of elems, ul_elems, and epscolnames now 
    ##       excludes the ionization state columns.
    elems = data.columns[31:-7]
    ul_elems = list(map(lambda x: "ul"+x, elems))
    epscolnames = list(map(lambda x: "eps"+x.lower(), elems))
    print("WARNING: Dropped the CaII, TiII, VII, CrII, MnII, FeII columns.")

    ## Rename the data element abundance columns with the prefix "eps" (e.g. "Fe" -> "epsfe")
    data.rename(columns=dict(zip(elems, epscolnames)), inplace=True)

    ## Separate the auxiliary columns (JINA_ID, Priority, etc.) from the element abundance columns (epsX) in 'data' and 'uls'
    auxdata_cols = data.columns[0:31].append(pd.Index([data.columns[-1]]))
    auxdata = data[auxdata_cols]
    data = data[epscolnames]

    auxuls_cols = uls.columns[0:5]
    auxuls = uls[auxuls_cols]
    uls = uls[ul_elems] # consist of 1s (upper limits) and NaNs (non-upper limits)

    ## Use the ulimits (uls) DataFrame to mask the data DataFrame
    uls_mask = pd.notnull(uls).to_numpy()  # Convert uls DataFrame to boolean array (True if not NaN)
    uls_table = data.where(uls_mask)  # Extract only the upper limit values (keep NaN for others)
    for col in uls_table.columns:
        uls_table.rename(columns={col: "ul"+col[3:]}, inplace=True) #same as ulcolnames (e.g. "ulFe" -> "ulfe")
    for col in uls_table.columns:
        uls_table[col] = uls_table[col].str.replace("<", "").astype(float)

    data_matrix = data.to_numpy()  # Convert data DataFrame to NumPy array
    data_matrix[uls_mask] = np.nan # Set values in `data_matrix` to NaN wherever `uls_mask` is True
    data = pd.DataFrame(data_matrix, columns=data.columns, index=data.index) # Convert the modified NumPy array back to a DataFrame

    ## Concatenate the upper limit values to the 'data' DataFrame
    if load_ul:
        data = pd.concat([data, uls_table], axis=1) # Concatenate the upper limit values to the data DataFrame

    ## Convert the element abundance and add the [X/H] and [X/Fe] columns
    if load_XH:
        XHcol_from_epscol(data)
        if load_ul:
            ulXHcol_from_ulcol(data)
    if load_XFe:
        XFecol_from_epscol(data)
        if load_ul:
            ulXFecol_from_ulcol(data)
            
    ## Combine the auxiliary columns with the element abundance columns
    if load_aux:
        data = pd.concat([auxdata, data],axis=1)
    else:
        data = pd.concat([auxdata[['Name','Ref','Priority','Ncap_key','C_key','MP_key','alpha_key'], data]], axis=1)

    ## Remove duplicate entries by using the 'Priority' column (1=high, 2=low)
    ## (dupicates entries originate from two papers referencing the same source)
    if priority==1 or priority==2:
        pmask = data['Priority'] == priority
        data = data[pmask]
        uls = uls[pmask]

    ## If a specific science key ('Ncap_key', 'C_key', 'MP_key', 'alpha_key') was provided, apply the mask to filter the data
    if sci_key is not None:
        sci_key_cols = ['MP_key', 'Ncap_key', 'C_key', 'alpha_key']
        sci_mask = None
        for key_col in sci_key_cols:
            if sci_key in data[key_col].dropna().unique():
                sci_mask = data[key_col] == sci_key
                break
        if sci_mask is None:
            raise ValueError(f"The provided sci_key '{sci_key}' is not valid. Choose from ('_MP', 'R1', 'R2', 'S', 'RS', 'I', 'CE', 'NO', 'alpha').")
        data = data[sci_mask]
        uls = uls[sci_mask]

    ## Finalize the DataFrame by dropping columns in the auxiliary columns
    if not load_aux:
        data.drop({'Priority','Ncap_key','C_key','MP_key','alpha_key'}, axis=1, inplace=True)

    ## Drop the log(eps) columns if not needed
    if not load_eps:
        data.drop(epscolnames, axis=1, inplace=True)

    ## Set the "Name" column as the index
    if name_as_index:
        data.index = data["Name"]

    ## Save the processed data to a CSV file
    data.to_csv(data_dir+"abundance_tables/JINAbase-yelland/JINAbase-yelland25-processed.csv", index=False)

    # Filter the dataframe based on desired version
    if version == "abohalima":
        data = data[data['Added_by'] == 'Abohalima']
    elif version == "ji":
        data = data[(data['Added_by'] == 'Abohalima') | (data['Added_by'] == 'Ji')]
    elif version == "mardini":
        data = data[(data['Ref'] == 'HANc18') | (data['Ref'] == 'KIR12') | (data['Added_by'] == 'Abohalima') | (data['Added_by'] == 'Mardini')]
    elif version == "yelland":
        pass  # use full dataset
    else:
        raise ValueError("Invalid version. Choose from ('abohalima', 'ji', 'mardini', 'yelland').")

    
    ## Filter by metallicity
    if feh_ulim is not None:
        if isinstance(feh_ulim, (int, float)):
            def feh_filter(val):
                if isinstance(val, (int, float)):
                    return val <= feh_ulim
                elif isinstance(val, str) and '<' in val:
                    return True  # treat upper limits as valid
                else:
                    return False  # ignore all other invalid entries

            data = data[data['Fe/H'].apply(feh_filter)]
        else:
            raise ValueError("Invalid value for feh_ulim. It should be a number (float or int).")

    return data

################################################################################
## Specific System's Data Read-in

def load_mw_halo(**kwargs):
    """
    Loads JINAbase and removes stars with loc='DW' or loc='UF' such that only halo stars remain
    Note: DW = dwarf galaxy, UF = ultra-faint galaxy
    """
    halo = load_jinabase(**kwargs)
    halo = halo[halo["Loc"] != "DW"]
    halo = halo[halo["Loc"] != "UF"]
    return halo

def load_atari_disk(**kwargs):
    """
    Atari Disk Stars

    Loads the data from Mardini et al. 2022 where they present the [Fe/H] metallicity
    and [C/Fe] abundance ratios of sources from various JINAbase references.
    """

    ## Manually add specific references
    # -------------------------------------------------- #
    ## Load Mardini+2022
    mardini2022_df = load_mardini2022()

    ## Combine the DataFrames
    atari_df = pd.concat([mardini2022_df], ignore_index=True, sort=False)

    return atari_df

def load_classical_dwarf_galaxies(add_all=False, **kwargs):
    """
    Loads JINAbase and extracts only the data for the classical dwarf galaxies.
    
    If 'add_all=True', then the data for following galaxies will be added:
    - Ursa Minor
    - Sculptor
    - Draco
    - Carina
    - Sextans
    - Sagittarius
    """
    
    cldw = load_jinabase(**kwargs)
    cldw = cldw[cldw["Loc"] == "DW"]
    
    # def get_gal(row):
        
    #     ## These are papers from a single galaxy
    #     refgalmap = {"AOK07b":"UMi","COH10":"UMi","URA15":"UMi",
    #                  "FRE10a":"Scl","GEI05":"Scl","JAB15":"Scl","SIM15":"Scl","SKU15":"Scl",
    #                  "AOK09":"Sex",
    #                  "FUL04":"Dra","COH09":"Dra","TSU15":"Dra","TSU17":"Dra",
    #                  "NOR17":"Car","VEN12":"Car",
    #                  "HAN18":"Sgr"}
    #     ref = row["Reference"]
    #     if ref in refgalmap:
    #         return refgalmap[ref]
        
    #     ## These are papers with multiple galaxies
    #     assert ref in ["SHE01","SHE03","TAF10","KIR12"], ref
    #     name = row["Name"]
    #     name = name[0].upper() + name[1:3].lower()
    #     if name == "Umi": return "UMi"
        
    #     return name
    
    # #allrefs = np.unique(cldw["Reference"])
    # #multirefs = ["SHE01","SHE03","TAF10","KIR12"]
    # gals = [get_gal(x) for i,x in cldw.iterrows()]
    # cldw["galaxy"] = gals

    # if add_all:
    #     fnx = load_letarte10_fornax()
    #     fnx2 = load_lemasle14_fornax()
    #     car = load_lemasle12_carina()
    #     sex = load_theler20_sextans()
    #     sgr = load_apogee_sgr()
    #     cldw = pd.concat([cldw,fnx,fnx2,scl,car,sex,sgr],axis=0)
    
    return cldw

def load_carina(**kwargs):
    """
    Loads Carina data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    carina_refs = [
        'SHE03','LEM12','NOR17','SUS17','VEN12','LUH24'#,'REI20'
        # 'REI20' could be included, but it lacks carbon abundances and does not have DW identifiers in jinabase
    ]

    ## JINAbase
    # -------------------------------------------------- #
    jinabase = load_jinabase(**kwargs)

    ## Filter JINAbase for Carina (by Reference) for References with single systems
    carina_refs1 = ['NOR17']#,'VEN12']
    jinabase_carina1 = jinabase[jinabase['Ref'].isin(carina_refs1)]

    ## Filter JINAbase for Carina (by Star Name) for References with multiple systems, including Carina 
    carina_refs2 = ['SHE03'] # 'REI20' could be included, but it lacks carbon abundances and does not have DW identifiers in jinabase
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_carina2 = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('car')]
    for ref in jinabase_carina2['Ref'].unique():
        if ref in carina_refs1:
            # drop rows from jinabase_carina2 that are already in jinabase_carina1
            jinabase_carina2 = jinabase_carina2[jinabase_carina2['Ref'] != ref]

    ## Concatenate the DataFrames
    carina_refs = carina_refs1 + carina_refs2
    jinabase_carina = pd.concat([jinabase_carina1, jinabase_carina2], ignore_index=True)

    ## Manually add specific references
    # -------------------------------------------------- #

    ## Lemasle 2012 (LEM12) -- no carbon abundances
    # data collected in SPAG, but function not created yet

    ## Susmitha 2017 (SUS17)
    # susmitha2017 = load_susmitha2017()

    ## Lucchetti 2024 (LUCC24)
    lucchetti2024 = load_lucchetti2024()
    lucchetti2024_carina = lucchetti2024[~lucchetti2024['Name'].str.lower().str.contains('fnx')]
    
    ## Combine the DataFrames
    # -------------------------------------------------- #
    carina_df = pd.concat([jinabase_carina, lucchetti2024_carina], ignore_index=True)

    if 'ul[C/Fe]' not in carina_df.columns:
        carina_df = pd.concat([carina_df, pd.Series(np.nan, index=carina_df.index, name='ul[C/Fe]')], axis=1)

    return carina_df

def load_sculptor(**kwargs):
    """
    Loads Sculptor data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    sculptor_refs = [
        'CHI18','GEI05','HIL19','JAB15','KIR12',#'REI20',
        'SIM15','SKU15','SKU17','SKU19','SKU24','SHE03',
        'TAF10'
        # 'REI20' could be included, but it lacks carbon abundances and does not have DW identifiers in jinabase
    ]

    ## JINAbase
    # -------------------------------------------------- #
    jinabase = load_jinabase(**kwargs)

    ## Filter JINAbase for Sculptor (by Star Name)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_sculptor1 = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('scl')]
    sculptor_refs = [ref for ref in sculptor_refs if ref not in jinabase_sculptor1['Ref'].unique()]

    ## Filter JINAbase for Sculptor (by Reference)
    jinabase_sculptor2 = jinabase[jinabase['Ref'].isin(sculptor_refs)]
    sculptor_refs = [ref for ref in sculptor_refs if ref not in jinabase_sculptor2['Ref'].unique()]

    ## Concatenate the DataFrames
    jinabase_sculptor = pd.concat([jinabase_sculptor1, jinabase_sculptor2], ignore_index=True)


    ## Manually add specific references
    # -------------------------------------------------- #

    ## Chiti+2018
    chiti2018a_df = load_chiti2018a(combine_tables=True)

    ## Skuladottir 2017 (SKU17)                                                                                
    # have not collected the data for SPAG yet

    ## Skuladottir 2024 (SKU24)
    # have not collected the data for SPAG yet

    ## Sestito 2023 (SES03)
    # have not collected the data for SPAG yet

    ## Frebel+2010b
    frebel2010b_df = load_frebel2010b()

    ## Combine the DataFrames
    # -------------------------------------------------- #
    sculptor_df = pd.concat([jinabase_sculptor, chiti2018a_df, frebel2010b_df], ignore_index=True, sort=False)

    return sculptor_df

def load_fornax(**kwargs):
    """
    Loads Fornax data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    fornax_refs = [
        'LEM14','LET07','LET10',#'REI20',
        'SHE03','TAF10'
        # 'REI20' could be included, but it lacks carbon abundances and does not have DW identifiers in jinabase
    ]

    ## JINAbase
    # -------------------------------------------------- #
    jinabase = load_jinabase(**kwargs)

    ## Filter JINAbase for Fornax (by Star Name)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_fornax1 = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('fnx')]
    fornax_refs = [ref for ref in fornax_refs if ref not in jinabase_fornax1['Ref'].unique()]
    
    ## Filter JINAbase for Fornax (by Reference)
    jinabase_fornax2 = jinabase[jinabase['Ref'].isin(fornax_refs)]
    fornax_refs = [ref for ref in fornax_refs if ref not in jinabase_fornax2['Ref'].unique()]

    ## Concatenate the DataFrames
    jinabase_fornax = pd.concat([jinabase_fornax1, jinabase_fornax2], ignore_index=True)


    ## Manually add specific references
    # -------------------------------------------------- #
    ## Lemasle 2014 (LEM14)
    lemasle2014 = load_lemasle2014()

    ## Letarte 2007 (LET07) -- no carbon abundances
    # thesis work, have not collected the data for SPAG yet

    ## Letarte 2010 (LET10)
    letarte2010 = load_letarte2010()

    ## Lucchetti 2024 (LUCC24)
    lucchetti2024 = load_lucchetti2024()
    lucchetti2024_fornax = lucchetti2024[lucchetti2024['Name'].str.lower().str.contains('fnx')]
    
    ## Combine the DataFrames
    # -------------------------------------------------- #
    fornax_df = pd.concat([jinabase_fornax, lemasle2014, letarte2010, lucchetti2024_fornax], ignore_index=True)

    if 'ul[C/Fe]' not in fornax_df.columns:
        fornax_df = pd.concat([fornax_df, pd.Series(np.nan, index=fornax_df.index, name='ul[C/Fe]')], axis=1)

    return fornax_df

def load_gse(**kwargs):
    """
    Gaia Sausage/Enceladus (GSE) Dwarf Galaxy Stars 

    Loads the data from Ou et al. (2024) for the Gaia Sausage/Enceladus (GSE) stars.
    This function reads in the data from the table and returns it as a pandas DataFrame.
    """

    ## Manually add specific references
    # -------------------------------------------------- #
    ## Load Ou+2024
    ou2024_df = load_ou2024()

    ## Combine the DataFrames
    gse_df = pd.concat([ou2024_df], ignore_index=True, sort=False)

    return gse_df

def load_sagittarius(include_lowres=False, include_apogee=False, **kwargs):
    """
    Sagittarius (Sgr) Dwarf Galaxy Stars 

    Loads the data from various references for the Sagittarius (Sgr) stars.
    """

    ## JINAbase
    # -------------------------------------------------- #
    jinabase = load_jinabase(**kwargs)

    sagittarius_refs = [
        'HANc18','SBO20',''#'REI20'
        # 'REI20' could be included, but it lacks carbon abundances and does not have DW identifiers in jinabase
    ]

    ## Filter JINAbase for Sagittarius (by Star Name)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_sagittarius1 = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('fnx')]
    sagittarius_refs = [ref for ref in sagittarius_refs if ref not in jinabase_sagittarius1['Ref'].unique()]
    
    ## Filter JINAbase for Sagittarius (by Reference)
    jinabase_sagittarius2 = jinabase[jinabase['Ref'].isin(sagittarius_refs)]
    sagittarius_refs = [ref for ref in sagittarius_refs if ref not in jinabase_sagittarius2['Ref'].unique()]

    ## Concatenate the DataFrames
    jinabase_sagittarius = pd.concat([jinabase_sagittarius1, jinabase_sagittarius2], ignore_index=True)

    ## Manually add specific references
    # -------------------------------------------------- #
    df_list = []
    ## APOGEE
    if include_apogee:
        apogee_df = load_apogee_sgr()   
        df_list.append(apogee_df)

    ## Ou+2025
    ou2025_df = load_ou2025()
    df_list.append(ou2025_df)

    ## Sestito+2024
    sestito2024_df = load_sestito2024()
    df_list.append(sestito2024_df)

    ## Sestito+2024b -- low/med resolution
    if include_lowres:
        sestito2024b_df = load_sestito2024b()
        df_list.append(sestito2024b_df)

    ## Sbordone+2007
    # sbordone2007_df = load_sbordone2007()
    # df_list.append(sbordone2007_df)

    ## Combine the DataFrames
    sagittarius_df = pd.concat(df_list, ignore_index=True, sort=False)

    return sagittarius_df

def load_lmc(**kwargs):
    """
    Load the Large Magellanic Cloud (LMC) Dwarf Galaxy Stars

    Loads the data from Chiti et al. 2024 and combines it with other
    references if needed.
    """

    ## Manually add specific references
    # -------------------------------------------------- #
    ## Load Chiti+2024
    chiti2024_df = load_chiti2024()

    ## Combine the DataFrames
    lmc_df = pd.concat([chiti2024_df], ignore_index=True, sort=False)

    return lmc_df

def load_ufds():
    """
    Load the UFD galaxies from Alexmods, parse abundance values and upper limits.

    Returns:
        pd.DataFrame: A cleaned DataFrame with numerical abundance columns and separate upper limit columns.
    """
    
    chiti2018b_df = rd.load_chiti2018b()
    chiti2023_df = rd.load_chiti2023()
    feltzing2009_df = rd.load_feltzing2009()
    francois2016_df = rd.load_francois2016()
    frebel2010a_df = rd.load_frebel2010a()
    frebel2014_df = rd.load_frebel2014()
    gilmore2013_df = rd.load_gilmore2013()
    hansent2017_df = rd.load_hansent2017()
    hansent2020_df = rd.load_hansent2020()
    hansent2024_df = rd.load_hansent2024()
    ishigaki2014_df = rd.load_ishigaki2014b()
    ji2016a_df = rd.load_ji2016a()
    ji2016b_df = rd.load_ji2016b()
    ji2019_df = rd.load_ji2019()
    ji2020_df = rd.load_ji2020()
    kirby2017_df = rd.load_kirby2017()
    koch2008c_df = rd.load_koch2008c()
    koch2013b_df = rd.load_koch2013b()
    lai2011_df = rd.load_lai2011()
    marshall2019_df = rd.load_marshall2019()
    nagasawa2018_df = rd.load_nagasawa2018()
    norris2010a_df = rd.load_norris2010a()
    norris2010b_df = rd.load_norris2010b()
    norris2010c_df = rd.load_norris2010c()
    roederer2014b_df = rd.load_roederer2014b()
    simon2010_df = rd.load_simon2010()
    spite2018_df = rd.load_spite2018()
    waller2023_df = rd.load_waller2023()
    webber2023_df = rd.load_webber2023()

    df_list = [
        chiti2018b_df,
        chiti2023_df,
        feltzing2009_df,
        francois2016_df,
        frebel2010a_df,
        frebel2014_df,
        gilmore2013_df,
        hansent2017_df,
        hansent2020_df,
        hansent2024_df,
        ishigaki2014_df,
        ji2016a_df,
        ji2016b_df,
        ji2019_df,
        ji2020_df,
        kirby2017_df,
        koch2008c_df,
        koch2013b_df,
        lai2011_df,
        marshall2019_df,
        nagasawa2018_df,
        norris2010a_df,
        norris2010b_df,
        norris2010c_df,
        roederer2014b_df,
        simon2010_df,
        spite2018_df,
        waller2023_df,
        webber2023_df
    ]

    ## Combine all dataframes into a single dataframe
    ufd_df = pd.DataFrame()
    for df in df_list:
        # print(df['Reference'].unique()[0])
        ufd_df = pd.concat([ufd_df, df], ignore_index=True)

    ## Drop all abundance ratio columns ([X/H], [X/Fe], etc.)
    abundance_cols = [col for col in ufd_df.columns if (('[' in col) or (']' in col))]
    ufd_df.drop(columns=abundance_cols, inplace=True, errors='ignore')

    ## Sort remaining columns
    epscols = [col for col in ufd_df.columns if col.startswith('eps')]
    ulcols = [col for col in ufd_df.columns if col.startswith('ul')]
    errcols = [col for col in ufd_df.columns if col.startswith('e_')]
    auxcols = [col for col in ufd_df.columns if col not in epscols + ulcols + errcols]

    ufd_df = ufd_df[auxcols + epscols + ulcols + errcols]

    # ## Convert abundance column names to Ion format (e.g. epsfe --> Fe I, epsc --> C-H)
    # for col in epscols:
    #     elem = col[3:]
    #     ion = ion_from_col(elem)
    #     if ion is not None:
    #         ufd_df.rename(columns={col: ion.replace(' ', '_')}, inplace=True)
    #     else:
    #         print(f"Warning: Could not convert column {col} to Ion format.")
        

    return ufd_df

def load_ufds_alexmods():
    """
    Load the UFD galaxies from Alexmods, parse abundance values and upper limits.

    Returns:
        pd.DataFrame: A cleaned DataFrame with numerical abundance columns and separate upper limit columns.
    """
    ufd_df = pd.read_csv(data_dir + "abundance_tables/alexmods_ufd/alexmods_ufd_yelland.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    # Identify abundance columns
    abundance_cols = [col for col in ufd_df.columns if col.startswith("[") and (col.endswith("Fe]") or col.endswith("H]"))]

    # Initialize upper limit columns
    for col in abundance_cols:
        ufd_df["ul" + col] = np.nan

    # Parse string values into numeric + upper limit
    for col in abundance_cols:
        # Strings with '<' are upper limits
        mask = ufd_df[col].astype(str).str.contains("<")
        ufd_df.loc[mask, "ul" + col] = ufd_df.loc[mask, col].astype(str).str.replace("<", "").astype(float)  # Extract upper limit values
        ufd_df.loc[mask, col] = np.nan  # Replace upper limits in main column with NaN

    ## Sort the columns to have the upper limit columns next to the abundance columns
    sorted_cols = []
    for col in abundance_cols:
        sorted_cols.append(col)
        sorted_cols.append("ul" + col)
    # Add other columns that are not abundance columns
    other_cols = [col for col in ufd_df.columns if col not in abundance_cols and not col.startswith("ul")]
    sorted_cols = other_cols + sorted_cols
    ufd_df = ufd_df[sorted_cols]

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in ufd_df.iterrows():
        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            ## pad RA_hms with leading zeros
            if len(row['RA_hms']) == 10:
                row['RA_hms'] = '0' + row['RA_hms']
                ufd_df.at[idx, 'RA_hms'] = row['RA_hms']
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            ufd_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            ufd_df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(float(row['RA_deg']), precision=2)
            ufd_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(float(row['DEC_deg']), precision=2)
            ufd_df.at[idx, 'DEC_dms'] = row['DEC_dms']

    return ufd_df

################################################################################
## Reference Read-in (Abundance Data)

def load_chiti2018a(combine_tables=True):
    """
    Sculptor (Scl) Dwarf Galaxy Stars

    Loads the data from Chiti et al. (2018) for the MagE and M2FS measurements.
    This function reads in the data from the two tables (table5/MagE and table6/M2FS) and
    returns them as pandas DataFrames. By default, the two tables are combined into a single
    DataFrame.
    """

    # Extract the limit columns
    def separate_limit_columns(df):
        """
        Separate the limit columns from the value columns in the given DataFrame.
        """
        limit_cols = [col for col in df.columns if col.startswith('l_')]
        for l_col in limit_cols:
            value_col = l_col.replace('l_', '')  # Corresponding value column

            mask_actual = (df[l_col] != '<') & (df[l_col] != '>')
            mask_lower = df[l_col] == '>'
            mask_upper = df[l_col] == '<'

            df[f"{value_col}_real"] = np.where(mask_actual, df[value_col], np.nan) # Actual values
            df[f"ll{value_col}"] = np.where(mask_lower, df[value_col], np.nan) # Lower limit
            df[f"ul{value_col}"] = np.where(mask_upper, df[value_col], np.nan) # Upper limit

            # Drop the original limit column & value column
            df = df.drop(columns=[value_col])
            df = df.drop(columns=[l_col])
            df = df.rename(columns={f"{value_col}_real": value_col})
        return df

    if combine_tables:

        ## Load the combined table (created by Alex Yelland)
        chiti2018a_df = pd.read_csv(data_dir+'abundance_tables/chiti2018a/chiti2018a_sculptor.csv', comment='#', header=0)
        
        ## Add columns and extract the limit columns
        chiti2018a_df = separate_limit_columns(chiti2018a_df)
        df_cols_reorder = [
            'Reference','Ref','Name','Loc','Type','Sci_key','RA_hms','DEC_dms','RA_deg','DEC_deg',
            'logg','Teff','[Ba/H]','Slit',
            'A(C)','llA(C)','ulA(C)','e_A(C)',
            '[Fe/H]','ll[Fe/H]','ul[Fe/H]','e_[Fe/H]',
            '[C/Fe]','ll[C/Fe]','ul[C/Fe]','e_[C/Fe]','[C/Fe]c',
            '[C/Fe]f','ll[C/Fe]f','ul[C/Fe]f','e_[C/Fe]f',
        ]
        chiti2018a_df = chiti2018a_df[df_cols_reorder]
        chiti2018a_df = chiti2018a_df.rename(columns={'Sci_key': 'Ncap_key'})  # Rename 'Sci_key' to 'Ncap_key' for consistency

        ## Fill the NaN values in the RA and DEC columns
        for idx, row in chiti2018a_df.iterrows():
            if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):  # Ensure RA_hms is not NaN
                ## pad RA_hms with leading zeros
                if len(row['RA_hms']) == 10:
                    row['RA_hms'] = '0' + row['RA_hms']
                    chiti2018a_df.at[idx, 'RA_hms'] = row['RA_hms']
                row['RA_deg'] = coord.ra_hms_to_deg(str(row['RA_hms']), precision=6)
                chiti2018a_df.at[idx, 'RA_deg'] = row['RA_deg']

            if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):  # Ensure DEC_dms is not NaN
                row['DEC_deg'] = coord.dec_dms_to_deg(str(row['DEC_dms']), precision=6)
                chiti2018a_df.at[idx, 'DEC_deg'] = row['DEC_deg']

            if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):  # Ensure RA_deg is not NaN
                row['RA_hms'] = coord.ra_deg_to_hms(row['RA_deg'], precision=2)
                chiti2018a_df.at[idx, 'RA_hms'] = row['RA_hms']

            if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):  # Ensure DEC_deg is not NaN
                row['DEC_dms'] = coord.dec_deg_to_dms(row['DEC_deg'], precision=2)
                chiti2018a_df.at[idx, 'DEC_dms'] = row['DEC_dms']

        ## Rename A(C) to epsc
        chiti2018a_df.rename(columns={'A(C)': 'epsc'}, inplace=True)
        chiti2018a_df.rename(columns={'llA(C)': 'llc'}, inplace=True)
        chiti2018a_df.rename(columns={'ulA(C)': 'ulc'}, inplace=True)
        chiti2018a_df.rename(columns={'e_A(C)': 'e_epsc'}, inplace=True)

        ## Add/Fill a epsfe column, calculating the value from [Fe/H]
        chiti2018a_df['epsfe'] = chiti2018a_df['[Fe/H]'].apply(lambda x: eps_from_XH(x, 'Fe', precision=2))

        ## Fill the Carbon Abundance Columns
        columns = [('epsc', '[Fe/H]', '[C/Fe]'), ('llc', '[Fe/H]', 'll[C/Fe]'), ('ulc', '[Fe/H]', 'ul[C/Fe]')]
        for eps_col, feh_col, cfe_col in columns:

            # Fill eps_col from [C/Fe], if eps_col is missing
            mask_eps = chiti2018a_df[eps_col].isna() & chiti2018a_df[cfe_col].notna() & chiti2018a_df[feh_col].notna()
            chiti2018a_df.loc[mask_eps, eps_col] = chiti2018a_df.loc[mask_eps].apply(
                lambda row: eps_from_XFe(row[cfe_col], row[feh_col], 'C'), axis=1)

            # Fill [C/Fe] from eps_col, if [C/Fe] is missing
            mask_cfe = chiti2018a_df[cfe_col].isna() & chiti2018a_df[eps_col].notna() & chiti2018a_df[feh_col].notna()
            chiti2018a_df.loc[mask_cfe, cfe_col] = chiti2018a_df.loc[mask_cfe].apply(
                lambda row: XFe_from_eps(row[eps_col], row[feh_col], 'C'), axis=1)

        ## Add/Fill a [C/H] column, calculating the value from epsc
        chiti2018a_df['[C/H]'] = chiti2018a_df['epsc'].apply(lambda x: XH_from_eps(x, 'C', precision=2))
        
        ## Manual changes --> Removing the Halo reference star from the sample
        chiti2018a_df = chiti2018a_df[chiti2018a_df['Name'] != 'CS29497-034']
        
        return chiti2018a_df

    else:
        ## Table 5: MagE Measurements
        mage_df = pd.read_csv(data_dir+'abundance_tables/chiti2018a/table5.csv', comment='#', header=None)
        mage_df.columns = [
            'ID', 'f5_ID', 'Slit', 'logg', 'Teff', 'l_[Fe/H]KP', '[Fe/H]KP', 'e_[Fe/H]KP', 
            'l_A(C)', 'A(C)', 'e_A(C)', 'l_[C/Fe]', '[C/Fe]', 'e_[C/Fe]', '[C/Fe]c',
            'l_[C/Fe]f','[C/Fe]f', 'e_[C/Fe]f', '[Ba/H]', 'RA_deg', 'DEC_deg'
        ]
        mage_df['Reference'] = 'Chiti+2018_MagE'
        mage_df['Ref'] = 'CHI18'
        mage_df = separate_limit_columns(mage_df)
        mage_cols_reorder = [
            'Reference','Ref','ID','f5_ID','RA_deg','DEC_deg','Slit','logg','Teff',
            '[Fe/H]KP','ll[Fe/H]KP','ul[Fe/H]KP','e_[Fe/H]KP',
            'A(C)','llA(C)','ulA(C)','e_A(C)',
            '[C/Fe]','ll[C/Fe]','ul[C/Fe]','e_[C/Fe]',
            '[C/Fe]c','[C/Fe]f','ll[C/Fe]f','ul[C/Fe]f','e_[C/Fe]f','[Ba/H]']
        mage_df = mage_df[mage_cols_reorder]

        ## Table 6: M2FS Measurements
        m2fs_df = pd.read_csv(data_dir+'abundance_tables/chiti2018a/table6.csv', comment='#', header=None)
        m2fs_df.columns = [
            'Type', 'ID', 'f6_ID', 'RA_hms', 'DEC_dms', 'logg', 'Teff',
            'l_[Fe/H]', '[Fe/H]', 'e_[Fe/H]', 
            'l_[C/Fe]', '[C/Fe]', 'e_[C/Fe]', '[C/Fe]c', 'l_[C/Fe]f', '[C/Fe]f'
        ]
        m2fs_df['Reference'] = 'Chiti+2018_M2FS'
        m2fs_df['Ref'] = 'CHI18'
        m2fs_df = separate_limit_columns(m2fs_df)
        m2fs_cols_reorder = [
            'Reference','Ref','ID','f6_ID','RA_hms','DEC_dms','Type','logg','Teff',
            '[Fe/H]','ll[Fe/H]','ul[Fe/H]','e_[Fe/H]',
            '[C/Fe]','ll[C/Fe]','ul[C/Fe]','e_[C/Fe]',
            '[C/Fe]c','[C/Fe]f','ll[C/Fe]f','ul[C/Fe]f']
        m2fs_df = m2fs_df[m2fs_cols_reorder]

        return mage_df, m2fs_df

def load_chiti2018b():
    """
    Load the Chiti et al. 2018b data for the Tucana II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + 'abundance_tables/chiti2018b/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/chiti2018b/table2.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/chiti2018b/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    
    abund_df['l_logepsX'] = abund_df['l_[X/H]']
    
    ## Make the new column names
    species = []
    for ion in abund_df['Species'].unique():
        species_i = ion_to_species(ion)
        elem_i = ion_to_element(ion)
        if species_i not in species:
            species.append(species_i)

    epscols = [make_epscol(s) for s in species]
    ulcols = [make_ulcol(s) for s in species]
    XHcols = [make_XHcol(s).replace(' ', '') for s in species]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(' ', '') for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    chiti2018b_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        chiti2018b_df.loc[i,'Name'] = name
        chiti2018b_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        chiti2018b_df.loc[i,'Reference'] = 'Chiti+2018b'
        chiti2018b_df.loc[i,'Ref'] = 'CHI18b'
        chiti2018b_df.loc[i,'Loc'] = 'UF'
        chiti2018b_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        chiti2018b_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        chiti2018b_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(chiti2018b_df.loc[i,'RA_hms'], precision=6)
        chiti2018b_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        chiti2018b_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(chiti2018b_df.loc[i,'DEC_dms'], precision=2)
        chiti2018b_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        chiti2018b_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        chiti2018b_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        chiti2018b_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row['Species']
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            row['logepsX'] = row['[X/H]'] + row['logepsX_sun']
            
            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                chiti2018b_df.loc[i, col] = row['logepsX'] if pd.isna(row['l_logepsX']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                chiti2018b_df.loc[i, col] = row['logepsX'] if pd.notna(row['l_logepsX']) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row['l_[X/H]']):
                    chiti2018b_df.loc[i, col] = row['[X/H]']
                    chiti2018b_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2018b_df.loc[i, col] = np.nan
                    chiti2018b_df.loc[i, 'ul'+col] = row['[X/H]']
                if 'e_[X/H]' in row.index:
                    chiti2018b_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_[X/Fe]']):
                    chiti2018b_df.loc[i, col] = row['[X/Fe]']
                    chiti2018b_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2018b_df.loc[i, col] = np.nan
                    chiti2018b_df.loc[i, 'ul'+col] = row['[X/Fe]']
                if 'e_[X/Fe]' in row.index:
                    chiti2018b_df.loc[i, 'e_'+col] = row['e_[X/Fe]']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get('e_logepsX', np.nan)
                if pd.notna(e_logepsX):
                    chiti2018b_df.loc[i, col] = e_logepsX
                else:
                    chiti2018b_df.loc[i, col] = np.nan

    return chiti2018b_df

def load_chiti2023():
    """
    Load the Chiti et al. 2023 data for the Tucana II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + 'abundance_tables/chiti2023/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/chiti2023/table2.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/chiti2023/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    
    abund_df['l_logepsX'] = abund_df['l_[X/H]']
    
    ## Make the new column names
    species = []
    for ion in abund_df['Species'].unique():
        species_i = ion_to_species(ion)
        elem_i = ion_to_element(ion)
        if species_i not in species:
            species.append(species_i)

    epscols = [make_epscol(s) for s in species]
    ulcols = [make_ulcol(s) for s in species]
    XHcols = [make_XHcol(s).replace(' ', '') for s in species]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(' ', '') for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    chiti2023_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        chiti2023_df.loc[i,'Name'] = name
        chiti2023_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        chiti2023_df.loc[i,'Reference'] = 'Chiti+2023'
        chiti2023_df.loc[i,'Ref'] = 'CHI23'
        chiti2023_df.loc[i,'Loc'] = 'UF'
        chiti2023_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        chiti2023_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        chiti2023_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(chiti2023_df.loc[i,'RA_hms'], precision=6)
        chiti2023_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        chiti2023_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(chiti2023_df.loc[i,'DEC_dms'], precision=2)
        chiti2023_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        chiti2023_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        chiti2023_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        chiti2023_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row['Species']
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            row['logepsX'] = row['[X/H]'] + row['logepsX_sun']
            
            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                chiti2023_df.loc[i, col] = row['logepsX'] if pd.isna(row['l_logepsX']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                chiti2023_df.loc[i, col] = row['logepsX'] if pd.notna(row['l_logepsX']) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row['l_[X/H]']):
                    chiti2023_df.loc[i, col] = row['[X/H]']
                    chiti2023_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2023_df.loc[i, col] = np.nan
                    chiti2023_df.loc[i, 'ul'+col] = row['[X/H]']
                if 'e_[X/H]' in row.index:
                    chiti2023_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_[X/Fe]']):
                    chiti2023_df.loc[i, col] = row['[X/Fe]']
                    chiti2023_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2023_df.loc[i, col] = np.nan
                    chiti2023_df.loc[i, 'ul'+col] = row['[X/Fe]']
                if 'e_[X/Fe]' in row.index:
                    chiti2023_df.loc[i, 'e_'+col] = row['e_[X/Fe]']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get('e_logepsX', np.nan)
                if pd.notna(e_logepsX):
                    chiti2023_df.loc[i, col] = e_logepsX
                else:
                    chiti2023_df.loc[i, col] = np.nan

    return chiti2023_df

def load_chiti2024():
    """
    Load the Chiti et al. 2024 data for the Large Magellanic Cloud (LMC).

    See `create_chiti2024_yelland.ipynb` for details on how the datafile was created.
    """

    chiti2024_df = pd.read_csv(data_dir + 'abundance_tables/chiti2024/chiti2024_yelland.csv', comment="#")

    ## Remove rows with 'MP_key' = NaN
    chiti2024_df = chiti2024_df[chiti2024_df['MP_key'].notna()] # no abundance data in these rows, only exposure data

    return chiti2024_df

def load_feltzing2009():
    """
    Load the Koch et al. 2008 data for the Hercules Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters (custom made table from the text)
    Table 1 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/feltzing2009/table1a.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/feltzing2009/table1b.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    feltzing2009_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        feltzing2009_df.loc[i,'Name'] = name
        feltzing2009_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        feltzing2009_df.loc[i,'Reference'] = 'Feltzing+2009'
        feltzing2009_df.loc[i,'Ref'] = 'FEL09'
        feltzing2009_df.loc[i,'Loc'] = 'UF'
        feltzing2009_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        feltzing2009_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        feltzing2009_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(feltzing2009_df.loc[i,'RA_hms'], precision=6)
        feltzing2009_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        feltzing2009_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(feltzing2009_df.loc[i,'DEC_dms'], precision=2)
        feltzing2009_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        feltzing2009_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        feltzing2009_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        feltzing2009_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                feltzing2009_df.loc[i, col] = normal_round(row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan, 2)

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                feltzing2009_df.loc[i, col] = normal_round(row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan, 2)

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    feltzing2009_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    feltzing2009_df.loc[i, 'ul'+col] = np.nan
                    # feltzing2009_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    feltzing2009_df.loc[i, col] = np.nan
                    feltzing2009_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    # feltzing2009_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    feltzing2009_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    feltzing2009_df.loc[i, 'ul'+col] = np.nan
                    # feltzing2009_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    feltzing2009_df.loc[i, col] = np.nan
                    feltzing2009_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    # feltzing2009_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    feltzing2009_df.loc[i, col] = e_logepsX
                else:
                    feltzing2009_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    feltzing2009_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return feltzing2009_df

def load_francois2016():
    """
    Load the Francois et al. 2016 data for the Bootes II, Leo IV, Cane Venatici I, Cane Venatici II, and Hercules Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table

    Note: The paper used the Grevesse & Sauval (1998) solar abundances. -- Published in a book chapter in 2000.
          Here, I convert the abundances to the Asplund et al. (2009) solar abundances, which are used in SPAG.
    """

    obs_df = pd.read_csv(data_dir + 'abundance_tables/francois2016/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/francois2016/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/francois2016/table6.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

    elements = []
    for col in abund_df.columns:
        if col.startswith('[') and col.endswith('/H]'): # only used for [Fe/H]
            elements.append(col.replace('[', '').replace('/H]', ''))
        if col.startswith('[') and col.endswith('/Fe]'):
            elements.append(col.replace('[', '').replace('/Fe]', ''))

    epscols = ['eps'+elem.lower() for elem in elements]
    ulcols = ['ul'+elem.lower() for elem in elements]
    XHcols = [f'[{elem}/H]' for elem in elements]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [f'[{elem}/Fe]' for elem in elements]
    ulXFecols = ['ul' + col for col in XFecols]
    # errcols = ['e_'+col for col in XFecols]

    ## New dataframe with proper columns
    francois2016_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols + ulXFecols) # + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        francois2016_df.loc[i,'Name'] = name
        francois2016_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        francois2016_df.loc[i,'Reference'] = 'Francois+2016'
        francois2016_df.loc[i,'Ref'] = 'FRA16'
        francois2016_df.loc[i,'Loc'] = 'UF'
        francois2016_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        francois2016_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        francois2016_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(francois2016_df.loc[i,'RA_hms'], precision=6)
        francois2016_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        francois2016_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(francois2016_df.loc[i,'DEC_dms'], precision=2)
        francois2016_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        francois2016_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        francois2016_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        francois2016_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

    for i, row in abund_df.iterrows():

        for col in abund_df.columns:

            if col in ['Name', 'Simbad_Identifier']: continue
            if col.startswith('l_'): continue

            elem = col.replace('[', '').replace('/H]', '').replace('/Fe]', '')
            epscol = 'eps' + elem.lower()
            ulcol = 'ul' + elem.lower()
            XHcol = f'[{elem}/H]'
            ulXHcol = f'ul[{elem}/H]'
            XFecol = f'[{elem}/Fe]'
            ulXFecol = f'ul[{elem}/Fe]'
            # errcol = 'e_' + XFecol

            ## epsX and ulX
            solar_logepsX_g1998 = get_solar(elem, version='grevesse1998')[0]
            if col.startswith('[') and col.endswith('/H]'): # only used for [Fe/H]
                if pd.isna(row['l_'+XHcol]):
                    francois2016_df.loc[i, epscol] = normal_round(row[XHcol] + solar_logepsX_g1998, 2)
                    francois2016_df.loc[i, ulcol] = np.nan
                else:
                    francois2016_df.loc[i, epscol] = np.nan
                    francois2016_df.loc[i, ulcol] = normal_round(row[XHcol] + solar_logepsX_g1998, 2)

            elif col.startswith('[') and col.endswith('/Fe]'):
                if pd.isna(row['l_'+col]):
                    francois2016_df.loc[i, epscol] = normal_round(row[col] + row['[Fe/H]'] + solar_logepsX_g1998, 2)
                    francois2016_df.loc[i, ulcol] = np.nan
                else:
                    francois2016_df.loc[i, epscol] = np.nan
                    francois2016_df.loc[i, ulcol] = normal_round(row[col] + row['[Fe/H]'] + solar_logepsX_g1998, 2)

            ## XH and ulXH
            solar_logepsX_a2009 = get_solar(elem, version='asplund2009')[0]
            if pd.isna(row['l_'+col]):
                francois2016_df.loc[i, XHcol] = normal_round(francois2016_df.loc[i, epscol] - solar_logepsX_a2009, 2)
                francois2016_df.loc[i, ulXHcol] = np.nan
            else:
                francois2016_df.loc[i, XHcol] = np.nan
                francois2016_df.loc[i, ulXHcol] = normal_round(francois2016_df.loc[i, ulcol] - solar_logepsX_a2009, 2)
            
            ## XFecol and ulXFecol
            if elem != 'Fe':
                if pd.isna(row['l_'+XFecol]):
                    francois2016_df.loc[i, XFecol] = normal_round(francois2016_df.loc[i, XHcol] - francois2016_df.loc[i, '[Fe/H]'], 2)
                    francois2016_df.loc[i, ulXFecol] = np.nan
                else:
                    francois2016_df.loc[i, XFecol] = np.nan
                    francois2016_df.loc[i, ulXFecol] = normal_round(francois2016_df.loc[i, ulXHcol] - francois2016_df.loc[i, '[Fe/H]'], 2)

    return francois2016_df

def load_frebel2010a():
    """
    Load the Frebel et al. 2010a data for the Ursa Major II and Coma Berenices Ultra-Faint Dwarf Galaxies.

    Table 1,2,5 - Observation and Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/frebel2010a/table1_table2_table5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/frebel2010a/table6_table7.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    frebel2010a_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        frebel2010a_df.loc[i,'Name'] = name
        frebel2010a_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        frebel2010a_df.loc[i,'Reference'] = 'Frebel+2010a'
        frebel2010a_df.loc[i,'Ref'] = 'FRE10a'
        frebel2010a_df.loc[i,'Loc'] = 'UF'
        frebel2010a_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]     
        frebel2010a_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        frebel2010a_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(frebel2010a_df.loc[i,'RA_hms'], precision=6)
        frebel2010a_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        frebel2010a_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(frebel2010a_df.loc[i,'DEC_dms'], precision=2)
        frebel2010a_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        frebel2010a_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        frebel2010a_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        frebel2010a_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                frebel2010a_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                frebel2010a_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    frebel2010a_df.loc[i, col] = row["[X/H]"]
                    frebel2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2010a_df.loc[i, col] = np.nan
                    frebel2010a_df.loc[i, 'ul'+col] = row["[X/H]"]
                if 'e_[X/H]' in row.index:
                    frebel2010a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    frebel2010a_df.loc[i, col] = row["[X/Fe]"]
                    frebel2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2010a_df.loc[i, col] = np.nan
                    frebel2010a_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                if 'e_[X/Fe]' in row.index:
                    frebel2010a_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    frebel2010a_df.loc[i, col] = e_logepsX
                else:
                    frebel2010a_df.loc[i, col] = np.nan

    return frebel2010a_df

def load_frebel2010b():
    """
    Sculptor (Scl) Dwarf Galaxy Star

    Load the data from Frebel+2010b, Table 1, for star S1020549.
    """
    
    csv_df = pd.read_csv(data_dir+'abundance_tables/frebel2010b/table1_S1020549.csv', comment='#')

    ## Column names
    species = []
    for ion in csv_df["Species"].unique():
        species_i = ion_to_species(ion)
        elem_i = ion_to_element(ion)
        # default_ion = f"{elem_i} {int_to_roman(get_default_ion(elem_i))}"
        
        if species_i not in species:
            species.append(species_i)

    epscols = [make_epscol(s) for s in species]
    ulcols = [make_ulcol(s) for s in species]
    XHcols = [make_XHcol(s).replace(" ", "") for s in species]
    XFecols = [make_XFecol(s).replace(" ", "") for s in species]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    frebel2010_df = pd.DataFrame(columns=['Name','Reference','Ref','Loc','RA_hms','RA_deg','DEC_dms','DEC_deg'] + epscols + ulcols + XHcols + XFecols + errcols)
    frebel2010_df.loc[0,'Name'] = 'S1020549'
    frebel2010_df.loc[0,'Reference'] = 'Frebel+2010b'
    frebel2010_df.loc[0,'Ref'] = 'FRE10b'
    frebel2010_df.loc[0,'Loc'] = 'DW'
    frebel2010_df.loc[0,'RA_hms'] = '01:00:47.80'
    frebel2010_df.loc[0,'RA_deg'] = coord.ra_hms_to_deg(frebel2010_df.loc[0,'RA_hms'])
    frebel2010_df.loc[0,'DEC_dms'] = '-33:41:03.0'
    frebel2010_df.loc[0,'DEC_deg'] = coord.dec_dms_to_deg(frebel2010_df.loc[0,'DEC_dms'])

    ## Fill in data
    for idx, row in csv_df.iterrows():
        ion = row["Species"]
        species_i = ion_to_species(ion)
        elem_i = ion_to_element(ion)

        ## Assign eps values
        col = make_epscol(species_i)
        if col in epscols:
            frebel2010_df.loc[0, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

        ## Assign upper limit values
        col = make_ulcol(species_i)
        if col in ulcols:
            frebel2010_df.loc[0, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

        ## Assign [X/H] values
        col = make_XHcol(species_i).replace(" ", "")
        if col in XHcols:
            if pd.isna(row["l_[X/H]"]):
                frebel2010_df.loc[0, col] = row["[X/H]"]
                # frebel2010_df.loc[0, 'ul'+col] = np.nan
            else:
                frebel2010_df.loc[0, col] = np.nan
                frebel2010_df.loc[0, 'ul'+col] = row["[X/H]"]

        ## Assign [X/Fe] values
        col = make_XFecol(species_i).replace(" ", "")
        if col in XFecols:
            if pd.isna(row["l_[X/Fe]"]):
                frebel2010_df.loc[0, col] = row["[X/Fe]"]
                # frebel2010_df.loc[0, 'ul'+col] = np.nan
            else:
                frebel2010_df.loc[0, col] = np.nan
                frebel2010_df.loc[0, 'ul'+col] = row["[X/Fe]"]

        ## Assign error values
        col = make_errcol(species_i)
        if col in errcols:
            err_random = row.get("e_random", np.nan)
            err_systematic = row.get("e_systematic", np.nan)
            
            if pd.notna(err_random) and pd.notna(err_systematic):
                frebel2010_df.loc[0, col] = np.sqrt(err_random**2 + err_systematic**2)
            elif pd.notna(err_random):
                frebel2010_df.loc[0, col] = err_random
            elif pd.notna(err_systematic):
                frebel2010_df.loc[0, col] = err_systematic
            else:
                frebel2010_df.loc[0, col] = np.nan

    frebel2010_df.loc[0,'Vel'] = 118.6
    frebel2010_df.loc[0,'Teff'] = 4550
    frebel2010_df.loc[0,'logg'] = 0.9
    frebel2010_df.loc[0,'Vmic'] = 2.8

    return frebel2010_df

def load_frebel2014():
    """
    Load the Frebel et al. 2014 data for the Segue 1 Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table
    """

    obs_df = pd.read_csv(data_dir + 'abundance_tables/frebel2014/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/frebel2014/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/frebel2014/table4.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

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
    frebel2014_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        frebel2014_df.loc[i,'Name'] = name
        frebel2014_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        frebel2014_df.loc[i,'Reference'] = 'Frebel+2014'
        frebel2014_df.loc[i,'Ref'] = 'FRE14'
        frebel2014_df.loc[i,'Loc'] = 'UF'
        frebel2014_df.loc[i,'System'] = 'Segue 1'
        frebel2014_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        frebel2014_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(frebel2014_df.loc[i,'RA_hms'], precision=6)
        frebel2014_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        frebel2014_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(frebel2014_df.loc[i,'DEC_dms'], precision=2)
        frebel2014_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        frebel2014_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        frebel2014_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        frebel2014_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                frebel2014_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                frebel2014_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    frebel2014_df.loc[i, col] = row["[X/H]"]
                    frebel2014_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2014_df.loc[i, col] = np.nan
                    frebel2014_df.loc[i, 'ul'+col] = row["[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    frebel2014_df.loc[i, col] = row["[X/Fe]"]
                    frebel2014_df.loc[i, 'ul'+col] = np.nan
                    # frebel2014_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    frebel2014_df.loc[i, col] = np.nan
                    frebel2014_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                    # frebel2014_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    frebel2014_df.loc[i, col] = e_logepsX
                else:
                    frebel2014_df.loc[i, col] = np.nan

    return frebel2014_df

def load_gilmore2013():
    '''
    Load the Gilmore et al. 2013 data for the Bootes I Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table
    '''

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + 'abundance_tables/gilmore2013/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/gilmore2013/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/gilmore2013/table6.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    
    ## Make the new column names
    species = []
    for ion in abund_df['Species'].unique():
        species_i = ion_to_species(ion)
        elem_i = ion_to_element(ion)
        if species_i not in species:
            species.append(species_i)

    epscols = [make_epscol(s) for s in species]
    ulcols = [make_ulcol(s) for s in species]
    XHcols = [make_XHcol(s).replace(' ', '') for s in species]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(' ', '') for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]
    
    ## New dataframe with proper columns
    gilmore2013_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        gilmore2013_df.loc[i,'Name'] = name
        gilmore2013_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        gilmore2013_df.loc[i,'Reference'] = 'Gilmore+2013'
        gilmore2013_df.loc[i,'Ref'] = 'GIL13'
        gilmore2013_df.loc[i,'Loc'] = 'UF'
        gilmore2013_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        gilmore2013_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        gilmore2013_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(gilmore2013_df.loc[i,'RA_hms'], precision=6)
        gilmore2013_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        gilmore2013_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(gilmore2013_df.loc[i,'DEC_dms'], precision=2)
        gilmore2013_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff_GM'].values[0]
        gilmore2013_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg_GM'].values[0]
        gilmore2013_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H_GM'].values[0]
        gilmore2013_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic_GM'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row['Species']
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                gilmore2013_df.loc[i, col] = row['logepsX_GM'] #if pd.isna(row['l_logepsX_GM']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                gilmore2013_df.loc[i, col] = row['logepsX_GM'] #if pd.notna(row['l_logepsX_GM']) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                # if pd.isna(row['l_[X/H]_GM']):
                gilmore2013_df.loc[i, col] = row['logepsX_GM'] - row['logepsX_sun']
                gilmore2013_df.loc[i, 'ul'+col] = np.nan
                # else:
                #     gilmore2013_df.loc[i, col] = np.nan
                #     gilmore2013_df.loc[i, 'ul'+col] = row['[X/H]_GM']
                if 'e_[X/H]_GM' in row.index:
                    gilmore2013_df.loc[i, 'e_'+col] = row['e_[X/H]_GM']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                # if pd.isna(row['l_[X/Fe]_GM']):
                gilmore2013_df.loc[i, col] = row['[X/Fe]_GM']
                gilmore2013_df.loc[i, 'ul'+col] = np.nan
                # else:
                #     gilmore2013_df.loc[i, col] = np.nan
                #     gilmore2013_df.loc[i, 'ul'+col] = row['[X/Fe]_GM']
                if 'e_[X/Fe]_GM' in row.index:
                    gilmore2013_df.loc[i, 'e_'+col] = row['e_[X/Fe]_GM']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get('e_logepsX_GM', np.nan)
                if pd.notna(e_logepsX):
                    gilmore2013_df.loc[i, col] = e_logepsX
                else:
                    gilmore2013_df.loc[i, col] = np.nan

    ## Get the Carbon data from Norris+2010c & Lai+2011
    norris2010c_df = load_norris2010c(load_gilmore2013=True)
    norris2010c_df = norris2010c_df[norris2010c_df['Reference'] == 'Gilmore+2013']
    for name in gilmore2013_df['Name'].unique():
        if name in norris2010c_df['Name'].values:
            idx = gilmore2013_df[gilmore2013_df['Name'] == name].index[0]
            gilmore2013_df.loc[idx, '[C/Fe]'] = norris2010c_df[norris2010c_df['Name'] == name]['[C/Fe]'].values[0]
            gilmore2013_df.loc[idx, 'epsc'] = norris2010c_df[norris2010c_df['Name'] == name]['epsc'].values[0]
            gilmore2013_df.loc[idx, '[C/H]'] = norris2010c_df[norris2010c_df['Name'] == name]['[C/H]'].values[0]
        elif name == 'BooI-119':
            ## Lai et al. 2011. 'BooI-119' was named 'Boo21' ([C/Fe] = 2.2)
            ### I calculated the epsc value, and converted back to [C/Fe] and [C/H] using Asplund+2009 solar abundances.
            idx = gilmore2013_df[gilmore2013_df['Name'] == name].index[0]
            gilmore2013_df.loc[idx, 'epsc'] = 6.8 # = (cfe + feh) + epsc_sun = ((2.2) + (-3.79)) + (8.39) w/ epsc_sun = 8.39 in Asplund+2005
            gilmore2013_df.loc[idx, '[C/H]'] = gilmore2013_df.loc[idx, 'epsc'] - get_solar('C')[0] # epsc_sun = 8.43 in Asplund+2009
            gilmore2013_df.loc[idx, '[C/Fe]'] = gilmore2013_df.loc[idx, '[C/H]'] - gilmore2013_df.loc[idx, '[Fe/H]'] 

    return gilmore2013_df

def load_hansent2017():
    '''
    Load the Hansen T. et al. 2017 data for the Tucana III Ultra-Faint Dwarf Galaxy.
    '''

    ## Read in the data tables
    data_df = pd.read_csv(data_dir + 'abundance_tables/hansent2017/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

    elements = [col.replace('[','').replace('/Fe]', '') for col in data_df.columns if ((col.endswith('/Fe]') & (col.startswith('['))))]
    elements += [col.replace('ul[','').replace('/Fe]', '') for col in data_df.columns if ((col.endswith('/Fe]') & (col.startswith('ul['))))]

    epscols = ['eps'+elem.lower() for elem in elements]
    ulcols = ['ul'+elem.lower() for elem in elements]
    XHcols = [f'[{elem}/H]' for elem in elements]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [f'[{elem}/Fe]' for elem in elements]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = ['e_'+col for col in XFecols]

    ## New dataframe with proper columns
    hansent2017_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    hansent2017_df['Name'] = data_df['Name'].astype(str)
    hansent2017_df['Simbad_Identifier'] = data_df['Simbad_Identifier'].astype(str)      
    hansent2017_df['Reference'] = 'HansenT+2017'
    hansent2017_df['Ref'] = 'HANt17'
    hansent2017_df['Loc'] = 'UF'
    hansent2017_df['System'] = 'Tucana III'
    hansent2017_df['RA_hms'] = data_df['RA_hms'].astype(str)   
    hansent2017_df['RA_deg'] = coord.ra_hms_to_deg(hansent2017_df['RA_hms'], precision=6)
    hansent2017_df['DEC_dms'] = data_df['DEC_dms'].astype(str)
    hansent2017_df['DEC_deg'] = coord.dec_dms_to_deg(hansent2017_df['DEC_dms'], precision=2)
    hansent2017_df['Teff'] = data_df['Teff'].astype(float)
    hansent2017_df['logg'] = data_df['logg'].astype(float)
    hansent2017_df['Fe/H'] = data_df['Fe/H'].astype(float)
    hansent2017_df['Vmic'] = data_df['Vmic'].astype(float)

    for elem in elements:
        epscol = 'eps' + elem.lower()
        ulcol = 'ul' + elem.lower()
        XHcol = f'[{elem}/H]'
        ulXHcol = f'ul[{elem}/H]'
        XFecol = f'[{elem}/Fe]'
        ulXFecol = f'ul[{elem}/Fe]'
        errcol = f'e_[{elem}/Fe]'

        # [X/Fe]
        hansent2017_df[XFecol] = data_df[XFecol].astype(float) if XFecol in data_df.columns else np.nan

        # ul[X/Fe]
        hansent2017_df[ulXFecol] = data_df[ulXFecol] if ulXFecol in data_df.columns else np.nan

        # [X/H]
        if XHcol in data_df.columns:
            hansent2017_df[XHcol] = data_df[XHcol].astype(float)
        elif XFecol in hansent2017_df and '[Fe/H]' in data_df.columns:
            hansent2017_df[XHcol] = hansent2017_df[XFecol] + data_df['[Fe/H]']
        else:
            hansent2017_df[XHcol] = np.nan

        # ul[X/H]
        if ulXHcol in data_df.columns:
            hansent2017_df[ulXHcol] = data_df[ulXHcol]
        elif ulXFecol in hansent2017_df and '[Fe/H]' in data_df.columns:
            hansent2017_df[ulXHcol] = hansent2017_df[ulXFecol] + data_df['[Fe/H]']
        else:
            hansent2017_df[ulXHcol] = np.nan

        # epsX
        if epscol in data_df.columns:
            hansent2017_df[epscol] = data_df[epscol].astype(float)
        else:
            hansent2017_df[epscol] = hansent2017_df[XHcol] + get_solar(elem)[0]

        # ulX
        if ulcol in data_df.columns:
            hansent2017_df[ulcol] = data_df[ulcol]
        else:
            hansent2017_df[ulcol] = hansent2017_df[ulXHcol] + get_solar(elem)[0]

        # e_[X/Fe]
        if errcol in data_df.columns:
            hansent2017_df[errcol] = data_df[errcol].astype(float)
        else:
            hansent2017_df[errcol] = np.nan
            
    return hansent2017_df

def load_hansent2020():
    """
    Load the Hansen T. et al. 2020 data for the Grus II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 5 - Abundance Table
    """

    obs_df = pd.read_csv(data_dir + 'abundance_tables/hansent2020/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/hansent2020/table2.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/hansent2020/table5.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

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
    hansent2020_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        hansent2020_df.loc[i,'Name'] = name
        hansent2020_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        hansent2020_df.loc[i,'Reference'] = 'HansenT+2020'
        hansent2020_df.loc[i,'Ref'] = 'HANt20'
        hansent2020_df.loc[i,'Loc'] = 'UF'
        hansent2020_df.loc[i,'System'] = 'Grus II'
        hansent2020_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        hansent2020_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(hansent2020_df.loc[i,'RA_hms'], precision=6)
        hansent2020_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        hansent2020_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(hansent2020_df.loc[i,'DEC_dms'], precision=2)
        hansent2020_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        hansent2020_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        hansent2020_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        hansent2020_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                hansent2020_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                hansent2020_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    hansent2020_df.loc[i, col] = row["[X/H]"]
                    hansent2020_df.loc[i, 'ul'+col] = np.nan
                else:
                    hansent2020_df.loc[i, col] = np.nan
                    hansent2020_df.loc[i, 'ul'+col] = row["[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    hansent2020_df.loc[i, col] = row["[X/Fe]"]
                    hansent2020_df.loc[i, 'ul'+col] = np.nan
                    # hansent2020_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    hansent2020_df.loc[i, col] = np.nan
                    hansent2020_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                    # hansent2020_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_stat", np.nan)
                if pd.notna(e_logepsX):
                    hansent2020_df.loc[i, col] = e_logepsX
                else:
                    hansent2020_df.loc[i, col] = np.nan

    return hansent2020_df

def load_hansent2024():
    """
    Load the Hansen T. et al. 2024 data for the Tucana V Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 4 - Abundance Table
    """

    obs_df = pd.read_csv(data_dir + 'abundance_tables/hansent2024/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/hansent2024/table2.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/hansent2024/table4.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

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
    hansent2024_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        hansent2024_df.loc[i,'Name'] = name
        hansent2024_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        hansent2024_df.loc[i,'Reference'] = 'HansenT+2024'
        hansent2024_df.loc[i,'Ref'] = 'HANt24'
        hansent2024_df.loc[i,'Loc'] = 'UF'
        hansent2024_df.loc[i,'System'] = 'Tucana V'
        hansent2024_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        hansent2024_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(hansent2024_df.loc[i,'RA_hms'], precision=6)
        hansent2024_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        hansent2024_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(hansent2024_df.loc[i,'DEC_dms'], precision=2)
        hansent2024_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        hansent2024_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        hansent2024_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        hansent2024_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                hansent2024_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                hansent2024_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    hansent2024_df.loc[i, col] = row["[X/H]"]
                    hansent2024_df.loc[i, 'ul'+col] = np.nan
                else:
                    hansent2024_df.loc[i, col] = np.nan
                    hansent2024_df.loc[i, 'ul'+col] = row["[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    hansent2024_df.loc[i, col] = row["[X/Fe]"]
                    hansent2024_df.loc[i, 'ul'+col] = np.nan
                    # hansent2024_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    hansent2024_df.loc[i, col] = np.nan
                    hansent2024_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                    # hansent2024_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row["e_[X/H]"] #eqivalent to e_logepsX
                if pd.notna(e_logepsX):
                    hansent2024_df.loc[i, col] = e_logepsX
                else:
                    hansent2024_df.loc[i, col] = np.nan

    return hansent2024_df

def load_ishigaki2014b(exclude_mw_halo_ref_stars=True):
    """
    Load the Francois et al. 2016 data for the Bootes I Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 5 - Abundance Table
    """

    obs_df = pd.read_csv(data_dir + 'abundance_tables/ishigaki2014b/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/ishigaki2014b/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/ishigaki2014b/table5.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    
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
    ishigaki2014b_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ishigaki2014b_df.loc[i,'Name'] = name
        ishigaki2014b_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ishigaki2014b_df.loc[i,'Reference'] = 'Ishigaki+2014b'
        ishigaki2014b_df.loc[i,'Ref'] = 'ISH14b'
        ishigaki2014b_df.loc[i,'Loc'] = 'UF'
        ishigaki2014b_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ishigaki2014b_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ishigaki2014b_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(ishigaki2014b_df.loc[i,'RA_hms'], precision=6)
        ishigaki2014b_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ishigaki2014b_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(ishigaki2014b_df.loc[i,'DEC_dms'], precision=2)
        ishigaki2014b_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ishigaki2014b_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        ishigaki2014b_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        ishigaki2014b_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                ishigaki2014b_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ishigaki2014b_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ishigaki2014b_df.loc[i, col] = row["[X/H]"]
                    ishigaki2014b_df.loc[i, 'ul'+col] = np.nan
                    # ishigaki2014b_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    ishigaki2014b_df.loc[i, col] = np.nan
                    ishigaki2014b_df.loc[i, 'ul'+col] = row["[X/H]"]
                    # ishigaki2014b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ishigaki2014b_df.loc[i, col] = row["[X/Fe]"]
                    ishigaki2014b_df.loc[i, 'ul'+col] = np.nan
                    # ishigaki2014b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    ishigaki2014b_df.loc[i, col] = np.nan
                    ishigaki2014b_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                    # ishigaki2014b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_tot", np.nan)
                if pd.notna(e_logepsX):
                    ishigaki2014b_df.loc[i, col] = e_logepsX
                else:
                    ishigaki2014b_df.loc[i, col] = np.nan

    ## Exclude the MW halo reference stars
    if exclude_mw_halo_ref_stars:
        ishigaki2014b_df = ishigaki2014b_df[~ishigaki2014b_df['Name'].isin(['HD216143', 'HD85773'])]

    return ishigaki2014b_df

def load_ji2016a():
    """
    Load the Ji et al. 2016a data for the Bootes II Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/ji2016a/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/ji2016a/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2016a/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    ji2016a_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2016a_df.loc[i,'Name'] = name
        ji2016a_df.loc[i,'Simbad_Identifier'] = name
        ji2016a_df.loc[i,'Reference'] = 'Ji+2016a'
        ji2016a_df.loc[i,'Ref'] = 'JI16a'
        ji2016a_df.loc[i,'Loc'] = 'UF'
        ji2016a_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ji2016a_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ji2016a_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(ji2016a_df.loc[i,'RA_hms'], precision=6)
        ji2016a_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ji2016a_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(ji2016a_df.loc[i,'DEC_dms'], precision=2)
        ji2016a_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ji2016a_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        ji2016a_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        ji2016a_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                ji2016a_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2016a_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ji2016a_df.loc[i, col] = row["[X/H]"]
                    ji2016a_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2016a_df.loc[i, col] = np.nan
                    ji2016a_df.loc[i, 'ul'+col] = row["[X/H]"]
                if 'e_[X/H]' in row.index:
                    ji2016a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2016a_df.loc[i, col] = row["[X/Fe]"]
                    ji2016a_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2016a_df.loc[i, col] = np.nan
                    ji2016a_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                if 'e_[X/Fe]' in row.index:
                    ji2016a_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    ji2016a_df.loc[i, col] = e_logepsX
                else:
                    ji2016a_df.loc[i, col] = np.nan

    return ji2016a_df

def load_ji2016b():
    """
    Load the Ji et al. 2016b data for the Reticulum II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation and Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/ji2016b/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2016b/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    ji2016b_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2016b_df.loc[i,'Name'] = name
        ji2016b_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        ji2016b_df.loc[i,'Reference'] = 'Ji+2016b'
        ji2016b_df.loc[i,'Ref'] = 'JI16b'
        ji2016b_df.loc[i,'Loc'] = 'UF'
        ji2016b_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]     
        ji2016b_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        ji2016b_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(ji2016b_df.loc[i,'RA_hms'], precision=6)
        ji2016b_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        ji2016b_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(ji2016b_df.loc[i,'DEC_dms'], precision=2)
        ji2016b_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        ji2016b_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        ji2016b_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        ji2016b_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                ji2016b_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2016b_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ji2016b_df.loc[i, col] = row["[X/H]"]
                    ji2016b_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2016b_df.loc[i, col] = np.nan
                    ji2016b_df.loc[i, 'ul'+col] = row["[X/H]"]
                if 'e_[X/H]' in row.index:
                    ji2016b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2016b_df.loc[i, col] = row["[X/Fe]"]
                    ji2016b_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2016b_df.loc[i, col] = np.nan
                    ji2016b_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                if 'e_[X/Fe]' in row.index:
                    ji2016b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    ji2016b_df.loc[i, col] = e_logepsX
                else:
                    ji2016b_df.loc[i, col] = np.nan

    return ji2016b_df

def load_ji2019():
    """
    Load the Ji et al. 2019 data for the Grus I AND Triangulum II Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/ji2019/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/ji2019/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2019/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    param_df = param_df[param_df['Ref'] == 'TW']
    
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
    ji2019_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2019_df.loc[i,'Name'] = name
        ji2019_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ji2019_df.loc[i,'Reference'] = 'Ji+2019'
        ji2019_df.loc[i,'Ref'] = 'JI19'
        ji2019_df.loc[i,'Loc'] = 'UF'
        ji2019_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ji2019_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ji2019_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(ji2019_df.loc[i,'RA_hms'], precision=6)
        ji2019_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ji2019_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(ji2019_df.loc[i,'DEC_dms'], precision=2)
        ji2019_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ji2019_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        ji2019_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        ji2019_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                ji2019_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2019_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ji2019_df.loc[i, col] = row["[X/H]"]
                    ji2019_df.loc[i, 'ul'+col] = np.nan
                    ji2019_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    ji2019_df.loc[i, col] = np.nan
                    ji2019_df.loc[i, 'ul'+col] = row["[X/H]"]
                    ji2019_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2019_df.loc[i, col] = row["[X/Fe]"]
                    ji2019_df.loc[i, 'ul'+col] = np.nan
                    ji2019_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    ji2019_df.loc[i, col] = np.nan
                    ji2019_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                    ji2019_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("stderr_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    ji2019_df.loc[i, col] = e_logepsX
                else:
                    ji2019_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    ji2019_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return ji2019_df

def load_ji2020():
    """
    Load the Ji et al. 2020 data for the Carina II and Carina III Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations
    Table 2 - Radial Velocities
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/ji2020/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    rv_df = pd.read_csv(data_dir + "abundance_tables/ji2020/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/ji2020/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2020/table6.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
    abund_df['l_[X/H]'] = abund_df['l_logepsX']
    abund_df['l_[X/Fe]'] = abund_df['l_logepsX']

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
    ji2020_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','M/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2020_df.loc[i,'Name'] = name
        ji2020_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ji2020_df.loc[i,'Reference'] = 'Ji+2020'
        ji2020_df.loc[i,'Ref'] = 'JI20'
        ji2020_df.loc[i,'Loc'] = 'UF'
        ji2020_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ji2020_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ji2020_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(ji2020_df.loc[i,'RA_hms'], precision=6)
        ji2020_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ji2020_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(ji2020_df.loc[i,'DEC_dms'], precision=2)
        ji2020_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ji2020_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        ji2020_df.loc[i,'M/H'] = param_df.loc[param_df['Name'] == name, '[M/H]'].values[0]
        ji2020_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                ji2020_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2020_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ji2020_df.loc[i, col] = row["[X/H]"]
                    ji2020_df.loc[i, 'ul'+col] = np.nan
                    ji2020_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    ji2020_df.loc[i, col] = np.nan
                    ji2020_df.loc[i, 'ul'+col] = row["[X/H]"]
                    ji2020_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2020_df.loc[i, col] = row["[X/Fe]"]
                    ji2020_df.loc[i, 'ul'+col] = np.nan
                    ji2020_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    ji2020_df.loc[i, col] = np.nan
                    ji2020_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                    ji2020_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_stat", np.nan)
                if pd.notna(e_logepsX):
                    ji2020_df.loc[i, col] = e_logepsX
                else:
                    ji2020_df.loc[i, col] = np.nan

    return ji2020_df

def load_kirby2017():
    """
    Load the Kirby et al. 2017 data for the Triangulum II Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters (custom made table from the text)
    Table 6 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/kirby2017/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/kirby2017/table6.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    kirby2017_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        kirby2017_df.loc[i,'Name'] = name
        kirby2017_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        kirby2017_df.loc[i,'Reference'] = 'Kirby+2017'
        kirby2017_df.loc[i,'Ref'] = 'KIR17'
        kirby2017_df.loc[i,'Loc'] = 'UF'
        kirby2017_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        kirby2017_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        kirby2017_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(kirby2017_df.loc[i,'RA_hms'], precision=6)
        kirby2017_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        kirby2017_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(kirby2017_df.loc[i,'DEC_dms'], precision=2)
        kirby2017_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        kirby2017_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        kirby2017_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        kirby2017_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                kirby2017_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                kirby2017_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    kirby2017_df.loc[i, col] = row["[X/H]"]
                    kirby2017_df.loc[i, 'ul'+col] = np.nan
                    # kirby2017_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    kirby2017_df.loc[i, col] = np.nan
                    kirby2017_df.loc[i, 'ul'+col] = row["[X/H]"]
                    # kirby2017_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    kirby2017_df.loc[i, col] = row["[X/Fe]"]
                    kirby2017_df.loc[i, 'ul'+col] = np.nan
                    # kirby2017_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    kirby2017_df.loc[i, col] = np.nan
                    kirby2017_df.loc[i, 'ul'+col] = row["[X/Fe]"]
                    # kirby2017_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    kirby2017_df.loc[i, col] = e_logepsX
                else:
                    kirby2017_df.loc[i, col] = np.nan

    return kirby2017_df

def load_koch2008c():
    """
    Load the Koch et al. 2008c data for the Hercules Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters (custom made table from the text)
    Table 1 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/koch2008c/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/koch2008c/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    koch2008c_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        koch2008c_df.loc[i,'Name'] = name
        koch2008c_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        koch2008c_df.loc[i,'Reference'] = 'Koch+2008c'
        koch2008c_df.loc[i,'Ref'] = 'KOC08c'
        koch2008c_df.loc[i,'Loc'] = 'UF'
        koch2008c_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        koch2008c_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        koch2008c_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(koch2008c_df.loc[i,'RA_hms'], precision=6)
        koch2008c_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        koch2008c_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(koch2008c_df.loc[i,'DEC_dms'], precision=2)
        koch2008c_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        koch2008c_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        koch2008c_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        koch2008c_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Calculate the logepsX value
            if ion not in ['Fe I', 'Fe II']:
                xfe_a05 = row["[X/Fe]"]
                feh_a05 = star_df.loc[star_df['Species'] == 'Fe I', '[X/H]'].values[0]
                logepsX_sun_a05 = get_solar(elem_i, version='asplund2005')[0]
                logepsX = normal_round(xfe_a05 + feh_a05 + logepsX_sun_a05, 2)
            else:
                logepsX_sun_a05 = get_solar(elem_i, version='asplund2005')[0]
                logepsX = normal_round(row["[X/H]"] + logepsX_sun_a05, 2)
            
            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                koch2008c_df.loc[i, col] = logepsX if pd.isna(row["l_[X/Fe]"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                koch2008c_df.loc[i, col] = logepsX if pd.notna(row["l_[X/Fe]"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    koch2008c_df.loc[i, col] = normal_round(logepsX - logepsX_sun_a09, 2)
                    koch2008c_df.loc[i, 'ul'+col] = np.nan
                    # koch2008c_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    koch2008c_df.loc[i, col] = np.nan
                    koch2008c_df.loc[i, 'ul'+col] = normal_round(logepsX - logepsX_sun_a09, 2)
                    # koch2008c_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            feh_a09 = koch2008c_df.loc[koch2008c_df['Name'] == name, '[Fe/H]'].values[0]
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    koch2008c_df.loc[i, col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    koch2008c_df.loc[i, 'ul'+col] = np.nan
                    # koch2008c_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    koch2008c_df.loc[i, col] = np.nan
                    koch2008c_df.loc[i, 'ul'+col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    # koch2008c_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_XFe = row.get("e_[X/Fe]", np.nan)
                if pd.notna(e_XFe):
                    koch2008c_df.loc[i, col] = e_XFe
                else:
                    koch2008c_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    koch2008c_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return koch2008c_df

def load_koch2013b():
    """
    Load the Koch et al. 2013b data for the Hercules Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters (custom made table from the text and Aden+2011)
    Table 1 - Abundance Table (chose to use the 3-sigma upper limits for Ba)
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/koch2013b/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/koch2013b/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    epscols = ['epsfe', 'epsca', 'epsba']
    ulcols = ['ulca', 'ulba']
    XHcols = ['[Fe/H]', '[Ca/H]', '[Ba/H]']
    ulXHcols = ['ul[Ca/H]', 'ul[Ba/H]']
    XFecols = ['[Ca/Fe]', '[Ba/Fe]']
    ulXFecols = ['ul[Ca/Fe]', 'ul[Ba/Fe]']

    ## New dataframe with proper columns
    koch2013b_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols)
    for i, name in enumerate(abund_df['Name'].unique()):
        koch2013b_df.loc[i,'Name'] = name
        koch2013b_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        koch2013b_df.loc[i,'Reference'] = 'Koch+2013b'
        koch2013b_df.loc[i,'Ref'] = 'KOC13b'
        koch2013b_df.loc[i,'Loc'] = 'UF'
        koch2013b_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        koch2013b_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        koch2013b_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(koch2013b_df.loc[i,'RA_hms'], precision=6)
        koch2013b_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        koch2013b_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(koch2013b_df.loc[i,'DEC_dms'], precision=2)
        koch2013b_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        koch2013b_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        koch2013b_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        koch2013b_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        logepsFe_sun_a09 = get_solar('Fe', version='asplund2009')[0]
        logepsCa_sun_a09 = get_solar('Ca', version='asplund2009')[0]
        logepsBa_sun_a09 = get_solar('Ba', version='asplund2009')[0]

        koch2013b_df.loc[i,'epsfe'] = normal_round(abund_df.loc[abund_df['Name'] == name, '[Fe/H]'].values[0] + logepsFe_sun_a09, 2)
        koch2013b_df.loc[i,'[Fe/H]'] = normal_round(abund_df.loc[abund_df['Name'] == name, '[Fe/H]'].values[0], 2)

        if pd.notna(abund_df.loc[abund_df['Name'] == name, '[Ca/H]'].values[0]):
            koch2013b_df.loc[i,'epsca'] = normal_round(abund_df.loc[abund_df['Name'] == name, '[Ca/H]'].values[0] + logepsCa_sun_a09, 2)
            koch2013b_df.loc[i,'[Ca/H]'] = normal_round(abund_df.loc[abund_df['Name'] == name, '[Ca/H]'].values[0], 2)
            koch2013b_df.loc[i,'[Ca/Fe]'] = normal_round(koch2013b_df.loc[i,'[Ca/H]'] - koch2013b_df.loc[i,'[Fe/H]'], 2)
            koch2013b_df.loc[i,'ulca'] = np.nan
            koch2013b_df.loc[i,'ul[Ca/H]'] = np.nan
            koch2013b_df.loc[i,'ul[Ca/Fe]'] = np.nan
        else:
            koch2013b_df.loc[i,'ulca'] = np.nan
            koch2013b_df.loc[i,'epsca'] = np.nan
            koch2013b_df.loc[i,'[Ca/H]'] = np.nan
            koch2013b_df.loc[i,'[Ca/Fe]'] = np.nan
            koch2013b_df.loc[i,'ul[Ca/H]'] = np.nan
            koch2013b_df.loc[i,'ul[Ca/Fe]'] = np.nan

        if pd.isna(abund_df.loc[abund_df['Name'] == name, 'l_logepsBa_3sig'].values[0]):
            koch2013b_df.loc[i,'epsba'] = normal_round(abund_df.loc[abund_df['Name'] == name, 'logepsBa_3sig'].values[0], 2)
            koch2013b_df.loc[i,'[Ba/H]'] = normal_round(abund_df.loc[abund_df['Name'] == name, 'logepsBa_3sig'].values[0] - logepsBa_sun_a09, 2)
            koch2013b_df.loc[i,'[Ba/Fe]'] = normal_round(koch2013b_df.loc[i,'[Ba/H]'] - koch2013b_df.loc[i,'[Fe/H]'], 2)
            koch2013b_df.loc[i,'ulba'] = np.nan
            koch2013b_df.loc[i,'ul[Ba/H]'] = np.nan
            koch2013b_df.loc[i,'ul[Ba/Fe]'] = np.nan
        else:
            koch2013b_df.loc[i,'epsba'] = np.nan
            koch2013b_df.loc[i,'[Ba/H]'] = np.nan
            koch2013b_df.loc[i,'[Ba/Fe]']  = np.nan
            koch2013b_df.loc[i,'ulba'] = normal_round(abund_df.loc[abund_df['Name'] == name, 'logepsBa_3sig'].values[0], 2)
            koch2013b_df.loc[i,'ul[Ba/H]'] = normal_round(abund_df.loc[abund_df['Name'] == name, 'logepsBa_3sig'].values[0] - logepsBa_sun_a09, 2)
            koch2013b_df.loc[i,'ul[Ba/Fe]'] = normal_round(koch2013b_df.loc[i,'ul[Ba/H]'] - koch2013b_df.loc[i,'[Fe/H]'], 2)

    return koch2013b_df

def load_lai2011():
    """
    Load the Lai et al. 2011 data for the Bootes I Ultra-Faint Dwarf Galaxy.

    Table 1 - All Data
    """

    ## Read in the data tables
    data_df = pd.read_csv(data_dir + "abundance_tables/lai2011/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    ## New dataframe with proper columns
    lai2011_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + ['epsfe', '[Fe/H]','ulfe', 'ul[Fe/H]', 'epsc', '[C/H]', '[C/Fe]', 'ulc', 'ul[C/H]', 'ul[C/Fe]'])
    for i, name in enumerate(data_df['Name'].unique()):
        lai2011_df.loc[i,'Name'] = name
        lai2011_df.loc[i,'Simbad_Identifier'] = data_df.loc[data_df['Name'] == name, 'Simbad_Identifier'].values[0]
        lai2011_df.loc[i,'Reference'] = 'Lai+2011'
        lai2011_df.loc[i,'Ref'] = 'LAI11'
        lai2011_df.loc[i,'Loc'] = 'UF'
        lai2011_df.loc[i,'System'] = data_df.loc[data_df['Name'] == name, 'System'].values[0]
        lai2011_df.loc[i,'RA_hms'] = data_df.loc[data_df['Name'] == name, 'RA_hms'].values[0]
        lai2011_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(lai2011_df.loc[i,'RA_hms'], precision=6)
        lai2011_df.loc[i,'DEC_dms'] = data_df.loc[data_df['Name'] == name, 'DEC_dms'].values[0]
        lai2011_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(lai2011_df.loc[i,'DEC_dms'], precision=2)
        lai2011_df.loc[i,'Teff'] = data_df.loc[data_df['Name'] == name, 'Teff'].values[0]
        lai2011_df.loc[i,'logg'] = data_df.loc[data_df['Name'] == name, 'logg'].values[0]
        lai2011_df.loc[i,'Vmic'] = data_df.loc[data_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in the Iron and Carbon Data
        logepsFe_sun_a05 = rd.get_solar('Fe', version='asplund2005')[0]
        logepsFe_sun_a09 = rd.get_solar('Fe', version='asplund2009')[0]
        logepsC_sun_a05 = rd.get_solar('C', version='asplund2005')[0]
        logepsC_sun_a09 = rd.get_solar('C', version='asplund2009')[0]

        feh_a05 = data_df.loc[data_df['Name'] == name, '[Fe/H]'].values[0]
        cfe_a05 = data_df.loc[data_df['Name'] == name, '[C/Fe]'].values[0]

        lai2011_df.loc[i,'epsfe'] = normal_round(feh_a05 + logepsFe_sun_a05, 2)
        lai2011_df.loc[i,'Fe/H'] = normal_round(lai2011_df.loc[i,'epsfe'] - logepsFe_sun_a09, 2)
        lai2011_df.loc[i,'[Fe/H]'] = normal_round(lai2011_df.loc[i,'epsfe'] - logepsFe_sun_a09, 2)
        lai2011_df.loc[i,'ulfe'] = np.nan
        lai2011_df.loc[i,'ul[Fe/H]'] = np.nan
        
        if pd.isna(data_df.loc[i, 'l_[C/Fe]']):
            lai2011_df.loc[i,'epsc'] = normal_round(cfe_a05 + feh_a05 + logepsC_sun_a05, 2)
            lai2011_df.loc[i,'[C/H]'] = normal_round(lai2011_df.loc[i,'epsc'] - logepsC_sun_a09, 2)
            lai2011_df.loc[i,'[C/Fe]'] = normal_round(lai2011_df.loc[i,'[C/H]'] - lai2011_df.loc[i,'[Fe/H]'], 2)
            lai2011_df.loc[i,'ulc'] = np.nan
            lai2011_df.loc[i,'ul[C/H]'] = np.nan
            lai2011_df.loc[i,'ul[C/Fe]'] = np.nan
        else:
            lai2011_df.loc[i,'epsc'] = np.nan
            lai2011_df.loc[i,'[C/H]'] = np.nan
            lai2011_df.loc[i,'[C/Fe]'] = np.nan
            lai2011_df.loc[i,'ulc'] = normal_round(cfe_a05 + feh_a05 + logepsC_sun_a05, 2)
            lai2011_df.loc[i,'ul[C/H]'] = normal_round(lai2011_df.loc[i,'ulc'] - logepsC_sun_a09, 2)
            lai2011_df.loc[i,'ul[C/Fe]'] = normal_round(lai2011_df.loc[i,'ul[C/H]'] - lai2011_df.loc[i,'[Fe/H]'], 2)

    return lai2011_df

def load_lemasle2014():
    """
    Fornax (Fnx) Dwarf Galaxy Stars

    Loads the data from Lemasle et al. (2014) for the Fornax stars.
    """
    ## Read FITS file and strip units
    tab = Table.read(data_dir+"abundance_tables/lemasle2014/lemasle14_fnx.txt",format='ascii.fixed_width')
    tab["Star"] = tab["Star"].astype(str)

    ## Remove all column units (prevents UnitsWarning)
    for col in tab.colnames:
        tab[col].unit = None

    ## Remove unwanted columns
    # tab.remove_columns(["[TiI/H]","e_ti1","fe2_h","e_fe2"])
    for col in tab.colnames:
        if col.startswith("N"): tab.remove_column(col)

    ## Rename the Star column
    tab.rename_column("Star","Name")

    elems = ["Na","Mg","Si","Ca","Sc","Ti","Cr","Ni","Y","Ba","La","Nd","Eu"]
    tab["ulfe"] = False
    for elem in elems:
        tab[XFecol(elem)] = tab[XHcol(elem)] - tab["[Fe/H]"]
        tab[ulcol(elem)] = False
    
    ## Convert to Pandas DataFrame
    lemasle2014_df = tab.to_pandas()
    epscol_from_XHcol(lemasle2014_df)

    lemasle2014_df["Loc"] = "DW"
    lemasle2014_df["Reference"] = "Lemasle+2014"
    lemasle2014_df["Ref"] = "LEM14"

    ## Reorder the columns
    cols = ["Name", "Reference", "Ref", "Loc"] + epscolnames(lemasle2014_df) + ulcolnames(lemasle2014_df) + XHcolnames(lemasle2014_df) + XFecolnames(lemasle2014_df) + errcolnames(lemasle2014_df)
    cols_missing = [col for col in cols if col not in lemasle2014_df.columns]
    lemasle2014_df = lemasle2014_df[cols + cols_missing]

    return lemasle2014_df

def load_letarte2010():
    """
    Fornax (Fnx) Dwarf Galaxy Stars

    Loads the data from Letarte et al. (2010) for the Fornax stars.
    """
    # Read FITS file and strip units
    # tab = Table.read(data_dir+"abundance_tables/letarte2010/letarte10_fornax.fits") # generates a UnitsWarning from '[-]'
    with fits.open(data_dir+"abundance_tables/letarte2010/letarte10_fornax.fits") as hdul:
        tab = Table(hdul[1].data)  # skip unit parsing
    tab["Star"] = tab["Star"].astype(str)
    tab.rename_column("Star", "Name")

    ## Remove all column units (prevents UnitsWarning)
    for col in tab.colnames:
        tab[col].unit = None  

    ## Remove unwanted columns
    for col in tab.colnames:
        if col.startswith("o__"): 
            tab.remove_column(col)

    ## Rename element columns
    elemmap = {"NaI": "Na", "MgI": "Mg", "SiI": "Si", "CaI": "Ca", "TiII": "Ti",
               "CrI": "Cr", "NiI": "Ni", "YII": "Y",
               "BaII": "Ba", "LaII": "La", "NdII": "Nd", "EuII": "Eu"}

    for e1, e2 in elemmap.items():
        tab.rename_column(f"__{e1}_Fe_", f"[{e2}/Fe]")
        tab.rename_column(f"e__{e1}_Fe_", f"e_{e2.lower()}")
        tab[ulcol(e2)] = False

    tab["ulfe"] = False
    tab.rename_column("__FeI_H_", "[Fe/H]")
    tab.rename_column("e__FeI_H_", "e_fe")

    ## Convert to Pandas DataFrame
    letarte2010_df = tab.to_pandas()
    XHcol_from_XFecol(letarte2010_df)
    epscol_from_XHcol(letarte2010_df)

    letarte2010_df.rename(columns={"__FeII_H_": "[Fe II/H]",
                       "e__FeII_H_": "e_fe2",
                       "__TiI_Fe_": "[Ti I/Fe]",
                       "e__TiI_Fe_": "e_ti1"},
              inplace=True)

    letarte2010_df["Loc"] = "DW"
    letarte2010_df["Reference"] = "Letarte+2010"
    letarte2010_df["Ref"] = "LET10"

    ## Reorder the columns
    cols = ["Name", "Reference", "Ref", "Loc"] + epscolnames(letarte2010_df) + ulcolnames(letarte2010_df) + XHcolnames(letarte2010_df) + XFecolnames(letarte2010_df) + errcolnames(letarte2010_df)
    cols_missing = [col for col in cols if col not in letarte2010_df.columns]
    letarte2010_df = letarte2010_df[cols + cols_missing]

    return letarte2010_df

def load_lucchetti2024():
    """
    Carina (Car) and Fornax (Fnx) Dwarf Galaxy Stars

    Loads the data from Lucchetti et al. (2024) for the Carina and Fornax stars. There
    are 4 stars in Carina and 2 stars in Fornax. The data is stored in a single table.
    This function reads in the data from the table and returns it as a pandas DataFrame.
    """
    
    csv_df = pd.read_csv(data_dir + 'abundance_tables/lucchesi2024/lucchesi2024_carina_fornax.csv')

    ## Extract element species from column headers (ignoring first two columns)
    ions = csv_df.columns[2:]
    species = [ion_to_species(ion) for ion in ions]

    ## Generate column names dynamically
    epscols = [make_epscol(s) for s in species]
    ulcols = [make_ulcol(s) for s in species]
    XHcols = [make_XHcol(s).replace(" ","") for s in species]
    XFecols = [make_XFecol(s).replace(" ","") for s in species]
    errcols = [make_errcol(s) for s in species]

    ## Create new DataFrame for final output
    lucchetti2024_df = pd.DataFrame(columns=['Name', 'Reference', 'Ref', 'Loc', 'RA_hms', 'RA_deg', 'DEC_dms', 'DEC_deg']
                                    + epscols + ulcols + XHcols + XFecols + errcols)

    ## Process each unique star
    star_data = []
    
    for star_name in csv_df["Star"].unique():
        star_df = csv_df[csv_df["Star"] == star_name].copy()  # Filter rows for this star
        
        ## Transpose data to make elements rows instead of columns
        species_df = star_df.copy()
        species_df.drop("Star", axis=1, inplace=True) # Drop the 'Star' column
        species_df = species_df.transpose().reset_index() # Transpose and reset index
        species_df.columns = species_df.iloc[0] # Set first row as header
        species_df = species_df[1:].reset_index(drop=True) # drop first row
        species_df.rename(columns={'Measure': 'Species'}, inplace=True) # Rename 'Measure' column to 'Species'

        ## Prepare row data for bulk assignment
        star_row = {'Name': star_name, 'Reference': 'Lucchetti+2024', 'Ref': 'LUH24', 'Loc': 'DW'}

        ## Fill in abundance data
        for _, row in species_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)

            ## Assign logepsX values
            col = make_epscol(species_i)
            if col in epscols and pd.notna(row["logepsX"]):
                star_row[col] = row["logepsX"] if "<" not in str(row["logepsX"]) else np.nan

            ## Assign upper limit values
            col = make_ulcol(species_i)
            if col in ulcols and pd.notna(row["logepsX"]):
                star_row[col] = row["logepsX"].split("<")[1] if "<" in str(row["logepsX"]) else np.nan

            ## Assign [X/H] values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols and pd.notna(row["[X/H]"]):
                if "<" in str(row["[X/H]"]):
                    star_row['ul'+col] = row["[X/H]"].split("<")[1]
                else:
                    star_row[col] = row["[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols and pd.notna(row["[X/Fe]"]):
                if "<" in str(row["[X/Fe]"]):
                    star_row['ul'+col] = row["[X/Fe]"].split("<")[1]
                else:
                    star_row[col] = row["[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols and pd.notna(row["e_[X/Fe]"]):
                star_row[col] = row["e_[X/Fe]"]

        ## Collect processed star data for bulk insertion
        star_data.append(star_row)

    ## Convert list of dicts to DataFrame
    lucchetti2024_df = pd.DataFrame(star_data)

    ## Drop all cols with 'Fe/Fe' in them
    for col in lucchetti2024_df.columns:
        if "Fe/Fe" in col:
            lucchetti2024_df.drop(col, axis=1, inplace=True)
        
    ## Add coordinates efficiently using a DataFrame merge
    aux_data = pd.read_csv(data_dir + 'abundance_tables/lucchesi2024/aux_data.csv')

    ## Convert RA/DEC to degrees
    aux_data['RA_deg'] = aux_data['RA_hms'].apply(coord.ra_hms_to_deg)
    aux_data['DEC_deg'] = aux_data['DEC_dms'].apply(coord.dec_dms_to_deg)

    ## Merge coordinates into the main DataFrame (fast operation)
    lucchetti2024_df = lucchetti2024_df.merge(aux_data, on=['Name'], how='left')
    
    return lucchetti2024_df

def load_mardini2022():

    mardini2022_df = pd.read_csv(data_dir+'abundance_tables/mardini2022/tab5_yelland.csv', comment='#')

    ## Add and rename the necessary columns
    # mardini2022_df.rename(columns={'source_id':'Name', 'ra':'RA_hms', 'dec':'DEC_deg', 'teff':'Teff'}, inplace=True)
    mardini2022_df['JINA_ID'] = mardini2022_df['JINA_ID'].astype(int)
    mardini2022_df['Name'] = mardini2022_df['Simbad_Identifier']
    # mardini2022_df['Reference'] = 'Mardini+2022'
    mardini2022_df['Ref'] = mardini2022_df['Reference'].str[:3].str.upper() + mardini2022_df['Reference'].str[-2:]
    mardini2022_df['Ncap_key'] = ''
    mardini2022_df['C_key'] = mardini2022_df['[C/Fe]'].apply(lambda cfe: classify_carbon_enhancement(cfe) if pd.notna(cfe) else np.nan)
    mardini2022_df['MP_key'] = mardini2022_df['[Fe/H]'].apply(lambda feh: classify_metallicity(feh) if pd.notna(feh) else np.nan)
    mardini2022_df['Loc'] = 'DS'
    mardini2022_df['RA_deg'] = np.nan
    mardini2022_df['DEC_deg'] = np.nan

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in mardini2022_df.iterrows():

        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            ## pad RA_hms with leading zeros
            if len(row['RA_hms']) == 10:
                row['RA_hms'] = '0' + row['RA_hms']
                mardini2022_df.at[idx, 'RA_hms'] = row['RA_hms']
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            mardini2022_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=6)
            mardini2022_df.at[idx, 'DEC_deg'] = row['DEC_deg']
            
        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(row['RA_deg'], precision=2)
            mardini2022_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(row['DEC_deg'], precision=2)
            mardini2022_df.at[idx, 'DEC_dms'] = row['DEC_dms']

    ## Get the JINAbase Data using the JINA_ID
    jina_ids = list(mardini2022_df['JINA_ID'])

    jinabase_df = load_jinabase(priority=None)
    sub_jinabase_df = jinabase_df[jinabase_df['JINA_ID'].isin(jina_ids)].copy()
    new_columns = [col for col in sub_jinabase_df.columns if col not in mardini2022_df.columns]
    # new_columns = ['logg']

    # Align on JINA_ID
    mardini2022_df = mardini2022_df.set_index('JINA_ID')
    sub_jinabase_df = sub_jinabase_df.set_index('JINA_ID')
    mardini2022_df = mardini2022_df.join(sub_jinabase_df[new_columns], how='left')

    # Fill in missing [C/Fe] values from JINAbase
    if '[C/Fe]' in mardini2022_df.columns and '[C/Fe]' in sub_jinabase_df.columns:
        mardini2022_df['[C/Fe]'] = mardini2022_df['[C/Fe]'].fillna(sub_jinabase_df['[C/Fe]'])
    if 'ul[C/Fe]' in mardini2022_df.columns and 'ul[C/Fe]' in sub_jinabase_df.columns:
        mardini2022_df['ul[C/Fe]'] = mardini2022_df['ul[C/Fe]'].fillna(sub_jinabase_df['ul[C/Fe]'])

    ## Manually added datafields
    mardini2022_df.loc[mardini2022_df['Name'] == 'SDSS J124502.68-073847.0', 'Ncap_key'] = 'S'  # halo reference star
    mardini2022_df.loc[mardini2022_df['Name'] == 'HE 0017-4346', 'Ncap_key'] = 'S'          # [C/Fe] = 3.02
    mardini2022_df.loc[mardini2022_df['Name'] == 'HE 1413-1954', 'C_key'] = 'NO'      # [C/Fe] = 1.44
    mardini2022_df.loc[mardini2022_df['Name'] == 'HE 1300+0157', 'C_key'] = 'NO'            # (HE 1300+0157, https://www.aanda.org/articles/aa/pdf/2019/03/aa34601-18.pdf)
    mardini2022_df.loc[mardini2022_df['Reference'] == 'Aguado+2017', 'logg'] = 4.9
    mardini2022_df.loc[mardini2022_df['Name'] == 'SDSS J124719.46-034152.4', 'logg'] = 4.0
    mardini2022_df.loc[mardini2022_df['Name'] == 'SDSS J105519.28+232234.0', 'logg'] = 4.9
    
    ## Reset the index
    sub_jinabase_df = sub_jinabase_df.reset_index()
    mardini2022_df = mardini2022_df.reset_index()

    ## Save the processed data to a CSV file
    mardini2022_df.to_csv(data_dir+'abundance_tables/mardini2022/tab5_processed.csv', index=False)

    return mardini2022_df

def load_marshall2019():
    """
    Load the Roederer et al. 2014 data for the Segue 1 Ultra-Faint Dwarf Galaxy.

    Table 1 - Observations
    Table 2 - Stellar Parameters
    Table 4 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/marshall2019/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/marshall2019/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/marshall2019/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    marshall2019_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        marshall2019_df.loc[i,'Name'] = name
        marshall2019_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        marshall2019_df.loc[i,'Reference'] = 'Marshall+2019'
        marshall2019_df.loc[i,'Ref'] = 'MAR19'
        marshall2019_df.loc[i,'Loc'] = 'UF'
        marshall2019_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        marshall2019_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        marshall2019_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(marshall2019_df.loc[i,'RA_hms'], precision=6)
        marshall2019_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        marshall2019_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(marshall2019_df.loc[i,'DEC_dms'], precision=2)
        marshall2019_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        marshall2019_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        marshall2019_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        marshall2019_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                marshall2019_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                marshall2019_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    marshall2019_df.loc[i, col] = row["logepsX"] - logepsX_sun_a09
                    marshall2019_df.loc[i, 'ul'+col] = np.nan
                    # marshall2019_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    marshall2019_df.loc[i, col] = np.nan
                    marshall2019_df.loc[i, 'ul'+col] = row["logepsX"] - logepsX_sun_a09
                    # marshall2019_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    marshall2019_df.loc[i, col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    marshall2019_df.loc[i, 'ul'+col] = np.nan
                    # marshall2019_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    marshall2019_df.loc[i, col] = np.nan
                    marshall2019_df.loc[i, 'ul'+col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    # marshall2019_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    marshall2019_df.loc[i, col] = e_logepsX
                else:
                    marshall2019_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    marshall2019_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return marshall2019_df

def load_nagasawa2018():
    """
    Load the Roederer et al. 2014 data for the Segue 1 Ultra-Faint Dwarf Galaxy.

    Table 1 & Table 2 - Observations
    Table 4 - Stellar Parameters
    Table 5 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/nagasawa2018/table1_table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/nagasawa2018/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/nagasawa2018/table5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    nagasawa2018_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        nagasawa2018_df.loc[i,'Name'] = name
        nagasawa2018_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        nagasawa2018_df.loc[i,'Reference'] = 'Nagasawa+2018'
        nagasawa2018_df.loc[i,'Ref'] = 'NAG18'
        nagasawa2018_df.loc[i,'Loc'] = 'UF'
        nagasawa2018_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        nagasawa2018_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        nagasawa2018_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(nagasawa2018_df.loc[i,'RA_hms'], precision=6)
        nagasawa2018_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        nagasawa2018_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(nagasawa2018_df.loc[i,'DEC_dms'], precision=2)
        nagasawa2018_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        nagasawa2018_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        nagasawa2018_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        nagasawa2018_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                nagasawa2018_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                nagasawa2018_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    nagasawa2018_df.loc[i, col] = row["logepsX"] - logepsX_sun_a09
                    nagasawa2018_df.loc[i, 'ul'+col] = np.nan
                    # nagasawa2018_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    nagasawa2018_df.loc[i, col] = np.nan
                    nagasawa2018_df.loc[i, 'ul'+col] = row["logepsX"] - logepsX_sun_a09
                    # nagasawa2018_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    nagasawa2018_df.loc[i, col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    nagasawa2018_df.loc[i, 'ul'+col] = np.nan
                    # nagasawa2018_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    nagasawa2018_df.loc[i, col] = np.nan
                    nagasawa2018_df.loc[i, 'ul'+col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    # nagasawa2018_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    nagasawa2018_df.loc[i, col] = e_logepsX
                else:
                    nagasawa2018_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    nagasawa2018_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return nagasawa2018_df

def load_norris2010a():
    """
    Load the Norris et al. 2010a data for the Bootes II (Boo-1137) Ultra-Faint Dwarf Galaxy.

    Table 0 - Observation and Stellar Parameters
    Table 2 - Abundance Table

    Note: The abundance ratios are using the Asplund+2005 solar abundances, not the Asplund+2009 solar abundances.
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + 'abundance_tables/norris2010a/table0_obs.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/norris2010a/table2.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    
    ## Make the new column names
    species = []
    for ion in abund_df['Species'].unique():
        species_i = ion_to_species(ion)
        elem_i = ion_to_element(ion)
        if species_i not in species:
            species.append(species_i)

    epscols = [make_epscol(s) for s in species]
    ulcols = [make_ulcol(s) for s in species]
    # XHcols = [make_XHcol(s).replace(' ', '') for s in species]
    # ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(' ', '') for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    norris2010a_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XFecols + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        norris2010a_df.loc[i,'Name'] = name
        norris2010a_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        norris2010a_df.loc[i,'Reference'] = 'Norris+2010a'
        norris2010a_df.loc[i,'Ref'] = 'NOR10a'
        norris2010a_df.loc[i,'Loc'] = 'UF'
        norris2010a_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        norris2010a_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        norris2010a_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(norris2010a_df.loc[i,'RA_hms'], precision=6)
        norris2010a_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        norris2010a_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(norris2010a_df.loc[i,'DEC_dms'], precision=2)
        norris2010a_df.loc[i,'Teff'] = obs_df.loc[obs_df['Name'] == name, 'Teff'].values[0]
        norris2010a_df.loc[i,'logg'] = obs_df.loc[obs_df['Name'] == name, 'logg'].values[0]
        norris2010a_df.loc[i,'M/H'] = obs_df.loc[obs_df['Name'] == name, 'M/H'].values[0]
        norris2010a_df.loc[i,'Vmic'] = obs_df.loc[obs_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row['Species']
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)
            
            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                norris2010a_df.loc[i, col] = row['logepsX'] if pd.isna(row['l_logepsX']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                norris2010a_df.loc[i, col] = row['logepsX'] if pd.notna(row['l_logepsX']) else np.nan

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_[X/Fe]']):
                    norris2010a_df.loc[i, col] = row['[X/Fe]']
                    norris2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    norris2010a_df.loc[i, col] = np.nan
                    norris2010a_df.loc[i, 'ul'+col] = row['[X/Fe]']
                if 'e_[X/Fe]' in row.index:
                    norris2010a_df.loc[i, 'e_'+col] = row['e_[X/Fe]']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get('e_logepsX', np.nan)
                if pd.notna(e_logepsX):
                    norris2010a_df.loc[i, col] = e_logepsX
                else:
                    norris2010a_df.loc[i, col] = np.nan

    return norris2010a_df

def load_norris2010b():
    """
    Load the Norris et al. 2010b data for the Segue 1 (Seg 1-7) Ultra-Faint Dwarf Galaxy.

    Table 0 - Observation and Stellar Parameters
    Table 2 - Abundance Table

    Note: Which solar abundances are used is not stated in the text, although I assume they are the Asplund+2005 solar abundances. Not the Asplund+2009 solar abundances.
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + 'abundance_tables/norris2010b/table0_obs.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/norris2010b/table2.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    
    ## Make the new column names
    species = []
    for ion in abund_df['Species'].unique():
        species_i = ion_to_species(ion)
        elem_i = ion_to_element(ion)
        if species_i not in species:
            species.append(species_i)

    epscols = [make_epscol(s) for s in species]
    ulcols = [make_ulcol(s) for s in species]
    # XHcols = [make_XHcol(s).replace(' ', '') for s in species]
    # ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(' ', '') for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    norris2010b_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XFecols + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        norris2010b_df.loc[i,'Name'] = name
        norris2010b_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        norris2010b_df.loc[i,'Reference'] = 'Norris+2010b'
        norris2010b_df.loc[i,'Ref'] = 'NOR10b'
        norris2010b_df.loc[i,'Loc'] = 'UF'
        norris2010b_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        norris2010b_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        norris2010b_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(norris2010b_df.loc[i,'RA_hms'], precision=6)
        norris2010b_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        norris2010b_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(norris2010b_df.loc[i,'DEC_dms'], precision=2)
        norris2010b_df.loc[i,'Teff'] = obs_df.loc[obs_df['Name'] == name, 'Teff'].values[0]
        norris2010b_df.loc[i,'logg'] = obs_df.loc[obs_df['Name'] == name, 'logg'].values[0]
        norris2010b_df.loc[i,'Fe/H'] = obs_df.loc[obs_df['Name'] == name, 'Fe/H'].values[0]
        norris2010b_df.loc[i,'Vmic'] = obs_df.loc[obs_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row['Species']
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)
            
            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                norris2010b_df.loc[i, col] = row['logepsX'] if pd.isna(row['l_logepsX']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                norris2010b_df.loc[i, col] = row['logepsX'] if pd.notna(row['l_logepsX']) else np.nan

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_[X/Fe]']):
                    norris2010b_df.loc[i, col] = row['[X/Fe]']
                    norris2010b_df.loc[i, 'ul'+col] = np.nan
                else:
                    norris2010b_df.loc[i, col] = np.nan
                    norris2010b_df.loc[i, 'ul'+col] = row['[X/Fe]']
                if 'e_[X/Fe]' in row.index:
                    norris2010b_df.loc[i, 'e_'+col] = row['e_[X/Fe]']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get('e_logepsX', np.nan)
                if pd.notna(e_logepsX):
                    norris2010b_df.loc[i, col] = e_logepsX
                else:
                    norris2010b_df.loc[i, col] = np.nan

    return norris2010b_df

def load_norris2010c(load_gilmore2013=False):
    """
    Load the Norris et al. 2010c data for the Bootes I (BooI) and Segue 1 (Seg1) Ultra-Faint Dwarf Galaxies.

    All relevant data was compiled together from the other tables and the text into `table0_combined.csv`.

    Note: Which solar abundances are used is not stated in the text, although I assume they are the Asplund+2005 solar abundances. Not the Asplund+2009 solar abundances.
    """

    ## Read in the data tables
    csv_df = pd.read_csv(data_dir + 'abundance_tables/norris2010c/table0_combined.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

    ## New dataframe with proper columns
    norris2010c_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + ['epsfe','epsc','[Fe/H]','[C/H]','[C/Fe]'])
    for i, name in enumerate(csv_df['Name'].unique()):
        norris2010c_df.loc[i,'Name'] = name
        norris2010c_df.loc[i,'Simbad_Identifier'] = csv_df.loc[csv_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        norris2010c_df.loc[i,'Reference'] = csv_df.loc[csv_df['Name'] == name, 'Reference'].values[0]
        norris2010c_df.loc[i,'Ref'] = 'NOR10c'
        norris2010c_df.loc[i,'Loc'] = 'UF'
        norris2010c_df.loc[i,'System'] = csv_df.loc[csv_df['Name'] == name, 'System'].values[0]     
        norris2010c_df.loc[i,'RA_hms'] = csv_df.loc[csv_df['Name'] == name, 'RA_hms'].values[0]
        norris2010c_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(norris2010c_df.loc[i,'RA_hms'], precision=6)
        norris2010c_df.loc[i,'DEC_dms'] = csv_df.loc[csv_df['Name'] == name, 'DEC_dms'].values[0]
        norris2010c_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(norris2010c_df.loc[i,'DEC_dms'], precision=2)
        norris2010c_df.loc[i,'Teff'] = csv_df.loc[csv_df['Name'] == name, 'Teff'].values[0]
        norris2010c_df.loc[i,'logg'] = csv_df.loc[csv_df['Name'] == name, 'logg'].values[0]
        norris2010c_df.loc[i,'Fe/H'] = csv_df.loc[csv_df['Name'] == name, '[Fe/H]'].values[0]
        norris2010c_df.loc[i,'Vmic'] = np.nan
        norris2010c_df.loc[i,'[Fe/H]'] = csv_df.loc[csv_df['Name'] == name, '[Fe/H]'].values[0]
        norris2010c_df.loc[i,'epsfe'] = csv_df.loc[csv_df['Name'] == name, 'epsfe'].values[0]
        norris2010c_df.loc[i,'epsc'] = csv_df.loc[csv_df['Name'] == name, 'epsc'].values[0]
        norris2010c_df.loc[i,'[C/H]'] = csv_df.loc[csv_df['Name'] == name, '[C/H]'].values[0]
        norris2010c_df.loc[i,'[C/Fe]'] = csv_df.loc[csv_df['Name'] == name, '[C/Fe]'].values[0]

    if not load_gilmore2013:
        norris2010c_df = norris2010c_df[norris2010c_df['Reference'] == 'Norris+2010c']

    return norris2010c_df

def load_ou2024():
    """
    Gaia Sausage Enceladus (GSE) Dwarf Galaxy Stars

    Load the data from Ou+2024, Table 1, for stars in the GSE dwarf galaxy.
    """
    ou2024_df = pd.read_csv(data_dir+'abundance_tables/ou2024/ou2024-yelland.csv', comment='#')

    return ou2024_df

def load_ou2025():
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Ou+2025 for stars in the Sagittarius dwarf galaxy.
    """
    ou2025_df = pd.read_csv(data_dir+'abundance_tables/ou2025/ou2025-yelland.csv', comment='#')

    return ou2025_df

def load_placco2014():
    file_path = data_dir+"abundance_tables/placco2014/table3.txt"
    placco2014_df = pd.read_fwf(file_path, skiprows=6, delimiter="|", header=None, skipinitialspace=True)

    ## Manually specify column names
    placco2014_df.columns = [
        "Name", "Teff", "log_g", "log_L", "[Fe/H]", "l_[N/Fe]", "[N/Fe]", "[C/Fe]", "Del[C/Fe]",
        "[C/Fe]c", "l_[Sr/Fe]", "[Sr/Fe]", "l_[Ba/Fe]", "[Ba/Fe]", "Class", "I/O", "Ref"
    ]

    ## Remove last row
    placco2014_df.drop(placco2014_df.index[-1], inplace=True)

    ## Remove leading and trailing whitespace from all string entries
    placco2014_df = placco2014_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    ## Modifying and Renaming Columns
    placco2014_df.rename(columns={"Ref": "Reference"}, inplace=True)
    placco2014_df["Reference"] = placco2014_df["Reference"].str.replace(r' et al\. \(', r'+', regex=True) \
                                                           .str.replace(r'\)', '', regex=True) \
                                                           .str.replace(r' and [^ ]+ \((\d{4})', r'+\1', regex=True)
    placco2014_df['Ref'] = placco2014_df['Reference'].str[:3].str.upper() + placco2014_df['Reference'].str[-2:]

    placco2014_df.rename(columns={"l_[N/Fe]": "ul[N/Fe]"}, inplace=True)
    placco2014_df.loc[placco2014_df['ul[N/Fe]'] == '{<=}', 'ul[N/Fe]'] = placco2014_df['[N/Fe]']
    placco2014_df.loc[placco2014_df['ul[N/Fe]'] == placco2014_df['[N/Fe]'], '[N/Fe]'] = ''
    
    placco2014_df.rename(columns={"l_[Sr/Fe]": "ul[Sr/Fe]"}, inplace=True)
    placco2014_df.loc[placco2014_df['ul[Sr/Fe]'] == '{<=}', 'ul[Sr/Fe]'] = placco2014_df['[Sr/Fe]']
    placco2014_df.loc[placco2014_df['ul[Sr/Fe]'] == placco2014_df['[Sr/Fe]'], '[Sr/Fe]'] = ''
    
    placco2014_df.rename(columns={"l_[Ba/Fe]": "ul[Ba/Fe]"}, inplace=True)
    placco2014_df.loc[placco2014_df['ul[Ba/Fe]'] == '{<=}', 'ul[Ba/Fe]'] = placco2014_df['[Ba/Fe]']
    placco2014_df.loc[placco2014_df['ul[Ba/Fe]'] == placco2014_df['[Ba/Fe]'], '[Ba/Fe]'] = ''

    placco2014_df.rename(columns={'[C/Fe]c': '[C/Fe]f'}, inplace=True)
    placco2014_df.rename(columns={'Del[C/Fe]': '[C/Fe]c'}, inplace=True)
    placco2014_df['epsc_c'] = placco2014_df['[C/Fe]c']

    placco2014_df.rename(columns={'log_g': 'logg'}, inplace=True)

    placco2014_df['MP_key'] = placco2014_df['[Fe/H]'].apply(lambda feh: classify_metallicity(float(feh)) if pd.notna(feh) else np.nan)
    placco2014_df['Ncap_key'] = ''
    placco2014_df['C_key'] = ''

    for name in placco2014_df['Name']:
        if placco2014_df.loc[placco2014_df['Name'] == name, 'Class'].values[0] == 'CEMP-no':
            placco2014_df.loc[placco2014_df['Name'] == name, 'C_key'] = 'NO'
        if placco2014_df.loc[placco2014_df['Name'] == name, 'Class'].values[0] == 'CEMP-s/rs':
            placco2014_df.loc[placco2014_df['Name'] == name, 'C_key'] = 'CE'
            placco2014_df.loc[placco2014_df['Name'] == name, 'Ncap_key'] = 'I'
        if placco2014_df.loc[placco2014_df['Name'] == name, 'Class'].values[0] == 'CEMP':
            placco2014_df.loc[placco2014_df['Name'] == name, 'C_key'] = 'CE'

    ## Convert columns to appropriate data types
    numeric_cols = ['Teff', 'logg', 'log_L', '[Fe/H]', '[N/Fe]', 'ul[N/Fe]', '[C/Fe]', 
                    '[C/Fe]f', '[C/Fe]c','[Sr/Fe]', 'ul[Sr/Fe]', '[Ba/Fe]', 'ul[Ba/Fe]']
    for col in numeric_cols:
        placco2014_df[col] = pd.to_numeric(placco2014_df[col], errors='coerce')

    ## Calculate the alternative carbon abundance columns
    placco2014_df["epsc"] = np.nan
    for i, row in placco2014_df.iterrows():
        placco2014_df.at[i, "epsc"] = eps_from_XFe(row["[C/Fe]"], row["[Fe/H]"], 'C')
        placco2014_df.at[i, "epsc_f"] = eps_from_XFe(row["[C/Fe]f"], row["[Fe/H]"], 'C')
    placco2014_df["[C/H]"] = (placco2014_df["[C/Fe]"] + placco2014_df["[Fe/H]"]).astype(float)
    placco2014_df["[C/H]f"] = (placco2014_df["[C/Fe]f"] + placco2014_df["[Fe/H]"]).astype(float)

    # ## Marking s-process stars (used for CEMP-s classification)
    # placco2014_df['CEMP'] = 0 
    # placco2014_df.loc[placco2014_df['[C/Fe]f'] >= 0.7, 'CEMP'] += 1 # CEMP-no stars, from Yoon+2016
    # placco2014_df.loc[(placco2014_df['CEMP'] == 1) & (placco2014_df['epsc_f'] >= 7.1), 'CEMP'] += 1 # CEMP-s stars, from Yoon+2016
    # placco2014_df.drop(columns=['CEMP'], inplace=True)

    ## Manual modifications for specific star entries (based on additional literature after Placco 2014)
    placco2014_df = placco2014_df[~placco2014_df["Name"].isin(["CS 22948-104"])] # considered to be apart of the Atari Disk
    placco2014_df.loc[placco2014_df['Name'] == 'HE 1300+0157', 'ul[Sr/Fe]'] = placco2014_df.loc[placco2014_df['Name'] == 'HE 1300+0157', '[Sr/Fe]']
    placco2014_df.loc[placco2014_df['Name'] == 'HE 1300+0157', '[Sr/Fe]'] = np.nan
    placco2014_df.loc[placco2014_df['Name'] == 'HK17435-00532', 'Ncap_key'] = 'RS'
    placco2014_df.loc[placco2014_df['Name'] == 'CS 31080-095', 'Ncap_key'] = 'S'
    placco2014_df.loc[placco2014_df['Name'] == 'CS 29528-041', 'Ncap_key'] = 'S'
    placco2014_df.loc[placco2014_df['Name'] == 'CS 22892-052', 'Ncap_key'] = 'R2'
    placco2014_df.loc[placco2014_df['Name'] == 'CS 29497-004', 'Ncap_key'] = 'R2'
    placco2014_df.loc[placco2014_df['Name'] == 'CS 31082-001', 'Ncap_key'] = 'R2'
    placco2014_df.loc[placco2014_df['Name'] == 'HE 0430-4901', 'Ncap_key'] = 'R2'
    placco2014_df.loc[placco2014_df['Reference'] == 'Simmerer+2004	', 'Ncap_key'] = 'S'

    ## [Sr/H] Column
    placco2014_df['[Sr/H]'] = np.nan
    for i, row in placco2014_df.iterrows():
        if row['[Sr/Fe]'] is not None and row['[Fe/H]'] is not None:
            placco2014_df.at[i, '[Sr/H]'] = row['[Sr/Fe]'] + row['[Fe/H]']
        else:
            placco2014_df.at[i, '[Sr/H]'] = np.nan

    placco2014_df['ul[Sr/H]'] = np.nan
    for i, row in placco2014_df.iterrows():
        if row['ul[Sr/Fe]'] is not None and row['[Fe/H]'] is not None:
            placco2014_df.at[i, 'ul[Sr/H]'] = row['ul[Sr/Fe]'] + row['[Fe/H]']
        else:
            placco2014_df.at[i, 'ul[Sr/H]'] = np.nan

    ## [Ba/H] Column
    placco2014_df['[Ba/H]'] = np.nan
    for i, row in placco2014_df.iterrows():
        if row['[Ba/Fe]'] is not None and row['[Fe/H]'] is not None:
            placco2014_df.at[i, '[Ba/H]'] = row['[Ba/Fe]'] + row['[Fe/H]']
        else:
            placco2014_df.at[i, '[Ba/H]'] = np.nan

    placco2014_df['ul[Ba/H]'] = np.nan
    for i, row in placco2014_df.iterrows():
        if row['ul[Ba/Fe]'] is not None and row['[Fe/H]'] is not None:
            placco2014_df.at[i, 'ul[Ba/H]'] = row['ul[Ba/Fe]'] + row['[Fe/H]']
        else:
            placco2014_df.at[i, 'ul[Ba/H]'] = np.nan
    
    ## [Sr/Ba] Column
    placco2014_df['[Sr/Ba]'] = np.nan
    for i, row in placco2014_df.iterrows():
        if row['[Sr/Fe]'] is not None and row['[Ba/Fe]'] is not None:
            placco2014_df.at[i, '[Sr/Ba]'] = row['[Sr/Fe]'] - row['[Ba/Fe]']
        else:
            placco2014_df.at[i, '[Sr/Ba]'] = np.nan

    placco2014_df['ul[Sr/Ba]'] = np.nan
    for i, row in placco2014_df.iterrows():
        
        srfe, ulsrfe = row['[Sr/Fe]'], row['ul[Sr/Fe]']
        bafe, ulbafe = row['[Ba/Fe]'], row['ul[Ba/Fe]']
        if (pd.notna(srfe) or pd.notna(ulsrfe)) and (pd.notna(bafe) or pd.notna(ulbafe)):

            if pd.isna(srfe) and pd.notna(ulsrfe):
                if pd.notna(bafe) and pd.isna(ulbafe):
                    placco2014_df.at[i, 'ul[Sr/Ba]'] = ulsrfe - bafe
                elif pd.isna(bafe) and pd.notna(ulbafe):
                    placco2014_df.at[i, 'ul[Sr/Ba]'] = ulsrfe - ulbafe

            elif pd.notna(srfe) and pd.isna(ulsrfe):
                if pd.isna(bafe) and pd.notna(ulbafe):
                    placco2014_df.at[i, 'ul[Sr/Ba]'] = srfe - ulbafe
                elif pd.notna(bafe) and pd.isna(ulbafe):
                    placco2014_df.at[i, 'ul[Sr/Ba]'] = np.nan  # Already defined, but still valid to be explicit

    
    ## Remove unnecessary columns
    placco2014_df.drop(columns=['Class'], inplace=True)
    # placco2014_df.drop(columns=['log_L'], inplace=True)    
    # placco2014_df.drop(columns=['I/O'], inplace=True)

    ## Convert columns to appropriate data types
    numeric_cols = ['Teff', 'logg', 'log_L', '[Fe/H]', '[N/Fe]', 'ul[N/Fe]', '[C/Fe]', 
                    '[C/Fe]c', '[C/Fe]f', '[Sr/Fe]', 'ul[Sr/Fe]', '[Ba/Fe]', 'ul[Ba/Fe]',
                    '[C/H]', '[C/H]f', '[Sr/H]', '[Ba/H]', '[Sr/Ba]', 'ul[Sr/Ba]', 'epsc', 'epsc_c', 'epsc_f']
    for col in numeric_cols:
        placco2014_df[col] = pd.to_numeric(placco2014_df[col], errors='coerce')


    placco2014_df.to_csv(data_dir+'abundance_tables/placco2014/placco2014_yelland.csv', index=False)
    return placco2014_df

def load_roederer2014b():
    """
    Load the Roederer et al. 2014 data for the Segue 1 Ultra-Faint Dwarf Galaxy.

    Table 0 - Observations & Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/roederer2014b/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/roederer2014b/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    roederer2014b_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        roederer2014b_df.loc[i,'Name'] = name
        roederer2014b_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        roederer2014b_df.loc[i,'Reference'] = 'Roederer+2014b'
        roederer2014b_df.loc[i,'Ref'] = 'ROE14b'
        roederer2014b_df.loc[i,'Loc'] = 'UF'
        roederer2014b_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        roederer2014b_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        roederer2014b_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(roederer2014b_df.loc[i,'RA_hms'], precision=6)
        roederer2014b_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        roederer2014b_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(roederer2014b_df.loc[i,'DEC_dms'], precision=2)
        roederer2014b_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        roederer2014b_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        roederer2014b_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        roederer2014b_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                roederer2014b_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                roederer2014b_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    roederer2014b_df.loc[i, col] = row["logepsX"] - logepsX_sun_a09
                    roederer2014b_df.loc[i, 'ul'+col] = np.nan
                    # roederer2014b_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    roederer2014b_df.loc[i, col] = np.nan
                    roederer2014b_df.loc[i, 'ul'+col] = row["logepsX"] - logepsX_sun_a09
                    # roederer2014b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    roederer2014b_df.loc[i, col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    roederer2014b_df.loc[i, 'ul'+col] = np.nan
                    # roederer2014b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    roederer2014b_df.loc[i, col] = np.nan
                    roederer2014b_df.loc[i, 'ul'+col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    # roederer2014b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_tot", np.nan)
                if pd.notna(e_logepsX):
                    roederer2014b_df.loc[i, col] = e_logepsX
                else:
                    roederer2014b_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    roederer2014b_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return roederer2014b_df

def load_sestito2024():
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Sestito for stars in the Sagittarius dwarf galaxy, focused on Carbon.
    PIGS IX (Table 4) is used for this dataset.
    """
    sestito2024_df = pd.read_csv(data_dir+'abundance_tables/sestito2024/sestito2024-yelland.csv', comment='#')

    return sestito2024_df

def load_sestito2024b():
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Sestito et al. 2024b for stars in the Sagittarius dwarf galaxy. This is low/med-resolution 
    photometry from the PIGS X survey.
    """
    sestito2024b_df = pd.read_csv(data_dir+'abundance_tables/sestito2024b/membpara.csv', comment='#')

    ## Add and rename the necessary columns
    sestito2024b_df.rename(columns={'PIGS':'Name', 'RAdeg':'RA_deg', 'DEdeg':'DEC_deg', 'GaiaDR3':'Simbad_Identifier', 'RV':'Vel', 'e_RV':'e_Vel'}, inplace=True)
    sestito2024b_df['Simbad_Identifier'] = 'Gaia DR3 ' + sestito2024b_df['Simbad_Identifier'].astype(str)
    sestito2024b_df['Reference'] = 'Sestito+2024b'
    sestito2024b_df['Ref'] = 'SES24b'
    sestito2024b_df['Loc'] = 'DW'
    sestito2024b_df['RA_hms'] = np.nan
    sestito2024b_df['DEC_dms'] = np.nan
    sestito2024b_df.drop(columns={'[C/Fe]corr', 'Unnamed: 18'}, inplace=True) # not trustworthy values
    sestito2024b_df['[C/Fe]c'] = np.nan
    sestito2024b_df['[C/Fe]f'] = np.nan

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in sestito2024b_df.iterrows():
        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            ## pad RA_hms with leading zeros
            if len(row['RA_hms']) == 10:
                row['RA_hms'] = '0' + row['RA_hms']
                sestito2024b_df.at[idx, 'RA_hms'] = row['RA_hms']
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            sestito2024b_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            sestito2024b_df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(float(row['RA_deg']), precision=2)
            sestito2024b_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(float(row['DEC_deg']), precision=2)
            sestito2024b_df.at[idx, 'DEC_dms'] = row['DEC_dms']

    # Categorize columns & reorder dataFrame
    columns = list(sestito2024b_df.columns)
    aux_cols = [
        'Reference','Ref','Name','Simbad_Identifier','Loc','RA_hms','DEC_dms','RA_deg','DEC_deg',
        'Field','Vel','e_Vel','Teff','e_Teff','logg','e_logg','pmRAred','pmDEred'
        ]
    carbon_cols = [col for col in columns if "[C/" in col]
    xh_cols = [col for col in columns if col.startswith("[") and col.endswith("/H]") and col not in carbon_cols]
    ul_xh_cols = [col for col in columns if col.startswith("ul[") and col.endswith("/H]") and col not in carbon_cols]
    e_xh_cols = [col for col in columns if col.startswith("e_[") and col.endswith("/H]") and col not in carbon_cols]
    xfe_cols = [col for col in columns if col.startswith("[") and col.endswith("/Fe]") and col not in carbon_cols]
    ul_xfe_cols = [col for col in columns if col.startswith("ul[") and col.endswith("/Fe]") and col not in carbon_cols]
    e_xfe_cols = [col for col in columns if col.startswith("e_[") and col.endswith("/Fe]") and col not in carbon_cols]
    xy_cols = [col for col in columns if (col.startswith("[") and ("/" in col)) and (col not in xh_cols + xfe_cols + carbon_cols)]
    remaining_cols = [col for col in columns if col not in aux_cols + carbon_cols + xh_cols + ul_xh_cols + e_xh_cols + xfe_cols + ul_xfe_cols + e_xfe_cols + xy_cols]

    ordered_cols = aux_cols + carbon_cols + xh_cols + ul_xh_cols + e_xh_cols + xfe_cols + ul_xfe_cols + e_xfe_cols + xy_cols + remaining_cols
    sestito2024b_df = sestito2024b_df[ordered_cols]

    return sestito2024b_df

def load_simon2010():
    """
    Load the Simon et al. 2010 data for the Leo IV Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters
    Table 2 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/simon2010/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/simon2010/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    simon2010_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        simon2010_df.loc[i,'Name'] = name
        simon2010_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        simon2010_df.loc[i,'Reference'] = 'Simon+2010'
        simon2010_df.loc[i,'Ref'] = 'SIM10'
        simon2010_df.loc[i,'Loc'] = 'UF'
        simon2010_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        simon2010_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        simon2010_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(simon2010_df.loc[i,'RA_hms'], precision=6)
        simon2010_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        simon2010_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(simon2010_df.loc[i,'DEC_dms'], precision=2)
        simon2010_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        simon2010_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        simon2010_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        simon2010_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                simon2010_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                simon2010_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    simon2010_df.loc[i, col] = row["logepsX"] - logepsX_sun_a09
                    simon2010_df.loc[i, 'ul'+col] = np.nan
                    # simon2010_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    simon2010_df.loc[i, col] = np.nan
                    simon2010_df.loc[i, 'ul'+col] = row["logepsX"] - logepsX_sun_a09
                    # simon2010_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    simon2010_df.loc[i, col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    simon2010_df.loc[i, 'ul'+col] = np.nan
                    # simon2010_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    simon2010_df.loc[i, col] = np.nan
                    simon2010_df.loc[i, 'ul'+col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    # simon2010_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = normal_round(np.sqrt(row.get("e_stat", 0.0)**2 + row.get("e_sys", 0.0)**2), 2)
                if pd.notna(e_logepsX):
                    simon2010_df.loc[i, col] = e_logepsX
                else:
                    simon2010_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    simon2010_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return simon2010_df

def load_spite2018():
    """
    Load the Spite et al. 2018 data for the Pisces II Ultra-Faint Dwarf Galaxy.

    Table 0 - Observations & Stellar Parameters
    Table 2 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/spite2018/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/spite2018/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    spite2018_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        spite2018_df.loc[i,'Name'] = name
        spite2018_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        spite2018_df.loc[i,'Reference'] = 'Spite+2018'
        spite2018_df.loc[i,'Ref'] = 'SPI18'
        spite2018_df.loc[i,'Loc'] = 'UF'
        spite2018_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        spite2018_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        spite2018_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(spite2018_df.loc[i,'RA_hms'], precision=6)
        spite2018_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        spite2018_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(spite2018_df.loc[i,'DEC_dms'], precision=2)
        spite2018_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        spite2018_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        spite2018_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        spite2018_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                spite2018_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                spite2018_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    spite2018_df.loc[i, col] = row["logepsX"] - logepsX_sun_a09
                    spite2018_df.loc[i, 'ul'+col] = np.nan
                    # spite2018_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    spite2018_df.loc[i, col] = np.nan
                    spite2018_df.loc[i, 'ul'+col] = row["logepsX"] - logepsX_sun_a09
                    # spite2018_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    spite2018_df.loc[i, col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    spite2018_df.loc[i, 'ul'+col] = np.nan
                    # spite2018_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    spite2018_df.loc[i, col] = np.nan
                    spite2018_df.loc[i, 'ul'+col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    # spite2018_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    spite2018_df.loc[i, col] = e_logepsX
                else:
                    spite2018_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    spite2018_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return spite2018_df

def load_waller2023():
    """
    Load the Roederer et al. 2014 data for the Segue 1 Ultra-Faint Dwarf Galaxy.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 7 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/waller2023/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/waller2023/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/waller2023/table7.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
    abund_df = abund_df[abund_df['LTE/NLTE'] == 'LTE']

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
    waller2023_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        waller2023_df.loc[i,'Name'] = name
        waller2023_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        waller2023_df.loc[i,'Reference'] = 'Waller+2023'
        waller2023_df.loc[i,'Ref'] = 'WAL23'
        waller2023_df.loc[i,'Loc'] = 'UF'
        waller2023_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        waller2023_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        waller2023_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(waller2023_df.loc[i,'RA_hms'], precision=6)
        waller2023_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        waller2023_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(waller2023_df.loc[i,'DEC_dms'], precision=2)
        waller2023_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        waller2023_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        waller2023_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        waller2023_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', '[X/H]'].values[0] + star_df.loc[star_df['Species'] == 'Fe I', 'logepsX_sun'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

            if ion == 'Fe I' or ion == 'Fe II':
                xfe = row["[X/Fe]"]
                feh = row["[X/H]"]
                logepsX_sun = row["logepsX_sun"]
                logepsX = normal_round(feh + logepsX_sun, 2)
            else:
                xfe = row["[X/Fe]"]
                feh = star_df.loc[star_df['Species'] == 'Fe I', '[X/H]'].values[0]
                logepsX_sun = star_df.loc[star_df['Species'] == ion, 'logepsX_sun'].values[0]
                logepsX = normal_round(xfe + feh + logepsX_sun, 2)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                waller2023_df.loc[i, col] = logepsX if pd.isna(row["l_[X/Fe]"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                waller2023_df.loc[i, col] = logepsX if pd.notna(row["l_[X/Fe]"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    waller2023_df.loc[i, col] = logepsX - logepsX_sun_a09
                    waller2023_df.loc[i, 'ul'+col] = np.nan
                    # waller2023_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    waller2023_df.loc[i, col] = np.nan
                    waller2023_df.loc[i, 'ul'+col] = logepsX - logepsX_sun_a09
                    # waller2023_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    waller2023_df.loc[i, col] = (logepsX - logepsX_sun_a09) - feh_a09
                    waller2023_df.loc[i, 'ul'+col] = np.nan
                    # waller2023_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    waller2023_df.loc[i, col] = np.nan
                    waller2023_df.loc[i, 'ul'+col] = (logepsX - logepsX_sun_a09) - feh_a09
                    # waller2023_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                if row["Species"] == 'Fe I' or row["Species"] == 'Fe II':
                    e_logepsX = row.get("e_[X/H]", np.nan)
                else:
                    e_logepsX = row.get("e_[X/Fe]", np.nan)
                if pd.notna(e_logepsX):
                    waller2023_df.loc[i, col] = e_logepsX
                else:
                    waller2023_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    waller2023_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return waller2023_df

def load_webber2023():
    """
    Load the Webber et al. 2023 data for the Cetus II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/webber2023/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/webber2023/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/webber2023/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    webber2023_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        webber2023_df.loc[i,'Name'] = name
        webber2023_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        webber2023_df.loc[i,'Reference'] = 'Webber+2023'
        webber2023_df.loc[i,'Ref'] = 'WEB23'
        webber2023_df.loc[i,'Loc'] = 'UF'
        webber2023_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        webber2023_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        webber2023_df.loc[i,'RA_deg'] = coord.ra_hms_to_deg(webber2023_df.loc[i,'RA_hms'], precision=6)
        webber2023_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        webber2023_df.loc[i,'DEC_deg'] = coord.dec_dms_to_deg(webber2023_df.loc[i,'DEC_dms'], precision=2)
        webber2023_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        webber2023_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        webber2023_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        webber2023_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                webber2023_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                webber2023_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    webber2023_df.loc[i, col] = row["logepsX"] - logepsX_sun_a09
                    webber2023_df.loc[i, 'ul'+col] = np.nan
                    # webber2023_df.loc[i, 'e_'+col] = row["e_[X/H]"]
                else:
                    webber2023_df.loc[i, col] = np.nan
                    webber2023_df.loc[i, 'ul'+col] = row["logepsX"] - logepsX_sun_a09
                    # webber2023_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    webber2023_df.loc[i, col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    webber2023_df.loc[i, 'ul'+col] = np.nan
                    # webber2023_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]
                else:
                    webber2023_df.loc[i, col] = np.nan
                    webber2023_df.loc[i, 'ul'+col] = (row["logepsX"] - logepsX_sun_a09) - feh_a09
                    # webber2023_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    webber2023_df.loc[i, col] = e_logepsX
                else:
                    webber2023_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    webber2023_df.drop(columns=['[Fe/Fe]','ul[Fe/Fe]','[FeII/Fe]','ul[FeII/Fe]'], inplace=True, errors='ignore')

    return webber2023_df

################################################################################
## Dataset Read-in (Abundance Data)

def load_apogee_sgr():
    """
    Loads the APOGEE data for Sgr from APOGEE_DR16
    
    STARFLAG == 0, ASPCAPFLAG == 0, VERR < 0.2, SNR > 70
    TEFF > 3700, LOGG < 3.5
    (142775 STARS)
    
    Within 1.5*342.7 arcmin of (RA, Dec) = (283.747, -30.4606)
    (2601 STARS)

    100 < VHELIO_AVG < 180
    -3.2 < GAIA_PMRA < -2.25
    -1.9 < GAIA_PMDEC < -0.9
    (400 STARS)
    """
    tab = Table.read(data_dir+"abundance_tables/APOGEE/apogee_sgr.fits")
    tab.rename_column("APOGEE_ID","Name")
    cols_to_keep = ["Name","RA","DEC","M_H_ERR","ALPHA_M","ALPHA_M_ERR","TEFF_ERR","LOGG_ERR"]
    tab.rename_columns(["TEFF","LOGG","VMICRO","M_H"], ["Teff","logg","Vmic","mh"])
    cols_to_keep.extend(["Teff","logg","Vmic","mh"])
    tab.rename_column("FE_H","[Fe/H]"); cols_to_keep.append("[Fe/H]")
    tab.rename_column("FE_H_ERR","e_fe"); cols_to_keep.append("e_fe")
    tab["ulfe"] = False; cols_to_keep.append("ulfe")
    
    for el in ["C","N","O","NA","MG","AL","SI","P","S","K","CA","TI","V","CR","MN","CO","NI","CU","CE"]:
        elem = getelem(el)
        tab["{}_FE_ERR".format(el)][tab["{}_FE".format(el)] < -9000] = np.nan
        tab["{}_FE".format(el)][tab["{}_FE".format(el)] < -9000] = np.nan
        tab.rename_column("{}_FE".format(el),"[{}/Fe]".format(elem))
        tab.rename_column("{}_FE_ERR".format(el),"e_{}".format(elem.lower()))
        tab[ulcol(elem)] = False
        cols_to_keep.extend(["[{}/Fe]".format(elem),"e_{}".format(elem.lower()),ulcol(elem)])
    
    df = tab[cols_to_keep].to_pandas()

    ## Adding/Modifying Columns
    df.rename(columns={
        'RA':'RA_deg',
        'DEC':'DEC_deg',
        'M_H_ERR': 'e_mh',
        'ALPHA_M': 'alpha_m',
        'ALPHA_M_ERR': 'e_alpha_m',
        'TEFF_ERR': 'e_Teff',
        'LOGG_ERR': 'e_logg',
        }, inplace=True)
    
    df["System"] = "Sgr"
    df["Loc"] = "DW"
    df["Reference"] = "APOGEE_DR16"
    df["Ref"] = "APOGEE"
    df['RA_hms'] = np.nan
    df['DEC_dms'] = np.nan

    for idx, row in df.iterrows():
        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            ## pad RA_hms with leading zeros
            if len(row['RA_hms']) == 10:
                row['RA_hms'] = '0' + row['RA_hms']
                df.at[idx, 'RA_hms'] = row['RA_hms']
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(float(row['RA_deg']), precision=2)
            df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(float(row['DEC_deg']), precision=2)
            df.at[idx, 'DEC_dms'] = row['DEC_dms']

    XHcol_from_XFecol(df)
    epscol_from_XHcol(df)
    ulXHcol_from_ulcol(df)
    ulXFecol_from_ulcol(df)

    # Categorize columns & reorder dataFrame
    columns = list(df.columns)
    aux_cols = [
        'Reference','Ref','Name','RA_hms','DEC_dms','RA_deg','DEC_deg',
        'Loc','System','Teff','e_Teff','logg','e_logg','Vmic','mh','e_mh',
        'alpha_m','e_alpha_m'
        ]
    carbon_cols = [col for col in columns if "[C/" in col]
    xh_cols = [col for col in columns if col.startswith("[") and col.endswith("/H]") and col not in carbon_cols]
    ul_xh_cols = [col for col in columns if col.startswith("ul[") and col.endswith("/H]") and col not in carbon_cols]
    e_xh_cols = [col for col in columns if col.startswith("e_[") and col.endswith("/H]") and col not in carbon_cols]
    xfe_cols = [col for col in columns if col.startswith("[") and col.endswith("/Fe]") and col not in carbon_cols]
    ul_xfe_cols = [col for col in columns if col.startswith("ul[") and col.endswith("/Fe]") and col not in carbon_cols]
    e_xfe_cols = [col for col in columns if col.startswith("e_[") and col.endswith("/Fe]") and col not in carbon_cols]
    xy_cols = [col for col in columns if (col.startswith("[") and ("/" in col)) and (col not in xh_cols + xfe_cols + carbon_cols)]
    remaining_cols = [col for col in columns if col not in aux_cols + carbon_cols + xh_cols + ul_xh_cols + e_xh_cols + xfe_cols + ul_xfe_cols + e_xfe_cols + xy_cols]

    ordered_cols = aux_cols + carbon_cols + xh_cols + ul_xh_cols + e_xh_cols + xfe_cols + ul_xfe_cols + e_xfe_cols + xy_cols + remaining_cols
    df = df[ordered_cols]

    return df