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
    # load the datafile into a pandas dataframe and strip all whitespaces in the columns names and values
    asplund09 = pd.read_csv(datafile_path, skipinitialspace=True)
    for col in asplund09.columns:
        if asplund09[col].dtype == "object":
            asplund09[col] = asplund09[col].str.strip()
        asplund09.rename(columns={col:col.strip()}, inplace=True)
    # print("Loading Datafile: ", datafile_path)
    
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

def solar_logepsX_asplund09_dict():
    """
    Returns the solar abundances from Asplund et al. 2009 as a dictionary.
    """

    asplund09 = solar_logepsX_asplund09()
    asplund09_dict = dict(zip(asplund09['elem'], asplund09['photosphere_logeps']))

    # if the value is NaN, use the meteoritic value
    for key in asplund09_dict.keys():
        if np.isnan(asplund09_dict[key]):
            asplund09_dict[key] = asplund09[asplund09['elem'] == key]['meteorite_logeps'].values[0]

    # appends a np.nan value for Tc
    asplund09_dict['Tc'] = np.nan

    return asplund09_dict

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

def get_solar(elems):
    """
    Returns the solar abundance of the elements in the list 'elems'.
    """
    
    elems = np.ravel(elems)
    good_elems = [getelem(elem) for elem in elems]

    asplund09 = solar_logepsX_asplund09_dict()

    return pd.Series([asplund09[elem] for elem in good_elems], index=elems, name='asplund09')

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

def load_jinabase(sci_key=None, priority=1, load_eps=True, load_ul=True, load_XH=True, load_XFe=True, load_aux=True, name_as_index=False):
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

    data_matrix = data.to_numpy()  # Convert data DataFrame to NumPy array
    data_matrix[uls_mask] = np.nan # Set values in `data_matrix` to NaN wherever `uls_mask` is True
    data = pd.DataFrame(data_matrix, columns=data.columns, index=data.index) # Convert the modified NumPy array back to a DataFrame

    ## Concatenate the upper limit values to the 'data' DataFrame
    if load_ul:
        data = pd.concat([data, uls_table], axis=1) # Concatenate the upper limit values to the data DataFrame

    ## Convert the element abundance and add the [X/H] and [X/Fe] columns
    if load_XH:
        XHcol_from_epscol(data)
    if load_XFe:
        XFecol_from_epscol(data)
            
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
        data.drop(eps_elems, axis=1, inplace=True)

    ## Set the "Name" column as the index
    if name_as_index:
        data.index = data["Name"]

    data.to_csv(data_dir+"abundance_tables/JINAbase-yelland/JINAbase-yelland25-processed.csv", index=False)
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
    jinabase = rd.load_jinabase(**kwargs)

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
    # susmitha2017 = rd.load_susmitha2017()

    ## Lucchetti 2024 (LUCC24)
    lucchetti2024 = rd.load_lucchetti2024()
    lucchetti2024_carina = lucchetti2024[~lucchetti2024['Name'].str.lower().str.contains('fnx')]
    
    ## Combine the DataFrames
    # -------------------------------------------------- #
    carina_df = pd.concat([jinabase_carina, lucchetti2024_carina], ignore_index=True)

    if '[C/Fe]_ul' not in carina_df.columns:
        carina_df = pd.concat([carina_df, pd.Series(np.nan, index=carina_df.index, name='[C/Fe]_ul')], axis=1)

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
    jinabase = rd.load_jinabase(**kwargs)

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
    chiti2018_df = load_chiti2018(combine_tables=True)
    chiti2018_df.rename(columns={'A(C)':'epsc'}, inplace=True)
    chiti2018_df.rename(columns={'A(C)_ul':'ulc'}, inplace=True)
    chiti2018_df.rename(columns={'A(C)_ll':'llc'}, inplace=True)
    chiti2018_df.rename(columns={'e_A(C)':'e_epsc'}, inplace=True)

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
    sculptor_df = pd.concat([jinabase_sculptor, chiti2018_df, frebel2010b_df], ignore_index=True, sort=False)

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
    jinabase = rd.load_jinabase(**kwargs)

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

    if '[C/Fe]_ul' not in fornax_df.columns:
        fornax_df = pd.concat([fornax_df, pd.Series(np.nan, index=fornax_df.index, name='[C/Fe]_ul')], axis=1)

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
    jinabase = rd.load_jinabase(**kwargs)

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

################################################################################
## Reference Read-in (Abundance Data)

def load_chiti2018(combine_tables=True):
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
            df[f"{value_col}_ll"] = np.where(mask_lower, df[value_col], np.nan) # Lower limit
            df[f"{value_col}_ul"] = np.where(mask_upper, df[value_col], np.nan) # Upper limit

            # Drop the original limit column & value column
            df = df.drop(columns=[value_col])
            df = df.drop(columns=[l_col])
            df = df.rename(columns={f"{value_col}_real": value_col})
        return df

    if combine_tables:

        ## Load the combined table (created by Alex Yelland)
        chiti2018_df = pd.read_csv(data_dir+'abundance_tables/chiti2018/chiti2018_sculptor.csv', comment='#', header=0)
        
        ## Add columns and extract the limit columns
        chiti2018_df = separate_limit_columns(chiti2018_df)
        df_cols_reorder = [
            'Reference','Ref','Name','Loc','Type','Sci_key','RA_hms','DEC_dms','RA_deg','DEC_deg',
            'logg','Teff','[Ba/H]','Slit',
            'A(C)','A(C)_ll','A(C)_ul','e_A(C)',
            '[Fe/H]','[Fe/H]_ll','[Fe/H]_ul','e_[Fe/H]',
            '[C/Fe]','[C/Fe]_ll','[C/Fe]_ul','e_[C/Fe]','[C/Fe]c',
            '[C/Fe]f','[C/Fe]f_ll','[C/Fe]f_ul','e_[C/Fe]f',
        ]
        chiti2018_df = chiti2018_df[df_cols_reorder]
        chiti2018_df = chiti2018_df.rename(columns={'Sci_key': 'Ncap_key'})  # Rename 'Sci_key' to 'Ncap_key' for consistency

        ## Fill the NaN values in the RA and DEC columns
        for idx, row in chiti2018_df.iterrows():
            if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):  # Ensure RA_hms is not NaN
                row['RA_deg'] = coord.ra_hms_to_deg(str(row['RA_hms']), precision=6)
                chiti2018_df.at[idx, 'RA_deg'] = row['RA_deg']

            if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):  # Ensure DEC_dms is not NaN
                row['DEC_deg'] = coord.dec_dms_to_deg(str(row['DEC_dms']), precision=6)
                chiti2018_df.at[idx, 'DEC_deg'] = row['DEC_deg']

            if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):  # Ensure RA_deg is not NaN
                row['RA_hms'] = coord.ra_deg_to_hms(row['RA_deg'], precision=2)
                chiti2018_df.at[idx, 'RA_hms'] = row['RA_hms']

            if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):  # Ensure DEC_deg is not NaN
                row['DEC_dms'] = coord.dec_deg_to_dms(row['DEC_deg'], precision=2)
                chiti2018_df.at[idx, 'DEC_dms'] = row['DEC_dms']

        columns = [('A(C)', '[C/Fe]'), ('A(C)_ll', '[C/Fe]_ll'), ('A(C)_ul', '[C/Fe]_ul')]
        for ac_col, cf_col in columns:
            # Fill A(C)* from [C/Fe]* if missing
            mask_ac = chiti2018_df[ac_col].isna()
            chiti2018_df.loc[mask_ac, ac_col] = chiti2018_df.loc[mask_ac, cf_col].apply(lambda x: XH_from_eps(x, 'C'))
            
            # Fill [C/Fe]* from A(C)* if missing
            mask_cf = chiti2018_df[cf_col].isna()
            chiti2018_df.loc[mask_cf, cf_col] = chiti2018_df.loc[mask_cf, ac_col].apply(lambda x: eps_from_XH(x, 'C'))

        return chiti2018_df

    else:
        ## Table 5: MagE Measurements
        mage_df = pd.read_csv(data_dir+'abundance_tables/chiti2018/table5.csv', comment='#', header=None)
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
            '[Fe/H]KP','[Fe/H]KP_ll','[Fe/H]KP_ul','e_[Fe/H]KP',
            'A(C)','A(C)_ll','A(C)_ul','e_A(C)',
            '[C/Fe]','[C/Fe]_ll','[C/Fe]_ul','e_[C/Fe]',
            '[C/Fe]c','[C/Fe]f','[C/Fe]f_ll','[C/Fe]f_ul','e_[C/Fe]f','[Ba/H]']
        mage_df = mage_df[mage_cols_reorder]

        ## Table 6: M2FS Measurements
        m2fs_df = pd.read_csv(data_dir+'abundance_tables/chiti2018/table6.csv', comment='#', header=None)
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
            '[Fe/H]','[Fe/H]_ll','[Fe/H]_ul','e_[Fe/H]',
            '[C/Fe]','[C/Fe]_ll','[C/Fe]_ul','e_[C/Fe]',
            '[C/Fe]c','[C/Fe]f','[C/Fe]f_ll','[C/Fe]f_ul']
        m2fs_df = m2fs_df[m2fs_cols_reorder]

        return mage_df, m2fs_df

def load_chiti2024():
    """
    Load the Chiti et al. 2024 data for the Large Magellanic Cloud (LMC).

    See `create_chiti2024_yelland.ipynb` for details on how the datafile was created.
    """

    chiti2024_df = pd.read_csv(data_dir + 'abundance_tables/chiti2024/chiti2024_yelland.csv', comment="#")

    ## Remove rows with 'MP_key' = NaN
    chiti2024_df = chiti2024_df[chiti2024_df['MP_key'].notna()] # no abundance data in these rows, only exposure data

    return chiti2024_df

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
    XHcols = [make_XHcol(s) for s in species]
    XFecols = [make_XFecol(s) for s in species]
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
        col = make_XHcol(species_i)
        if col in XHcols:
            if pd.isna(row["l_[X/H]"]):
                frebel2010_df.loc[0, col] = row["[X/H]"]
                # frebel2010_df.loc[0, col+'_ul'] = np.nan
            else:
                frebel2010_df.loc[0, col] = np.nan
                frebel2010_df.loc[0, col+'_ul'] = row["[X/H]"]

        ## Assign [X/Fe] values
        col = make_XFecol(species_i)
        if col in XFecols:
            if pd.isna(row["l_[X/Fe]"]):
                frebel2010_df.loc[0, col] = row["[X/Fe]"]
                # frebel2010_df.loc[0, col+'_ul'] = np.nan
            else:
                frebel2010_df.loc[0, col] = np.nan
                frebel2010_df.loc[0, col+'_ul'] = row["[X/Fe]"]

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
    XHcols = [make_XHcol(s) for s in species]
    XFecols = [make_XFecol(s) for s in species]
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
            col = make_XHcol(species_i)
            if col in XHcols and pd.notna(row["[X/H]"]):
                if "<" in str(row["[X/H]"]):
                    star_row[col+'_ul'] = row["[X/H]"].split("<")[1]
                else:
                    star_row[col] = row["[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i)
            if col in XFecols and pd.notna(row["[X/Fe]"]):
                if "<" in str(row["[X/Fe]"]):
                    star_row[col+'_ul'] = row["[X/Fe]"].split("<")[1]
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
    mardini2022_df['Ref'] = mardini2022_df['Reference'].str[:3].str.upper() + mardini2022_df['Reference'].str[-2:]
    mardini2022_df['Ncap_key'] = ''
    mardini2022_df['C_key'] = mardini2022_df['[C/Fe]'].apply(lambda cfe: classify_carbon_enhancement(cfe) if pd.notna(cfe) else np.nan)
    mardini2022_df['MP_key'] = mardini2022_df['[Fe/H]'].apply(lambda feh: classify_metallicity(feh) if pd.notna(feh) else np.nan)
    mardini2022_df['Loc'] = 'DS'
    mardini2022_df['RA_deg'] = np.nan
    mardini2022_df['DEC_deg'] = np.nan

    ## Manually added datafields
    mardini2022_df.loc[mardini2022_df['Name'] == 'SDSS J124502.68-073847.0', 'Ncap_key'] = 'S'  # [C/Fe] = 2.54
    mardini2022_df.loc[mardini2022_df['Name'] == 'UCAC4 233-000355', 'Ncap_key'] = 'S'          # [C/Fe] = 3.02
    mardini2022_df.loc[mardini2022_df['Name'] == '2MASS J14160471-208540', 'C_key'] = 'NO'    # [C/Fe] = 1.44
    mardini2022_df.loc[mardini2022_df['Name'] == 'UCAC4 459-050836', 'C_key'] = 'NO'            # (HE 1300+0157, https://www.aanda.org/articles/aa/pdf/2019/03/aa34601-18.pdf)

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in mardini2022_df.iterrows():
        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(row['RA_deg'], precision=2)
            mardini2022_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            mardini2022_df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            mardini2022_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(row['DEC_deg'], precision=2)
            mardini2022_df.at[idx, 'DEC_dms'] = row['DEC_dms']

    ## Get the JINAbase Data using the JINA_ID
    jina_ids = list(mardini2022_df['JINA_ID'])

    jinabase_df = rd.load_jinabase(priority=None)
    sub_jinabase_df = jinabase_df[jinabase_df['JINA_ID'].isin(jina_ids)].copy()
    new_columns = [col for col in sub_jinabase_df.columns if col not in mardini2022_df.columns]

    # Align on JINA_ID
    mardini2022_df = mardini2022_df.set_index('JINA_ID')
    sub_jinabase_df = sub_jinabase_df.set_index('JINA_ID')
    mardini2022_df = mardini2022_df.join(sub_jinabase_df[new_columns], how='left')

    # Fill in missing [C/Fe] values from JINAbase
    if '[C/Fe]' in mardini2022_df.columns and '[C/Fe]' in sub_jinabase_df.columns:
        mardini2022_df['[C/Fe]'] = mardini2022_df['[C/Fe]'].fillna(sub_jinabase_df['[C/Fe]'])

    ## Reset the index
    sub_jinabase_df = sub_jinabase_df.reset_index()
    mardini2022_df = mardini2022_df.reset_index()

    return mardini2022_df

def load_ou2024():
    """
    Gaia Sausage Enceladus (GSE) Dwarf Galaxy Stars

    Load the data from Ou+2024, Table 1, for stars in the GSE dwarf galaxy.
    """
    ou2024_df = pd.read_csv(data_dir+'abundance_tables/ou2024/full_tab.csv', comment='#')

    ## Add and rename the necessary columns
    ou2024_df.rename(columns={'source_id':'Name', 'ra':'RA_hms', 'dec':'DEC_deg', 'teff':'Teff'}, inplace=True)
    ou2024_df['Reference'] = 'Ou+2024'
    ou2024_df['Ref'] = 'OU24'
    ou2024_df['Loc'] = 'DW'
    ou2024_df['RA_deg'] = np.nan
    ou2024_df['DEC_dms'] = np.nan

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in ou2024_df.iterrows():
        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(row['RA_deg'], precision=2)
            ou2024_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            ou2024_df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            ou2024_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(row['DEC_deg'], precision=2)
            ou2024_df.at[idx, 'DEC_dms'] = row['DEC_dms']

    return ou2024_df

def load_ou2025():
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Ou+2025 for stars in the Sagittarius dwarf galaxy.
    """
    ou2025_df = pd.read_csv(data_dir+'abundance_tables/ou2025/full_tab_v4.csv', comment='#')

    ## Add and rename the necessary columns
    ou2025_df.rename(columns={'name':'Name', 'teff':'Teff'}, inplace=True)
    ou2025_df['Reference'] = 'Ou+2025'
    ou2025_df['Ref'] = 'OU25'
    ou2025_df['Loc'] = 'DW'
    ou2025_df['RA_deg'] = np.nan
    ou2025_df['DEC_dms'] = np.nan

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in ou2025_df.iterrows():
        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(float(row['RA_deg']), precision=2)
            ou2025_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            ou2025_df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            ou2025_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(float(row['DEC_deg']), precision=2)
            ou2025_df.at[idx, 'DEC_dms'] = row['DEC_dms']

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

    placco2014_df.rename(columns={"l_[N/Fe]": "[N/Fe]_ul"}, inplace=True)
    placco2014_df.loc[placco2014_df['[N/Fe]_ul'] == '{<=}', '[N/Fe]_ul'] = placco2014_df['[N/Fe]']
    placco2014_df.loc[placco2014_df['[N/Fe]_ul'] == placco2014_df['[N/Fe]'], '[N/Fe]'] = ''
    
    placco2014_df.rename(columns={"l_[Sr/Fe]": "[Sr/Fe]_ul"}, inplace=True)
    placco2014_df.loc[placco2014_df['[Sr/Fe]_ul'] == '{<=}', '[Sr/Fe]_ul'] = placco2014_df['[Sr/Fe]']
    placco2014_df.loc[placco2014_df['[Sr/Fe]_ul'] == placco2014_df['[Sr/Fe]'], '[Sr/Fe]'] = ''
    
    placco2014_df.rename(columns={"l_[Ba/Fe]": "[Ba/Fe]_ul"}, inplace=True)
    placco2014_df.loc[placco2014_df['[Ba/Fe]_ul'] == '{<=}', '[Ba/Fe]_ul'] = placco2014_df['[Ba/Fe]']
    placco2014_df.loc[placco2014_df['[Ba/Fe]_ul'] == placco2014_df['[Ba/Fe]'], '[Ba/Fe]'] = ''

    placco2014_df.rename(columns={'[C/Fe]c': '[C/Fe]f'}, inplace=True)
    placco2014_df.rename(columns={'Del[C/Fe]': '[C/Fe]c'}, inplace=True)

    placco2014_df.rename(columns={'log_g': 'logg'}, inplace=True)

    placco2014_df['MP_key'] = placco2014_df['[Fe/H]'].apply(lambda feh: classify_metallicity(float(feh)) if pd.notna(feh) else np.nan)
    placco2014_df['Ncap_key'] = ''
    placco2014_df['C_key'] = ''

    for name in placco2014_df['Name']:
        if placco2014_df.loc[placco2014_df['Name'] == name, 'Class'].values[0] == 'CEMP-no':
            placco2014_df.loc[placco2014_df['Name'] == name, 'C_key'] = 'NO'
        if placco2014_df.loc[placco2014_df['Name'] == name, 'Class'].values[0] == 'CEMP-s/rs':
            placco2014_df.loc[placco2014_df['Name'] == name, 'C_key'] = 'CE'
            placco2014_df.loc[placco2014_df['Name'] == name, 'Ncap_key'] = 'RS'
        if placco2014_df.loc[placco2014_df['Name'] == name, 'Class'].values[0] == 'CEMP':
            placco2014_df.loc[placco2014_df['Name'] == name, 'C_key'] = 'CE'
    
    ## Remove unnecessary columns
    placco2014_df.drop(columns=['Class'], inplace=True)
    # placco2014_df.drop(columns=['log_L'], inplace=True)    
    # placco2014_df.drop(columns=['I/O'], inplace=True)

    ## Convert columns to appropriate data types
    numeric_cols = ['Teff', 'logg', 'log_L', '[Fe/H]', '[N/Fe]', '[N/Fe]_ul', '[C/Fe]', 
                    '[C/Fe]f', '[C/Fe]c','[Sr/Fe]', '[Sr/Fe]_ul', '[Ba/Fe]', '[Ba/Fe]_ul']
    for col in numeric_cols:
        placco2014_df[col] = pd.to_numeric(placco2014_df[col], errors='coerce')

    # placco2014_df.to_csv(data_dir+'abundance_tables/placco2014/placco2014_yelland.csv', index=False)

    return placco2014_df

def load_sestito2024():
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Sestito for stars in the Sagittarius dwarf galaxy, focused on Carbon.
    PIGS IX (Table 4) is used for this dataset.
    """
    sestito2024_df = pd.read_csv(data_dir+'abundance_tables/sestito2024/pigs_ix_tab4-ay.csv', comment='#')

    ## Add and rename the necessary columns
    sestito2024_df['Reference'] = 'Sestito+2024'
    sestito2024_df['Ref'] = 'SES24'
    sestito2024_df['Loc'] = 'DW'

    ## Calculate the RA and DEC in hms and dms
    sestito2024_df['RA_hms'] = sestito2024_df['RA_deg'].apply(lambda x: coord.ra_deg_to_hms(float(x), precision=2) if pd.notna(x) else np.nan)
    sestito2024_df['DEC_dms'] = sestito2024_df['DEC_deg'].apply(lambda x: coord.dec_deg_to_dms(float(x), precision=2) if pd.notna(x) else np.nan)

    return sestito2024_df

def load_sestito2024b():
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Sestito et al. 2024b for stars in the Sagittarius dwarf galaxy. This is low/med-resolution 
    photometry from the PIGS X survey.
    """
    sestito2024b_df = pd.read_csv(data_dir+'abundance_tables/sestito2024b/membpara.csv', comment='#')

    ## Add and rename the necessary columns
    sestito2024b_df.rename(columns={'PIGS':'Name', 'RAdeg':'RA_deg', 'DEdeg':'DEC_deg'}, inplace=True)
    sestito2024b_df['Reference'] = 'Sestito+2024b'
    sestito2024b_df['Ref'] = 'SES24b'
    sestito2024b_df['Loc'] = 'DW'
    sestito2024b_df['RA_hms'] = np.nan
    sestito2024b_df['DEC_dms'] = np.nan

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in sestito2024b_df.iterrows():
        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = coord.ra_deg_to_hms(float(row['RA_deg']), precision=2)
            sestito2024b_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = coord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            sestito2024b_df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            row['RA_deg'] = coord.ra_hms_to_deg(row['RA_hms'], precision=6)
            sestito2024b_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = coord.dec_deg_to_dms(float(row['DEC_deg']), precision=2)
            sestito2024b_df.at[idx, 'DEC_dms'] = row['DEC_dms']


    return sestito2024b_df

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
    tab.rename_columns(["TEFF","LOGG","VMICRO","M_H"], ["Teff","logg","vturb","Z"])
    cols_to_keep.extend(["Teff","logg","vturb","Z"])
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

    XHcol_from_XFecol(df)
    epscol_from_XHcol(df)

    df["Galaxy"] = "Sgr"
    df["Loc"] = "DW"
    df["Reference"] = "APOGEE_DR16"
    df["Ref"] = "APOGEE"

    return df