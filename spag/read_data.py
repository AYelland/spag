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
from spag.utils_data import *

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

def load_jinabase(sci_key=None, priority=1, load_eps=True, load_ul=True, load_XH=True, load_XFe=True, load_aux=True, name_as_index=True):
    """
    sci_key: str or None
        A label used for interesting stars in the JINAbase database. (nan, 'R2', 'R1', 'S', 'I')
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
    data = pd.read_csv(data_dir+"abundance_tables/JINAbase-updated/JINAbase_2021_yelland.csv", header=0, na_values=["*"]) #index_col=0
    uls  = pd.read_csv(data_dir+"abundance_tables/JINAbase-updated/JINAbase_2021_ulimits.csv", header=0, na_values=["*"]) #index_col=0
    Nstars = len(data)

    ## Get the list of elements & create the corresponding column names for the element abundance columns
    ## NOTE: The ionization state columns are dropped and not used in this analysis. Hence, the -7 
    ##       in the column slicing. Every application of elems, ul_elems, and epscolnames now 
    ##       excludes the ionization state columns.
    elems = data.columns[26:-7]
    ul_elems = list(map(lambda x: "ul"+x, elems))
    epscolnames = list(map(lambda x: "eps"+x.lower(), elems))
    print("WARNING: Dropped the CaII, TiII, VII, CrII, MnII, FeII columns.")

    ## Rename the data element abundance columns with the prefix "eps" (e.g. "Fe" -> "epsfe")
    data.rename(columns=dict(zip(elems, epscolnames)), inplace=True)

    ## Separate the auxiliary columns (JINA_ID, Priority, etc.) from the element abundance columns (epsX) in 'data' and 'uls'
    auxdata_cols = data.columns[0:26].append(pd.Index([data.columns[-1]]))
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
        XH_from_eps(data)
    if load_XFe:
        XFe_from_eps(data)

    ## Use a specific `Sci_key` to filter the data
    if sci_key != None:
        assert sci_key in auxdata["Sci_key"].unique(), auxdata["Sci_key"].unique() #if sci_key is not in the list of unique Sci_keys, print the list of unique Sci_keys
        auxdata = auxdata[auxdata["Sci_key"]==sci_key]
    else:
        assert len(data) == Nstars, (len(data), Nstars)

    ## Combine the auxiliary columns with the element abundance columns
    if load_aux:
        data = pd.concat([data, auxdata],axis=1)
        # Remove duplicate entries by using the "Priority" column (1=high, 2=low)
        # (dupicates entries originate from two papers referencing the same source)
        if priority==1 or priority==2:
            pmask = data["Priority"] == priority
            data = data[pmask]
            uls = uls[pmask]
    else:
        data = pd.concat([data, auxdata[["Name","Ref","Priority"]]], axis=1)
        # Remove duplicate entries by using the "Priority" column (1=high, 2=low)
        # (dupicates entries originate from two papers referencing the same source)
        if priority==1 or priority==2:
            pmask = data["Priority"] == priority
            data = data[pmask]
            uls = uls[pmask]
        data.drop("Priority", axis=1, inplace=True)

    ## Drop the log(eps) columns if not needed
    if not load_eps:
        data.drop(eps_elems, axis=1, inplace=True)

    ## Set the "Name" column as the index
    if name_as_index:
        data.index = data["Name"]

    return data

def jinabase_create_the_upperlimits_file():
    """
    Create a Jinabase upper limits file from the JINAbase data, updated in 2021.
    The file is effectively a mask of the JINAbase abundances, where each upper
    limit is marked with a 1 (identified by the '<'), and all other values are 
    marked with a *.
    """

    # Load and make a copy of the JINAbase data
    jinabase_df = pd.read_csv(data_dir+"abundance_tables/JINAbase-updated/JINAbase_2021_yelland.csv", na_values=["*"])
    ul_jinabase_df = jinabase_df.copy()

    # Remove unnecessary columns
    ul_jinabase_df = ul_jinabase_df.drop(columns=[
        'NLTE','Sci_key','C_key','Loc','Type','RA','DEC','Vel','Vel_bibcode',
        'U_mag','B_mag','V_mag','R_mag','I_mag','J_mag','H_mag','K_mag',
        'Teff','logg','Fe/H','Vmic','Added_by'
        ])

    # At the 'ul' prefix to the columns with element symbols (e.g. Fe -> ulFe)
    for col in ul_jinabase_df.columns:
        if col in pt_list+['CaII', 'TiII', 'VII', 'CrII', 'MnII', 'FeII']:
            ul_jinabase_df.rename(columns={col: 'ul'+col}, inplace=True)

    # Replace values containing "<" with 1, and replace all other values with *
    ul_columns = [col for col in ul_jinabase_df.columns if col.startswith('ul')]

    ul_jinabase_df[ul_columns] = ul_jinabase_df[ul_columns].applymap(
        lambda x: 1 if isinstance(x, str) and "<" in x else "*"
    )

    # save the dataframe to a csv file
    ul_jinabase_df.to_csv(data_dir+"abundance_tables/JINAbase-updated/JINAbase_2021_ulimits.csv", index=False)


def jinabase_assign_ids_and_priorities():
    # Read the data into a DataFrame
    jinabase_df = pd.read_csv(data_dir + "abundance_tables/JINAbase-updated/JINAbase_2021.csv", na_values=["*"])

    # Assign JINA_ID to rows where it is missing
    jinabase_df['JINA_ID'] = jinabase_df['JINA_ID'].fillna(pd.Series(jinabase_df.index.map(lambda x: x+1), index=jinabase_df.index))

    # Columns containing element abundances (starting from 'Li' to the last element column)
    abundance_columns = jinabase_df.columns[jinabase_df.columns.get_loc('Li'):-1]

    # Count non-NaN values in element abundance columns for each row
    jinabase_df['nonNaN_abundances'] = jinabase_df[abundance_columns].notna().sum(axis=1)

    # Separate rows with NaN in Simbad_Identifier
    rows_with_simbad_identifier = jinabase_df[jinabase_df['Simbad_Identifier'].notna()]
    rows_without_simbad_identifier = jinabase_df[jinabase_df['Simbad_Identifier'].isna()]

    # Assign Priority to rows with Simbad_Identifier
    def assign_priority(group):
        # Sort the group by the number of non-NaN abundances (descending order)
        group = group.sort_values('nonNaN_abundances', ascending=False)
        # Assign Priority 1.0 to the row with the most non-NaN values, others get 2.0
        group['Priority'] = [1.0] + [2.0] * (len(group) - 1)
        return group

    rows_with_simbad_identifier = rows_with_simbad_identifier.groupby('Simbad_Identifier', group_keys=False).apply(assign_priority)

    # Assign default Priority (1.0) to rows without Simbad_Identifier
    rows_without_simbad_identifier.loc[:, 'Priority'] = 1.0

    # Combine both parts back together
    jinabase_df = pd.concat([rows_with_simbad_identifier, rows_without_simbad_identifier], ignore_index=True)

    # Drop the helper column 'nonNaN_abundances'
    jinabase_df = jinabase_df.drop(columns=['nonNaN_abundances'])

    # Sort by JINA_ID
    jinabase_df = jinabase_df.sort_values('JINA_ID')

    # Save the updated DataFrame to a new file
    jinabase_df.to_csv(data_dir + "abundance_tables/JINAbase-updated/JINAbase_2021_ay.csv", index=False)

    return jinabase_df


################################################################################
## Specific System's Data Read-in

def load_MW_halo(**kwargs):
    """
    Loads JINAbase and removes stars with loc='DW' or loc='UF' such that only halo stars remain
    Note: DW = dwarf galaxy, UF = ultra-faint galaxy
    """
    halo = load_jinabase(**kwargs)
    halo = halo[halo["Loc"] != "DW"]
    halo = halo[halo["Loc"] != "UF"]
    return halo

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
    
    def get_gal(row):
        
        ## These are papers from a single galaxy
        refgalmap = {"AOK07b":"UMi","COH10":"UMi","URA15":"UMi",
                     "FRE10a":"Scl","GEI05":"Scl","JAB15":"Scl","SIM15":"Scl","SKU15":"Scl",
                     "AOK09":"Sex",
                     "FUL04":"Dra","COH09":"Dra","TSU15":"Dra","TSU17":"Dra",
                     "NOR17":"Car","VEN12":"Car",
                     "HAN18":"Sgr"}
        ref = row["Reference"]
        if ref in refgalmap:
            return refgalmap[ref]
        
        ## These are papers with multiple galaxies
        assert ref in ["SHE01","SHE03","TAF10","KIR12"], ref
        name = row["Name"]
        name = name[0].upper() + name[1:3].lower()
        if name == "Umi": return "UMi"
        
        return name
    
    #allrefs = np.unique(cldw["Reference"])
    #multirefs = ["SHE01","SHE03","TAF10","KIR12"]
    gals = [get_gal(x) for i,x in cldw.iterrows()]
    cldw["galaxy"] = gals

    if add_all:
        fnx = load_letarte10_fornax()
        fnx2 = load_lemasle14_fornax()
        car = load_lemasle12_carina()
        sex = load_theler20_sextans()
        sgr = load_apogee_sgr()
        cldw = pd.concat([cldw,fnx,fnx2,scl,car,sex,sgr],axis=0)
    
    return cldw