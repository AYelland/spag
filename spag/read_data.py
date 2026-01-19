#!/usr/bin/env python
# -*- coding: utf-8 -*-
# add to top of read_data.py temporarily

from __future__ import (division, print_function, absolute_import, unicode_literals)

import  sys, os, glob, time

import numpy as np
import pandas as pd
import seaborn as sns
from astropy.io import fits
from astropy.table import Table

from spag.convert import *
from spag.classification import *
from spag.utils import *
import spag.coordinates as scoord
from spag.solar import *

sns.set_palette("colorblind")
sns_palette = sns.color_palette()

################################################################################
## Directory Variables

# script_dir = "/".join(IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]) + "/" # use this if in ipython
script_dir = os.path.dirname(os.path.realpath(__file__))+"/" # use this if not in ipython (i.e. terminal script)
data_dir = script_dir+"data/"
plots_dir = script_dir+"plots/"
linelist_dir = script_dir+"linelists/"

################################################################################
## Group of Systems Read-in

def load_mw_halo(**kwargs):
    """
    Loads JINAbase and removes stars with loc='DW' or loc='UF' such that only halo stars remain
    Note: DW = dwarf galaxy, UF = ultra-faint galaxy
    """
    halo = load_jinabase(**kwargs)
    halo = halo[halo["Loc"] != "DW"]
    halo = halo[halo["Loc"] != "UF"]
    return halo

def load_accreted_dwarfs(io=None, **kwargs):
    """
    Loads the fully accreted dwarf spheroidal galaxies (dSph), including
    the Atari Disk and the Gaia-Sausage/Enceladus.
    """
    # jinabase_df = load_jinabase(io=io, **kwargs)

    df_list = [
        load_atari(), #jinabase=jinabase_df),
        load_gse(), #jinabase=jinabase_df)
        load_sass_stars()
    ]

    ## Combine all dataframes into a single dataframe
    ads_df = pd.concat(df_list, ignore_index=True)

    if ~ads_df['System'].any():
        print("Warning: Some stars are missing the 'System' value in the ads_df dataframe. Please check the data.")
    if ~ads_df['Loc'].any():
        print("Warning: Some stars are missing the 'Loc' value in the ads_df dataframe. Please check the data.")

    return ads_df

def load_classical_dwarfs(io=1, **kwargs):
    """
    Loads all of the classical dwarf galaxy functions into a single dataframe.
    """

    if io == 1 or io == 0 or io is None:
        jinabase_df = load_jinabase(io=io, **kwargs)
    else:
        raise ValueError("Invalid value for io. It should be 0, 1, or None. (Default is None)")

    df_list = [
        load_carina(jinabase=jinabase_df),
        load_draco(jinabase=jinabase_df),
        load_fornax(jinabase=jinabase_df),
        load_leoI(jinabase=jinabase_df),
        load_lmc(jinabase=jinabase_df),
        load_sagittarius(jinabase=jinabase_df),
        load_sculptor(jinabase=jinabase_df),
        load_sextans(jinabase=jinabase_df),
        load_ursaminor(jinabase=jinabase_df)
    ]

    ## Combine all dataframes into a single dataframe
    cds_df = pd.concat(df_list, ignore_index=True)

    if ~cds_df['System'].any():
        print("Warning: Some stars are missing the 'System' value in the cds_df dataframe. Please check the data.")
    if ~cds_df['Loc'].any():
        print("Warning: Some stars are missing the 'Loc' value in the cds_df dataframe. Please check the data.")

    return cds_df

def load_ufds(io=None, **kwargs):
    """
    Load the UFD galaxies from Alexmods, parse abundance values and upper limits.

    Returns:
        pd.DataFrame: A cleaned DataFrame with numerical abundance columns and separate upper limit columns.
    """

    df_list = [
        load_chiti2018b(),
        load_chiti2023(),
        load_chiti2025a(),
        load_feltzing2009(),
        load_francois2016(),
        load_frebel2010a(),
        load_frebel2013c(),
        load_frebel2014(),
        load_frebel2016(),
        load_gilmore2013(),
        load_hansent2017(),
        load_hansent2020a(),
        load_hansent2024(),
        load_ishigaki2014(),
        load_ji2016a(),
        load_ji2016b(),
        load_ji2018(),
        load_ji2019a(),
        load_ji2020a(),
        load_kirby2017b(),
        load_koch2008c(),
        load_koch2013b(),
        load_lai2011b(),
        load_marshall2019(),
        load_nagasawa2018(),
        load_norris2010a(),
        load_norris2010b(),
        load_norris2010c(),
        load_roederer2014b(),
        load_roederer2016b(),
        load_simon2010(),
        load_spite2018(),
        load_waller2023(),
        load_webber2023(),
    ]

    for i, df in enumerate(df_list):
        dupes = df.columns[df.columns.duplicated()].tolist()
        if len(dupes) > 0:
            raise ValueError(f"Warning: Duplicate columns found in dataframe {i}: {dupes}")

    ## Combine all dataframes into a single dataframe
    ufd_df = pd.concat(df_list, ignore_index=True)

    ## Drop all abundance ratio columns ([X/H], [X/Fe], etc.)
    # (this is done such that we can standardize which solar abundances are used --> Asplund 2009)
    abundance_cols = [col for col in ufd_df.columns if (('[' in col) or (']' in col) or (col.startswith('e_')))]
    ufd_df.drop(columns=abundance_cols, inplace=True, errors='ignore')

    ## Classify/Sort remaining columns
    epscols = [col for col in ufd_df.columns if col.startswith('eps')]
    ulcols = [col for col in ufd_df.columns if col.startswith('ul')]
    auxcols = [col for col in ufd_df.columns if col not in epscols + ulcols]
    ufd_df = ufd_df[auxcols + epscols + ulcols]

    ## Compute [Fe/H] and ul[Fe/H]
    epsfe_sun_a09 = get_solar('Fe', version='asplund2009')[0]
    newcols = {
        '[Fe/H]': ufd_df['epsfe'] - epsfe_sun_a09,
        'ul[Fe/H]': ufd_df['ulfe'] - epsfe_sun_a09,
        '[FeII/H]': ufd_df['epsfe2'] - epsfe_sun_a09,
        'ul[FeII/H]': ufd_df['ulfe2'] - epsfe_sun_a09,
    }

    ## Process each element
    for col in epscols:
        
        ## Skip Fe columns, already processed above
        if col in ['epsfe', 'epsfe2']: continue 

        elem = col[3:]
        ion = ion_from_col(elem)
        X_name = elem.title().replace('1','I').replace('2','II')
        el = elem.title().replace('1','').replace('2','')

        try:
            epsX_sun_09 = get_solar(el, version='asplund2009')[0]
        except:
            print(f"Warning: Could not get solar abundance for {el}, skipping...")
            continue

        epsX = ufd_df[f'eps{elem}']
        ulX = ufd_df[f'ul{elem}']
        XH = epsX - epsX_sun_09
        ulXH = ulX - epsX_sun_09
          
        FeH = newcols['[Fe/H]']
        ulFeH = newcols['ul[Fe/H]']
                
        ## Create [X/H] and [X/Fe] abundance columns
        newcols[f'[{X_name}/H]'] = XH
        newcols[f'ul[{X_name}/H]'] = ulXH

        mask_FeH = pd.notna(FeH)
        mask_ulFeH = pd.isna(FeH) & pd.notna(ulFeH)
        mask_XH = pd.notna(XH) & pd.isna(ulXH)
        mask_ulXH = pd.isna(XH) & pd.notna(ulXH)
        
        ll_XFe = np.full_like(XH, np.nan) # Default: all NaN
        ll_XFe[mask_ulFeH & mask_XH] = XH[mask_ulFeH & mask_XH] - ulFeH[mask_ulFeH & mask_XH]

        XFe = np.full_like(XH, np.nan) # Default: all NaN
        XFe[mask_FeH & mask_XH] = XH[mask_FeH & mask_XH] - FeH[mask_FeH & mask_XH]

        ul_XFe = np.full_like(XH, np.nan) # Default: all NaN
        ul_XFe[mask_ulFeH & mask_ulXH] = np.nan ## cannot determine if upper or lower limit since FeH and XH are both upper limits
        ul_XFe[mask_FeH & mask_ulXH] = ulXH[mask_FeH & mask_ulXH] - FeH[mask_FeH & mask_ulXH]

        newcols[f'll[{X_name}/Fe]'] = ll_XFe
        newcols[f'[{X_name}/Fe]'] = XFe
        newcols[f'ul[{X_name}/Fe]'] = ul_XFe
            
    ## Concatenate new columns
    newcols_df = pd.DataFrame(newcols, index=ufd_df.index)
    ufd_df = pd.concat([ufd_df, newcols_df], axis=1)

    ## Remove duplicate stars in the UFD data
    dups = [
        ('Feltzing+2009' , 'BooI-007'),
        ('Feltzing+2009' , 'BooI-033'),
        ('Feltzing+2009' , 'BooI-094'),
        ('Feltzing+2009' , 'BooI-117'),
        ('Feltzing+2009' , 'BooI-121'),
        ('Feltzing+2009' , 'BooI-127'),
        ('Feltzing+2009' , 'BooI-911'),
        ('Francois+2016' , 'LeoIV-S1'),
        ('Francois+2016' , 'BooII-7'),
        ('Francois+2016' , 'BooII-15'),
        ('Gilmore+2013'  , 'BooI-127'),
        ('Ishigaki+2014' , 'BooI-094'),
        ('Ishigaki+2014' , 'BooI-117'),
        ('Ishigaki+2014' , 'BooI-127'),
        ('Ji+2016b'      , 'DES J033523-540407'),
        ('Ji+2016b'      , 'DES J033607-540235'),
        ('Ji+2016b'      , 'DES J033531-540148'),
        ('Ji+2019a'      , 'TriII-40'),
        ('Koch+2013b'    , '42795'),
        ('Koch+2013b'    , '42241'),
        ('Koch+2013b'    , '42149'),
        ('Koch+2013b'    , '41460'),
        ('Lai+2011b'     , 'BooI-01'),
        ('Lai+2011b'     , 'BooI-24'),
        ('Lai+2011b'     , 'BooI-21'),
        ('Norris+2010c'  , 'BooI-911'),
        ('Norris+2010c'  , 'Seg1-71'),
        ('Norris+2010c'  , 'Seg1-31'),
        ('Norris+2010c'  , 'BooI-980'),
        ('Norris+2010c'  , 'Seg1-7'),
        ('Roederer+2016b', 'Star 2'),
        ('Roederer+2016b', 'Star 1'),

        ## temporary choices for carbon abundances
        # ('Norris+2010c'  , 'BooI-121'), # comment if you need carbon abundances
        ('Ishigaki+2014' , 'BooI-121'), # comment otherwise
        # ('Norris+2010c'  , 'BooI-9'), # comment if you need carbon abundances
        ('Ishigaki+2014' , 'BooI-009'), # comment otherwise
    ]

    for ref, name in dups:
        ufd_df.loc[(ufd_df['Name'] == name) & (ufd_df['Reference'] == ref), 'I/O'] = 0
    ufd_df = ufd_df[ufd_df['I/O'] == 1].reset_index(drop=True)

    return ufd_df

def load_stellar_streams(**kwargs):
    """
    Load the stellar streams data from JINAbase and other sources.
    
    Returns:
        pd.DataFrame: A DataFrame containing the stellar streams data.
    """

    # ## Load JINAbase data
    # jinabase_df = load_jinabase(**kwargs)

    ## Load additional references
    df_list = [
        load_gull2021(), ## Helmi, omega-Centauri
        load_ji2020b(), ## ATLAS, Aliqa Uma, Chenab, Elqui, Indus, Jhelum, and Phoenix
        load_martin2022a(), ## C-19
        load_roederer2010a(), ## Helmi
        load_roederer2019() ## Sylgr
    ]
    
    for i, df in enumerate(df_list):
        dupes = df.columns[df.columns.duplicated()].tolist()
        if len(dupes) > 0:
            raise ValueError(f"Warning: Duplicate columns found in dataframe {i}: {dupes}")

    ## Combine all dataframes into a single dataframe
    ss_df = pd.concat(df_list, ignore_index=True)

    ## Drop all abundance ratio columns ([X/H], [X/Fe], etc.)
    # (this is done such that we can standardize which solar abundances are used --> Asplund 2009)
    abundance_cols = [col for col in ss_df.columns if (('[' in col) or (']' in col) or (col.startswith('e_')))]
    ss_df.drop(columns=abundance_cols, inplace=True, errors='ignore')

    ## Classify/Sort remaining columns
    epscols = [col for col in ss_df.columns if col.startswith('eps')]
    ulcols = [col for col in ss_df.columns if col.startswith('ul')]
    auxcols = [col for col in ss_df.columns if col not in epscols + ulcols]
    ss_df = ss_df[auxcols + epscols + ulcols]

    ## Compute [Fe/H] and ul[Fe/H]
    epsfe_sun_a09 = get_solar('Fe', version='asplund2009')[0]
    newcols = {
        '[Fe/H]': ss_df['epsfe'] - epsfe_sun_a09,
        'ul[Fe/H]': ss_df['ulfe'] - epsfe_sun_a09,
        '[FeII/H]': ss_df['epsfe2'] - epsfe_sun_a09,
        'ul[FeII/H]': ss_df['ulfe2'] - epsfe_sun_a09,
    }

    ## Process each element
    for col in epscols:
        ## Skip Fe columns, already processed above
        if col in ['epsfe', 'epsfe2']: continue 

        elem = col[3:]
        ion = ion_from_col(elem)
        X_name = elem.title().replace('1','I').replace('2','II')
        el = elem.title().replace('1','').replace('2','')

        try:
            epsX_sun_09 = get_solar(el, version='asplund2009')[0]
        except:
            print(f"Warning: Could not get solar abundance for {el}, skipping...")
            continue

        epsX = ss_df[f'eps{elem}']
        ulX = ss_df[f'ul{elem}']
        XH = epsX - epsX_sun_09
        ulXH = ulX - epsX_sun_09

        ## Create abundance columns
        newcols[f'[{X_name}/H]'] = XH
        newcols[f'ul[{X_name}/H]'] = ulXH
        newcols[f'[{X_name}/Fe]'] = XH - newcols['[Fe/H]']
        newcols[f'ul[{X_name}/Fe]'] = ulXH - newcols['[Fe/H]']

    ## Concatenate new columns
    newcols_df = pd.DataFrame(newcols, index=ss_df.index)
    ss_df = pd.concat([ss_df, newcols_df], axis=1)

    ## Remove duplicate stars in the stellar streams data
    dups = [
        ('Roederer+2010a' , 'HD 128279'),
        ('Roederer+2010a' , 'CD-36 1052'),
    ]
    for ref, name in dups:
        ss_df.loc[(ss_df['Name'] == name) & (ss_df['Reference'] == ref), 'I/O'] = 0
    ss_df = ss_df[ss_df['I/O'] == 1].reset_index(drop=True)

    return ss_df

################################################################################
## Specific System's Data Read-in

def load_atari(jinabase=None, **kwargs):
    """
    Atari Disk Stars

    Loads the data from Mardini et al. 2022 where they present the [Fe/H] metallicity
    and [C/Fe] abundance ratios of sources from various JINAbase references.
    """

    ## Load References
    mardini2022a_df = load_mardini2022a()
    mardini2024b_df = load_mardini2024b()

    ## Combine the DataFrames
    atari_df = pd.concat([
            mardini2022a_df, 
            mardini2024b_df
        ], ignore_index=True, sort=False)

    if 'ul[C/Fe]' not in atari_df.columns:
        atari_df = pd.concat([atari_df, pd.Series(np.nan, index=atari_df.index, name='ul[C/Fe]')], axis=1)

    return atari_df

def load_carina(jinabase=None, **kwargs):
    """
    Loads Carina data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    ## JINAbase
    if jinabase is None:
        jinabase = load_jinabase(**kwargs)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_car = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('car')]
    # print(jinabase_car['Reference'].unique())

    ## Load References
    lemasle2012_df = jinabase[jinabase['Reference'] == 'Lemasle+2012'].copy() #load_lemasle2012()
    lucchesi2024_df = load_lucchesi2024()
    norris2017b_df = jinabase[jinabase['Reference'] == 'Norris+2017b'].copy() #load_norris2017b()
    # reichert2020_df = jinabase[jinabase['Reference'] == 'Reichert+2020'] # complication of other references, somewhat unreliable.copy()
    shetrone2003_df = jinabase[jinabase['Reference'] == 'Shetrone+2003'].copy() #load_shetrone2003()
    # susmitha2017_df = load_susmitha2017() ## not created yet
    venn2012_df = jinabase[jinabase['Reference'] == 'Venn+2012'].copy() #load_venn2012()
    
    ## Add filters for specific references
    lucchesi2024_df = lucchesi2024_df[lucchesi2024_df['System'] == 'Carina']
    # reichert2020_df = reichert2020_df[reichert2020_df['System'] == 'Carina']
    shetrone2003_df = shetrone2003_df[shetrone2003_df['System'] == 'Carina']
    venn2012_df = venn2012_df[venn2012_df['System'] == 'Carina']
    
    ## Combine the DataFrames
    carina_df = pd.concat([
            lemasle2012_df,
            lucchesi2024_df,
            norris2017b_df, 
            # reichert2020_df,
            shetrone2003_df,
            # susmitha2017_df,
            venn2012_df, 
        ], ignore_index=True)
    # print(carina_df['Reference'].unique())
    
    ## Add upperlimit C/Fe column if needed.
    if 'ul[C/Fe]' not in carina_df.columns:
        carina_df = pd.concat([carina_df, pd.Series(np.nan, index=carina_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in carina_df.columns if col.startswith('eps')]
    XHcols    = [col for col in carina_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in carina_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in carina_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in carina_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in carina_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in carina_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in carina_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in carina_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in carina_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    carina_df = carina_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]
    
    ## Removing Duplicate stars 
    dups = [
        ('Venn+2012', 'Car-7002'), # duplicate with a LUC24 star
        ('Lemasle+2012', 'MKV0925') # duplicate with a LUC24 star
    ]
    for ref, name in dups:
        carina_df.loc[(carina_df['Name'] == name) & (carina_df['Reference'] == ref), 'I/O'] = 0
    carina_df = carina_df[carina_df['I/O'] == 1].reset_index(drop=True)

    return carina_df

def load_draco(jinabase=None, **kwargs):
    """
    Loads Draco data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    ## JINAbase
    if jinabase is None:
        jinabase = load_jinabase(**kwargs)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_dra = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('dra')]
    # print(jinabase_dra['Reference'].unique())

    ## Load References
    cohen2009_df = jinabase[jinabase['Reference'] == 'Cohen+2009']
    fulbright2004_df = jinabase[jinabase['Reference'] == 'Fulbright+2004']
    # reichert2020_df = jinabase[jinabase['Reference'] == 'Reichert+2020'] # complication of other references, somewhat unreliable
    shetrone2001_df = jinabase[jinabase['Reference'] == 'Shetrone+2001']
    tsujimoto2015_df = jinabase[jinabase['Reference'] == 'Tsujimoto+2015']
    tsujimoto2017_df = jinabase[jinabase['Reference'] == 'Tsujimoto+2017']

    ## Add filters for specific references
    # reichert2020_df = reichert2020_df[reichert2020_df['System'] == 'Draco']
    shetrone2001_df = shetrone2001_df[shetrone2001_df['System'] == 'Draco']

    ## Combine the DataFrames
    draco_df = pd.concat([
            cohen2009_df,
            fulbright2004_df,
            # reichert2020_df,
            shetrone2001_df,
            tsujimoto2015_df,
            tsujimoto2017_df
        ], ignore_index=True)
    # print(draco_df['Reference'].unique())
    
    ## Add upperlimit C/Fe column if needed.
    if 'ul[C/Fe]' not in draco_df.columns:
        draco_df = pd.concat([draco_df, pd.Series(np.nan, index=draco_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in draco_df.columns if col.startswith('eps')]
    XHcols    = [col for col in draco_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in draco_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in draco_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in draco_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in draco_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in draco_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in draco_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in draco_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in draco_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    draco_df = draco_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    ## Removing Duplicate stars 
    dups = []
    for ref, name in dups:
        draco_df.loc[(draco_df['Name'] == name) & (draco_df['Reference'] == ref), 'I/O'] = 0
    draco_df = draco_df[draco_df['I/O'] == 1].reset_index(drop=True)

    return draco_df

def load_fornax(jinabase=None, **kwargs):
    """
    Loads Fornax data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    ## JINAbase
    if jinabase is None:
        jinabase = load_jinabase(**kwargs)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_fnx = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('fnx')]
    # print(jinabase_fnx['Reference'].unique())
    
    ## Load references
    # letarte2007_df = load_letarte2007() ## not created yet
    letarte2010_df = load_letarte2010()
    lemasle2014_df = load_lemasle2014()
    lucchesi2024_df = load_lucchesi2024()
    # reichert2020_df = jinabase[jinabase['Reference'] == 'Reichert+2020'] # complication of other references, somewhat unreliable.copy()
    shetrone2003_df = jinabase[jinabase['Reference'] == 'Shetrone+2003'].copy()
    tafelmeyer2010_df = jinabase[jinabase['Reference'] == 'Tafelmeyer+2010'].copy()
    
    ## Add filters for specific references
    lucchesi2024_df = lucchesi2024_df[lucchesi2024_df['System'] == 'Fornax']
    # reichert2020_df = reichert2020_df[reichert2020_df['System'] == 'Fornax']
    shetrone2003_df = shetrone2003_df[shetrone2003_df['System'] == 'Fornax']
    tafelmeyer2010_df = tafelmeyer2010_df[tafelmeyer2010_df['System'] == 'Fornax']
    
    ## Combine the DataFrames
    fornax_df = pd.concat([
            # letarte2007_df,
            letarte2010_df,
            lemasle2014_df,
            lucchesi2024_df,
            # reichert2020_df,
            shetrone2003_df,
            tafelmeyer2010_df, 
        ], ignore_index=True)
    # print(fornax_df['Reference'].unique())

    if 'ul[C/Fe]' not in fornax_df.columns:
        fornax_df = pd.concat([fornax_df, pd.Series(np.nan, index=fornax_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in fornax_df.columns if col.startswith('eps')]
    XHcols    = [col for col in fornax_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in fornax_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in fornax_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in fornax_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in fornax_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in fornax_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in fornax_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in fornax_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in fornax_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    fornax_df = fornax_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    ## Removing Duplicate stars 
    dups = [
        ('Letarte+2010', 'BL239'),
        ('Letarte+2010', 'BL266'),
        ('Letarte+2010', 'BL278'),
    ]
    for ref, name in dups:
        fornax_df.loc[(fornax_df['Name'] == name) & (fornax_df['Reference'] == ref), 'I/O'] = 0
    fornax_df = fornax_df[fornax_df['I/O'] == 1].reset_index(drop=True)

    return fornax_df

def load_gse(jinabase=None, **kwargs):
    """
    Gaia Sausage/Enceladus (GSE) Dwarf Galaxy Stars 

    Loads the data from Ou et al. (2024) for the Gaia Sausage/Enceladus (GSE) stars.
    This function reads in the data from the table and returns it as a pandas DataFrame.
    """

    ## Load References
    ou2024c_df = load_ou2024c()

    ## Combine the DataFrames
    gse_df = pd.concat([
            ou2024c_df
        ], ignore_index=True, sort=False)

    if 'ul[C/Fe]' not in gse_df.columns:
        gse_df = pd.concat([gse_df, pd.Series(np.nan, index=gse_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in gse_df.columns if col.startswith('eps')]
    XHcols    = [col for col in gse_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in gse_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in gse_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in gse_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in gse_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in gse_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in gse_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in gse_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in gse_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    gse_df = gse_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    return gse_df

def load_leoI(jinabase=None, **kwargs):
    """
    Loads Sextans data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    ## JINAbase
    if jinabase is None:
        jinabase = load_jinabase(**kwargs)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_leoI = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('leoI')]
    # print(jinabase_umi['Reference'].unique())

    ## Load References
    # reichert2020_df = jinabase[jinabase['Reference'] == 'Reichert+2020'] # complication of other references, somewhat unreliable
    shetrone2003_df = jinabase[jinabase['Reference'] == 'Shetrone+2003']
    # theler2020_df = load_theler() ## not created yet

    ## Add filters for specific references
    # reichert2020_df = reichert2020_df[reichert2020_df['System'] == 'Leo I']
    shetrone2003_df = shetrone2003_df[shetrone2003_df['System'] == 'Leo I']

    ## Combine the DataFrames
    leoI_df = pd.concat([
            # reichert2020_df,
            shetrone2003_df
        ], ignore_index=True)
    # print(ursaminor_df['Reference'].unique())
    
    ## Add upperlimit C/Fe column if needed.
    if 'ul[C/Fe]' not in leoI_df.columns:
        leoI_df = pd.concat([leoI_df, pd.Series(np.nan, index=leoI_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in leoI_df.columns if col.startswith('eps')]
    XHcols    = [col for col in leoI_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in leoI_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in leoI_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in leoI_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in leoI_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in leoI_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in leoI_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in leoI_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in leoI_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    leoI_df = leoI_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    return leoI_df

def load_lmc(jinabase=None, **kwargs):
    """
    Load the Large Magellanic Cloud (LMC) Dwarf Galaxy Stars

    Loads the data from Chiti et al. 2024 and combines it with other
    references if needed.
    """

    ## Load References
    chiti2024_df = load_chiti2024()
    reggiani2021_df = load_reggiani2021()
    ji2025_df = load_ji2025()

    ## Add filters for specific references
    reggiani2021_df = reggiani2021_df[reggiani2021_df['System'] == 'Large Magellanic Cloud']

    ## Combine the DataFrames
    lmc_df = pd.concat([
            chiti2024_df,
            reggiani2021_df,
            ji2025_df
        ], ignore_index=True, sort=False)

    if 'ul[C/Fe]' not in lmc_df.columns:
        lmc_df = pd.concat([lmc_df, pd.Series(np.nan, index=lmc_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in lmc_df.columns if col.startswith('eps')]
    XHcols    = [col for col in lmc_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in lmc_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in lmc_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in lmc_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in lmc_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in lmc_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in lmc_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in lmc_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in lmc_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    lmc_df = lmc_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    return lmc_df

def load_sagittarius(jinabase=None, include_medres=True, include_apogee=False, **kwargs):
    """
    Sagittarius (Sgr) Dwarf Galaxy Stars 

    Loads the data from various references for the Sagittarius (Sgr) stars.
    """

    ## JINAbase
    if jinabase is None:
        jinabase = load_jinabase(**kwargs)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_sgr = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('sgr')]
    # print(jinabase_sgr['Reference'].unique())

    ## Load references
    apogee_df = load_apogee_sgr() if include_apogee else pd.DataFrame()
    hansenc2018_df = jinabase[jinabase['Reference'] == 'Hansen_C+2018']
    ou2025_df = load_ou2025()
    # reichert2020_df = jinabase[jinabase['Reference'] == 'Reichert+2020'] # complication of other references, somewhat unreliable
    sbordone2007_df = load_sbordone2007()
    sbordone2020_df = jinabase[jinabase['Reference'] == 'Sbordone+2020']
    sestito2024b_df = load_sestito2024b()
    sestito2024d_df = load_sestito2024d() if include_medres else pd.DataFrame()

    ## Add filters for specific references
    # reichert2020_df = reichert2020_df[reichert2020_df['System'] == 'Sagittarius']
    sbordone2007_df = sbordone2007_df[sbordone2007_df['System'] == 'Sagittarius']

    ## Combine the DataFrames
    sagittarius_df = pd.concat([
            apogee_df,
            hansenc2018_df,
            ou2025_df,
            # reichert2020_df,
            sbordone2007_df,
            sbordone2020_df,
            sestito2024b_df,
            sestito2024d_df,
        ], ignore_index=True, sort=False)
    # print(sagittarius_df['Reference'].unique())

    if 'ul[C/Fe]' not in sagittarius_df.columns:
        sagittarius_df = pd.concat([sagittarius_df, pd.Series(np.nan, index=sagittarius_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in sagittarius_df.columns if col.startswith('eps')]
    XHcols    = [col for col in sagittarius_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in sagittarius_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in sagittarius_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in sagittarius_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in sagittarius_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in sagittarius_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in sagittarius_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in sagittarius_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in sagittarius_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    sagittarius_df = sagittarius_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    ## Removing Duplicate stars 
    dups = [
        ('Sestito+2024d', 'Pristine_185538.63-302704.3'),
        ('Sestito+2024b', 'Pristine_185053.71-313317.7'),
        ('Sestito+2024d', 'Pristine_185210.30-315413.2'),
        ('Sestito+2024b', 'Pristine_185210.30-315413.2'),
        ('Sestito+2024d', 'Pristine_185248.45-293223.4'),
        ('Sestito+2024d', 'Pristine_185704.51-301021.6'),
        ('Sestito+2024b', 'Pristine_185704.51-301021.6'),
        ('Sestito+2024b', 'Pristine_190612.10-315504.4'),
        ('Sestito+2024d', 'Pristine_190612.10-315504.4'),
        ('Sestito+2024b', 'Pristine_184431.86-293145.0'), # chose Sestito+2024b for carbon abundance over Sestito+2024d
        ('Sestito+2024d', 'Pristine_184759.63-315322.5'),
        ('Sestito+2024d', 'Pristine_184843.24-314626.8'),
        ('Sestito+2024b', 'Pristine_184853.44-302718.4'), # chose Sestito+2024b for carbon abundance over Sestito+2024d
        ('Sestito+2024b', 'Pristine_184957.04-291425.1'), # chose Sestito+2024b for carbon abundance over Sestito+2024d
        ('Sestito+2024b', 'Pristine_185129.00-300942.8'), # chose Sestito+2024b for carbon abundance over Sestito+2024d
        ('Sestito+2024b', 'Pristine_185347.87-314747.6'), # chose Sestito+2024b for carbon abundance over Sestito+2024d
        ('Sestito+2024b', 'Pristine_185855.01-301522.2'), # chose Sestito+2024b for carbon abundance over Sestito+2024d
    ]
    for ref, name in dups:
        sagittarius_df.loc[(sagittarius_df['Name'] == name) & (sagittarius_df['Reference'] == ref), 'I/O'] = 0
    sagittarius_df = sagittarius_df[sagittarius_df['I/O'] == 1].reset_index(drop=True)

    return sagittarius_df

def load_sculptor(jinabase=None, **kwargs):
    """
    Sculptor (Scl) Dwarf Galaxy Stars 

    Loads the data from various references for the sculptor (Sgr) stars.
    """

    ## JINAbase
    if jinabase is None:
        jinabase = load_jinabase(**kwargs)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_scl = jinabase_non_nan[(jinabase_non_nan['Name'].str.lower().str.contains('scl')) | (jinabase_non_nan['System'].str.lower().str.contains('scl'))]
    # print(jinabase_scl['Reference'].unique())

    ## Load references
    chiti2018a_df = load_chiti2018a()
    frebel2010b_df = load_frebel2010b()
    geisler2005_df = jinabase[jinabase['Reference'] == 'Geisler+2005']
    hill2019_df = jinabase[jinabase['Reference'] == 'Hill+2019']
    jablonka2015_df = jinabase[jinabase['Reference'] == 'Jablonka+2015']
    kirby2012c_df = jinabase[jinabase['Reference'] == 'Kirby+2012c']
    # reichert2020_df = jinabase[jinabase['Reference'] == 'Reichert+2020'] # complication of other references, somewhat unreliable
    # sestito2023_df = load_sestito2023() ## not created yet
    shetrone2003_df = jinabase[jinabase['Reference'] == 'Shetrone+2003']
    simon2015_df = jinabase[jinabase['Reference'] == 'Simon+2015']
    skuladottir2015_df = jinabase[jinabase['Reference'] == 'Skuladottir+2015a']
    # skuladottir2017_df = load_skuladottir2017() ## not created yet
    skuladottir2019_df = jinabase[jinabase['Reference'] == 'Skuladottir+2019']
    # skuladottir2024_df = load_skuladottir2024() ## not created yet
    tafelmayer2010_df = jinabase[jinabase['Reference'] == 'Tafelmeyer+2010']

    ## Add filters for specific references
    kirby2012c_df = kirby2012c_df[kirby2012c_df['System'] == 'Sculptor']
    # reichert2020_df = reichert2020_df[reichert2020_df['System'] == 'Sculptor']
    shetrone2003_df = shetrone2003_df[shetrone2003_df['System'] == 'Sculptor']
    tafelmayer2010_df = tafelmayer2010_df[tafelmayer2010_df['System'] == 'Sculptor']

    ## Combine the DataFrames
    sculptor_df = pd.concat([
            chiti2018a_df,
            frebel2010b_df,
            geisler2005_df,
            hill2019_df,
            jablonka2015_df,
            kirby2012c_df,
            # reichert2020_df,
            # sestito2023_df,
            shetrone2003_df,
            simon2015_df,
            skuladottir2015_df,
            # skuladottir2017_df,
            skuladottir2019_df,
            # skuladottir2024_df,
            tafelmayer2010_df
        ], ignore_index=True, sort=False)
    # print(sculptor_df['Reference'].unique())

    if 'ul[C/Fe]' not in sculptor_df.columns:
        sculptor_df = pd.concat([sculptor_df, pd.Series(np.nan, index=sculptor_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in sculptor_df.columns if col.startswith('eps')]
    XHcols    = [col for col in sculptor_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in sculptor_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in sculptor_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in sculptor_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in sculptor_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in sculptor_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in sculptor_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in sculptor_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in sculptor_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    sculptor_df = sculptor_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    ## Removing Duplicate stars 
    dups = [     
        ('Chiti+2018a', '10_7_923'), 
        ('Chiti+2018a', '11_1_4296'),
        ('Chiti+2018a', '6_6_402'),
        ('Simon+2015', 'SclS1020549'),
        ('Chiti+2018a', '10_8_1072'),
        ('Chiti+2018a', '10_8_320'),
        ('Chiti+2018a', '10_8_2818'),
        
        ('Hill+2019', 'Scl_ET0237'), # comment otherwise
        # ('Chiti+2018a', '7_3_243'),    # comment for carbon abundance
        
        ('Hill+2019', 'Scl_ET0232'), # comment otherwise
        # ('Chiti+2018a', '7_4_1514'),   # comment for carbon abundance
        
        ('Hill+2019', 'Scl_ET0369'), # comment otherwise
        # ('Chiti+2018a', '10_8_2908'),  # comment for carbon abundance
        
        ('Hill+2019', 'Scl_ET0320'), # comment otherwise
        # ('Chiti+2018a', '11_1_3738'),  # comment for carbon abundance
        
        ('Hill+2019', 'Scl_ET0238'), # comment otherwise
        # ('Chiti+2018a', '11_1_2583'), # comment for carbon abundance
        
        ('Hill+2019', 'Scl_ET0322'), # comment otherwise
        # ('Chiti+2018a', '10_8_3315'), # comment for carbon abundance
        
        ## previously considered duplicates, but I don't know why they were...
        # ('Hill+2019', 'Scl_ET0236'),
        # ('Hill+2019', 'Scl_ET0051'),
        # ('Hill+2019', 'Scl_ET0239'),
        # ('Chiti+2018a', '11_1_4824'),
    ]
    for ref, name in dups:
        sculptor_df.loc[(sculptor_df['Name'] == name) & (sculptor_df['Reference'] == ref), 'I/O'] = 0
    sculptor_df = sculptor_df[sculptor_df['I/O'] == 1].reset_index(drop=True)

    return sculptor_df

def load_sextans(jinabase=None, **kwargs):
    """
    Loads Sextans data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    ## JINAbase
    if jinabase is None:
        jinabase = load_jinabase(**kwargs)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_sex = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('sex')]
    # print(jinabase_umi['Reference'].unique())

    ## Load References
    aoki2009b_df = jinabase[jinabase['Reference'] == 'Aoki+2009b']
    # reichert2020_df = jinabase[jinabase['Reference'] == 'Reichert+2020'] # complication of other references, somewhat unreliable
    shetrone2001_df = jinabase[jinabase['Reference'] == 'Shetrone+2001']
    tafelmeyer2010_df = jinabase[jinabase['Reference'] == 'Tafelmeyer+2010']
    # theler2020_df = load_theler() ## not created yet

    ## Add filters for specific references
    # reichert2020_df = reichert2020_df[reichert2020_df['System'] == 'Sextans']
    shetrone2001_df = shetrone2001_df[shetrone2001_df['System'] == 'Sextans']
    tafelmeyer2010_df = tafelmeyer2010_df[tafelmeyer2010_df['System'] == 'Sextans']

    ## Combine the DataFrames
    sextans_df = pd.concat([
            aoki2009b_df,
            # reichert2020_df,
            shetrone2001_df,
            tafelmeyer2010_df,
            # theler2020_df
        ], ignore_index=True)
    # print(ursaminor_df['Reference'].unique())
    
    ## Add upperlimit C/Fe column if needed.
    if 'ul[C/Fe]' not in sextans_df.columns:
        sextans_df = pd.concat([sextans_df, pd.Series(np.nan, index=sextans_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in sextans_df.columns if col.startswith('eps')]
    XHcols    = [col for col in sextans_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in sextans_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in sextans_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in sextans_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in sextans_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in sextans_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in sextans_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in sextans_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in sextans_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    sextans_df = sextans_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    ## Removing Duplicate stars
    dups = []
    for ref, name in dups:
        sextans_df.loc[(sextans_df['Name'] == name) & (sextans_df['Reference'] == ref), 'I/O'] = 0
    sextans_df = sextans_df[sextans_df['I/O'] == 1].reset_index(drop=True)

    return sextans_df

def load_ursaminor(jinabase=None, **kwargs):
    """
    Loads Ursa Minor data from JINAbase and adds data from specific references. All data
    is stored in a single DataFrame. Find datasets in SPAG directories.
    """

    ## JINAbase
    if jinabase is None:
        jinabase = load_jinabase(**kwargs)
    jinabase_nan = jinabase[jinabase['Name'].isna()]  # Rows where 'Name' is NaN
    jinabase_non_nan = jinabase[jinabase['Name'].notna()]  # Rows where 'Name' is not NaN
    jinabase_umi = jinabase_non_nan[jinabase_non_nan['Name'].str.lower().str.contains('umi')]
    # print(jinabase_umi['Reference'].unique())

    ## Load References
    aoki2007c_df = jinabase[jinabase['Reference'] == 'Aoki+2007c']
    cohen2010_df = jinabase[jinabase['Reference'] == 'Cohen+2010']
    kirby2012c_df = jinabase[jinabase['Reference'] == 'Kirby+2012c']
    # reichert2020_df = jinabase[jinabase['Reference'] == 'Reichert+2020'] # complication of other references, somewhat unreliable
    sadakane2004_df = jinabase[jinabase['Reference'] == 'Sadakane+2004']
    sestito2023b_df = jinabase[jinabase['Reference'] == 'Sestito+2023b']
    shetrone2001_df = jinabase[jinabase['Reference'] == 'Shetrone+2001']
    ural2015_df = jinabase[jinabase['Reference'] == 'Ural+2015']

    ## Add filters for specific references
    kirby2012c_df = kirby2012c_df[kirby2012c_df['System'] == 'Ursa Minor']
    # reichert2020_df = reichert2020_df[reichert2020_df['System'] == 'Ursa Minor']
    shetrone2001_df = shetrone2001_df[shetrone2001_df['System'] == 'Ursa Minor']

    ## Combine the DataFrames
    ursaminor_df = pd.concat([
            aoki2007c_df,
            cohen2010_df,
            kirby2012c_df,
            # reichert2020_df,
            sadakane2004_df,
            sestito2023b_df,
            shetrone2001_df,
            ural2015_df
        ], ignore_index=True)
    # print(ursaminor_df['Reference'].unique())
    
    ## Add upperlimit C/Fe column if needed.
    if 'ul[C/Fe]' not in ursaminor_df.columns:
        ursaminor_df = pd.concat([ursaminor_df, pd.Series(np.nan, index=ursaminor_df.index, name='ul[C/Fe]')], axis=1)

    ## Sort the columns
    epscols   = [col for col in ursaminor_df.columns if col.startswith('eps')]
    XHcols    = [col for col in ursaminor_df.columns if col.startswith('[') and col.endswith('/H]')]
    XFecols   = [col for col in ursaminor_df.columns if col.startswith('[') and col.endswith('/Fe]')]
    ulXHcols  = [col for col in ursaminor_df.columns if col.startswith('ul[') and col.endswith('/H]')]
    ulXFecols = [col for col in ursaminor_df.columns if col.startswith('ul[') and col.endswith('/Fe]')]
    ulcols    = [col for col in ursaminor_df.columns if col.startswith('ul') and col not in ulXHcols and col not in ulXFecols]
    eXHcols   = [col for col in ursaminor_df.columns if col.startswith('e_[') and col.endswith('/H]')]
    eXFecols  = [col for col in ursaminor_df.columns if col.startswith('e_[') and col.endswith('/Fe]')]
    ecols     = [col for col in ursaminor_df.columns if col.startswith('e_') and col not in eXHcols and col not in eXFecols]
    auxcols   = [col for col in ursaminor_df.columns if col not in epscols + XHcols + XFecols + ulXHcols + ulXFecols + ulcols + eXHcols + eXFecols + ecols]
    ursaminor_df = ursaminor_df[auxcols + epscols + ulcols + XHcols + XFecols + ulXHcols + ulXFecols  + ecols + eXHcols + eXFecols]

    ## Removing Duplicate stars
    dups = [
        ('Shetrone+2001', 'UMi199'),
    ]
    for ref, name in dups:
        ursaminor_df.loc[(ursaminor_df['Name'] == name) & (ursaminor_df['Reference'] == ref), 'I/O'] = 0
    ursaminor_df = ursaminor_df[ursaminor_df['I/O'] == 1]

    return ursaminor_df

def load_sass_stars(remove_dups_io=1, **kwargs):
    """
    Load the SASS stars data from JINAbase, using selection filters and criteria.
    """
    jinabase_df = load_jinabase(io=None)
    hughes2025_df = load_hughes2025()
    francois2007_df = load_francois2007()
    nordlander2019_df = load_nordlander2019()

    ## Selects only halo stars (or more like everything unclassified in JINAbase)
    halo_df = jinabase_df[(jinabase_df['Loc'] == 'HA') | (jinabase_df['Loc'].isin(['', 'nan', np.nan]))]
    halo_df = pd.concat([halo_df, francois2007_df, nordlander2019_df], ignore_index=True, sort=False)

    ## Has C measurements
    # halo_df = halo_df[
    #     (halo_df['[C/H]'].notna() | halo_df['ul[C/H]'].notna())
    # ]

    ## Has Sr and/or Ba measurements
    halo_w_sr_ba_df = halo_df[
        (halo_df['[Sr/H]'].notna() | halo_df['ul[Sr/H]'].notna()) &
        (halo_df['[Ba/H]'].notna() | halo_df['ul[Ba/H]'].notna())
    ]

    ## Has low Sr and Ba abundances
    low_sr_ba_df = halo_w_sr_ba_df[
        (halo_w_sr_ba_df['[Sr/H]'].notna()) & (halo_w_sr_ba_df['[Sr/H]'].astype(float) <= -4.5) & 
        (halo_w_sr_ba_df['[Ba/H]'].notna()) & (halo_w_sr_ba_df['[Ba/H]'].astype(float) <= -4)
    ]
    low_ulsr_ba_df = halo_w_sr_ba_df[
        (halo_w_sr_ba_df['ul[Sr/H]'].notna()) & (halo_w_sr_ba_df['ul[Sr/H]'].astype(float) <= -4.5) & 
        (halo_w_sr_ba_df['[Ba/H]'].notna()) & (halo_w_sr_ba_df['[Ba/H]'].astype(float) <= -4)
    ]
    low_sr_ulba_df = halo_w_sr_ba_df[
        (halo_w_sr_ba_df['[Sr/H]'].notna()) & (halo_w_sr_ba_df['[Sr/H]'].astype(float) <= -4.5) & 
        (halo_w_sr_ba_df['ul[Ba/H]'].notna()) & (halo_w_sr_ba_df['ul[Ba/H]'].astype(float) <= -4)
    ]
    low_ulsr_ulba_df = halo_w_sr_ba_df[
        (halo_w_sr_ba_df['ul[Sr/H]'].notna()) & (halo_w_sr_ba_df['ul[Sr/H]'].astype(float) <= -4.5) & 
        (halo_w_sr_ba_df['ul[Ba/H]'].notna()) & (halo_w_sr_ba_df['ul[Ba/H]'].astype(float) <= -4)
    ]

    ## Concatenate the dataframes
    jinabase_sass_df = pd.concat([low_sr_ba_df, low_ulsr_ba_df, low_sr_ulba_df, low_ulsr_ulba_df], ignore_index=True)
    jinabase_sass_df['System'] = 'SASS'
    
    ## Remove all Roederer+2014b stars, due to low temperature and questionable abundances
    # jinabase_sass_df = jinabase_sass_df[jinabase_sass_df['Reference'] != 'Roederer+2014b']
    
    ## Combine with other Datasets
    sass_df = pd.concat([jinabase_sass_df, hughes2025_df], ignore_index=True, sort=False)
    sass_df.reset_index(drop=True, inplace=True)
    
    ## Removing Duplicate stars 
    sass_df['I/O'] = 1  # Initialize I/O column to 1
    dups = [
        ('Norris+2001', 'CS22172-002'),
        ('Ryan+1996', 'CS22172-002'), # note: doesn't have carbon
        ('Holmbeck+2020', 'J03142084-1035112'),
        ('Roederer+2014a', 'HE1012-1540'),
        ('Li+2015c', 'LAMOSTJ1313-0552'),
        ('Hansen_T+2014', 'HE1310-0536'),
        ('Aoki+2005c', 'BS16084-160'),
        ('Roederer+2014a', 'CS22891-200'),
        ('Roederer+2014c', 'CS22891-200'),
        ('McWilliam+1995', 'CS22891-200'),
        ('Roederer+2014c', 'CS22885-096'),
        ('Norris+2001', 'CS22885-096'),
        ('McWilliam+1995', 'CS22885-096'),
        ('Ryan+1996', 'CS22885-096'), # note: doesn't have carbon
        ('Yong+2013a', 'CS30336-049'),
        ('Aoki+2005c', 'CS29516-041'),
        ('McWilliam+1995', 'CS22949-048'),
        ('Roederer+2014a', 'BD+44493'),
        ('Roederer+2014c', 'CD-38245'),
        ('Ezzeddine+2020', '2MASS J00463619-3739335'),
        ('Norris+2001', 'CD-38245'),
        ('McWilliam+1995', 'CD-38245'),
        ('Ryan+1996', 'CD-38245'), # note: doesn't have carbon
        ('Yong+2013a', 'HE0057-5959'),
        ('Cohen+2008', 'HE1347-1025'),
        ('Cohen+2008', 'HE1356-0622'),
        ('Rasmussen+2020', 'RAVE J071234.0-481405'),
        ('Roederer+2014c', 'CS22968-014'),
        ('Cohen+2013', 'CS22968-014'),
        ('McWilliam+1995', 'CS22968-014'),
        ('Ryan+1996', 'CS22968-014'), # note: doesn't have carbon
        ('Aoki+2005c', 'CS30325-094'),
        ('Frebel+2008a', 'HE1327-23263D'),
        ('Collet+2006', 'HE1327-23261D'), # note: doesn't have carbon
        ('Collet+2006', 'HE1327-23263D'), # note: doesn't have carbon
        ('Cohen+2013', 'BS16467-062'),
        ('Cohen+2008', 'BS16467-062'),
        ('Hansen_T+2014', 'HE2239-5019'),
        ('Collet+2006', 'HE0107-52401D'),
        ('Collet+2006', 'HE0107-52403D'),
        
        ('Roederer+2014c', 'CS22952-015'), # we have measurements from Francois+2007 that don't make the cut (Sr too high), so we cut this star here
        ('Roederer+2014c', 'CS22189-009'), # we have measurements from Francois+2007 that don't make the cut (Sr too high), so we cut this star here
        
        ## not a duplicate, but sometimes removed due to upper limit in iron (has carbon)
        # ('Keller+2014', 'NAMESMSSJ031300.36-670839.3')
    ]
    for ref, name in dups:
        sass_df.loc[(sass_df['Name'] == name) & (sass_df['Reference'] == ref), 'I/O'] = 0

    ## Using the I/O column to filter the data
    if remove_dups_io == 0 or remove_dups_io == 1:
        sass_df = sass_df[sass_df['I/O'] == remove_dups_io].reset_index(drop=True)
    elif remove_dups_io is None:
        pass
    else:
        raise ValueError("Invalid value for 'remove_dups_io'. It should be 0, 1, or None.")
    
    return sass_df

################################################################################
## Reference Read-in (Abundance Data)

### JINAbase Data Read-in

def load_jinabase(sci_key=None, io=1, load_eps=True, load_ll=True, load_ul=True, load_XH=True, load_XFe=True, load_aux=True, name_as_index=False, feh_ulim=None, version="yelland"):
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
    io: int
        The flag for which duplicate entries, prioritizing some sources/observations over others. (0/1)
    load_eps: bool
        Load the log(eps) columns from the JINAbase database.
    load_ll: bool
        Load the lower limit value columns from the JINAbase database.
    load_ul: bool
        Load the upper limit value columns from the JINAbase database.
    load_XH: bool
        Calculate the [X/H] columns from the log(eps) columns, using Asplund et al. (2009) solar abundances.
    load_XFe: bool
        Calculate the [X/Fe] columns from the log(eps) columns, using Asplund et al. (2009) solar abundances.
    load_aux: bool
        Load the auxiliary columns from the JINAbase database. (e.g. JINA_ID, Name, Ref, I/O, stellar parameters, etc.)
    name_as_index: bool
        Set the "Name" column as the index of the DataFrame.
    version: str
        The version of the JINAbase data to load. Options are "abohalima", "ji", "mardini", or "yelland".

    Load the JINAbase data from the local copy of the JINAbase-updated repository. 
    Speak with Mohammad Mardini for more details.
    https://github.com/Mohammad-Mardini/JINAbase-updated
    """

    ## Read data
    data = pd.read_csv(data_dir+"abundance_tables/JINAbase-4-yelland/JINAbase-yelland25.csv", header=0, na_values=["*"]) #index_col=0
    uls  = pd.read_csv(data_dir+"abundance_tables/JINAbase-4-yelland/JINAbase-yelland25-ulimits.csv", header=0, na_values=["*"]) #index_col=0
    Nstars = len(data)

    ## Gather groups of column names
    abund_cols = data.columns[data.columns.get_loc('Li I'):-1]
    auxdata_cols = [col for col in data.columns if col not in abund_cols]
    epscols = [epscol(ion) for ion in abund_cols]
    ulcols = [ulcol(ion) for ion in abund_cols]
    
    ## Separate the auxiliary columns from the abundance columns (also rename for epsX and ulX columns)
    auxdata = data[auxdata_cols]
    data = data[abund_cols]
    data.rename(columns=dict(zip(abund_cols, epscols)), inplace=True)

    ## Rename the abundance columns in 'uls' DataFrame
    auxuls_cols = uls.columns[0:5]
    auxuls = uls[auxuls_cols]
    uls = uls[ulcols] # consist of 1s (upper limits) and NaNs (non-upper limits)

    ## Use the ulimits (uls) DataFrame to mask the data DataFrame
    uls_mask = pd.notnull(uls).to_numpy()  # Convert uls DataFrame to boolean array (True if not NaN)
    uls_values = data.where(uls_mask)  # Extract only the upper limit values from 'data' (keep NaN for others)
    for col in uls_values.columns:
        uls_values.rename(columns={col: "ul"+col[3:]}, inplace=True) # from epsX to ulX
    uls_values = uls_values.applymap(lambda x: float(x.strip("<")) if isinstance(x, str) and x.strip().startswith("<") else np.nan)

    data_matrix = data.to_numpy()  # Convert data DataFrame to NumPy array
    data_matrix[uls_mask] = np.nan # Set values in `data_matrix` to NaN wherever `uls_mask` is True
    data = pd.DataFrame(data_matrix, columns=data.columns, index=data.index) # Convert the modified NumPy array back to a DataFrame

    ## Concatenate the 'uls_values' Dataframe to the 'data' DataFrame
    if load_ul:
        data = pd.concat([data, uls_values], axis=1) # Concatenate the upper limit values to the data DataFrame

    ## Convert the element abundance and add the [X/H] and [X/Fe] columns
    if load_XH:
        XHcol_from_epscol(data)
        if load_ul:
            ulXHcol_from_ulcol(data)
    if load_XFe:
        XFecol_from_epscol(data)
        if load_ul:
            ulXFecol_from_ulcol(data)
        if load_ll:
            llXFecol_from_ulfecol(data)
            
    ## Combine the auxiliary columns with the element abundance columns
    if load_aux:
        data = pd.concat([auxdata, data],axis=1)
    else:
        data = pd.concat([auxdata[['Name','Ref','I/O','Ncap_key','C_key','MP_key','alpha_key'], data]], axis=1)

    ## Remove duplicate entries by using the 'I/O' column (1=keep, 0=remove)
    ## (duplicate entries originate from two papers referencing the same target)
    if io==1 or io==0:
        pmask = data['I/O'] == io
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
            raise ValueError(f"The provided sci_key '{sci_key}' is not valid. Choose from ('MP', 'R1', 'R2', 'S', 'RS', 'I', 'CE', 'NO', 'alpha').")
        data = data[sci_mask]
        uls = uls[sci_mask]

    ## Finalize the DataFrame by dropping columns in the auxiliary columns
    if not load_aux:
        data.drop({'Ncap_key','C_key','MP_key','alpha_key'}, axis=1, inplace=True)

    ## Drop the log(eps) columns if not needed
    if not load_eps:
        data.drop(epscols, axis=1, inplace=True)

    ## Set the "Name" column as the index
    if name_as_index:
        data.index = data["Name"]

    ## Set I/O as integer type
    data['I/O'] = data['I/O'].astype(int)

    ## Save the processed data to a CSV file
    data.to_csv(data_dir+"abundance_tables/JINAbase-4-yelland/JINAbase-yelland25-processed.csv", index=False)

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

### milky way (MW)

def load_placco2014c(remove_atari=True, remove_sass=True, remove_dups=True, use_jinabase_sass=False, io=1):
    """
    Load the Placco et al. (2014) abundance data for the Milky Way (MW) halo stars.

    616 stars - total number of stars in Placco et al. (2014c) data-set
    505 stars - with all CEMP-s/i stars removed
        (-111; following Placco+2014c work with [Ba/Fe] > 0.6)
    497 stars - with duplicates removed 
        (-8;   actually -11, but 3 were already removed from CEMP-s/i cut)
    478 stars - with all atari stars removed 
        (-19;  actually -21, but 2 were already removed from CEMP-s/i cut)
    437 stars - with all SASS stars removed 
        (-41;  actually -43 of the 77 SASS stars, but 2 were already removed from duplicates cut)

    By default, we read-in the filtered/cleaned data-set (437 stars) with 111 stars removed by Placco 
    and 68 stars removed by Yelland, though with the `io` argument, you can choose to read-in the 
    original data-set (616 stars) as well or versions of the data-set with only some of the filters applied.
    """

    placco2014c_df = pd.read_csv(data_dir+"abundance_tables/placco2014c/cds_files/table3_mod.csv") # using modified table for correct reference labeling

    ## Rename, Clean-up, and Add-to Reference Columns
    placco2014c_df.rename(columns={"Ref": "Reference"}, inplace=True)
    placco2014c_df["Reference"] = placco2014c_df["Reference"].str.replace(r' et al\. \(', r'+', regex=True) \
                                                             .str.replace(r'\)', '', regex=True) \
                                                             .str.replace(r' and [^ ]+ \((\d{4})', r'+\1', regex=True)
    placco2014c_df["Reference"] = placco2014c_df["Reference"].str.replace('Hansen', 'Hansen_T')                                                 
    placco2014c_df['Ref'] = placco2014c_df['Reference'].str[:3].str.upper() + np.where(
        placco2014c_df['Reference'].str[-1].str.isalpha(), 
        placco2014c_df['Reference'].str[-3:], 
        placco2014c_df['Reference'].str[-2:]
    )
    placco2014c_df['Ref'] = placco2014c_df['Ref'].str.replace('HAN', 'HANt')
    placco2014c_df['Ref'] = placco2014c_df['Ref'].str.replace('HOL11', 'HOLj11')
    placco2014c_df['Ref'] = placco2014c_df['Ref'].str.replace('HOL20', 'HOLe20')
    
    ## Modifying and Renaming Abundance Columns
    placco2014c_df.rename(columns={"l_[N/Fe]": "ul[N/Fe]"}, inplace=True)
    placco2014c_df.loc[placco2014c_df['ul[N/Fe]'] == '{<=}', 'ul[N/Fe]'] = placco2014c_df['[N/Fe]']
    placco2014c_df.loc[placco2014c_df['ul[N/Fe]'] == placco2014c_df['[N/Fe]'], '[N/Fe]'] = ''
    
    placco2014c_df.rename(columns={"l_[Sr/Fe]": "ul[Sr/Fe]"}, inplace=True)
    placco2014c_df.loc[placco2014c_df['ul[Sr/Fe]'] == '{<=}', 'ul[Sr/Fe]'] = placco2014c_df['[Sr/Fe]']
    placco2014c_df.loc[placco2014c_df['ul[Sr/Fe]'] == placco2014c_df['[Sr/Fe]'], '[Sr/Fe]'] = ''
    
    placco2014c_df.rename(columns={"l_[Ba/Fe]": "ul[Ba/Fe]"}, inplace=True)
    placco2014c_df.loc[placco2014c_df['ul[Ba/Fe]'] == '{<=}', 'ul[Ba/Fe]'] = placco2014c_df['[Ba/Fe]']
    placco2014c_df.loc[placco2014c_df['ul[Ba/Fe]'] == placco2014c_df['[Ba/Fe]'], '[Ba/Fe]'] = ''

    placco2014c_df.rename(columns={'[C/Fe]c': '[C/Fe]f'}, inplace=True)
    placco2014c_df.rename(columns={'Del[C/Fe]': '[C/Fe]c'}, inplace=True)
    placco2014c_df['epsc_c'] = placco2014c_df['[C/Fe]c']

    ## Other column renames
    placco2014c_df.rename(columns={'log(g)': 'logg'}, inplace=True)
    placco2014c_df.rename(columns={'log(L)': 'logL'}, inplace=True)

    ## Convert columns to appropriate data types
    numeric_cols = ['Teff', 'logg', 'logL', '[Fe/H]', '[N/Fe]', 'ul[N/Fe]', '[C/Fe]', 
                    '[C/Fe]f', '[C/Fe]c','[Sr/Fe]', 'ul[Sr/Fe]', '[Ba/Fe]', 'ul[Ba/Fe]']
    for col in numeric_cols:
        placco2014c_df[col] = pd.to_numeric(placco2014c_df[col], errors='coerce')
        
    ## Adding Classification Columns
    placco2014c_df['MP_key'] = placco2014c_df['[Fe/H]'].apply(lambda feh: classify_metallicity(float(feh)) if pd.notna(feh) else np.nan)
    placco2014c_df['Ncap_key'] = ''
    placco2014c_df['C_key'] = ''
    for name in placco2014c_df['Name']:
        if placco2014c_df.loc[placco2014c_df['Name'] == name, 'Class'].values[0] == 'CEMP-no':
            placco2014c_df.loc[placco2014c_df['Name'] == name, 'C_key'] = 'NO'
        if placco2014c_df.loc[placco2014c_df['Name'] == name, 'Class'].values[0] == 'CEMP-s/rs':
            placco2014c_df.loc[placco2014c_df['Name'] == name, 'C_key'] = 'CE'
            placco2014c_df.loc[placco2014c_df['Name'] == name, 'Ncap_key'] = 'I'
        if placco2014c_df.loc[placco2014c_df['Name'] == name, 'Class'].values[0] == 'CEMP':
            placco2014c_df.loc[placco2014c_df['Name'] == name, 'C_key'] = 'CE'
            
    # ## Marking s-process stars (used for CEMP-s classification)
    # placco2014c_df['CEMP'] = 0 
    # placco2014c_df.loc[placco2014c_df['[C/Fe]f'] >= 0.7, 'CEMP'] += 1 # CEMP-no stars, from Yoon+2016
    # placco2014c_df.loc[(placco2014c_df['CEMP'] == 1) & (placco2014c_df['epsc_f'] >= 7.1), 'CEMP'] += 1 # CEMP-s stars, from Yoon+2016
    # placco2014c_df.drop(columns=['CEMP'], inplace=True)
    
    ### Manual modifications for specific star entries (based on additional literature after Placco 2014)
    placco2014c_df.loc[placco2014c_df['Name'] == 'HE 1300+0157', 'ul[Sr/Fe]'] = placco2014c_df.loc[placco2014c_df['Name'] == 'HE 1300+0157', '[Sr/Fe]']
    placco2014c_df.loc[placco2014c_df['Name'] == 'HE 1300+0157', '[Sr/Fe]'] = np.nan
    placco2014c_df.loc[placco2014c_df['Name'] == 'HK17435-00532', 'Ncap_key'] = 'RS'
    placco2014c_df.loc[placco2014c_df['Name'] == 'CS 31080-095', 'Ncap_key'] = 'S'
    placco2014c_df.loc[placco2014c_df['Name'] == 'CS 29528-041', 'Ncap_key'] = 'S'
    placco2014c_df.loc[placco2014c_df['Name'] == 'CS 22892-052', 'Ncap_key'] = 'R2'
    placco2014c_df.loc[placco2014c_df['Name'] == 'CS 29497-004', 'Ncap_key'] = 'R2'
    placco2014c_df.loc[placco2014c_df['Name'] == 'CS 31082-001', 'Ncap_key'] = 'R2'
    placco2014c_df.loc[placco2014c_df['Name'] == 'HE 0430-4901', 'Ncap_key'] = 'R2'
    placco2014c_df.loc[placco2014c_df['Reference'] == 'Simmerer+2004	', 'Ncap_key'] = 'S'

    ## Calculate the alternative carbon abundance columns
    placco2014c_df["epsc"] = np.nan
    for i, row in placco2014c_df.iterrows():
        placco2014c_df.at[i, "epsc"] = eps_from_XFe(row["[C/Fe]"], row["[Fe/H]"], 'C')
        placco2014c_df.at[i, "epsc_f"] = eps_from_XFe(row["[C/Fe]f"], row["[Fe/H]"], 'C')
    placco2014c_df["[C/H]"] = (placco2014c_df["[C/Fe]"] + placco2014c_df["[Fe/H]"]).astype(float)
    placco2014c_df["[C/H]f"] = (placco2014c_df["[C/Fe]f"] + placco2014c_df["[Fe/H]"]).astype(float)

    ## [Sr/H] Column
    placco2014c_df['[Sr/H]'] = np.nan
    for i, row in placco2014c_df.iterrows():
        if row['[Sr/Fe]'] is not None and row['[Fe/H]'] is not None:
            placco2014c_df.at[i, '[Sr/H]'] = row['[Sr/Fe]'] + row['[Fe/H]']
        else:
            placco2014c_df.at[i, '[Sr/H]'] = np.nan

    placco2014c_df['ul[Sr/H]'] = np.nan
    for i, row in placco2014c_df.iterrows():
        if row['ul[Sr/Fe]'] is not None and row['[Fe/H]'] is not None:
            placco2014c_df.at[i, 'ul[Sr/H]'] = row['ul[Sr/Fe]'] + row['[Fe/H]']
        else:
            placco2014c_df.at[i, 'ul[Sr/H]'] = np.nan

    ## [Ba/H] Column
    placco2014c_df['[Ba/H]'] = np.nan
    for i, row in placco2014c_df.iterrows():
        if row['[Ba/Fe]'] is not None and row['[Fe/H]'] is not None:
            placco2014c_df.at[i, '[Ba/H]'] = row['[Ba/Fe]'] + row['[Fe/H]']
        else:
            placco2014c_df.at[i, '[Ba/H]'] = np.nan

    placco2014c_df['ul[Ba/H]'] = np.nan
    for i, row in placco2014c_df.iterrows():
        if row['ul[Ba/Fe]'] is not None and row['[Fe/H]'] is not None:
            placco2014c_df.at[i, 'ul[Ba/H]'] = row['ul[Ba/Fe]'] + row['[Fe/H]']
        else:
            placco2014c_df.at[i, 'ul[Ba/H]'] = np.nan
    
    ## [Sr/Ba] Column
    placco2014c_df['[Sr/Ba]'] = np.nan
    for i, row in placco2014c_df.iterrows():
        if row['[Sr/Fe]'] is not None and row['[Ba/Fe]'] is not None:
            placco2014c_df.at[i, '[Sr/Ba]'] = row['[Sr/Fe]'] - row['[Ba/Fe]']
        else:
            placco2014c_df.at[i, '[Sr/Ba]'] = np.nan

    placco2014c_df['ul[Sr/Ba]'] = np.nan
    for i, row in placco2014c_df.iterrows():
        
        srfe, ulsrfe = row['[Sr/Fe]'], row['ul[Sr/Fe]']
        bafe, ulbafe = row['[Ba/Fe]'], row['ul[Ba/Fe]']
        if (pd.notna(srfe) or pd.notna(ulsrfe)) and (pd.notna(bafe) or pd.notna(ulbafe)):

            if pd.isna(srfe) and pd.notna(ulsrfe):
                if pd.notna(bafe) and pd.isna(ulbafe):
                    placco2014c_df.at[i, 'ul[Sr/Ba]'] = ulsrfe - bafe
                elif pd.isna(bafe) and pd.notna(ulbafe):
                    placco2014c_df.at[i, 'ul[Sr/Ba]'] = ulsrfe - ulbafe

            elif pd.notna(srfe) and pd.isna(ulsrfe):
                if pd.isna(bafe) and pd.notna(ulbafe):
                    placco2014c_df.at[i, 'ul[Sr/Ba]'] = srfe - ulbafe
                elif pd.notna(bafe) and pd.isna(ulbafe):
                    placco2014c_df.at[i, 'ul[Sr/Ba]'] = np.nan  # Already defined, but still valid to be explicit
    
    ## Remove unnecessary columns
    placco2014c_df.drop(columns=['Class'], inplace=True)
    # placco2014c_df.drop(columns=['logL'], inplace=True)    
    # placco2014c_df.drop(columns=['I/O'], inplace=True)

    ## Convert columns to appropriate data types
    numeric_cols = ['Teff', 'logg', 'logL', 'I/O', '[Fe/H]', '[N/Fe]', 'ul[N/Fe]', '[C/Fe]', 
                    '[C/Fe]c', '[C/Fe]f', '[Sr/Fe]', 'ul[Sr/Fe]', '[Ba/Fe]', 'ul[Ba/Fe]',
                    '[C/H]', '[C/H]f', '[Sr/H]', '[Ba/H]', '[Sr/Ba]', 'ul[Sr/Ba]', 'epsc', 'epsc_c', 'epsc_f']
    for col in numeric_cols:
        placco2014c_df[col] = pd.to_numeric(placco2014c_df[col], errors='coerce')

    ## Add the Simbad_Identifier, RA_hms, DEC_dms, RA_deg, DEC_deg columns
    simbad_df = pd.read_csv(data_dir+'abundance_tables/placco2014c/simbad_data.csv')
    for name in simbad_df['Name']:
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'Simbad_Identifier'] = simbad_df.loc[simbad_df['Name'] == name, 'MAIN_ID'].values[0]
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'RA_hms'] = simbad_df.loc[simbad_df['Name'] == name, 'RA'].values[0].replace(' ', ':')
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'DEC_dms'] = simbad_df.loc[simbad_df['Name'] == name, 'DEC'].values[0].replace(' ', ':')
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'RA_deg'] = scoord.ra_hms_to_deg(placco2014c_df.loc[placco2014c_df['Name'] == name, 'RA_hms'], precision=4)
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'DEC_deg'] = scoord.dec_dms_to_deg(placco2014c_df.loc[placco2014c_df['Name'] == name, 'DEC_dms'], precision=2)
    new_columns = ['Simbad_Identifier', 'RA_hms', 'DEC_dms', 'RA_deg', 'DEC_deg']
    placco2014c_df = placco2014c_df[[placco2014c_df.columns[0]] + new_columns + list(placco2014c_df.columns[1:-len(new_columns)])]

    ## Save the pre-filtered DataFrame
    placco2014c_df.to_csv(data_dir+'abundance_tables/placco2014c/placco2014c.csv', index=False)

    ## Removing Atari Stars
    if remove_atari: # 21 stars, but only 19 removed here since 2 were already removed in the CEMP-s/i cut
        atari_stars = [
            ('Yong+2013a', 'BPS BS 16928-0053'),
            ('Barklem+2005b', 'BPS CS 22186-0023'),
            ('Masseron+2012', 'BPS CS 22948-0104'),
            ('Roederer+2014a', 'BPS CS 22960-0064'),
            ('Yong+2013a', 'BPS CS 29506-0007'),
            ('Yong+2013a', 'BPS CS 30306-0132'),
            ('Yong+2013a', 'BPS CS 31079-0028'),
            ('Yong+2013a', 'HD   2796'),
            ('Simmerer+2004', 'HD  23798'),
            ('Simmerer+2004', 'HD 119516'),
            ('Cohen+2013', 'HE 0017-4346'),
            ('Barklem+2005b', 'HE 0023-4825'),
            ('Hollek+2011', 'TYC 4928-1438-1'),
            ('Yong+2013a', 'HE 1300+0157'),
            ('Hollek+2011', 'TYC 4961-1053-1'),
            ('Barklem+2005b', 'HE 1413-1954'),
            ('Yong+2013a', 'HE 1424-0241'),
            ('Barklem+2005b', 'HE 2259-3407'),
            ('Placco+2014a', 'HE 2318-1621'),
            ('Caffau+2011d', 'UCAC3 215-112497'),
            ('Aoki+2013a', '2MASS J12450268-0738469')
        ]
        for ref, simbad_id in atari_stars:
            placco2014c_df.loc[(placco2014c_df['Simbad_Identifier'] == simbad_id) & (placco2014c_df['Reference'] == ref), 'I/O'] = 0

    # Removing Duplicate stars
    if remove_dups: # 11 stars, but only 8 removed here since 3 were already removed in the CEMP-s/i cut
        dups = [
            ('Cohen+2013', 'HE 0058-0244'),
            ('Roederer+2014a', 'CS 22948-066'),
            ('Masseron+2012', 'CS 22949-008b'),
            ('Roederer+2014a', 'CS 22949-037'),
            ('Roederer+2014a', 'CS 22957-027'),
            ('Thompson+2008', 'CS 22964-161b'),
            ('Cohen+2013', 'HE 0305-5442'),
            ('Barklem+2005b', 'CS 29493-090'),
            ('Lai+2007', 'CS 29497-040'),
            ('Aoki+2005c', 'CS 30327-038'),
            ('Yong+2013a', 'HE 0132-2439'),
        ]
        for ref, name in dups:
            placco2014c_df.loc[(placco2014c_df['Name'] == name) & (placco2014c_df['Reference'] == ref), 'I/O'] = 0
    
    ## Removing SASS stars -- yes, there are SASS stars in the Placco+2014 dataset
    sass_df = load_sass_stars()
    if remove_sass:
        mw_sass_stars = []
        for simbad_id in placco2014c_df['Simbad_Identifier']:
            if simbad_id in sass_df['Simbad_Identifier'].values:
                mw_sass_stars.append(simbad_id)
        for name in mw_sass_stars:
            placco2014c_df.loc[placco2014c_df['Simbad_Identifier'] == name, 'I/O'] = 0
        if io == 1: print("Note: SASS stars are excluded. You are using only the Placco et al. (2014) abundance values.")
    else:
        ## If you want to include the SASS stars, you can choose to use either their
        ## JINAbase abundances or their Placco et al. (2014) abundances.
        if use_jinabase_sass:
            # print("Number of stars before SASS substitution:", len(placco2014c_df))
            placco2014c_sass_rows = []
            for simbad_id in placco2014c_df['Simbad_Identifier']:
                if simbad_id in sass_df['Simbad_Identifier'].values:
                    ## Use the JINAbase abundance values for SASS stars
                    row = sass_df[sass_df['Simbad_Identifier'] == simbad_id].iloc[0:1].copy()
                    row['I/O'] = placco2014c_df.loc[placco2014c_df['Simbad_Identifier'] == simbad_id, 'I/O'].values[0]
                    placco2014c_sass_rows.append(row)
                else:
                    ## Use the Placco+2014 abundance values for non-SASS stars
                    row = placco2014c_df[placco2014c_df['Simbad_Identifier'] == simbad_id].iloc[0:1].copy()
                    placco2014c_sass_rows.append(row)
            placco2014c_sass_df = pd.concat(placco2014c_sass_rows, ignore_index=True)
            placco2014c_df = placco2014c_sass_df.copy()
            # print("Number of stars after SASS substitution:", len(placco2014c_df))
            if io == 1: print("Note: SASS stars are included. You are using their JINAbase abundance values.")
        else:
            if io == 1: print("Note: SASS stars are included. You are using their Placco et al. (2014) abundance values.")
    print()
    
    ## Using the I/O column to filter the data
    if io == 0 or io == 1:
        placco2014c_df = placco2014c_df[placco2014c_df['I/O'] == io].reset_index(drop=True)
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    ## Save the final DataFrame
    placco2014c_df.to_csv(data_dir+'abundance_tables/placco2014c/placco2014c-processed.csv', index=False)
    
    return placco2014c_df

def load_cayrel2004():
    """
    Loads the Cayrel et al. 2004 & Francois et al. 2007 data for Milky Way halo data. 
    This paper is part of "First Stars." series (First Stars. V.), where the heavy element
    abundance values are taken from Francois et al. 2007 (First Stars. VIII.).

    Table 2 - Cayrel+2004 Observation Table
    Table 4 - Cayrel+2004 Stellar Parameters
    Table 8 - Cayrel+2004 Abundance Table
    Table 3,4,5 - Francois+2007 Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/cayrel2004/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/cayrel2004/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    c04_abund_df = pd.read_csv(data_dir + "abundance_tables/cayrel2004/table8.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    f07_abund_df = pd.read_csv(data_dir + "abundance_tables/francois2007/table3_4_5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
    # -------------------------------------
    ## Modify Cayrel+2004 Abundance Table
    c04_abund_df = c04_abund_df.drop(columns=['Seq', '[Fe/H]'])
    c04_abund_df = c04_abund_df.rename(columns={'Nlines': 'N'})
    c04_abund_df = c04_abund_df[['Name', 'Species', 'N', 'l_logepsX', 'logepsX', 'l_[X/H]', '[X/H]', 'l_[X/Fe]', '[X/Fe]', 'e_[X/H]']]
    # -------------------------------------
    ## Modify François+2007 Abundance Table
    f07_abund_list = []
    for i, row in f07_abund_df.iterrows():
        for col in row.index:
            if '/Fe]' in col:
                elem = col.replace("[", "").replace("/Fe]", "").split("/")[0]
                N_elem = row['N_'+elem]
                ion = ion_from_col(col)
                
                ## Converting between Solar Abundances: Grevesse & Sauval 1998 --> Asplund 2009
                FeH_g98 = row['[Fe/H]'] # Note: these [Fe/H] values are [Fe/H]_c values from Cayrel+2004 (avg of Fe I and Fe II)
                logepsFe = eps_from_XH(FeH_g98, 'Fe', version='grevesse1998')
                FeH_a09 = XH_from_eps(logepsFe, 'Fe', version='asplund2009')
                
                ## Limit Flags
                l_logepsX = '<' if '<' in str(row[col]) else np.nan
                l_XH = '<' if '<' in str(row[col]) else np.nan
                l_XFe = '<' if '<' in str(row[col]) else np.nan

                ## Abundance Calculations (Asplund 2009)
                XFe_g98 = float(row[col].replace('<', '')) if l_XFe == '<' else float(row[col])
                logepsX = eps_from_XFe(XFe_g98, FeH_g98, elem, version='grevesse1998')
                XH = XH_from_eps(logepsX, elem, version='asplund2009')
                XFe = XFe_from_eps(logepsX, FeH_a09, elem, version='asplund2009')
                e_XH = np.nan

                ## Add to list, converted into DataFrame after loop
                f07_abund_list.append((row['Name'], ion, N_elem, l_logepsX, logepsX, l_XH, XH, l_XFe, XFe, e_XH))

    f07_abund_df = pd.DataFrame(f07_abund_list, columns=['Name', 'Species', 'N', 'l_logepsX', 'logepsX', 'l_[X/H]', '[X/H]', 'l_[X/Fe]', '[X/Fe]', 'e_[X/H]'])
    # -------------------------------------
    ## Combine the two abundance tables
    abund_df = (
        pd.concat([c04_abund_df, f07_abund_df], ignore_index=True)
        .assign(Z=lambda df: df['Species'].map(ion_to_atomic_number))
        .sort_values(['Name', 'Z'])
        .drop(columns='Z')
        .reset_index(drop=True)
    )
    # -------------------------------------

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
    cayrel2004_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        cayrel2004_df.loc[i,'Name'] = name
        cayrel2004_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        cayrel2004_df.loc[i,'Reference'] = 'Cayrel+2004'
        cayrel2004_df.loc[i,'Ref'] = 'CAY04'
        cayrel2004_df.loc[i,'I/O'] = 1
        cayrel2004_df.loc[i,'Loc'] = 'HA'
        cayrel2004_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        cayrel2004_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        cayrel2004_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(cayrel2004_df.loc[i,'RA_hms'], precision=6)
        cayrel2004_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        cayrel2004_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(cayrel2004_df.loc[i,'DEC_dms'], precision=2)
        cayrel2004_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        cayrel2004_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        cayrel2004_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H_m'].values[0]
        cayrel2004_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                cayrel2004_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                cayrel2004_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    cayrel2004_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    cayrel2004_df.loc[i, 'ul'+col] = np.nan
                else:
                    cayrel2004_df.loc[i, col] = np.nan
                    cayrel2004_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                # if 'e_[X/H]' in row.index:
                #     cayrel2004_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    cayrel2004_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    cayrel2004_df.loc[i, 'ul'+col] = np.nan
                else:
                    cayrel2004_df.loc[i, col] = np.nan
                    cayrel2004_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                # if 'e_[X/Fe]' in row.index:
                #     cayrel2004_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_[X/H]", np.nan)
                if pd.notna(e_logepsX):
                    cayrel2004_df.loc[i, col] = e_logepsX
                else:
                    cayrel2004_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    cayrel2004_df.drop(columns=[col for col in cayrel2004_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Sort by RA_deg, DEC_deg
    cayrel2004_df = cayrel2004_df.sort_values(['RA_deg', 'DEC_deg']).reset_index(drop=True)
    
    return cayrel2004_df

def load_francois2007():
    """
    Loads the Francois et al. 2007 & Cayrel et al. 2004 data for Milky Way halo data. 
    This paper is part of "First Stars." series (First Stars. VIII.), where the stellar parameters 
    are taken from Cayrel et al. 2004 (First Stars. V.) along with the [Fe/H] and 
    light chemical abundances.

    Table 2 - Cayrel+2004 Observation Table
    Table 4 - Cayrel+2004 Stellar Parameters
    Table 8 - Cayrel+2004 Abundance Table
    Table 3,4,5 - Francois+2007 Abundance Table
    """
    francois2007_df = load_cayrel2004()
    
    ## Change out the Reference and Ref columns
    francois2007_df['Reference'] = 'Francois+2007'
    francois2007_df['Ref'] = 'FRA07'

    return francois2007_df

### milky way accreted dwarf galaxies (Acc. dSph)

def load_mardini2022a(io=None):
    """
    Atari Disk (Atr) Stars

    Load the data from Mardini et al. (2022), Table 5, for stars in the Atari Disk (Atr) region.
    """

    mardini2022a_df = pd.read_csv(data_dir+'abundance_tables/mardini2022a/tab5_yelland.csv', comment='#')

    ## Add and rename the necessary columns
    # mardini2022a_df.rename(columns={'source_id':'Name', 'ra':'RA_hms', 'dec':'DEC_deg', 'teff':'Teff'}, inplace=True)
    mardini2022a_df['JINA_ID'] = mardini2022a_df['JINA_ID'].astype(int)
    mardini2022a_df['Name'] = mardini2022a_df['Simbad_Identifier']
    mardini2022a_df['Reference'] = mardini2022a_df['Reference']
    mardini2022a_df['Ref'] = mardini2022a_df['Reference'].str[:3].str.upper() + np.where(mardini2022a_df['Reference'].str[-1].str.isalpha(), mardini2022a_df['Reference'].str[-3:], mardini2022a_df['Reference'].str[-2:])
    mardini2022a_df['Ref'] = mardini2022a_df['Ref'].str.replace('HAN', 'HANt')
    mardini2022a_df['Ref'] = mardini2022a_df['Ref'].str.replace('HOL11', 'HOLj11')
    mardini2022a_df['Ref'] = mardini2022a_df['Ref'].str.replace('HOL20', 'HOLe20')
    # mardini2022a_df['Reference_2'] = 'Mardini+2022a'
    # mardini2022a_df['Ref_2'] = 'MARm22a'
    mardini2022a_df['I/O'] = 1
    mardini2022a_df['Ncap_key'] = ''
    mardini2022a_df['C_key'] = mardini2022a_df['[C/Fe]'].apply(lambda cfe: classify_carbon_enhancement(cfe) if pd.notna(cfe) else np.nan)
    mardini2022a_df['MP_key'] = mardini2022a_df['[Fe/H]'].apply(lambda feh: classify_metallicity(feh) if pd.notna(feh) else np.nan)
    mardini2022a_df['Loc'] = 'aDW'
    mardini2022a_df['System'] = 'Atari'
    mardini2022a_df['RA_deg'] = np.nan
    mardini2022a_df['DEC_deg'] = np.nan

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in mardini2022a_df.iterrows():

        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            ## pad RA_hms with leading zeros
            if len(row['RA_hms']) == 10:
                row['RA_hms'] = '0' + row['RA_hms']
                mardini2022a_df.at[idx, 'RA_hms'] = row['RA_hms']
            row['RA_deg'] = scoord.ra_hms_to_deg(row['RA_hms'], precision=6)
            mardini2022a_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = scoord.dec_dms_to_deg(row['DEC_dms'], precision=6)
            mardini2022a_df.at[idx, 'DEC_deg'] = row['DEC_deg']
            
        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = scoord.ra_deg_to_hms(row['RA_deg'], precision=2)
            mardini2022a_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = scoord.dec_deg_to_dms(row['DEC_deg'], precision=2)
            mardini2022a_df.at[idx, 'DEC_dms'] = row['DEC_dms']

    ## Get the JINAbase Data using the JINA_ID
    jina_ids = list(mardini2022a_df['JINA_ID'])

    jinabase_df = load_jinabase(io=None)
    sub_jinabase_df = jinabase_df[jinabase_df['JINA_ID'].isin(jina_ids)].copy()
    new_columns = [col for col in sub_jinabase_df.columns if col not in mardini2022a_df.columns]
    # new_columns = ['logg']

    # Align on JINA_ID
    mardini2022a_df = mardini2022a_df.set_index('JINA_ID')
    sub_jinabase_df = sub_jinabase_df.set_index('JINA_ID')
    mardini2022a_df = mardini2022a_df.join(sub_jinabase_df[new_columns], how='left')

    # Fill in missing [C/Fe] values from JINAbase
    if '[C/Fe]' in mardini2022a_df.columns and '[C/Fe]' in sub_jinabase_df.columns:
        mardini2022a_df['[C/Fe]'] = mardini2022a_df['[C/Fe]'].fillna(sub_jinabase_df['[C/Fe]'])
    if 'ul[C/Fe]' in mardini2022a_df.columns and 'ul[C/Fe]' in sub_jinabase_df.columns:
        mardini2022a_df['ul[C/Fe]'] = mardini2022a_df['ul[C/Fe]'].fillna(sub_jinabase_df['ul[C/Fe]'])

    ## Manually added datafields
    mardini2022a_df.loc[mardini2022a_df['Name'] == '2MASS J12450268-0738469', 'Ncap_key'] = 'S'  # halo reference star
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'HE 0017-4346', 'Ncap_key'] = 'S'          # [C/Fe] = 3.02
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'HE 1413-1954', 'C_key'] = 'NO'      # [C/Fe] = 1.44
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'HE 1300+0157', 'C_key'] = 'NO'            # (HE 1300+0157, https://www.aanda.org/articles/aa/pdf/2019/03/aa34601-18.pdf)
    mardini2022a_df.loc[mardini2022a_df['Reference'] == 'Aguado+2017', 'logg'] = 4.9
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'SDSS J124719.46-034152.4', 'logg'] = 4.0
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'SDSS J105519.28+232234.0', 'logg'] = 4.9
    
    ## Reset the index
    sub_jinabase_df = sub_jinabase_df.reset_index()
    mardini2022a_df = mardini2022a_df.reset_index()

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        mardini2022a_df = mardini2022a_df[mardini2022a_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    ## Save the processed data to a CSV file
    mardini2022a_df.sort_values(by='[Fe/H]', ascending=False, inplace=True)
    mardini2022a_df.to_csv(data_dir+'abundance_tables/mardini2022a/tab5_processed.csv', index=False)

    return mardini2022a_df

def load_mardini2024b(io=None):
    """
    Load the Koch et al. 2008 data for the Hercules Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations & Stellar Parameters (custom made table from the text)
    Table 2 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/mardini2024b/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/mardini2024b/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    mardini2024b_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        mardini2024b_df.loc[i,'Name'] = name
        mardini2024b_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        mardini2024b_df.loc[i,'Reference'] = 'Mardini+2024b'
        mardini2024b_df.loc[i,'Ref'] = 'MARm24b'
        mardini2024b_df['I/O'] = 1
        mardini2024b_df.loc[i,'Loc'] = 'DW'
        mardini2024b_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        mardini2024b_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        mardini2024b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(mardini2024b_df.loc[i,'RA_hms'], precision=6)
        mardini2024b_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        mardini2024b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(mardini2024b_df.loc[i,'DEC_dms'], precision=2)
        mardini2024b_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        mardini2024b_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        mardini2024b_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        mardini2024b_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                mardini2024b_df.loc[i, col] = normal_round(row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan, 2)

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                mardini2024b_df.loc[i, col] = normal_round(row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan, 2)

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    mardini2024b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    mardini2024b_df.loc[i, 'ul'+col] = np.nan
                else:
                    mardini2024b_df.loc[i, col] = np.nan
                    mardini2024b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    mardini2024b_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    mardini2024b_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    mardini2024b_df.loc[i, 'ul'+col] = np.nan
                else:
                    mardini2024b_df.loc[i, col] = np.nan
                    mardini2024b_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    mardini2024b_df.loc[i, 'e_'+col] = row['e_[X/Fe]']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    mardini2024b_df.loc[i, col] = e_logepsX
                else:
                    mardini2024b_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    mardini2024b_df.drop(columns=[col for col in mardini2024b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        mardini2024b_df = mardini2024b_df[mardini2024b_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return mardini2024b_df

def load_ou2024c(io=None):
    """
    Gaia Sausage Enceladus (GSE) Dwarf Galaxy Stars

    Load the data from Ou+2024, Table 1, for stars in the GSE dwarf galaxy.
    """
    ou2024c_df = pd.read_csv(data_dir+'abundance_tables/ou2024c/ou2024c-yelland.csv', comment='#')
    ou2024c_df['I/O'] = 1

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        ou2024c_df = ou2024c_df[ou2024c_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")

    return ou2024c_df

### small accreted stellar systems (SASS)

def load_hughes2025(io=None):
    """
    Load the Hughes et al. 2025 data for the 10 SASS stars.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/hughes2025/obs_table.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/hughes2025/param_table.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/hughes2025/abund_table.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    hughes2025_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        hughes2025_df.loc[i,'Name'] = name
        hughes2025_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        hughes2025_df.loc[i,'Reference'] = 'Hughes+2025'
        hughes2025_df.loc[i,'Ref'] = 'HUG25'
        hughes2025_df.loc[i,'I/O'] = 1
        hughes2025_df.loc[i,'Loc'] = ''
        hughes2025_df.loc[i,'System'] = 'SASS' # obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        hughes2025_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        hughes2025_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(hughes2025_df.loc[i,'RA_hms'], precision=6)
        hughes2025_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        hughes2025_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(hughes2025_df.loc[i,'DEC_dms'], precision=2)
        hughes2025_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        hughes2025_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        hughes2025_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        hughes2025_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            feh_a09 = star_df.loc[star_df['Species'] == 'Fe I', '[X/H]'].values[0]
            logepsX = normal_round(row["[X/H]"] + logepsX_sun_a09, 2)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                hughes2025_df.loc[i, col] = logepsX if pd.isna(row["l_[X/H]"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                hughes2025_df.loc[i, col] = logepsX if pd.notna(row["l_[X/H]"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    hughes2025_df.loc[i, col] = normal_round(logepsX - logepsX_sun_a09, 2)
                    hughes2025_df.loc[i, 'ul'+col] = np.nan
                else:
                    hughes2025_df.loc[i, col] = np.nan
                    hughes2025_df.loc[i, 'ul'+col] = normal_round(logepsX - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    hughes2025_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    hughes2025_df.loc[i, col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    hughes2025_df.loc[i, 'ul'+col] = np.nan
                else:
                    hughes2025_df.loc[i, col] = np.nan
                    hughes2025_df.loc[i, 'ul'+col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    hughes2025_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            # col = make_errcol(species_i)
            # if col in errcols:
            #     e_logepsX = row.get("e_logepsX", np.nan)
            #     if pd.notna(e_logepsX):
            #         hughes2025_df.loc[i, col] = e_logepsX
            #     else:
            #         hughes2025_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    hughes2025_df.drop(columns=[col for col in hughes2025_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return hughes2025_df

def load_nordlander2019(io=None):
    """
    Load the Nordlander et al. 2019 data for the halo/SASS star SMSS J160540.18-144323.1 (SMSS 1605-1443).

    Table 1 - Observations
    Table 2 - Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/nordlander2019/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/nordlander2019/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    nordlander2019_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        nordlander2019_df.loc[i,'Name'] = name
        nordlander2019_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        nordlander2019_df.loc[i,'Reference'] = 'Nordlander+2019'
        nordlander2019_df.loc[i,'Ref'] = 'NORt19'
        nordlander2019_df.loc[i,'I/O'] = 1
        nordlander2019_df.loc[i,'Loc'] = 'UF' # [HA, BU, DS, DW, UF, GC]
        nordlander2019_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        nordlander2019_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        nordlander2019_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(nordlander2019_df.loc[i,'RA_hms'], precision=6)
        nordlander2019_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        nordlander2019_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(nordlander2019_df.loc[i,'DEC_dms'], precision=2)
        nordlander2019_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        nordlander2019_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        nordlander2019_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        nordlander2019_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                nordlander2019_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                nordlander2019_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    nordlander2019_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    nordlander2019_df.loc[i, 'ul'+col] = np.nan
                else:
                    nordlander2019_df.loc[i, col] = np.nan
                    nordlander2019_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    nordlander2019_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    nordlander2019_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    nordlander2019_df.loc[i, 'ul'+col] = np.nan
                else:
                    nordlander2019_df.loc[i, col] = np.nan
                    nordlander2019_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    nordlander2019_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = np.sqrt(row.get("e_logepsX_stat", np.nan)**2 + row.get("e_logepsX_sys", np.nan)**2)
                if pd.notna(e_logepsX):
                    nordlander2019_df.loc[i, col] = e_logepsX
                else:
                    nordlander2019_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    nordlander2019_df.drop(columns=[col for col in nordlander2019_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return nordlander2019_df

### classical and dwarf spheroidal galaxies (dSph)

def load_chiti2018a(data_subset='merged'):
    """
    Sculptor (Scl) Dwarf Galaxy Stars

    Loads the data from Chiti et al. (2018) for the MagE and M2FS measurements.
    This function reads in the data from the two tables (table5/MagE and table6/M2FS) and
    returns them as pandas DataFrames. By default, the two tables are combined into a single
    DataFrame.
    """

    ## Load the combined table (created by Alex Yelland)
    chiti2018a_df = pd.read_csv(data_dir+'abundance_tables/chiti2018a/chiti2018a.csv', comment='#')
    chiti2018a_df['I/O'] = chiti2018a_df['I/O'].astype(int)

    ## Add/Fill a epsfe column, calculating the value from [Fe/H]
    chiti2018a_df['epsfe'] = chiti2018a_df['[Fe/H]'].apply(lambda x: eps_from_XH(x, 'Fe', precision=2))

    ## Fill the Carbon Abundance Columns
    columns = [('epsc', '[Fe/H]', '[C/Fe]'), ('ulc', '[Fe/H]', 'ul[C/Fe]')]
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
    chiti2018a_df['ul[C/H]'] = chiti2018a_df['ulc'].apply(lambda x: XH_from_eps(x, 'C', precision=2))

    ## Choose the data subset to return
    if data_subset == 'all':
        # Return the full DataFrame
        return chiti2018a_df
    elif data_subset == 'mage':
        # Return only the MagE data
        return chiti2018a_df[chiti2018a_df['Instrument'] == 'MagE'].reset_index(drop=True)
    elif data_subset == 'm2fs':
        # Return only the M2FS data
        return chiti2018a_df[chiti2018a_df['Instrument'] == 'M2FS'].reset_index(drop=True)
    elif data_subset == 'merged':
        # Return the merged data (MagE and M2FS combined)
        return chiti2018a_df[chiti2018a_df['I/O'] == 1].reset_index(drop=True)
    else:
        raise ValueError("Invalid data_subset value. Choose from 'all', 'mage', 'm2fs', or 'merged'.")

def load_chiti2024(io=None):
    """
    Load the Chiti et al. 2024 data for the Large Magellanic Cloud (LMC) stars.

    table_obs - Observations (custom made from Extended Data Table 1)
    table_param - Stellar Parameters (custom made from Table 1 and Extended Data Table 2)
    table_abund1 - Abundance Table (modified from Supplementary Data Table 1)
    table_abund2 - Additional Abundances not in the Supplementary Data Table 1
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/chiti2024/table_obs.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/chiti2024/table_param.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df1 = pd.read_csv(data_dir + "abundance_tables/chiti2024/table_abund1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    ## Make the new column names
    species = []
    for species_i in abund_df1["Species"].unique():
        # species_i = ion_to_species(ion)
        # elem_i = ion_to_element(ion)
        elem_i = species_to_element(species_i)
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
    chiti2024_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Reference','Ref','Instrument','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df1['Name'].unique()):
        chiti2024_df.loc[i,'Name'] = name
        chiti2024_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        chiti2024_df.loc[i,'Reference'] = 'Chiti+2024'
        chiti2024_df.loc[i,'Ref'] = 'CHI24'
        chiti2024_df.loc[i,'I/O'] = int(1)
        chiti2024_df.loc[i,'Instrument'] = obs_df.loc[obs_df['Name'] == name, 'Instrument'].values[0]       
        chiti2024_df.loc[i,'Loc'] = 'DW'
        chiti2024_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        chiti2024_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        chiti2024_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(chiti2024_df.loc[i,'RA_hms'], precision=6)
        chiti2024_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        chiti2024_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(chiti2024_df.loc[i,'DEC_dms'], precision=2)
        chiti2024_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        chiti2024_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        chiti2024_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, '[Fe/H]'].values[0]
        chiti2024_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df1[abund_df1['Name'] == name]
        for j, row in star_df.iterrows():
            # ion = row["Species"]
            # species_i = ion_to_species(ion)
            # elem_i = ion_to_element(ion)

            species_i = row["Species"]
            ion_i = species_to_ion(species_i)
            elem_i = species_to_element(species_i)

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 26.0, 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                chiti2024_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                chiti2024_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    chiti2024_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    chiti2024_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2024_df.loc[i, col] = np.nan
                    chiti2024_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]_tot' in row.index:
                    chiti2024_df.loc[i, 'e_'+col] = row["e_[X/H]_tot"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    chiti2024_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    chiti2024_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2024_df.loc[i, col] = np.nan
                    chiti2024_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]_tot' in row.index:
                    chiti2024_df.loc[i, 'e_'+col] = row["e_[X/Fe]_tot"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_[X/H]_tot", np.nan)
                if pd.notna(e_logepsX):
                    chiti2024_df.loc[i, col] = e_logepsX
                else:
                    chiti2024_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    chiti2024_df.drop(columns=[col for col in chiti2024_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Add the addtional abundances not in the Supplementary Data Table 1
    abund_df2 = pd.read_csv(data_dir + "abundance_tables/chiti2024/table_abund2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
    for i, name in enumerate(abund_df2['Name'].unique()):
        if name not in chiti2024_df['Name'].values:
            # If the star is not already in the dataframe, add a new row
            new_row = {}
            new_row['Name'] = name
            new_row['Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
            new_row['Reference'] = 'Chiti+2024'
            new_row['Ref'] = 'CHI24'
            new_row['I/O'] = int(1)
            new_row['Instrument'] = obs_df.loc[obs_df['Name'] == name, 'Instrument'].values[0]
            new_row['Loc'] = 'DW'
            new_row['System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
            new_row['RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
            new_row['RA_deg'] = scoord.ra_hms_to_deg(obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0], precision=6)
            new_row['DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
            new_row['DEC_deg'] = scoord.dec_dms_to_deg(obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0], precision=2)
            new_row['Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
            new_row['logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
            new_row['Fe/H'] = param_df.loc[param_df['Name'] == name, '[Fe/H]'].values[0]
            new_row['Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]
            for col in abund_df2.columns:
                
                if col == 'Name': continue

                species_i = species_from_col(col)
                elem_i = species_to_element(species_i)
                logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
                feh_a09 = abund_df2.loc[abund_df2['Name'] == name, '[Fe/H]'].values[0]

                if col.startswith('[') and col.endswith('/H]'):
                    new_row[make_epscol(species_i)] = abund_df2.loc[abund_df2['Name'] == name, col].values[0] + logepsX_sun_a09
                    new_row[make_XHcol(species_i).replace(" ", "")] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]
                    new_row[make_XFecol(species_i).replace(" ", "")] = abund_df2.loc[abund_df2['Name'] == name, col].values[0] - feh_a09

                elif col.startswith('ul[') and col.endswith('/H]'):
                    new_row[make_ulcol(species_i)] = abund_df2.loc[abund_df2['Name'] == name, col].values[0] + logepsX_sun_a09
                    new_row['ul' + make_XHcol(species_i).replace(" ", "")] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]
                    new_row['ul' + make_XFecol(species_i).replace(" ", "")] = abund_df2.loc[abund_df2['Name'] == name, col].values[0] - feh_a09

                elif col.startswith('[') and col.endswith('/Fe]'):
                    new_row[make_epscol(species_i)] = abund_df2.loc[abund_df2['Name'] == name, col].values[0] + feh_a09 + logepsX_sun_a09
                    new_row[make_XHcol(species_i).replace(" ", "")] = abund_df2.loc[abund_df2['Name'] == name, col].values[0] + feh_a09
                    new_row[make_XFecol(species_i).replace(" ", "")] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]

                elif col.startswith('ul[') and col.endswith('/Fe]'):
                    new_row[make_ulcol(species_i)] = abund_df2.loc[abund_df2['Name'] == name, col].values[0] + feh_a09 + logepsX_sun_a09
                    new_row['ul' + make_XHcol(species_i).replace(" ", "")] = abund_df2.loc[abund_df2['Name'] == name, col].values[0] + feh_a09
                    new_row['ul' + make_XFecol(species_i).replace(" ", "")] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]

                elif col.startswith('e_[') and col.endswith('/H]'):
                    new_row[make_errcol(species_i)] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]
                    new_row[col] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]
                    
                elif col.startswith('e_[') and col.endswith('/Fe]'):
                    new_row[make_errcol(species_i)] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]
                    new_row[col] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]

            chiti2024_df = pd.concat([chiti2024_df, pd.DataFrame([new_row])], ignore_index=True)

        else:
            # If the star is already in the dataframe, update the existing row
            for col in abund_df2.columns:
                if chiti2024_df.loc[chiti2024_df['Name'] == name, col].empty:
                    chiti2024_df.loc[chiti2024_df['Name'] == name, col] = abund_df2.loc[abund_df2['Name'] == name, col].values[0]

    ## Drop the Fe/Fe columns
    chiti2024_df.drop(columns=[col for col in chiti2024_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        chiti2024_df = chiti2024_df[chiti2024_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return chiti2024_df

def load_chiti2025a(io=None):
    """
    Load the Chiti et al. 2025 data for the Pictor II Ultra-Faint Dwarf Galaxy.

    Table 0 - Observation Table & Stellar Parameters
    Table 2 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + 'abundance_tables/chiti2025a/table0.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/chiti2025a/table2.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

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
    llXFecols = ['ll' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    chiti2025a_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + llXFecols + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        chiti2025a_df.loc[i,'Name'] = name
        chiti2025a_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        chiti2025a_df.loc[i,'Reference'] = 'Chiti+2025a'
        chiti2025a_df.loc[i,'Ref'] = 'CHI25a'
        chiti2025a_df.loc[i,'I/O'] = 1
        chiti2025a_df.loc[i,'Loc'] = 'UF'
        chiti2025a_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]     
        chiti2025a_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        chiti2025a_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(chiti2025a_df.loc[i,'RA_hms'], precision=6)
        chiti2025a_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        chiti2025a_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(chiti2025a_df.loc[i,'DEC_dms'], precision=2)
        chiti2025a_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        chiti2025a_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        chiti2025a_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        chiti2025a_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row['Species']
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            l_FeH = star_df.loc[star_df['Species'] == 'Fe I', 'l_[X/H]'].values[0]
            feh_a09 = star_df.loc[star_df['Species'] == 'Fe I', '[X/H]'].values[0]
            
            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                chiti2025a_df.loc[i, col] = row['[X/H]'] + logepsX_sun_a09 if pd.isna(row['l_[X/H]']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                chiti2025a_df.loc[i, col] = row['[X/H]'] + logepsX_sun_a09 if pd.notna(row['l_[X/H]']) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row['l_[X/H]']):
                    chiti2025a_df.loc[i, col] = normal_round(row["[X/H]"], 2)
                    chiti2025a_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2025a_df.loc[i, col] = np.nan
                    chiti2025a_df.loc[i, 'ul'+col] = normal_round(row["[X/H]"], 2)
                if 'e_[X/H]' in row.index:
                    chiti2025a_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if 'Fe/Fe' in col: continue
                if pd.notna(l_FeH) and pd.isna(row['l_[X/H]']):
                    if l_FeH == '<':
                        chiti2025a_df.loc[i, 'll'+col] = normal_round((row["[X/H]"]) - feh_a09, 2)
                        chiti2025a_df.loc[i, col] = np.nan
                        chiti2025a_df.loc[i, 'ul'+col] = np.nan
                    if l_FeH == '>':
                        chiti2025a_df.loc[i, 'll'+col] = np.nan
                        chiti2025a_df.loc[i, col] = np.nan
                        chiti2025a_df.loc[i, 'ul'+col] = normal_round((row["[X/H]"]) - feh_a09, 2)
                elif pd.isna(l_FeH) and pd.notna(row['l_[X/H]']):
                    if row['l_[X/H]'] == '<':
                        chiti2025a_df.loc[i, 'll'+col] = np.nan
                        chiti2025a_df.loc[i, col] = np.nan
                        chiti2025a_df.loc[i, 'ul'+col] = normal_round((row["[X/H]"]) - feh_a09, 2)
                    if row['l_[X/H]'] == '>':
                        chiti2025a_df.loc[i, 'll'+col] = normal_round((row["[X/H]"]) - feh_a09, 2)
                        chiti2025a_df.loc[i, col] = np.nan
                        chiti2025a_df.loc[i, 'ul'+col] = np.nan
                elif pd.isna(l_FeH) and pd.isna(row['l_[X/H]']):
                    chiti2025a_df.loc[i, 'll'+col] = np.nan
                    chiti2025a_df.loc[i, col] = normal_round((row["[X/H]"]) - feh_a09, 2)
                    chiti2025a_df.loc[i, 'ul'+col] = np.nan      
                elif pd.notna(l_FeH) and pd.notna(row['l_[X/H]']): ## unknown how to resolve -- upperlimit on iron and X element
                    chiti2025a_df.loc[i, 'll'+col] = np.nan
                    chiti2025a_df.loc[i, col] = np.nan
                    chiti2025a_df.loc[i, 'ul'+col] = np.nan
                if 'e_[X/H]' in row.index:
                    chiti2025a_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get('e_[X/H]', np.nan)
                if pd.notna(e_logepsX):
                    chiti2025a_df.loc[i, col] = e_logepsX
                else:
                    chiti2025a_df.loc[i, col] = np.nan
    
    ## Drop the Fe/Fe columns
    chiti2025a_df.drop(columns=[col for col in chiti2025a_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return chiti2025a_df

def load_frebel2010b(io=None):
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
    XHcols = [make_XHcol(s).replace(' ', '') for s in species]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(' ', '') for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    llXFecols = ['ll' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    frebel2010_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + llXFecols + ulXFecols + errcols)
    frebel2010_df.loc[0,'Name'] = 'S1020549'
    frebel2010_df.loc[0,'Simbad_Identifier'] = '2MASS J01004785-3341029'
    frebel2010_df.loc[0,'Reference'] = 'Frebel+2010b'
    frebel2010_df.loc[0,'Ref'] = 'FRE10b'
    frebel2010_df.loc[0,'I/O'] = 1
    frebel2010_df.loc[0,'Loc'] = 'DW'
    frebel2010_df.loc[0,'System'] = 'Sculptor'
    frebel2010_df.loc[0,'RA_hms'] = '01:00:47.80'
    frebel2010_df.loc[0,'RA_deg'] = scoord.ra_hms_to_deg(frebel2010_df.loc[0,'RA_hms'])
    frebel2010_df.loc[0,'DEC_dms'] = '-33:41:03.0'
    frebel2010_df.loc[0,'DEC_deg'] = scoord.dec_dms_to_deg(frebel2010_df.loc[0,'DEC_dms'])
    frebel2010_df.loc[0,'Teff'] = 4550
    frebel2010_df.loc[0,'logg'] = 0.9
    frebel2010_df.loc[0,'Fe/H'] = -3.81
    frebel2010_df.loc[0,'Vmic'] = 2.8

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

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        frebel2010_df = frebel2010_df[frebel2010_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return frebel2010_df

def load_ji2025(io=None):
    """
    Load the Ji et al. 2025 data for a Milky Way star.

    Table 1 - Observations
    Table 2 - Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/ji2025/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    obs_df = pd.read_csv(data_dir + "abundance_tables/ji2025/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/ji2025/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2025/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    ji2025_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','M/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2025_df.loc[i,'Name'] = name
        ji2025_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ji2025_df.loc[i,'Reference'] = 'Ji+2025'
        ji2025_df.loc[i,'Ref'] = 'JI25'
        ji2025_df.loc[i,'I/O'] = 1
        ji2025_df.loc[i,'Loc'] = "DW" # [HA, BU, DS, DW, UF, GC]
        ji2025_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ji2025_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ji2025_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(ji2025_df.loc[i,'RA_hms'], precision=6)
        ji2025_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ji2025_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(ji2025_df.loc[i,'DEC_dms'], precision=2)
        ji2025_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ji2025_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        # ji2025_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        ji2025_df.loc[i,'M/H'] = param_df.loc[param_df['Name'] == name, 'M/H'].values[0]
        ji2025_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                ji2025_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2025_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ji2025_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    ji2025_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2025_df.loc[i, col] = np.nan
                    ji2025_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    ji2025_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2025_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    ji2025_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2025_df.loc[i, col] = np.nan
                    ji2025_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    ji2025_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    ji2025_df.loc[i, col] = e_logepsX
                else:
                    ji2025_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    ji2025_df.drop(columns=[col for col in ji2025_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return ji2025_df

def load_lemasle2012(io=None):
    """
    Load the Lemasle et al. 2012 data for the Carina Classical Dwarf Galaxy.

    Table 3 - Observations
    Table 5 - Stellar Parameters
    Table 7 & 8 - Abundance Tables
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/lemasle2012/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/lemasle2012/table5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/lemasle2012/table7_table8.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    obs_df = obs_df[~(obs_df['Memb'] == 'non-member')]
    
    ## Make the new column names
    species = []
    for col in abund_df.columns:
        if col.startswith('[') and col.endswith('/H]'):
            ion = col[1:-3].strip()
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
    lemasle2012_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        lemasle2012_df.loc[i,'Name'] = name
        lemasle2012_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        lemasle2012_df.loc[i,'Reference'] = 'Lemasle+2012'
        lemasle2012_df.loc[i,'Ref'] = 'LEM12'
        lemasle2012_df.loc[i,'I/O'] = 1
        lemasle2012_df.loc[i,'Loc'] = 'DW'
        lemasle2012_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        lemasle2012_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        lemasle2012_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(lemasle2012_df.loc[i,'RA_hms'], precision=6)
        lemasle2012_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        lemasle2012_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(lemasle2012_df.loc[i,'DEC_dms'], precision=2)
        lemasle2012_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        lemasle2012_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        lemasle2012_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        lemasle2012_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            
            for col in abund_df.columns:
                if col.startswith('[') and col.endswith('/H]'):
                    ion = col[1:-3].strip()
                    species_i = ion_to_species(ion)
                    elem_i = ion_to_element(ion)
                else:
                    continue

                logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
                logepsFe_a09 = star_df['[Fe I/H]'].values[0] + get_solar('Fe', version='asplund2009')[0]
                feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

                logepsX = row[col] + logepsX_sun_a09

                ## Assign epsX values
                col = make_epscol(species_i)
                if col in epscols:
                    lemasle2012_df.loc[i, col] = logepsX

                ## Assign ulX values
                col = make_ulcol(species_i)
                if col in ulcols:
                    lemasle2012_df.loc[i, col] = np.nan

                ## Assign [X/H] and ul[X/H]values
                col = make_XHcol(species_i).replace(" ", "")
                if col in XHcols:
                    # if pd.isna(row["l_[X/H]"]):
                    lemasle2012_df.loc[i, col] = normal_round(logepsX - logepsX_sun_a09, 2)
                    lemasle2012_df.loc[i, 'ul'+col] = np.nan
                    # else:
                    #     lemasle2012_df.loc[i, col] = np.nan
                    #     lemasle2012_df.loc[i, 'ul'+col] = normal_round(logepsX - logepsX_sun_a09, 2)
                    if "e_["+ion+"/H]" in row.index:
                        lemasle2012_df.loc[i, "e_["+ion.replace(" ", "")+"/H]"] = row["e_["+ion+"/H]"]

                ## Assign [X/Fe] values
                col = make_XFecol(species_i).replace(" ", "")
                if col in XFecols:
                    # if pd.isna(row["l_[X/Fe]"]):
                    lemasle2012_df.loc[i, col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    lemasle2012_df.loc[i, 'ul'+col] = np.nan
                    # else:
                    #     lemasle2012_df.loc[i, col] = np.nan
                    #     lemasle2012_df.loc[i, 'ul'+col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    if "e_["+ion+"/Fe]" in row.index:
                        lemasle2012_df.loc[i, "e_["+ion.replace(" ", "")+"/Fe]"] = row["e_["+ion+"/Fe]"]

                    # ## Assign error values
                    # col = make_errcol(species_i)
                    # if col in errcols:
                    #     e_logepsX = row.get("stderr_logepsX", np.nan)
                    #     if pd.notna(e_logepsX):
                    #         lemasle2012_df.loc[i, col] = e_logepsX
                    #     else:
                    #         lemasle2012_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    lemasle2012_df.drop(columns=[col for col in lemasle2012_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        lemasle2012_df = lemasle2012_df[lemasle2012_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return lemasle2012_df

def load_lemasle2014(io=None):
    """
    Load the Lemasle et al. 2014 data for the Fornax dwarf spheroidal galaxy.

    Table A.3 - Observation Parameters
    Table 3 - Stellar Parameters (custom made table from the text)
    Table A.5 - Abundance Table

    Note: Which solar abundances used in this dataset is unclear/not mentioned in the paper. Assuming they follow Aslpund et al. 2009
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/lemasle2014/tablea3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/lemasle2014/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/lemasle2014/tablea5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
    ## Make the new column names
    species = []
    for col in abund_df.columns:
        if col.startswith('[') and col.endswith(']'):
            ion = col.replace('[', '').replace('/Fe]', '').replace('/H]', '')
            if 'II' in ion: 
                ion = ion.replace('II', ' II')
            elif 'I' in ion: 
                ion = ion.replace('I', ' I')
            else:
                ion = ion + ' ' + 'I'*get_default_ion(ion)
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
    lemasle2014_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        lemasle2014_df.loc[i,'Name'] = name
        lemasle2014_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        lemasle2014_df.loc[i,'Reference'] = 'Lemasle+2014'
        lemasle2014_df.loc[i,'Ref'] = 'LEM14'
        lemasle2014_df.loc[i,'I/O'] = 1
        lemasle2014_df.loc[i,'Loc'] = 'DW'
        lemasle2014_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        lemasle2014_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        lemasle2014_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(lemasle2014_df.loc[i,'RA_hms'], precision=6)
        lemasle2014_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        lemasle2014_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(lemasle2014_df.loc[i,'DEC_dms'], precision=2)
        lemasle2014_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        lemasle2014_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        lemasle2014_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        lemasle2014_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]
    
        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():

            logepsFe_sun_a09 = get_solar('Fe', version='asplund2009')[0]
            feh_a09 = star_df['[FeI/H]'].values[0]
            logepsFe_a09 = feh_a09 + logepsFe_sun_a09

            
            for j_col in row.index:
                if j_col.startswith('o_[') or j_col == 'Name': continue

                ## Get the ion/species/element
                if j_col.startswith('[') and j_col.endswith(']'):
                    ion = j_col.replace('[', '').replace('/Fe]', '').replace('/H]', '')
                    if 'II' in ion: 
                        ion = ion.replace('II', ' II')
                    elif 'I' in ion: 
                        ion = ion.replace('I', ' I')
                    else:
                        ion = ion + ' ' + 'I'*get_default_ion(ion)
                    species_i = ion_to_species(ion)
                    elem_i = ion_to_element(ion)

                    logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]

                
                if j_col.startswith('[') and j_col.endswith('/H]'):
                    
                    ## Assign epsX values
                    col = make_epscol(species_i)
                    if col in epscols:
                        lemasle2014_df.loc[i, col] = normal_round(row[j_col] + logepsX_sun_a09, 2) # if pd.isna(row["l_logepsX"]) else np.nan, 2)

                    ## Assign ulX values
                    col = make_ulcol(species_i)
                    if col in ulcols:
                        lemasle2014_df.loc[i, col] = np.nan ## no upper limits in this dataset

                    ## Assign [X/H] and ul[X/H]values
                    col = make_XHcol(species_i).replace(" ", "")
                    if col in XHcols:
                        lemasle2014_df.loc[i, col] = normal_round(row[j_col], 2)
                        lemasle2014_df.loc[i, 'ul'+col] = np.nan

                    ## Assign [X/Fe] values
                    col = make_XFecol(species_i).replace(" ", "")
                    if col in XFecols:
                        lemasle2014_df.loc[i, col] = normal_round(row[j_col] - feh_a09, 2)
                        lemasle2014_df.loc[i, 'ul'+col] = np.nan

                    ## Assign error values
                    col = make_errcol(species_i)
                    if col in errcols:
                        e_logepsX = row.get("e_"+j_col, np.nan)
                        if pd.notna(e_logepsX):
                            lemasle2014_df.loc[i, col] = e_logepsX
                        else:
                            lemasle2014_df.loc[i, col] = np.nan

                if j_col.startswith('[') and j_col.endswith('/Fe]'):
                    
                    ## Assign epsX values
                    col = make_epscol(species_i)
                    if col in epscols:
                        lemasle2014_df.loc[i, col] = normal_round(row[j_col] + feh_a09 + logepsX_sun_a09, 2) # if pd.isna(row["l_logepsX"]) else np.nan, 2)

                    ## Assign ulX values
                    col = make_ulcol(species_i)
                    if col in ulcols:
                        lemasle2014_df.loc[i, col] = np.nan ## no upper limits in this dataset

                    ## Assign [X/H] and ul[X/H]values
                    col = make_XHcol(species_i).replace(" ", "")
                    if col in XHcols:
                        lemasle2014_df.loc[i, col] = normal_round(row[j_col] + feh_a09, 2)
                        lemasle2014_df.loc[i, 'ul'+col] = np.nan

                    ## Assign [X/Fe] values
                    col = make_XFecol(species_i).replace(" ", "")
                    if col in XFecols:
                        lemasle2014_df.loc[i, col] = normal_round(row[j_col], 2)
                        lemasle2014_df.loc[i, 'ul'+col] = np.nan

                    ## Assign error values
                    col = make_errcol(species_i)
                    if col in errcols:
                        e_logepsX = row.get("e_"+j_col, np.nan)
                        if pd.notna(e_logepsX):
                            lemasle2014_df.loc[i, col] = e_logepsX
                        else:
                            lemasle2014_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    lemasle2014_df.drop(columns=[col for col in lemasle2014_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        lemasle2014_df = lemasle2014_df[lemasle2014_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
        
    return lemasle2014_df

def load_letarte2010(io=None):
    """
    Load the Letarte et al. 2010 data for the Fornax Classical Dwarf Galaxy.

    Table A.2 & A.3 - Observations & Stellar Parameters (custom made table from the text)
    Table A.5 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/letarte2010/tablea2_tablea3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/letarte2010/tablea5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
    ## Make the new column names
    species = []
    for col in abund_df.columns:
        if col.startswith('[') and col.endswith(']'):
            ion = col.replace('[', '').replace('/Fe]', '').replace('/H]', '')
            if 'II' in ion: 
                ion = ion.replace('II', ' II')
            elif 'I' in ion: 
                ion = ion.replace('I', ' I')
            else:
                ion = ion + ' ' + 'I'*get_default_ion(ion)
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
    letarte2010_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        letarte2010_df.loc[i,'Name'] = name
        letarte2010_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        letarte2010_df.loc[i,'Reference'] = 'Letarte+2010'
        letarte2010_df.loc[i,'Ref'] = 'LET10'
        letarte2010_df.loc[i,'I/O'] = 1        
        letarte2010_df.loc[i,'Loc'] = 'DW'
        letarte2010_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        letarte2010_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        letarte2010_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(letarte2010_df.loc[i,'RA_hms'], precision=6)
        letarte2010_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        letarte2010_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(letarte2010_df.loc[i,'DEC_dms'], precision=2)
        letarte2010_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        letarte2010_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        letarte2010_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        letarte2010_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]
    
        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():

            logepsFe_sun_a89 = get_solar(elem_i, version='anders1989')[0]
            feh_a89 = star_df['[FeI/H]'].values[0]
            logepsFe = feh_a89 + logepsFe_sun_a89
            feh_a09 = logepsFe - get_solar('Fe', version='asplund2009')[0]

            
            for j_col in row.index:
                if j_col.startswith('o_[') or j_col == 'Name': continue

                ## Get the ion/species/element
                if j_col.startswith('[') and j_col.endswith(']'):
                    ion = j_col.replace('[', '').replace('/Fe]', '').replace('/H]', '')
                    if 'II' in ion: 
                        ion = ion.replace('II', ' II')
                    elif 'I' in ion: 
                        ion = ion.replace('I', ' I')
                    else:
                        ion = ion + ' ' + 'I'*get_default_ion(ion)
                    species_i = ion_to_species(ion)
                    elem_i = ion_to_element(ion)

                    if elem_i in ['Ti','Fe','La']:
                        ## using the Grevesse & Sauval 1998 values for these 3 elements (in text)
                        logepsX_sun_a89 = get_solar(elem_i, version='grevesse1998')[0] 
                    else:
                        logepsX_sun_a89 = get_solar(elem_i, version='anders1989')[0]
                    logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]

                
                if j_col.startswith('[') and j_col.endswith('/H]'):
                    
                    ## Assign epsX values
                    col = make_epscol(species_i)
                    if col in epscols:
                        letarte2010_df.loc[i, col] = normal_round(row[j_col] + logepsX_sun_a89, 2) # if pd.isna(row["l_logepsX"]) else np.nan, 2)

                    ## Assign ulX values
                    col = make_ulcol(species_i)
                    if col in ulcols:
                        letarte2010_df.loc[i, col] = np.nan ## no upper limits in this dataset

                    ## Assign [X/H] and ul[X/H]values
                    col = make_XHcol(species_i).replace(" ", "")
                    if col in XHcols:
                        letarte2010_df.loc[i, col] = normal_round((row[j_col] + logepsX_sun_a89) - logepsX_sun_a09, 2)
                        letarte2010_df.loc[i, 'ul'+col] = np.nan

                    ## Assign [X/Fe] values
                    col = make_XFecol(species_i).replace(" ", "")
                    if col in XFecols:
                        letarte2010_df.loc[i, col] = normal_round(((row[j_col] + logepsX_sun_a89) - logepsX_sun_a09) - feh_a09, 2)
                        letarte2010_df.loc[i, 'ul'+col] = np.nan

                    ## Assign error values
                    col = make_errcol(species_i)
                    if col in errcols:
                        e_logepsX = row.get("e_"+j_col, np.nan)
                        if pd.notna(e_logepsX):
                            letarte2010_df.loc[i, col] = e_logepsX
                        else:
                            letarte2010_df.loc[i, col] = np.nan

                if j_col.startswith('[') and j_col.endswith('/Fe]'):
                    
                    ## Assign epsX values
                    col = make_epscol(species_i)
                    if col in epscols:
                        letarte2010_df.loc[i, col] = normal_round(row[j_col] + feh_a89 + logepsX_sun_a89, 2) # if pd.isna(row["l_logepsX"]) else np.nan, 2)

                    ## Assign ulX values
                    col = make_ulcol(species_i)
                    if col in ulcols:
                        letarte2010_df.loc[i, col] = np.nan ## no upper limits in this dataset

                    ## Assign [X/H] and ul[X/H]values
                    col = make_XHcol(species_i).replace(" ", "")
                    if col in XHcols:
                        letarte2010_df.loc[i, col] = normal_round((row[j_col] + feh_a89  + logepsX_sun_a89) - logepsX_sun_a09, 2)
                        letarte2010_df.loc[i, 'ul'+col] = np.nan

                    ## Assign [X/Fe] values
                    col = make_XFecol(species_i).replace(" ", "")
                    if col in XFecols:
                        letarte2010_df.loc[i, col] = normal_round(((row[j_col] + feh_a89  + logepsX_sun_a89) - logepsX_sun_a09) - feh_a09, 2)
                        letarte2010_df.loc[i, 'ul'+col] = np.nan

                    ## Assign error values
                    col = make_errcol(species_i)
                    if col in errcols:
                        e_logepsX = row.get("e_"+j_col, np.nan)
                        if pd.notna(e_logepsX):
                            letarte2010_df.loc[i, col] = e_logepsX
                        else:
                            letarte2010_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    letarte2010_df.drop(columns=[col for col in letarte2010_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        letarte2010_df = letarte2010_df[letarte2010_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return letarte2010_df
    
def load_lucchesi2024(io=None):
    """
    Carina (Car) and Fornax (Fnx) Dwarf Galaxy Stars
    --------------------------------------------
    Load the Lucchesi et al. 2024 data for the Carina & Fornax dSph galaxies. There
    are 4 stars in Carina and 2 stars in Fornax.

    Table 0 - Observations & Stellar Parameters (created from Table 1, 2, 3)
    Table A.4 - Abundance Table (restructured from the original Table A.4)
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/lucchesi2024/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/lucchesi2024/tablea4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    lucchesi2024_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        lucchesi2024_df.loc[i,'Name'] = name
        lucchesi2024_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        lucchesi2024_df.loc[i,'Reference'] = 'Lucchesi+2024'
        lucchesi2024_df.loc[i,'Ref'] = 'LUC24'
        lucchesi2024_df.loc[i,'I/O'] = 1
        lucchesi2024_df.loc[i,'Loc'] = 'DW'
        lucchesi2024_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        lucchesi2024_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        lucchesi2024_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(lucchesi2024_df.loc[i,'RA_hms'], precision=6)
        lucchesi2024_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        lucchesi2024_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(lucchesi2024_df.loc[i,'DEC_dms'], precision=2)
        lucchesi2024_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        lucchesi2024_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        lucchesi2024_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        lucchesi2024_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                lucchesi2024_df.loc[i, col] = normal_round(row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan, 2)

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                lucchesi2024_df.loc[i, col] = normal_round(row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan, 2)

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    lucchesi2024_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    lucchesi2024_df.loc[i, 'ul'+col] = np.nan
                else:
                    lucchesi2024_df.loc[i, col] = np.nan
                    lucchesi2024_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    lucchesi2024_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    lucchesi2024_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    lucchesi2024_df.loc[i, 'ul'+col] = np.nan
                else:
                    lucchesi2024_df.loc[i, col] = np.nan
                    lucchesi2024_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    lucchesi2024_df.loc[i, 'e_'+col] = row['e_[X/Fe]']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_[X/Fe]", np.nan)
                if pd.notna(e_logepsX):
                    lucchesi2024_df.loc[i, col] = e_logepsX
                else:
                    lucchesi2024_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    lucchesi2024_df.drop(columns=[col for col in lucchesi2024_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        lucchesi2024_df = lucchesi2024_df[lucchesi2024_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return lucchesi2024_df

def load_norris2017b(io=None):
    """
    Load the Norris et al. 2017b data for the Carina Classical Dwarf Spheroidal Galaxies.
    Paper reports on 63 stars, but only 32 new stars are from this study. The other 31
    stars are from Venn+2012, Shetrone+2003, and Lemasle+2012.

    Table 1 - Observations
    Table 5 - Stellar Parameters
    Table 6 - Abundance Table
    """
    
    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/norris2017b/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/norris2017b/table5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/norris2017b/table6.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    norris2017b_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        norris2017b_df.loc[i,'Name'] = name
        norris2017b_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        norris2017b_df.loc[i,'Reference'] = 'Norris+2017b'
        norris2017b_df.loc[i,'Ref'] = 'NOR17b'
        norris2017b_df.loc[i,'I/O'] = 1
        norris2017b_df.loc[i,'Loc'] = 'DW'
        norris2017b_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        norris2017b_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        norris2017b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(norris2017b_df.loc[i,'RA_hms'], precision=6)
        norris2017b_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        norris2017b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(norris2017b_df.loc[i,'DEC_dms'], precision=2)
        norris2017b_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        norris2017b_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        norris2017b_df.loc[i,'M/H'] = param_df.loc[param_df['Name'] == name, 'M/H'].values[0]
        norris2017b_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                norris2017b_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                norris2017b_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    norris2017b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    norris2017b_df.loc[i, 'ul'+col] = np.nan
                else:
                    norris2017b_df.loc[i, col] = np.nan
                    norris2017b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    norris2017b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    norris2017b_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    norris2017b_df.loc[i, 'ul'+col] = np.nan
                else:
                    norris2017b_df.loc[i, col] = np.nan
                    norris2017b_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    norris2017b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    norris2017b_df.loc[i, col] = e_logepsX
                else:
                    norris2017b_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    norris2017b_df.drop(columns=[col for col in norris2017b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')
    
    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        norris2017b_df = norris2017b_df[norris2017b_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return norris2017b_df

def load_ou2025(io=None):
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Ou+2025 for stars in the Sagittarius dwarf galaxy.
    """
    ou2025_df = pd.read_csv(data_dir+'abundance_tables/ou2025/ou2025-yelland.csv', comment='#')
    ou2025_df['I/O'] = 1

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        ou2025_df = ou2025_df[ou2025_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")

    return ou2025_df

def load_reggiani2021(io=None):
    """
    Load the Reggiani et al. 2021 data for the Small and Large Magellanic Clouds.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 5 - Abundance Table
    """

    obs_df = pd.read_csv(data_dir + 'abundance_tables/reggiani2021/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/reggiani2021/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/reggiani2021/table5.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

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
    reggiani2021_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        reggiani2021_df.loc[i,'Name'] = name
        reggiani2021_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        reggiani2021_df.loc[i,'Reference'] = 'Reggiani+2021'
        reggiani2021_df.loc[i,'Ref'] = 'REG21'
        reggiani2021_df.loc[i,'I/O'] = 1
        reggiani2021_df.loc[i,'Loc'] = 'DW'
        reggiani2021_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        reggiani2021_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        reggiani2021_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(reggiani2021_df.loc[i,'RA_hms'], precision=6)
        reggiani2021_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        reggiani2021_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(reggiani2021_df.loc[i,'DEC_dms'], precision=2)
        reggiani2021_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        reggiani2021_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        reggiani2021_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        reggiani2021_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                reggiani2021_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                reggiani2021_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    reggiani2021_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    reggiani2021_df.loc[i, 'ul'+col] = np.nan
                else:
                    reggiani2021_df.loc[i, col] = np.nan
                    reggiani2021_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    reggiani2021_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    reggiani2021_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    reggiani2021_df.loc[i, 'ul'+col] = np.nan
                else:
                    reggiani2021_df.loc[i, col] = np.nan
                    reggiani2021_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    reggiani2021_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    reggiani2021_df.loc[i, col] = e_logepsX
                else:
                    reggiani2021_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    reggiani2021_df.drop(columns=[col for col in reggiani2021_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        reggiani2021_df = reggiani2021_df[reggiani2021_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    return reggiani2021_df

def load_sestito2024b(io=None):
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Sestito for stars in the Sagittarius dwarf galaxy, focused on Carbon.
    PIGS IX (Table 4) is used for this dataset.
    """
    sestito2024b_df = pd.read_csv(data_dir+'abundance_tables/sestito2024b/sestito2024b-yelland.csv', comment='#')
    sestito2024b_df['I/O'] = 1

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        sestito2024b_df = sestito2024b_df[sestito2024b_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")

    return sestito2024b_df

def load_sestito2024d(io=None):
    """
    Sagittarius (Sag) Dwarf Galaxy Stars

    Load the data from Sestito et al. 2024b for stars in the Sagittarius dwarf galaxy. This is low/med-resolution 
    photometry from the PIGS X survey.
    """
    sestito2024d_df = pd.read_csv(data_dir+'abundance_tables/sestito2024d/membpara.csv', comment='#')

    ## Add and rename the necessary columns
    sestito2024d_df.rename(columns={'PIGS':'Name', 'RAdeg':'RA_deg', 'DEdeg':'DEC_deg', 'GaiaDR3':'Simbad_Identifier', 'RV':'Vel', 'e_RV':'e_Vel'}, inplace=True)
    sestito2024d_df['Simbad_Identifier'] = 'Gaia DR3 ' + sestito2024d_df['Simbad_Identifier'].astype(str)
    sestito2024d_df['Reference'] = 'Sestito+2024d'
    sestito2024d_df['Ref'] = 'SES24d'
    sestito2024d_df['I/O'] = 1
    sestito2024d_df['Loc'] = 'DW'
    sestito2024d_df['System'] = 'Sagittarius'
    sestito2024d_df['RA_hms'] = np.nan
    sestito2024d_df['DEC_dms'] = np.nan
    sestito2024d_df.drop(columns={'[C/Fe]corr', 'Unnamed: 18'}, inplace=True) # not trustworthy values
    sestito2024d_df['[C/Fe]c'] = np.nan
    sestito2024d_df['[C/Fe]f'] = np.nan

    ## Fill the NaN values in the RA and DEC columns
    for idx, row in sestito2024d_df.iterrows():
        if pd.isna(row['RA_deg']) and pd.notna(row['RA_hms']):
            ## pad RA_hms with leading zeros
            if len(row['RA_hms']) == 10:
                row['RA_hms'] = '0' + row['RA_hms']
                sestito2024d_df.at[idx, 'RA_hms'] = row['RA_hms']
            row['RA_deg'] = scoord.ra_hms_to_deg(row['RA_hms'], precision=6)
            sestito2024d_df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = scoord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            sestito2024d_df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = scoord.ra_deg_to_hms(float(row['RA_deg']), precision=2)
            sestito2024d_df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = scoord.dec_deg_to_dms(float(row['DEC_deg']), precision=2)
            sestito2024d_df.at[idx, 'DEC_dms'] = row['DEC_dms']

    # Categorize columns & reorder dataFrame
    columns = list(sestito2024d_df.columns)
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
    sestito2024d_df = sestito2024d_df[ordered_cols]

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        sestito2024d_df = sestito2024d_df[sestito2024d_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return sestito2024d_df

def load_shetrone2003(io=None):
    """
    Load the Shetrone et al. 2003 data for Carina, Fornax, Leo I, Sculptor, M30, M55, and M68.
    (Note: M55-283 and M55-76 do not have coordinates or Simbad_Identifier since I could
    not find them in the references or on SIMBAD.)

    Table 1 - Observations
    Table 5 - Stellar Parameters
    Table 7,8,9,10 - Abundance Table
    """
    
    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/shetrone2003/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/shetrone2003/table5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/shetrone2003/table7_8_9_10.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    shetrone2003_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        shetrone2003_df.loc[i,'Name'] = name
        shetrone2003_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        shetrone2003_df.loc[i,'Reference'] = 'Shetrone+2003'
        shetrone2003_df.loc[i,'Ref'] = 'SHE03'
        shetrone2003_df.loc[i,'I/O'] = 1
        shetrone2003_df.loc[i,'Loc'] = 'DW'
        shetrone2003_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        if pd.notna(obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]):
            shetrone2003_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
            shetrone2003_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(shetrone2003_df.loc[i,'RA_hms'], precision=6)
            shetrone2003_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
            shetrone2003_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(shetrone2003_df.loc[i,'DEC_dms'], precision=2)
        shetrone2003_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        shetrone2003_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        shetrone2003_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        shetrone2003_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Calculate the logepsX value
            if ion not in ['Fe I', 'Fe II']:
                xfe_she03 = row["[X/Fe]"]
                feh_she03 = star_df.loc[star_df['Species'] == 'Fe I', '[X/H]'].values[0]
                logepsX_sun_she03 = row['logepsX_sun']
                logepsX = normal_round(xfe_she03 + feh_she03 + logepsX_sun_she03, 2)
            else:
                logepsX_sun_she03 = row['logepsX_sun']
                logepsX = normal_round(row["[X/H]"] + logepsX_sun_she03, 2)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                shetrone2003_df.loc[i, col] = logepsX if pd.isna(row["l_[X/Fe]"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                shetrone2003_df.loc[i, col] = logepsX if pd.notna(row["l_[X/Fe]"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    shetrone2003_df.loc[i, col] = normal_round(logepsX - logepsX_sun_a09, 2)
                    shetrone2003_df.loc[i, 'ul'+col] = np.nan
                else:
                    shetrone2003_df.loc[i, col] = np.nan
                    shetrone2003_df.loc[i, 'ul'+col] = normal_round(logepsX - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    shetrone2003_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            feh_a09 = shetrone2003_df.loc[shetrone2003_df['Name'] == name, '[Fe/H]'].values[0]
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    shetrone2003_df.loc[i, col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    shetrone2003_df.loc[i, 'ul'+col] = np.nan
                else:
                    shetrone2003_df.loc[i, col] = np.nan
                    shetrone2003_df.loc[i, 'ul'+col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    shetrone2003_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_XFe = row.get("e_[X/Fe]", np.nan)
                if pd.notna(e_XFe):
                    shetrone2003_df.loc[i, col] = e_XFe
                else:
                    shetrone2003_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    shetrone2003_df.drop(columns=[col for col in shetrone2003_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')
    
    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        shetrone2003_df = shetrone2003_df[shetrone2003_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return shetrone2003_df

def load_venn2012(io=None):
    """
    Load the Venn et al. 2012 data for the Carina Classical Dwarf Spheroidal Galaxies.

    Table 1 - Observations
    Table 6 - Stellar Parameters
    Table 10,11,12,13 - Abundance Table
    """
    
    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/venn2012/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/venn2012/table6.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/venn2012/table10_11_12_13.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    venn2012_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        venn2012_df.loc[i,'Name'] = name
        venn2012_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        venn2012_df.loc[i,'Reference'] = 'Venn+2012'
        venn2012_df.loc[i,'Ref'] = 'VEN12'
        venn2012_df.loc[i,'I/O'] = 1
        venn2012_df.loc[i,'Loc'] = 'DW'
        venn2012_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        venn2012_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        venn2012_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(venn2012_df.loc[i,'RA_hms'], precision=6)
        venn2012_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        venn2012_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(venn2012_df.loc[i,'DEC_dms'], precision=2)
        venn2012_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        venn2012_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        venn2012_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        venn2012_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            ## Calculate the logepsX value
            if ion not in ['Fe I']:
                xfe_a09 = row["[X/Fe]"]
                feh_a09 = star_df.loc[star_df['Species'] == 'Fe I', '[X/H]'].values[0]
                logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
                logepsX = normal_round(xfe_a09 + feh_a09 + logepsX_sun_a09, 2)
            else:
                logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
                logepsX = normal_round(row["[X/H]"] + logepsX_sun_a09, 2)

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                venn2012_df.loc[i, col] = logepsX if pd.isna(row["l_[X/Fe]"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                venn2012_df.loc[i, col] = logepsX if pd.notna(row["l_[X/Fe]"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    venn2012_df.loc[i, col] = normal_round(logepsX - logepsX_sun_a09, 2)
                    venn2012_df.loc[i, 'ul'+col] = np.nan
                else:
                    venn2012_df.loc[i, col] = np.nan
                    venn2012_df.loc[i, 'ul'+col] = normal_round(logepsX - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    venn2012_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            feh_a09 = venn2012_df.loc[venn2012_df['Name'] == name, '[Fe/H]'].values[0]
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    venn2012_df.loc[i, col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    venn2012_df.loc[i, 'ul'+col] = np.nan
                else:
                    venn2012_df.loc[i, col] = np.nan
                    venn2012_df.loc[i, 'ul'+col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    venn2012_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_XFe = row.get("e_[X/Fe]", np.nan)
                if pd.notna(e_XFe):
                    venn2012_df.loc[i, col] = e_XFe
                else:
                    venn2012_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    venn2012_df.drop(columns=[col for col in venn2012_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')
    
    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        venn2012_df = venn2012_df[venn2012_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return venn2012_df

### ultra-faint dwarf galaxies (UFD)

def load_chiti2018b(io=None):
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
    abund_df['logepsX'] = abund_df['[X/H]'] + abund_df['logepsX_sun']

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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        chiti2018b_df.loc[i,'Name'] = name
        chiti2018b_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        chiti2018b_df.loc[i,'Reference'] = 'Chiti+2018b'
        chiti2018b_df.loc[i,'Ref'] = 'CHI18b'
        chiti2018b_df.loc[i,'I/O'] = 1
        chiti2018b_df.loc[i,'Loc'] = 'UF'
        chiti2018b_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        chiti2018b_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        chiti2018b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(chiti2018b_df.loc[i,'RA_hms'], precision=6)
        chiti2018b_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        chiti2018b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(chiti2018b_df.loc[i,'DEC_dms'], precision=2)
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

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

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
                    chiti2018b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    chiti2018b_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2018b_df.loc[i, col] = np.nan
                    chiti2018b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    chiti2018b_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_[X/Fe]']):
                    chiti2018b_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    chiti2018b_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2018b_df.loc[i, col] = np.nan
                    chiti2018b_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
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
    
    ## Drop the Fe/Fe columns
    chiti2018b_df.drop(columns=[col for col in chiti2018b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return chiti2018b_df

def load_chiti2023(io=None):
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
    abund_df['logepsX'] = abund_df['[X/H]'] + abund_df['logepsX_sun']

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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        chiti2023_df.loc[i,'Name'] = name
        chiti2023_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        chiti2023_df.loc[i,'Reference'] = 'Chiti+2023'
        chiti2023_df.loc[i,'Ref'] = 'CHI23'
        chiti2023_df.loc[i,'I/O'] = 1
        chiti2023_df.loc[i,'Loc'] = 'UF'
        chiti2023_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        chiti2023_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        chiti2023_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(chiti2023_df.loc[i,'RA_hms'], precision=6)
        chiti2023_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        chiti2023_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(chiti2023_df.loc[i,'DEC_dms'], precision=2)
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

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

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
                    chiti2023_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    chiti2023_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2023_df.loc[i, col] = np.nan
                    chiti2023_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    chiti2023_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_[X/Fe]']):
                    chiti2023_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    chiti2023_df.loc[i, 'ul'+col] = np.nan
                else:
                    chiti2023_df.loc[i, col] = np.nan
                    chiti2023_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
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

    ## Drop the Fe/Fe columns
    chiti2023_df.drop(columns=[col for col in chiti2023_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return chiti2023_df

def load_feltzing2009(io=None):
    """
    Load the Feltzing et al. 2009 data for the Bootes I Ultra-Faint Dwarf Galaxies.

    Table 1a - Observations & Stellar Parameters (custom made table from the text)
    Table 1b - Abundance Table
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        feltzing2009_df.loc[i,'Name'] = name
        feltzing2009_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        feltzing2009_df.loc[i,'Reference'] = 'Feltzing+2009'
        feltzing2009_df.loc[i,'Ref'] = 'FEL09'
        feltzing2009_df.loc[i,'I/O'] = 1
        feltzing2009_df.loc[i,'Loc'] = 'UF'
        feltzing2009_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        feltzing2009_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        feltzing2009_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(feltzing2009_df.loc[i,'RA_hms'], precision=6)
        feltzing2009_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        feltzing2009_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(feltzing2009_df.loc[i,'DEC_dms'], precision=2)
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
                else:
                    feltzing2009_df.loc[i, col] = np.nan
                    feltzing2009_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    feltzing2009_df.loc[i, 'e_'+col] = row['e_[X/H]']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    feltzing2009_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    feltzing2009_df.loc[i, 'ul'+col] = np.nan
                else:
                    feltzing2009_df.loc[i, col] = np.nan
                    feltzing2009_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    feltzing2009_df.loc[i, 'e_'+col] = row['e_[X/Fe]']

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    feltzing2009_df.loc[i, col] = e_logepsX
                else:
                    feltzing2009_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    feltzing2009_df.drop(columns=[col for col in feltzing2009_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return feltzing2009_df

def load_francois2016(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols + ulXFecols) # + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        francois2016_df.loc[i,'Name'] = name
        francois2016_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        francois2016_df.loc[i,'Reference'] = 'Francois+2016'
        francois2016_df.loc[i,'Ref'] = 'FRA16'
        francois2016_df.loc[i,'I/O'] = 1
        francois2016_df.loc[i,'Loc'] = 'UF'
        francois2016_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        francois2016_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        francois2016_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(francois2016_df.loc[i,'RA_hms'], precision=6)
        francois2016_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        francois2016_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(francois2016_df.loc[i,'DEC_dms'], precision=2)
        francois2016_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        francois2016_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        francois2016_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        francois2016_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

    for i, row in abund_df.iterrows():

        for col in abund_df.columns:

            if col in ['Name', 'Simbad_Identifier']: continue
            if col.startswith('l_'): continue

            elem = col.replace('[', '').replace('/H]', '').replace('/Fe]', '')
            epsX = 'eps' + elem.lower()
            ulX = 'ul' + elem.lower()
            XH = f'[{elem}/H]'
            ulXH = f'ul[{elem}/H]'
            XFe = f'[{elem}/Fe]'
            ulXFe = f'ul[{elem}/Fe]'
            # errcol = 'e_' + XFecol

            ## epsX and ulX
            solar_logepsX_g1998 = get_solar(elem, version='grevesse1998')[0]
            if col.startswith('[') and col.endswith('/H]'): # only used for [Fe/H]
                if pd.isna(row['l_'+XH]):
                    francois2016_df.loc[i, epsX] = normal_round(row[col] + solar_logepsX_g1998, 2)
                    francois2016_df.loc[i, ulX] = np.nan
                else:
                    francois2016_df.loc[i, epsX] = np.nan
                    francois2016_df.loc[i, ulX] = normal_round(row[col] + solar_logepsX_g1998, 2)
            elif col.startswith('[') and col.endswith('/Fe]'):
                if pd.isna(row['l_'+col]):
                    francois2016_df.loc[i, epsX] = normal_round(row[col] + row['[Fe/H]'] + solar_logepsX_g1998, 2)
                    francois2016_df.loc[i, ulX] = np.nan
                else:
                    francois2016_df.loc[i, epsX] = np.nan
                    francois2016_df.loc[i, ulX] = normal_round(row[col] + row['[Fe/H]'] + solar_logepsX_g1998, 2)

            ## XH and ulXH
            solar_logepsX_a2009 = get_solar(elem, version='asplund2009')[0]
            if pd.isna(row['l_'+col]):
                francois2016_df.loc[i, XH] = normal_round(francois2016_df.loc[i, epsX] - solar_logepsX_a2009, 2)
                francois2016_df.loc[i, ulXH] = np.nan
            else:
                francois2016_df.loc[i, XH] = np.nan
                francois2016_df.loc[i, ulXH] = normal_round(francois2016_df.loc[i, ulX] - solar_logepsX_a2009, 2)
            
            ## XFecol and ulXFecol
            if elem != 'Fe':
                if pd.isna(row['l_'+XFe]):
                    francois2016_df.loc[i, XFe] = normal_round(francois2016_df.loc[i, XH] - francois2016_df.loc[i, '[Fe/H]'], 2)
                    francois2016_df.loc[i, ulXFe] = np.nan
                else:
                    francois2016_df.loc[i, XFe] = np.nan
                    francois2016_df.loc[i, ulXFe] = normal_round(francois2016_df.loc[i, ulXH] - francois2016_df.loc[i, '[Fe/H]'], 2)

    return francois2016_df

def load_frebel2010a(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        frebel2010a_df.loc[i,'Name'] = name
        frebel2010a_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        frebel2010a_df.loc[i,'Reference'] = 'Frebel+2010a'
        frebel2010a_df.loc[i,'Ref'] = 'FRE10a'
        frebel2010a_df.loc[i,'I/O'] = 1
        frebel2010a_df.loc[i,'Loc'] = 'UF'
        frebel2010a_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]     
        frebel2010a_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        frebel2010a_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(frebel2010a_df.loc[i,'RA_hms'], precision=6)
        frebel2010a_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        frebel2010a_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(frebel2010a_df.loc[i,'DEC_dms'], precision=2)
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

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

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
                    frebel2010a_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    frebel2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2010a_df.loc[i, col] = np.nan
                    frebel2010a_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    frebel2010a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    frebel2010a_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    frebel2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2010a_df.loc[i, col] = np.nan
                    frebel2010a_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
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

    ## Drop the Fe/Fe columns
    frebel2010a_df.drop(columns=[col for col in frebel2010a_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return frebel2010a_df

def load_frebel2013c(io=None):
    """
    Load the Frebel et al. 2016 data for the Bootes I Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations (& Stellar Parameters, manually added from the text)
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/frebel2013c/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/frebel2013c/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/frebel2013c/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    frebel2013c_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        frebel2013c_df.loc[i,'Name'] = name
        frebel2013c_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        frebel2013c_df.loc[i,'Reference'] = 'Frebel+2013c'
        frebel2013c_df.loc[i,'Ref'] = 'FRE13c'
        frebel2013c_df.loc[i,'I/O'] = 1
        frebel2013c_df.loc[i,'Loc'] = 'UF'
        frebel2013c_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        frebel2013c_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        frebel2013c_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(frebel2013c_df.loc[i,'RA_hms'], precision=6)
        frebel2013c_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        frebel2013c_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(frebel2013c_df.loc[i,'DEC_dms'], precision=2)
        frebel2013c_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        frebel2013c_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        frebel2013c_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        frebel2013c_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                frebel2013c_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                frebel2013c_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    frebel2013c_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    frebel2013c_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2013c_df.loc[i, col] = np.nan
                    frebel2013c_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    frebel2013c_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    frebel2013c_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    frebel2013c_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2013c_df.loc[i, col] = np.nan
                    frebel2013c_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    frebel2013c_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    frebel2013c_df.loc[i, col] = e_logepsX
                else:
                    frebel2013c_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    frebel2013c_df.drop(columns=[col for col in frebel2013c_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return frebel2013c_df

def load_frebel2014(io=None):
    """
    Load the Frebel et al. 2014 data for the Segue 1 Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table

    J100714+160154 is an s-process star.
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        frebel2014_df.loc[i,'Name'] = name
        frebel2014_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        frebel2014_df.loc[i,'Reference'] = 'Frebel+2014'
        frebel2014_df.loc[i,'Ref'] = 'FRE14'
        frebel2014_df.loc[i,'I/O'] = 1
        frebel2014_df.loc[i,'Loc'] = 'UF'
        frebel2014_df.loc[i,'System'] = 'Segue 1'
        frebel2014_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        frebel2014_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(frebel2014_df.loc[i,'RA_hms'], precision=6)
        frebel2014_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        frebel2014_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(frebel2014_df.loc[i,'DEC_dms'], precision=2)
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

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

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
                    frebel2014_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    frebel2014_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2014_df.loc[i, col] = np.nan
                    frebel2014_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    frebel2014_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    frebel2014_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    frebel2014_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2014_df.loc[i, col] = np.nan
                    frebel2014_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    frebel2014_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    frebel2014_df.loc[i, col] = e_logepsX
                else:
                    frebel2014_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    frebel2014_df.drop(columns=[col for col in frebel2014_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return frebel2014_df

def load_frebel2016(io=None):
    """
    Load the Frebel et al. 2016 data for the Bootes I Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations (& Stellar Parameters, manually added from the text)
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/frebel2016/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/frebel2016/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    frebel2016_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        frebel2016_df.loc[i,'Name'] = name
        frebel2016_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        frebel2016_df.loc[i,'Reference'] = 'Frebel+2016'
        frebel2016_df.loc[i,'Ref'] = 'FRE16'
        frebel2016_df.loc[i,'I/O'] = 1
        frebel2016_df.loc[i,'Loc'] = 'UF'
        frebel2016_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        frebel2016_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        frebel2016_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(frebel2016_df.loc[i,'RA_hms'], precision=6)
        frebel2016_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        frebel2016_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(frebel2016_df.loc[i,'DEC_dms'], precision=2)
        frebel2016_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        frebel2016_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        frebel2016_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        frebel2016_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                frebel2016_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                frebel2016_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    frebel2016_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    frebel2016_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2016_df.loc[i, col] = np.nan
                    frebel2016_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    frebel2016_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    frebel2016_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    frebel2016_df.loc[i, 'ul'+col] = np.nan
                else:
                    frebel2016_df.loc[i, col] = np.nan
                    frebel2016_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    frebel2016_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    frebel2016_df.loc[i, col] = e_logepsX
                else:
                    frebel2016_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    frebel2016_df.drop(columns=[col for col in frebel2016_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return frebel2016_df

def load_gilmore2013(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        gilmore2013_df.loc[i,'Name'] = name
        gilmore2013_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        gilmore2013_df.loc[i,'Reference'] = 'Gilmore+2013'
        gilmore2013_df.loc[i,'Ref'] = 'GIL13'
        gilmore2013_df.loc[i,'I/O'] = 1
        gilmore2013_df.loc[i,'Loc'] = 'UF'
        gilmore2013_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        gilmore2013_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        gilmore2013_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(gilmore2013_df.loc[i,'RA_hms'], precision=6)
        gilmore2013_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        gilmore2013_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(gilmore2013_df.loc[i,'DEC_dms'], precision=2)
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

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX_GM'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                gilmore2013_df.loc[i, col] = row['logepsX_GM'] if pd.isna(row['l_logepsX_GM']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                gilmore2013_df.loc[i, col] = row['logepsX_GM'] if pd.notna(row['l_logepsX_GM']) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row['l_logepsX_GM']):
                    gilmore2013_df.loc[i, col] = normal_round(row['logepsX_GM'] - logepsX_sun_a09, 2)
                    gilmore2013_df.loc[i, 'ul'+col] = np.nan
                else:
                    gilmore2013_df.loc[i, col] = np.nan
                    gilmore2013_df.loc[i, 'ul'+col] = normal_round(row['logepsX_GM'] - logepsX_sun_a09, 2)
                if 'e_[X/H]_GM' in row.index:
                    gilmore2013_df.loc[i, 'e_'+col] = row['e_[X/H]_GM']

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_logepsX_GM']):
                    gilmore2013_df.loc[i, col] = normal_round((row['logepsX_GM'] - logepsX_sun_a09) - feh_a09, 2)
                    gilmore2013_df.loc[i, 'ul'+col] = np.nan
                else:
                    gilmore2013_df.loc[i, col] = np.nan
                    gilmore2013_df.loc[i, 'ul'+col] = normal_round((row['logepsX_GM'] - logepsX_sun_a09) - feh_a09, 2)
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

    ## Get the Carbon data from Norris+2010c & Lai+2011b
    norris2010c_df = load_norris2010c(load_gilmore2013=True)
    norris2010c_df = norris2010c_df[norris2010c_df['Reference'] == 'Gilmore+2013']
    for name in gilmore2013_df['Name'].unique():
        if name in norris2010c_df['Name'].values:
            idx = gilmore2013_df[gilmore2013_df['Name'] == name].index[0]
            gilmore2013_df.loc[idx, '[C/Fe]'] = norris2010c_df[norris2010c_df['Name'] == name]['[C/Fe]'].values[0]
            gilmore2013_df.loc[idx, 'epsc'] = norris2010c_df[norris2010c_df['Name'] == name]['epsc'].values[0]
            gilmore2013_df.loc[idx, '[C/H]'] = norris2010c_df[norris2010c_df['Name'] == name]['[C/H]'].values[0]
        elif name == 'BooI-119':
            ## Lai et al. 2011b. 'BooI-119' was named 'Boo21' ([C/Fe] = 2.2)
            ### I manually calculated the epsc value, and converted back to [C/Fe] and [C/H] using Asplund+2009 solar abundances.
            idx = gilmore2013_df[gilmore2013_df['Name'] == name].index[0]
            gilmore2013_df.loc[idx, 'epsc'] = 6.8 # = (cfe + feh) + epsc_sun = ((2.2) + (-3.79)) + (8.39) w/ epsc_sun = 8.39 in Asplund+2005
            gilmore2013_df.loc[idx, '[C/H]'] = gilmore2013_df.loc[idx, 'epsc'] - get_solar('C')[0] # epsc_sun = 8.43 in Asplund+2009
            gilmore2013_df.loc[idx, '[C/Fe]'] = gilmore2013_df.loc[idx, '[C/H]'] - gilmore2013_df.loc[idx, '[Fe/H]'] 

    ## Drop the Fe/Fe columns
    gilmore2013_df.drop(columns=[col for col in gilmore2013_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return gilmore2013_df

def load_roederer2016b(io=None):
    """
    Load the Roederer et al. 2016b data for stars in Reticulum II.

    Table 1 - Observations
    Table 4 - Stellar Parameters
    Table 6,7 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/roederer2016b/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/roederer2016b/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/roederer2016b/table67.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    roederer2016b_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','M/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        roederer2016b_df.loc[i,'Name'] = name
        roederer2016b_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        roederer2016b_df.loc[i,'Reference'] = 'Roederer+2016b'
        roederer2016b_df.loc[i,'Ref'] = 'ROE16b'
        roederer2016b_df.loc[i,'I/O'] = 1
        roederer2016b_df.loc[i,'Loc'] = 'UF'
        roederer2016b_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        roederer2016b_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        roederer2016b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(roederer2016b_df.loc[i,'RA_hms'], precision=6)
        roederer2016b_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        roederer2016b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(roederer2016b_df.loc[i,'DEC_dms'], precision=2)
        roederer2016b_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        roederer2016b_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        roederer2016b_df.loc[i,'M/H'] = param_df.loc[param_df['Name'] == name, 'M/H'].values[0]
        roederer2016b_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                roederer2016b_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                roederer2016b_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    roederer2016b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    roederer2016b_df.loc[i, 'ul'+col] = np.nan
                else:
                    roederer2016b_df.loc[i, col] = np.nan
                    roederer2016b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    roederer2016b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    roederer2016b_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    roederer2016b_df.loc[i, 'ul'+col] = np.nan
                else:
                    roederer2016b_df.loc[i, col] = np.nan
                    roederer2016b_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    roederer2016b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("sigma_tot", np.nan)
                if pd.notna(e_logepsX):
                    roederer2016b_df.loc[i, col] = e_logepsX
                else:
                    roederer2016b_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    roederer2016b_df.drop(columns=[col for col in roederer2016b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return roederer2016b_df

def load_hansent2017(io=None):
    '''
    Load the Hansen T. et al. 2017 data for the Tucana III Ultra-Faint Dwarf Galaxy.
    '''

    ## Read in the data tables
    data_df = pd.read_csv(data_dir + 'abundance_tables/hansent2017/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

    elements = [col.replace('[','').replace('/H]', '') for col in data_df.columns if ((col.startswith('[')) & (col.endswith('/H]')))]
    elements += [col.replace('ul[','').replace('/H]', '') for col in data_df.columns if ((col.startswith('ul[')) & (col.endswith('/H]')))]
    elements += [col.replace('[','').replace('/Fe]', '') for col in data_df.columns if ((col.startswith('[')) & (col.endswith('/Fe]')))]
    elements += [col.replace('ul[','').replace('/Fe]', '') for col in data_df.columns if ((col.startswith('ul[')) & (col.endswith('/Fe]')))]
    elements = list(set(elements))  # Remove duplicates
    
    epscols = ['eps'+elem.lower() for elem in elements]
    ulcols = ['ul'+elem.lower() for elem in elements]
    XHcols = [f'[{elem}/H]' for elem in elements]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [f'[{elem}/Fe]' for elem in elements]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = ['e_'+col for col in XFecols]

    ## New dataframe with proper columns
    hansent2017_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    hansent2017_df['Name'] = data_df['Name'].astype(str)
    hansent2017_df['Simbad_Identifier'] = data_df['Simbad_Identifier'].astype(str)      
    hansent2017_df['Reference'] = 'Hansen_T+2017'
    hansent2017_df['Ref'] = 'HANt17'
    hansent2017_df['I/O'] = 1
    hansent2017_df['Loc'] = 'UF'
    hansent2017_df['System'] = 'Tucana III'
    hansent2017_df['RA_hms'] = data_df['RA_hms'].astype(str)   
    hansent2017_df['RA_deg'] = scoord.ra_hms_to_deg(hansent2017_df['RA_hms'], precision=6)
    hansent2017_df['DEC_dms'] = data_df['DEC_dms'].astype(str)
    hansent2017_df['DEC_deg'] = scoord.dec_dms_to_deg(hansent2017_df['DEC_dms'], precision=2)
    hansent2017_df['Teff'] = data_df['Teff'].astype(float)
    hansent2017_df['logg'] = data_df['logg'].astype(float)
    hansent2017_df['Fe/H'] = data_df['Fe/H'].astype(float)
    hansent2017_df['Vmic'] = data_df['Vmic'].astype(float)

    for elem in elements:
        epsX = 'eps' + elem.lower()
        ulX = 'ul' + elem.lower()
        XH = f'[{elem}/H]'
        ulXH = f'ul[{elem}/H]'
        XFe = f'[{elem}/Fe]'
        ulXFe = f'ul[{elem}/Fe]'
        errX = f'e_[{elem}/Fe]'

        # [X/Fe]
        hansent2017_df[XFe] = data_df[XFe].astype(float) if XFe in data_df.columns else np.nan

        # ul[X/Fe]
        hansent2017_df[ulXFe] = data_df[ulXFe] if ulXFe in data_df.columns else np.nan

        # [X/H]
        if XH in data_df.columns:
            hansent2017_df[XH] = data_df[XH].astype(float)
        elif XFe in hansent2017_df and '[Fe/H]' in data_df.columns:
            hansent2017_df[XH] = hansent2017_df[XFe] + data_df['[Fe/H]']
        else:
            hansent2017_df[XH] = np.nan

        # ul[X/H]
        if ulXH in data_df.columns:
            hansent2017_df[ulXH] = data_df[ulXH]
        elif ulXFe in hansent2017_df and '[Fe/H]' in data_df.columns:
            hansent2017_df[ulXH] = hansent2017_df[ulXFe] + data_df['[Fe/H]']
        else:
            hansent2017_df[ulXH] = np.nan

        # epsX
        if epsX in data_df.columns:
            hansent2017_df[epsX] = data_df[epsX].astype(float)
        else:
            hansent2017_df[epsX] = hansent2017_df[XH] + get_solar(elem, version='asplund2009')[0]

        # ulX
        if ulX in data_df.columns:
            hansent2017_df[ulX] = data_df[ulX]
        else:
            hansent2017_df[ulX] = hansent2017_df[ulXH] + get_solar(elem, version='asplund2009')[0]

        # e_[X/Fe]
        if errX in data_df.columns:
            hansent2017_df[errX] = data_df[errX].astype(float)
        else:
            hansent2017_df[errX] = np.nan

    ## Drop the Fe/Fe columns
    hansent2017_df.drop(columns=[col for col in hansent2017_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return hansent2017_df

def load_hansent2020a(io=None):
    """
    Load the Hansen T. et al. 2020a data for the Grus II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 5 - Abundance Table
    """

    obs_df = pd.read_csv(data_dir + 'abundance_tables/hansent2020a/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/hansent2020a/table2.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/hansent2020a/table5.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

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
    hansent2020a_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        hansent2020a_df.loc[i,'Name'] = name
        hansent2020a_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        hansent2020a_df.loc[i,'Reference'] = 'Hansen_T+2020a'
        hansent2020a_df.loc[i,'Ref'] = 'HANt20a'
        hansent2020a_df.loc[i,'I/O'] = 1
        hansent2020a_df.loc[i,'Loc'] = 'UF'
        hansent2020a_df.loc[i,'System'] = 'Grus II'
        hansent2020a_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        hansent2020a_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(hansent2020a_df.loc[i,'RA_hms'], precision=6)
        hansent2020a_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        hansent2020a_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(hansent2020a_df.loc[i,'DEC_dms'], precision=2)
        hansent2020a_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        hansent2020a_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        hansent2020a_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        hansent2020a_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                hansent2020a_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                hansent2020a_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    hansent2020a_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    hansent2020a_df.loc[i, 'ul'+col] = np.nan
                else:
                    hansent2020a_df.loc[i, col] = np.nan
                    hansent2020a_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    hansent2020a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    hansent2020a_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    hansent2020a_df.loc[i, 'ul'+col] = np.nan
                else:
                    hansent2020a_df.loc[i, col] = np.nan
                    hansent2020a_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    hansent2020a_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    hansent2020a_df.loc[i, col] = e_logepsX
                else:
                    hansent2020a_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    hansent2020a_df.drop(columns=[col for col in hansent2020a_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return hansent2020a_df

def load_hansent2024(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        hansent2024_df.loc[i,'Name'] = name
        hansent2024_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        hansent2024_df.loc[i,'Reference'] = 'Hansen_T+2024'
        hansent2024_df.loc[i,'Ref'] = 'HANt24'
        hansent2024_df.loc[i,'I/O'] = 1
        hansent2024_df.loc[i,'Loc'] = 'UF'
        hansent2024_df.loc[i,'System'] = 'Tucana V'
        hansent2024_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        hansent2024_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(hansent2024_df.loc[i,'RA_hms'], precision=6)
        hansent2024_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        hansent2024_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(hansent2024_df.loc[i,'DEC_dms'], precision=2)
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

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

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
                    hansent2024_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    hansent2024_df.loc[i, 'ul'+col] = np.nan
                else:
                    hansent2024_df.loc[i, col] = np.nan
                    hansent2024_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    hansent2024_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    hansent2024_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    hansent2024_df.loc[i, 'ul'+col] = np.nan
                else:
                    hansent2024_df.loc[i, col] = np.nan
                    hansent2024_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    hansent2024_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row["e_[X/H]"] #eqivalent to e_logepsX
                if pd.notna(e_logepsX):
                    hansent2024_df.loc[i, col] = e_logepsX
                else:
                    hansent2024_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    hansent2024_df.drop(columns=[col for col in hansent2024_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return hansent2024_df

def load_ishigaki2014(exclude_mw_halo_ref_stars=True, io=None):
    """
    Load the Ishigaki et al. 2014 data for the Bootes I Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 5 - Abundance Table
    """

    obs_df = pd.read_csv(data_dir + 'abundance_tables/ishigaki2014/table1.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    param_df = pd.read_csv(data_dir + 'abundance_tables/ishigaki2014/table3.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    abund_df = pd.read_csv(data_dir + 'abundance_tables/ishigaki2014/table5.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])
    
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
    ishigaki2014_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ishigaki2014_df.loc[i,'Name'] = name
        ishigaki2014_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ishigaki2014_df.loc[i,'Reference'] = 'Ishigaki+2014'
        ishigaki2014_df.loc[i,'Ref'] = 'ISH14'
        ishigaki2014_df.loc[i,'I/O'] = 1
        ishigaki2014_df.loc[i,'Loc'] = 'UF'
        ishigaki2014_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ishigaki2014_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ishigaki2014_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(ishigaki2014_df.loc[i,'RA_hms'], precision=6)
        ishigaki2014_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ishigaki2014_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(ishigaki2014_df.loc[i,'DEC_dms'], precision=2)
        ishigaki2014_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ishigaki2014_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        ishigaki2014_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        ishigaki2014_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                ishigaki2014_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ishigaki2014_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ishigaki2014_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    ishigaki2014_df.loc[i, 'ul'+col] = np.nan
                else:
                    ishigaki2014_df.loc[i, col] = np.nan
                    ishigaki2014_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    ishigaki2014_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ishigaki2014_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    ishigaki2014_df.loc[i, 'ul'+col] = np.nan
                else:
                    ishigaki2014_df.loc[i, col] = np.nan
                    ishigaki2014_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    ishigaki2014_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_tot", np.nan)
                if pd.notna(e_logepsX):
                    ishigaki2014_df.loc[i, col] = e_logepsX
                else:
                    ishigaki2014_df.loc[i, col] = np.nan

    ## Exclude the MW halo reference stars
    if exclude_mw_halo_ref_stars:
        ishigaki2014_df = ishigaki2014_df[~ishigaki2014_df['Name'].isin(['HD216143', 'HD85773'])]

    ## Drop the Fe/Fe columns
    ishigaki2014_df.drop(columns=[col for col in ishigaki2014_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return ishigaki2014_df

def load_ji2016a(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2016a_df.loc[i,'Name'] = name
        ji2016a_df.loc[i,'Simbad_Identifier'] = name
        ji2016a_df.loc[i,'Reference'] = 'Ji+2016a'
        ji2016a_df.loc[i,'Ref'] = 'JI16a'
        ji2016a_df.loc[i,'I/O'] = 1
        ji2016a_df.loc[i,'Loc'] = 'UF'
        ji2016a_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ji2016a_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ji2016a_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(ji2016a_df.loc[i,'RA_hms'], precision=6)
        ji2016a_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ji2016a_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(ji2016a_df.loc[i,'DEC_dms'], precision=2)
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

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

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
                    ji2016a_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    ji2016a_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2016a_df.loc[i, col] = np.nan
                    ji2016a_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    ji2016a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2016a_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    ji2016a_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2016a_df.loc[i, col] = np.nan
                    ji2016a_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
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

    ## Drop the Fe/Fe columns
    ji2016a_df.drop(columns=[col for col in ji2016a_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return ji2016a_df

def load_ji2016b(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2016b_df.loc[i,'Name'] = name
        ji2016b_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        ji2016b_df.loc[i,'Reference'] = 'Ji+2016b'
        ji2016b_df.loc[i,'Ref'] = 'JI16b'
        ji2016b_df.loc[i,'I/O'] = 1
        ji2016b_df.loc[i,'Loc'] = 'UF'
        ji2016b_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]     
        ji2016b_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        ji2016b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(ji2016b_df.loc[i,'RA_hms'], precision=6)
        ji2016b_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        ji2016b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(ji2016b_df.loc[i,'DEC_dms'], precision=2)
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

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

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
                    ji2016b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    ji2016b_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2016b_df.loc[i, col] = np.nan
                    ji2016b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    ji2016b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2016b_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    ji2016b_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2016b_df.loc[i, col] = np.nan
                    ji2016b_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
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

    ## Drop the Fe/Fe columns
    ji2016b_df.drop(columns=[col for col in ji2016b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return ji2016b_df

def load_ji2018(io=None):
    """
    Load the Ji et al. 2018 data for the brighted star in Reticulum II.

    Table 0 - Observations & Stellar Parameters
    Table 2 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/ji2018/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2018/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    ji2018_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2018_df.loc[i,'Name'] = name
        ji2018_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ji2018_df.loc[i,'Reference'] = 'Ji+2018'
        ji2018_df.loc[i,'Ref'] = 'JI18'
        ji2018_df.loc[i,'I/O'] = 1
        ji2018_df.loc[i,'Loc'] = 'UF'
        ji2018_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        ji2018_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        ji2018_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(ji2018_df.loc[i,'RA_hms'], precision=6)
        ji2018_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        ji2018_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(ji2018_df.loc[i,'DEC_dms'], precision=2)
        ji2018_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        ji2018_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        ji2018_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        ji2018_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in data
        star_df = abund_df[abund_df['Name'] == name]
        for j, row in star_df.iterrows():
            ion = row["Species"]
            species_i = ion_to_species(ion)
            elem_i = ion_to_element(ion)

            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX_w'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                ji2018_df.loc[i, col] = row["logepsX_w"] if pd.isna(row["l_logepsX_w"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2018_df.loc[i, col] = row["logepsX_w"] if pd.notna(row["l_logepsX_w"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX_w"]):
                    ji2018_df.loc[i, col] = normal_round(row["logepsX_w"] - logepsX_sun_a09, 2)
                    ji2018_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2018_df.loc[i, col] = np.nan
                    ji2018_df.loc[i, 'ul'+col] = normal_round(row["logepsX_w"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    ji2018_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX_w"]):
                    ji2018_df.loc[i, col] = normal_round((row["logepsX_w"] - logepsX_sun_a09) - feh_a09, 2)
                    ji2018_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2018_df.loc[i, col] = np.nan
                    ji2018_df.loc[i, 'ul'+col] = normal_round((row["logepsX_w"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    ji2018_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("sigma_adopted", np.nan)
                if pd.notna(e_logepsX):
                    ji2018_df.loc[i, col] = e_logepsX
                else:
                    ji2018_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    ji2018_df.drop(columns=[col for col in ji2018_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return ji2018_df

def load_ji2019a(io=None):
    """
    Load the Ji et al. 2019a data for the Grus I, Triangulum II Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/ji2019a/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/ji2019a/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2019a/table4.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    ji2019a_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2019a_df.loc[i,'Name'] = name
        ji2019a_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ji2019a_df.loc[i,'Reference'] = 'Ji+2019a'
        ji2019a_df.loc[i,'Ref'] = 'JI19a'
        ji2019a_df.loc[i,'I/O'] = 1
        ji2019a_df.loc[i,'Loc'] = 'UF'
        ji2019a_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ji2019a_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ji2019a_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(ji2019a_df.loc[i,'RA_hms'], precision=6)
        ji2019a_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ji2019a_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(ji2019a_df.loc[i,'DEC_dms'], precision=2)
        ji2019a_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ji2019a_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        ji2019a_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        ji2019a_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                ji2019a_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2019a_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ji2019a_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    ji2019a_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2019a_df.loc[i, col] = np.nan
                    ji2019a_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    ji2019a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2019a_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    ji2019a_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2019a_df.loc[i, col] = np.nan
                    ji2019a_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    ji2019a_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("stderr_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    ji2019a_df.loc[i, col] = e_logepsX
                else:
                    ji2019a_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    ji2019a_df.drop(columns=[col for col in ji2019a_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return ji2019a_df

def load_ji2020a(io=None):
    """
    Load the Ji et al. 2020 data for the Carina II and Carina III Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations
    Table 2 - Radial Velocities
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/ji2020a/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    rv_df = pd.read_csv(data_dir + "abundance_tables/ji2020a/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/ji2020a/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2020a/table6.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    ji2020a_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','M/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2020a_df.loc[i,'Name'] = name
        ji2020a_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ji2020a_df.loc[i,'Reference'] = 'Ji+2020a'
        ji2020a_df.loc[i,'Ref'] = 'JI20a'
        ji2020a_df.loc[i,'I/O'] = 1
        ji2020a_df.loc[i,'Loc'] = 'UF'
        ji2020a_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ji2020a_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ji2020a_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(ji2020a_df.loc[i,'RA_hms'], precision=6)
        ji2020a_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ji2020a_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(ji2020a_df.loc[i,'DEC_dms'], precision=2)
        ji2020a_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ji2020a_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        ji2020a_df.loc[i,'M/H'] = param_df.loc[param_df['Name'] == name, '[M/H]'].values[0]
        ji2020a_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                ji2020a_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2020a_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    ji2020a_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    ji2020a_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2020a_df.loc[i, col] = np.nan
                    ji2020a_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    ji2020a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    ji2020a_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    ji2020a_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2020a_df.loc[i, col] = np.nan
                    ji2020a_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    ji2020a_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_stat", np.nan)
                if pd.notna(e_logepsX):
                    ji2020a_df.loc[i, col] = e_logepsX
                else:
                    ji2020a_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    ji2020a_df.drop(columns=[col for col in ji2020a_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return ji2020a_df

def load_kirby2017b(io=None):
    """
    Load the Kirby et al. 2017b data for the Triangulum II Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters (custom made table from the text)
    Table 6 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/kirby2017b/table0.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/kirby2017b/table6.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    kirby2017b_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        kirby2017b_df.loc[i,'Name'] = name
        kirby2017b_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        kirby2017b_df.loc[i,'Reference'] = 'Kirby+2017b'
        kirby2017b_df.loc[i,'Ref'] = 'KIR17b'
        kirby2017b_df.loc[i,'I/O'] = 1
        kirby2017b_df.loc[i,'Loc'] = 'UF'
        kirby2017b_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        kirby2017b_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        kirby2017b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(kirby2017b_df.loc[i,'RA_hms'], precision=6)
        kirby2017b_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        kirby2017b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(kirby2017b_df.loc[i,'DEC_dms'], precision=2)
        kirby2017b_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        kirby2017b_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        kirby2017b_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        kirby2017b_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                kirby2017b_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                kirby2017b_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_[X/H]"]):
                    kirby2017b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    kirby2017b_df.loc[i, 'ul'+col] = np.nan
                else:
                    kirby2017b_df.loc[i, col] = np.nan
                    kirby2017b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    kirby2017b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    kirby2017b_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    kirby2017b_df.loc[i, 'ul'+col] = np.nan
                else:
                    kirby2017b_df.loc[i, col] = np.nan
                    kirby2017b_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    kirby2017b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    kirby2017b_df.loc[i, col] = e_logepsX
                else:
                    kirby2017b_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    kirby2017b_df.drop(columns=[col for col in kirby2017b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return kirby2017b_df

def load_koch2008c(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        koch2008c_df.loc[i,'Name'] = name
        koch2008c_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        koch2008c_df.loc[i,'Reference'] = 'Koch+2008c'
        koch2008c_df.loc[i,'Ref'] = 'KOC08c'
        koch2008c_df.loc[i,'I/O'] = 1
        koch2008c_df.loc[i,'Loc'] = 'UF'
        koch2008c_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        koch2008c_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        koch2008c_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(koch2008c_df.loc[i,'RA_hms'], precision=6)
        koch2008c_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        koch2008c_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(koch2008c_df.loc[i,'DEC_dms'], precision=2)
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
                else:
                    koch2008c_df.loc[i, col] = np.nan
                    koch2008c_df.loc[i, 'ul'+col] = normal_round(logepsX - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    koch2008c_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            feh_a09 = koch2008c_df.loc[koch2008c_df['Name'] == name, '[Fe/H]'].values[0]
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    koch2008c_df.loc[i, col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    koch2008c_df.loc[i, 'ul'+col] = np.nan
                else:
                    koch2008c_df.loc[i, col] = np.nan
                    koch2008c_df.loc[i, 'ul'+col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    koch2008c_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_XFe = row.get("e_[X/Fe]", np.nan)
                if pd.notna(e_XFe):
                    koch2008c_df.loc[i, col] = e_XFe
                else:
                    koch2008c_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    koch2008c_df.drop(columns=[col for col in koch2008c_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return koch2008c_df

def load_koch2013b(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols)
    for i, name in enumerate(abund_df['Name'].unique()):
        koch2013b_df.loc[i,'Name'] = str(name)
        koch2013b_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        koch2013b_df.loc[i,'Reference'] = 'Koch+2013b'
        koch2013b_df.loc[i,'Ref'] = 'KOC13b'
        koch2013b_df.loc[i,'I/O'] = 1
        koch2013b_df.loc[i,'Loc'] = 'UF'
        koch2013b_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        koch2013b_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        koch2013b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(koch2013b_df.loc[i,'RA_hms'], precision=6)
        koch2013b_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        koch2013b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(koch2013b_df.loc[i,'DEC_dms'], precision=2)
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

def load_lai2011b(io=None):
    """
    Load the Lai et al. 2011 data for the Bootes I Ultra-Faint Dwarf Galaxy.

    Table 1 - All Data
    """

    ## Read in the data tables
    data_df = pd.read_csv(data_dir + "abundance_tables/lai2011b/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

    ## New dataframe with proper columns
    lai2011b_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + ['epsfe', '[Fe/H]','ulfe', 'ul[Fe/H]', 'epsc', '[C/H]', '[C/Fe]', 'ulc', 'ul[C/H]', 'ul[C/Fe]'])
    for i, name in enumerate(data_df['Name'].unique()):
        lai2011b_df.loc[i,'Name'] = name
        lai2011b_df.loc[i,'Simbad_Identifier'] = data_df.loc[data_df['Name'] == name, 'Simbad_Identifier'].values[0]
        lai2011b_df.loc[i,'Reference'] = 'Lai+2011b'
        lai2011b_df.loc[i,'Ref'] = 'LAI11b'
        lai2011b_df.loc[i,'I/O'] = 1
        lai2011b_df.loc[i,'Loc'] = 'UF'
        lai2011b_df.loc[i,'System'] = data_df.loc[data_df['Name'] == name, 'System'].values[0]
        lai2011b_df.loc[i,'RA_hms'] = data_df.loc[data_df['Name'] == name, 'RA_hms'].values[0]
        lai2011b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(lai2011b_df.loc[i,'RA_hms'], precision=6)
        lai2011b_df.loc[i,'DEC_dms'] = data_df.loc[data_df['Name'] == name, 'DEC_dms'].values[0]
        lai2011b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(lai2011b_df.loc[i,'DEC_dms'], precision=2)
        lai2011b_df.loc[i,'Teff'] = data_df.loc[data_df['Name'] == name, 'Teff'].values[0]
        lai2011b_df.loc[i,'logg'] = data_df.loc[data_df['Name'] == name, 'logg'].values[0]
        lai2011b_df.loc[i,'Vmic'] = data_df.loc[data_df['Name'] == name, 'Vmic'].values[0]

        ## Fill in the Iron and Carbon Data
        logepsFe_sun_a05 = get_solar('Fe', version='asplund2005')[0]
        logepsFe_sun_a09 = get_solar('Fe', version='asplund2009')[0]
        logepsC_sun_a05 = get_solar('C', version='asplund2005')[0]
        logepsC_sun_a09 = get_solar('C', version='asplund2009')[0]

        feh_a05 = data_df.loc[data_df['Name'] == name, '[Fe/H]'].values[0]
        cfe_a05 = data_df.loc[data_df['Name'] == name, '[C/Fe]'].values[0]

        lai2011b_df.loc[i,'epsfe'] = normal_round(feh_a05 + logepsFe_sun_a05, 2)
        lai2011b_df.loc[i,'Fe/H'] = normal_round(lai2011b_df.loc[i,'epsfe'] - logepsFe_sun_a09, 2)
        lai2011b_df.loc[i,'[Fe/H]'] = normal_round(lai2011b_df.loc[i,'epsfe'] - logepsFe_sun_a09, 2)
        lai2011b_df.loc[i,'ulfe'] = np.nan
        lai2011b_df.loc[i,'ul[Fe/H]'] = np.nan
        
        if pd.isna(data_df.loc[i, 'l_[C/Fe]']):
            lai2011b_df.loc[i,'epsc'] = normal_round(cfe_a05 + feh_a05 + logepsC_sun_a05, 2)
            lai2011b_df.loc[i,'[C/H]'] = normal_round(lai2011b_df.loc[i,'epsc'] - logepsC_sun_a09, 2)
            lai2011b_df.loc[i,'[C/Fe]'] = normal_round(lai2011b_df.loc[i,'[C/H]'] - lai2011b_df.loc[i,'[Fe/H]'], 2)
            lai2011b_df.loc[i,'ulc'] = np.nan
            lai2011b_df.loc[i,'ul[C/H]'] = np.nan
            lai2011b_df.loc[i,'ul[C/Fe]'] = np.nan
        else:
            lai2011b_df.loc[i,'epsc'] = np.nan
            lai2011b_df.loc[i,'[C/H]'] = np.nan
            lai2011b_df.loc[i,'[C/Fe]'] = np.nan
            lai2011b_df.loc[i,'ulc'] = normal_round(cfe_a05 + feh_a05 + logepsC_sun_a05, 2)
            lai2011b_df.loc[i,'ul[C/H]'] = normal_round(lai2011b_df.loc[i,'ulc'] - logepsC_sun_a09, 2)
            lai2011b_df.loc[i,'ul[C/Fe]'] = normal_round(lai2011b_df.loc[i,'ul[C/H]'] - lai2011b_df.loc[i,'[Fe/H]'], 2)

    return lai2011b_df

def load_marshall2019(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        marshall2019_df.loc[i,'Name'] = name
        marshall2019_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        marshall2019_df.loc[i,'Reference'] = 'Marshall+2019'
        marshall2019_df.loc[i,'Ref'] = 'MARj19'
        marshall2019_df.loc[i,'I/O'] = 1
        marshall2019_df.loc[i,'Loc'] = 'UF'
        marshall2019_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        marshall2019_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        marshall2019_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(marshall2019_df.loc[i,'RA_hms'], precision=6)
        marshall2019_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        marshall2019_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(marshall2019_df.loc[i,'DEC_dms'], precision=2)
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
                    marshall2019_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    marshall2019_df.loc[i, 'ul'+col] = np.nan
                else:
                    marshall2019_df.loc[i, col] = np.nan
                    marshall2019_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    marshall2019_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    marshall2019_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    marshall2019_df.loc[i, 'ul'+col] = np.nan
                else:
                    marshall2019_df.loc[i, col] = np.nan
                    marshall2019_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    marshall2019_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    marshall2019_df.loc[i, col] = e_logepsX
                else:
                    marshall2019_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    marshall2019_df.drop(columns=[col for col in marshall2019_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return marshall2019_df

def load_nagasawa2018(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        nagasawa2018_df.loc[i,'Name'] = name
        nagasawa2018_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        nagasawa2018_df.loc[i,'Reference'] = 'Nagasawa+2018'
        nagasawa2018_df.loc[i,'Ref'] = 'NAG18'
        nagasawa2018_df.loc[i,'I/O'] = 1
        nagasawa2018_df.loc[i,'Loc'] = 'UF'
        nagasawa2018_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        nagasawa2018_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        nagasawa2018_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(nagasawa2018_df.loc[i,'RA_hms'], precision=6)
        nagasawa2018_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        nagasawa2018_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(nagasawa2018_df.loc[i,'DEC_dms'], precision=2)
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
                    nagasawa2018_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    nagasawa2018_df.loc[i, 'ul'+col] = np.nan
                else:
                    nagasawa2018_df.loc[i, col] = np.nan
                    nagasawa2018_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    nagasawa2018_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    nagasawa2018_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    nagasawa2018_df.loc[i, 'ul'+col] = np.nan
                else:
                    nagasawa2018_df.loc[i, col] = np.nan
                    nagasawa2018_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    nagasawa2018_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    nagasawa2018_df.loc[i, col] = e_logepsX
                else:
                    nagasawa2018_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    nagasawa2018_df.drop(columns=[col for col in nagasawa2018_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return nagasawa2018_df

def load_norris2010a(io=None):
    """
    Load the Norris et al. 2010a data for the Bootes I (Boo-1137) Ultra-Faint Dwarf Galaxy.

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
    XHcols = [make_XHcol(s).replace(' ', '') for s in species]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(' ', '') for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    norris2010a_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XFecols + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        norris2010a_df.loc[i,'Name'] = name
        norris2010a_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        norris2010a_df.loc[i,'Reference'] = 'Norris+2010a'
        norris2010a_df.loc[i,'Ref'] = 'NOR10a'
        norris2010a_df.loc[i,'I/O'] = 1
        norris2010a_df.loc[i,'Loc'] = 'UF'
        norris2010a_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        norris2010a_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        norris2010a_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(norris2010a_df.loc[i,'RA_hms'], precision=6)
        norris2010a_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        norris2010a_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(norris2010a_df.loc[i,'DEC_dms'], precision=2)
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
            
            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                norris2010a_df.loc[i, col] = row['logepsX'] if pd.isna(row['l_logepsX']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                norris2010a_df.loc[i, col] = row['logepsX'] if pd.notna(row['l_logepsX']) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    norris2010a_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    norris2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    norris2010a_df.loc[i, col] = np.nan
                    norris2010a_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    norris2010a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_[X/Fe]']):
                    norris2010a_df.loc[i, col] = normal_round((row['logepsX'] - logepsX_sun_a09) - feh_a09, 2)
                    norris2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    norris2010a_df.loc[i, col] = np.nan
                    norris2010a_df.loc[i, 'ul'+col] = normal_round((row['logepsX'] - logepsX_sun_a09) - feh_a09, 2)
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

    ## Drop the Fe/Fe columns
    norris2010a_df.drop(columns=[col for col in norris2010a_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return norris2010a_df

def load_norris2010b(io=None):
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
    XHcols = [make_XHcol(s).replace(' ', '') for s in species]
    ulXHcols = ['ul' + col for col in XHcols]
    XFecols = [make_XFecol(s).replace(' ', '') for s in species]
    ulXFecols = ['ul' + col for col in XFecols]
    errcols = [make_errcol(s) for s in species]

    ## New dataframe with proper columns
    norris2010b_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XFecols + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        norris2010b_df.loc[i,'Name'] = name
        norris2010b_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        norris2010b_df.loc[i,'Reference'] = 'Norris+2010b'
        norris2010b_df.loc[i,'Ref'] = 'NOR10b'
        norris2010b_df.loc[i,'I/O'] = 1
        norris2010b_df.loc[i,'Loc'] = 'UF'
        norris2010b_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]     
        norris2010b_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        norris2010b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(norris2010b_df.loc[i,'RA_hms'], precision=6)
        norris2010b_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        norris2010b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(norris2010b_df.loc[i,'DEC_dms'], precision=2)
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
            
            logepsX_sun_a09 = get_solar(elem_i, version='asplund2009')[0]
            logepsFe_a09 = star_df.loc[star_df['Species'] == 'Fe I', 'logepsX'].values[0]
            feh_a09 = logepsFe_a09 - get_solar('Fe', version='asplund2009')[0]

            ## Assign epsX values
            col = make_epscol(species_i)
            if col in epscols:
                norris2010b_df.loc[i, col] = row['logepsX'] if pd.isna(row['l_logepsX']) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                norris2010b_df.loc[i, col] = row['logepsX'] if pd.notna(row['l_logepsX']) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    norris2010b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    norris2010b_df.loc[i, 'ul'+col] = np.nan
                else:
                    norris2010b_df.loc[i, col] = np.nan
                    norris2010b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    norris2010b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row['l_[X/Fe]']):
                    norris2010b_df.loc[i, col] = normal_round((row['logepsX'] - logepsX_sun_a09) - feh_a09, 2)
                    norris2010b_df.loc[i, 'ul'+col] = np.nan
                else:
                    norris2010b_df.loc[i, col] = np.nan
                    norris2010b_df.loc[i, 'ul'+col] = normal_round((row['logepsX'] - logepsX_sun_a09) - feh_a09, 2)
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

    ## Drop the Fe/Fe columns
    norris2010b_df.drop(columns=[col for col in norris2010b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return norris2010b_df

def load_norris2010c(load_gilmore2013=False, io=None):
    """
    Load the Norris et al. 2010c data for the Bootes I (BooI) and Segue 1 (Seg1) Ultra-Faint Dwarf Galaxies.

    All relevant data was compiled together from the other tables and the text into `table0_combined.csv`.

    Note: Which solar abundances are used is not stated in the text, although I assume they are the Asplund+2005 solar 
          abundances. Not the Asplund+2009 solar abundances.
    """

    ## Read in the data tables
    csv_df = pd.read_csv(data_dir + 'abundance_tables/norris2010c/table0_combined.csv', comment='#', na_values=['', ' ', 'nan', 'NaN', 'N/A', 'n/a'])

    ## New dataframe with proper columns
    norris2010c_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + ['epsfe','epsc','[Fe/H]','[C/H]','[C/Fe]'])
    for i, name in enumerate(csv_df['Name'].unique()):
        norris2010c_df.loc[i,'Name'] = name
        norris2010c_df.loc[i,'Simbad_Identifier'] = csv_df.loc[csv_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        norris2010c_df.loc[i,'Reference'] = csv_df.loc[csv_df['Name'] == name, 'Reference'].values[0]
        norris2010c_df.loc[i,'Ref'] = 'NOR10c'
        norris2010c_df.loc[i,'I/O'] = 1
        norris2010c_df.loc[i,'Loc'] = 'UF'
        norris2010c_df.loc[i,'System'] = csv_df.loc[csv_df['Name'] == name, 'System'].values[0]     
        norris2010c_df.loc[i,'RA_hms'] = csv_df.loc[csv_df['Name'] == name, 'RA_hms'].values[0]
        norris2010c_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(norris2010c_df.loc[i,'RA_hms'], precision=6)
        norris2010c_df.loc[i,'DEC_dms'] = csv_df.loc[csv_df['Name'] == name, 'DEC_dms'].values[0]
        norris2010c_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(norris2010c_df.loc[i,'DEC_dms'], precision=2)
        norris2010c_df.loc[i,'Teff'] = csv_df.loc[csv_df['Name'] == name, 'Teff'].values[0]
        norris2010c_df.loc[i,'logg'] = csv_df.loc[csv_df['Name'] == name, 'logg'].values[0]
        norris2010c_df.loc[i,'Fe/H'] = csv_df.loc[csv_df['Name'] == name, '[Fe/H]'].values[0]
        norris2010c_df.loc[i,'Vmic'] = np.nan

        logepsFe_sun_a05 = get_solar('Fe', version='asplund2005')[0]
        logepsFe_sun_a09 = get_solar('Fe', version='asplund2009')[0]
        logepsC_sun_a05 = get_solar('C', version='asplund2005')[0]
        logepsC_sun_a09 = get_solar('C', version='asplund2009')[0]

        norris2010c_df.loc[i,'epsfe'] = normal_round(csv_df.loc[csv_df['Name'] == name,'[Fe/H]'].values[0] + logepsFe_sun_a05, 2)
        norris2010c_df.loc[i,'[Fe/H]'] = normal_round(norris2010c_df.loc[i,'epsfe'] - logepsFe_sun_a09, 2)
        norris2010c_df.loc[i,'epsc'] = normal_round(csv_df.loc[csv_df['Name'] == name,'[C/H]'].values[0] + logepsC_sun_a05, 2)
        norris2010c_df.loc[i,'[C/H]'] = normal_round(norris2010c_df.loc[i,'epsc'] - logepsC_sun_a09, 2)
        norris2010c_df.loc[i,'[C/Fe]'] = normal_round(norris2010c_df.loc[i,'[C/H]'] - norris2010c_df.loc[i,'[Fe/H]'], 2)

    if not load_gilmore2013:
        norris2010c_df = norris2010c_df[norris2010c_df['Reference'] == 'Norris+2010c']

    return norris2010c_df

def load_roederer2014b(io=None):
    """
    Load the Roederer et al. 2014 data for the Segue 2 Ultra-Faint Dwarf Galaxy.

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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        roederer2014b_df.loc[i,'Name'] = name
        roederer2014b_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        roederer2014b_df.loc[i,'Reference'] = 'Roederer+2014b'
        roederer2014b_df.loc[i,'Ref'] = 'ROE14b'
        roederer2014b_df.loc[i,'I/O'] = 1
        roederer2014b_df.loc[i,'Loc'] = 'UF'
        roederer2014b_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        roederer2014b_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        roederer2014b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(roederer2014b_df.loc[i,'RA_hms'], precision=6)
        roederer2014b_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        roederer2014b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(roederer2014b_df.loc[i,'DEC_dms'], precision=2)
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
                    roederer2014b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    roederer2014b_df.loc[i, 'ul'+col] = np.nan
                else:
                    roederer2014b_df.loc[i, col] = np.nan
                    roederer2014b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    roederer2014b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    roederer2014b_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    roederer2014b_df.loc[i, 'ul'+col] = np.nan
                else:
                    roederer2014b_df.loc[i, col] = np.nan
                    roederer2014b_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    roederer2014b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_tot", np.nan)
                if pd.notna(e_logepsX):
                    roederer2014b_df.loc[i, col] = e_logepsX
                else:
                    roederer2014b_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    roederer2014b_df.drop(columns=[col for col in roederer2014b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return roederer2014b_df

def load_simon2010(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        simon2010_df.loc[i,'Name'] = name
        simon2010_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        simon2010_df.loc[i,'Reference'] = 'Simon+2010'
        simon2010_df.loc[i,'Ref'] = 'SIM10'
        simon2010_df.loc[i,'I/O'] = 1
        simon2010_df.loc[i,'Loc'] = 'UF'
        simon2010_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        simon2010_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        simon2010_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(simon2010_df.loc[i,'RA_hms'], precision=6)
        simon2010_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        simon2010_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(simon2010_df.loc[i,'DEC_dms'], precision=2)
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
                    simon2010_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    simon2010_df.loc[i, 'ul'+col] = np.nan
                else:
                    simon2010_df.loc[i, col] = np.nan
                    simon2010_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    simon2010_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    simon2010_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    simon2010_df.loc[i, 'ul'+col] = np.nan
                else:
                    simon2010_df.loc[i, col] = np.nan
                    simon2010_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    simon2010_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = normal_round(np.sqrt(row.get("e_stat", 0.0)**2 + row.get("e_sys", 0.0)**2), 2)
                if pd.notna(e_logepsX):
                    simon2010_df.loc[i, col] = e_logepsX
                else:
                    simon2010_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    simon2010_df.drop(columns=[col for col in simon2010_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return simon2010_df

def load_sbordone2007(io=None):
    """
    Load the Sbordone et al. 2007 data for the Sagittarius dSph galaxy.

    Table 1 - Observation and Stellar Parameters
    Table 4,5,6 - Abundance Tables (merged into one table)
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/sbordone2007/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/sbordone2007/table456a_long.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    
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
    sbordone2007_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        sbordone2007_df.loc[i,'Name'] = name
        sbordone2007_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]        
        sbordone2007_df.loc[i,'Reference'] = 'Sbordone+2007'
        sbordone2007_df.loc[i,'Ref'] = 'SBO07'
        sbordone2007_df.loc[i,'I/O'] = 1
        sbordone2007_df.loc[i,'Loc'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Loc'].values[0]
        sbordone2007_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]     
        sbordone2007_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        sbordone2007_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(sbordone2007_df.loc[i,'RA_hms'], precision=6)
        sbordone2007_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        sbordone2007_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(sbordone2007_df.loc[i,'DEC_dms'], precision=2)
        sbordone2007_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        sbordone2007_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        sbordone2007_df.loc[i,'Fe/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Fe/H'].values[0]
        sbordone2007_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                sbordone2007_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                sbordone2007_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    sbordone2007_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    sbordone2007_df.loc[i, 'ul'+col] = np.nan
                else:
                    sbordone2007_df.loc[i, col] = np.nan
                    sbordone2007_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    sbordone2007_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    sbordone2007_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    sbordone2007_df.loc[i, 'ul'+col] = np.nan
                else:
                    sbordone2007_df.loc[i, col] = np.nan
                    sbordone2007_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    sbordone2007_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    sbordone2007_df.loc[i, col] = e_logepsX
                else:
                    sbordone2007_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    sbordone2007_df.drop(columns=[col for col in sbordone2007_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return sbordone2007_df

def load_spite2018(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        spite2018_df.loc[i,'Name'] = name
        spite2018_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        spite2018_df.loc[i,'Reference'] = 'Spite+2018'
        spite2018_df.loc[i,'Ref'] = 'SPI18'
        spite2018_df.loc[i,'I/O'] = 1
        spite2018_df.loc[i,'Loc'] = 'UF'
        spite2018_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        spite2018_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        spite2018_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(spite2018_df.loc[i,'RA_hms'], precision=6)
        spite2018_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        spite2018_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(spite2018_df.loc[i,'DEC_dms'], precision=2)
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
                    spite2018_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    spite2018_df.loc[i, 'ul'+col] = np.nan
                else:
                    spite2018_df.loc[i, col] = np.nan
                    spite2018_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    spite2018_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    spite2018_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    spite2018_df.loc[i, 'ul'+col] = np.nan
                else:
                    spite2018_df.loc[i, col] = np.nan
                    spite2018_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    spite2018_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    spite2018_df.loc[i, col] = e_logepsX
                else:
                    spite2018_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    spite2018_df.drop(columns=[col for col in spite2018_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return spite2018_df

def load_waller2023(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        waller2023_df.loc[i,'Name'] = name
        waller2023_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        waller2023_df.loc[i,'Reference'] = 'Waller+2023'
        waller2023_df.loc[i,'Ref'] = 'WAL23'
        waller2023_df.loc[i,'I/O'] = 1
        waller2023_df.loc[i,'Loc'] = 'UF'
        waller2023_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        waller2023_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        waller2023_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(waller2023_df.loc[i,'RA_hms'], precision=6)
        waller2023_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        waller2023_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(waller2023_df.loc[i,'DEC_dms'], precision=2)
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
                    waller2023_df.loc[i, col] = normal_round(logepsX - logepsX_sun_a09, 2)
                    waller2023_df.loc[i, 'ul'+col] = np.nan
                else:
                    waller2023_df.loc[i, col] = np.nan
                    waller2023_df.loc[i, 'ul'+col] = normal_round(logepsX - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    waller2023_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_[X/Fe]"]):
                    waller2023_df.loc[i, col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                    waller2023_df.loc[i, 'ul'+col] = np.nan
                else:
                    waller2023_df.loc[i, col] = np.nan
                    waller2023_df.loc[i, 'ul'+col] = normal_round((logepsX - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    waller2023_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

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
    waller2023_df.drop(columns=[col for col in waller2023_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return waller2023_df

def load_webber2023(io=None):
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
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        webber2023_df.loc[i,'Name'] = name
        webber2023_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        webber2023_df.loc[i,'Reference'] = 'Webber+2023'
        webber2023_df.loc[i,'Ref'] = 'WEB23'
        webber2023_df.loc[i,'I/O'] = 1
        webber2023_df.loc[i,'Loc'] = 'UF'
        webber2023_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        webber2023_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        webber2023_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(webber2023_df.loc[i,'RA_hms'], precision=6)
        webber2023_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        webber2023_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(webber2023_df.loc[i,'DEC_dms'], precision=2)
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
                    webber2023_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    webber2023_df.loc[i, 'ul'+col] = np.nan
                else:
                    webber2023_df.loc[i, col] = np.nan
                    webber2023_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    webber2023_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    webber2023_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    webber2023_df.loc[i, 'ul'+col] = np.nan
                else:
                    webber2023_df.loc[i, col] = np.nan
                    webber2023_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    webber2023_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    webber2023_df.loc[i, col] = e_logepsX
                else:
                    webber2023_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    webber2023_df.drop(columns=[col for col in webber2023_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    return webber2023_df

## stellar streams (SS)

def load_gull2021(io=None):
    """
    Load the Gull et al. 2021 data for the Helmi debris stream, Helmi trail stream, and omega Centauri stream.
    
    Helmi debris stream (Helmi et al. 1999)
        Helmi & White (1999) found 13 members of the now so-called debris stream.
        Roederer et al. (2010) performed a detailed abundance analysis of 12 of those 13 members.
        The Helmi debris stars manifest themselves in a well-defined stream, 
         with prominent negative vz motion (Myeong et al. 2019).
    
    Helmi trail stream (Helmi et al. 1999)
        Chiba & Beers (2000) 9 stars apart of a secondary stream associated with the Helmi debris stream trail stream.
        The Helmi trail stream distinguishes itself from the Helmi debris stream kinematically (Yuan et al. 2020). 
         by displaying a positive vz (vertical velocity) motions, slightly higher energy, larger radial motions, 
         and are more diffuse without clear features on kinematic diagrams

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 5 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/gull2021/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/gull2021/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/gull2021/table5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    gull2021_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        gull2021_df.loc[i,'Name'] = name
        gull2021_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        gull2021_df.loc[i,'Reference'] = 'Gull+2021'
        gull2021_df.loc[i,'Ref'] = 'GUL21'
        gull2021_df.loc[i,'I/O'] = 1
        gull2021_df.loc[i,'Loc'] = 'SS'
        gull2021_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        gull2021_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        gull2021_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(gull2021_df.loc[i,'RA_hms'], precision=6)
        gull2021_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        gull2021_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(gull2021_df.loc[i,'DEC_dms'], precision=2)
        gull2021_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        gull2021_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        gull2021_df.loc[i,'Fe/H'] = param_df.loc[param_df['Name'] == name, 'Fe/H'].values[0]
        gull2021_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                gull2021_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                gull2021_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    gull2021_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    gull2021_df.loc[i, 'ul'+col] = np.nan
                else:
                    gull2021_df.loc[i, col] = np.nan
                    gull2021_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    gull2021_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    gull2021_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    gull2021_df.loc[i, 'ul'+col] = np.nan
                else:
                    gull2021_df.loc[i, col] = np.nan
                    gull2021_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    gull2021_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    gull2021_df.loc[i, col] = e_logepsX
                else:
                    gull2021_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    gull2021_df.drop(columns=[col for col in gull2021_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        gull2021_df = gull2021_df[gull2021_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return gull2021_df

def load_ji2020b(io=None):
    """
    Load the Ji et al. 2020 data for the 7 stellar streams in the Milky Way.
    These streams include: ATLAS, Aliqa Uma, Chenab, Elqui, Indus, Jhelum, and Phoenix

    Table 1 - Observations
    Table 2 - Radial Velocities
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/ji2020b/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/ji2020b/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/ji2020b/table6.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    ji2020b_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','M/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        ji2020b_df.loc[i,'Name'] = name
        ji2020b_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        ji2020b_df.loc[i,'Reference'] = 'Ji+2020b'
        ji2020b_df.loc[i,'Ref'] = 'JI20b'
        ji2020b_df.loc[i,'I/O'] = 1
        ji2020b_df.loc[i,'Loc'] = 'SS'
        ji2020b_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        ji2020b_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        ji2020b_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(ji2020b_df.loc[i,'RA_hms'], precision=6)
        ji2020b_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        ji2020b_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(ji2020b_df.loc[i,'DEC_dms'], precision=2)
        ji2020b_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        ji2020b_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        ji2020b_df.loc[i,'M/H'] = param_df.loc[param_df['Name'] == name, 'M/H'].values[0]
        ji2020b_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                ji2020b_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                ji2020b_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    ji2020b_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    ji2020b_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2020b_df.loc[i, col] = np.nan
                    ji2020b_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    ji2020b_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    ji2020b_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    ji2020b_df.loc[i, 'ul'+col] = np.nan
                else:
                    ji2020b_df.loc[i, col] = np.nan
                    ji2020b_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    ji2020b_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    ji2020b_df.loc[i, col] = e_logepsX
                else:
                    ji2020b_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    ji2020b_df.drop(columns=[col for col in ji2020b_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        ji2020b_df = ji2020b_df[ji2020b_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return ji2020b_df

def load_martin2022a(io=None):
    """
    Load the Martin et al. 2022 data for the C-19 Stream.

    Table 2 - Observations Table
    Table 3 - Abundance Table 1, for Gemini/GRACES observations
    Table 5 - Abundance Table 2, for OSIRIS observations
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/martin2022a/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df1 = pd.read_csv(data_dir + "abundance_tables/martin2022a/table3_mod.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df2 = pd.read_csv(data_dir + "abundance_tables/martin2022a/table5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df2 = abund_df2[abund_df2['Code'] == 'W']

    ## New dataframe with proper columns
    martin2022a_df = pd.DataFrame(
                    columns=['Name','Simbad_Identifier','Pristine_Name','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','Fe/H','M/H','Vmic'] )#+ epscols + ulcols + XHcols + ulXHcols + XFecols + ulXFecols + errcols)
    for i, name in enumerate(obs_df['Name'].unique()):
        martin2022a_df.loc[i,'Name'] = name
        martin2022a_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        martin2022a_df.loc[i,'Pristine_Name'] = obs_df.loc[obs_df['Name'] == name, 'Pristine_Name'].values[0]
        martin2022a_df.loc[i,'Reference'] = 'Martin+2022'
        martin2022a_df.loc[i,'I/O'] = 1
        martin2022a_df.loc[i,'Ref'] = 'MARn22'
        martin2022a_df.loc[i,'Loc'] = 'SS'
        martin2022a_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        martin2022a_df.loc[i,'RA_deg'] = obs_df.loc[obs_df['Name'] == name, 'RA_deg'].values[0]
        martin2022a_df.loc[i,'RA_hms'] = scoord.ra_deg_to_hms(martin2022a_df.loc[i,'RA_deg'], precision=6)
        martin2022a_df.loc[i,'DEC_deg'] = obs_df.loc[obs_df['Name'] == name, 'DEC_deg'].values[0]
        martin2022a_df.loc[i,'DEC_dms'] = scoord.dec_deg_to_dms(martin2022a_df.loc[i,'DEC_deg'], precision=2)

        ## Abundance Table 1, for Gemini/GRACES observations
        if name in abund_df1['Name'].values:

            martin2022a_df.loc[i,'Teff'] = abund_df1.loc[abund_df1['Name'] == name, 'Teff'].values[0]
            martin2022a_df.loc[i,'logg'] = abund_df1.loc[abund_df1['Name'] == name, 'logg'].values[0]
            martin2022a_df.loc[i,'Fe/H'] = abund_df1.loc[abund_df1['Name'] == name, '[FeI/H]'].values[0]
            martin2022a_df.loc[i,'Vmic'] = abund_df1.loc[abund_df1['Name'] == name, 'Vmic'].values[0]

            martin2022a_df.loc[i, '[Fe/H]'] = abund_df1.loc[abund_df1['Name'] == name, '[FeI/H]'].values[0]
            martin2022a_df.loc[i, '[FeII/H]'] = abund_df1.loc[abund_df1['Name'] == name, '[FeII/H]'].values[0]
            martin2022a_df.loc[i, '[Na/H]'] = abund_df1.loc[abund_df1['Name'] == name, '[NaI/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']
            martin2022a_df.loc[i, '[Mg/H]'] = abund_df1.loc[abund_df1['Name'] == name, '[MgI/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']
            martin2022a_df.loc[i, '[Ca/H]'] = abund_df1.loc[abund_df1['Name'] == name, '[CaI/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']
            martin2022a_df.loc[i, '[Cr/H]'] = abund_df1.loc[abund_df1['Name'] == name, '[CrI/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']
            martin2022a_df.loc[i, '[Ba/H]'] = abund_df1.loc[abund_df1['Name'] == name, '[BaII/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']

            martin2022a_df.loc[i, 'ul[Fe/H]'] = abund_df1.loc[abund_df1['Name'] == name, 'ul[FeI/H]'].values[0]
            martin2022a_df.loc[i, 'ul[FeII/H]'] = abund_df1.loc[abund_df1['Name'] == name, 'ul[FeII/H]'].values[0]
            martin2022a_df.loc[i, 'ul[Na/H]'] = abund_df1.loc[abund_df1['Name'] == name, 'ul[NaI/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']
            martin2022a_df.loc[i, 'ul[Mg/H]'] = abund_df1.loc[abund_df1['Name'] == name, 'ul[MgI/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']
            martin2022a_df.loc[i, 'ul[Ca/H]'] = abund_df1.loc[abund_df1['Name'] == name, 'ul[CaI/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']
            martin2022a_df.loc[i, 'ul[Cr/H]'] = abund_df1.loc[abund_df1['Name'] == name, 'ul[CrI/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']
            martin2022a_df.loc[i, 'ul[Ba/H]'] = abund_df1.loc[abund_df1['Name'] == name, 'ul[BaII/FeII]'].values[0] + martin2022a_df.loc[i, '[FeII/H]']

            martin2022a_df.loc[i, '[Na/Fe]'] = martin2022a_df.loc[i, '[Na/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, '[Mg/Fe]'] = martin2022a_df.loc[i, '[Mg/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, '[Ca/Fe]'] = martin2022a_df.loc[i, '[Ca/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, '[Cr/Fe]'] = martin2022a_df.loc[i, '[Cr/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, '[Ba/Fe]'] = martin2022a_df.loc[i, '[Ba/H]'] - martin2022a_df.loc[i, '[Fe/H]']

            martin2022a_df.loc[i, 'ul[Na/Fe]'] = martin2022a_df.loc[i, 'ul[Na/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, 'ul[Mg/Fe]'] = martin2022a_df.loc[i, 'ul[Mg/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, 'ul[Ca/Fe]'] = martin2022a_df.loc[i, 'ul[Ca/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, 'ul[Cr/Fe]'] = martin2022a_df.loc[i, 'ul[Cr/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, 'ul[Ba/Fe]'] = martin2022a_df.loc[i, 'ul[Ba/H]'] - martin2022a_df.loc[i, '[Fe/H]']

            logepsFe_a09 = get_solar('Fe', version='asplund2009')[0]
            logepsNa_a09 = get_solar('Na', version='asplund2009')[0]
            logepsMg_a09 = get_solar('Mg', version='asplund2009')[0]
            logepsCa_a09 = get_solar('Ca', version='asplund2009')[0]
            logepsCr_a09 = get_solar('Cr', version='asplund2009')[0]
            logepsBa_a09 = get_solar('Ba', version='asplund2009')[0]

            martin2022a_df.loc[i, 'epsfe'] = abund_df1.loc[abund_df1['Name'] == name, '[FeI/H]'].values[0] + logepsFe_a09
            martin2022a_df.loc[i, 'epsfe2'] = abund_df1.loc[abund_df1['Name'] == name, '[FeII/H]'].values[0] + logepsFe_a09
            martin2022a_df.loc[i, 'epsna'] = martin2022a_df.loc[i, '[Na/H]'] + logepsNa_a09
            martin2022a_df.loc[i, 'epsmg'] = martin2022a_df.loc[i, '[Mg/H]'] + logepsMg_a09
            martin2022a_df.loc[i, 'epsca'] = martin2022a_df.loc[i, '[Ca/H]'] + logepsCa_a09
            martin2022a_df.loc[i, 'epscr'] = martin2022a_df.loc[i, '[Cr/H]'] + logepsCr_a09
            martin2022a_df.loc[i, 'epsba'] = martin2022a_df.loc[i, '[Ba/H]'] + logepsBa_a09

            martin2022a_df.loc[i, 'ulfe'] = martin2022a_df.loc[i, 'ul[Fe/H]'] + logepsFe_a09
            martin2022a_df.loc[i, 'ulfe2'] = martin2022a_df.loc[i, 'ul[FeII/H]'] + logepsFe_a09
            martin2022a_df.loc[i, 'ulna'] = martin2022a_df.loc[i, 'ul[Na/H]'] + logepsNa_a09
            martin2022a_df.loc[i, 'ulmg'] = martin2022a_df.loc[i, 'ul[Mg/H]'] + logepsMg_a09
            martin2022a_df.loc[i, 'ulca'] = martin2022a_df.loc[i, 'ul[Ca/H]'] + logepsCa_a09
            martin2022a_df.loc[i, 'ulcr'] = martin2022a_df.loc[i, 'ul[Cr/H]'] + logepsCr_a09
            martin2022a_df.loc[i, 'ulba'] = martin2022a_df.loc[i, 'ul[Ba/H]'] + logepsBa_a09

        ## Abundance Table 2, for OSIRIS observations
        if name in abund_df2['Name'].values:

            martin2022a_df.loc[i,'Teff'] = abund_df2.loc[abund_df2['Name'] == name, 'Teff'].values[0]
            martin2022a_df.loc[i,'logg'] = abund_df2.loc[abund_df2['Name'] == name, 'logg'].values[0]
            martin2022a_df.loc[i,'M/H'] = abund_df2.loc[abund_df2['Name'] == name, '[M/H]'].values[0]
            martin2022a_df.loc[i,'Vmic'] = np.nan

            logepsFe_a09 = get_solar('Fe', version='asplund2009')[0]
            logepsCa_a09 = get_solar('Ca', version='asplund2009')[0]
            logepsC_a09 = get_solar('C', version='asplund2009')[0]

            if (pd.isna(martin2022a_df.loc[i, '[Fe/H]']) and pd.isna(martin2022a_df.loc[i, 'ul[Fe/H]'])):
                martin2022a_df.loc[i, '[Fe/H]'] = abund_df2.loc[abund_df2['Name'] == name, '[Fe/H]'].values[0]
                martin2022a_df.loc[i, 'ul[Fe/H]'] = abund_df2.loc[abund_df2['Name'] == name, 'ul[Fe/H]'].values[0]
                martin2022a_df.loc[i, 'e_[Fe/H]'] = abund_df2.loc[abund_df2['Name'] == name, 'e_[Fe/H]'].values[0]

                martin2022a_df.loc[i, 'epsfe'] = martin2022a_df.loc[i, '[Fe/H]'] + logepsFe_a09
                martin2022a_df.loc[i, 'ulfe'] = martin2022a_df.loc[i, 'ul[Fe/H]'] + logepsFe_a09

            if (pd.isna(martin2022a_df.loc[i, '[Ca/H]']) and pd.isna(martin2022a_df.loc[i, 'ul[Ca/H]'])):
                martin2022a_df.loc[i, '[Ca/H]'] = abund_df2.loc[abund_df2['Name'] == name, '[Ca/H]'].values[0]
                martin2022a_df.loc[i, 'ul[Ca/H]'] = abund_df2.loc[abund_df2['Name'] == name, 'ul[Ca/H]'].values[0]
                martin2022a_df.loc[i, 'e_[Ca/H]'] = abund_df2.loc[abund_df2['Name'] == name, 'e_[Ca/H]'].values[0]
                
                martin2022a_df.loc[i, '[Ca/Fe]'] = martin2022a_df.loc[i, '[Ca/H]'] - martin2022a_df.loc[i, '[Fe/H]']
                martin2022a_df.loc[i, 'ul[Ca/Fe]'] = martin2022a_df.loc[i, 'ul[Ca/H]'] - martin2022a_df.loc[i, '[Fe/H]']
                martin2022a_df.loc[i, 'e_[Ca/Fe]'] = np.nan #(martin2022a_df.loc[i, 'e_[Fe/H]'])**2 + (martin2022a_df.loc[i, 'e_[Ca/H]'])**2

                martin2022a_df.loc[i, 'epsca'] = martin2022a_df.loc[i, '[Ca/H]'] + logepsCa_a09
                martin2022a_df.loc[i, 'ulca'] = martin2022a_df.loc[i, 'ul[Ca/H]'] + logepsCa_a09

            martin2022a_df.loc[i, '[C/H]'] = abund_df2.loc[abund_df2['Name'] == name, '[C/Fe]'].values[0] + martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, 'ul[C/H]'] = abund_df2.loc[abund_df2['Name'] == name, 'ul[C/Fe]'].values[0] + martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, 'e_[C/H]'] = np.nan #abund_df2.loc[abund_df2['Name'] == name, 'e_[C/H]'].values[0]

            martin2022a_df.loc[i, '[C/Fe]'] = martin2022a_df.loc[i, '[C/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, 'ul[C/Fe]'] = martin2022a_df.loc[i, 'ul[C/H]'] - martin2022a_df.loc[i, '[Fe/H]']
            martin2022a_df.loc[i, 'e_[C/Fe]'] = np.nan #(martin2022a_df.loc[i, 'e_[Fe/H]'])**2 + (martin2022a_df.loc[i, 'e_[C/H]'])**2

            martin2022a_df.loc[i, 'epsc'] = martin2022a_df.loc[i, '[C/H]'] + logepsC_a09
            martin2022a_df.loc[i, 'ulc'] = martin2022a_df.loc[i, 'ul[C/H]'] + logepsC_a09

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        martin2022a_df = martin2022a_df[martin2022a_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return martin2022a_df

def load_roederer2010a(io=None):
    """
    Load the Roederer et al. 2010a data for the Helmi stellar stream.

    Table 2 - Observations
    Table 5 - Stellar Parameters
    Table 7,8,9,10 - Abundance Table
    """

    ## Read in the data tables
    obs_df = pd.read_csv(data_dir + "abundance_tables/roederer2010a/table2.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    param_df = pd.read_csv(data_dir + "abundance_tables/roederer2010a/table5.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/roederer2010a/table7-8-9-10.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    roederer2010a_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','M/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        roederer2010a_df.loc[i,'Name'] = name
        roederer2010a_df.loc[i,'Simbad_Identifier'] = obs_df.loc[obs_df['Name'] == name, 'Simbad_Identifier'].values[0]
        roederer2010a_df.loc[i,'Reference'] = 'Roederer+2010a'
        roederer2010a_df.loc[i,'Ref'] = 'ROE10a'
        roederer2010a_df.loc[i,'I/O'] = 1
        roederer2010a_df.loc[i,'Loc'] = 'SS'
        roederer2010a_df.loc[i,'System'] = obs_df.loc[obs_df['Name'] == name, 'System'].values[0]
        roederer2010a_df.loc[i,'RA_hms'] = obs_df.loc[obs_df['Name'] == name, 'RA_hms'].values[0]
        roederer2010a_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(roederer2010a_df.loc[i,'RA_hms'], precision=6)
        roederer2010a_df.loc[i,'DEC_dms'] = obs_df.loc[obs_df['Name'] == name, 'DEC_dms'].values[0]
        roederer2010a_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(roederer2010a_df.loc[i,'DEC_dms'], precision=2)
        roederer2010a_df.loc[i,'Teff'] = param_df.loc[param_df['Name'] == name, 'Teff'].values[0]
        roederer2010a_df.loc[i,'logg'] = param_df.loc[param_df['Name'] == name, 'logg'].values[0]
        roederer2010a_df.loc[i,'M/H'] = param_df.loc[param_df['Name'] == name, 'M/H'].values[0]
        roederer2010a_df.loc[i,'Vmic'] = param_df.loc[param_df['Name'] == name, 'Vmic'].values[0]

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
                roederer2010a_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                roederer2010a_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    roederer2010a_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    roederer2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    roederer2010a_df.loc[i, col] = np.nan
                    roederer2010a_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    roederer2010a_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    roederer2010a_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    roederer2010a_df.loc[i, 'ul'+col] = np.nan
                else:
                    roederer2010a_df.loc[i, col] = np.nan
                    roederer2010a_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    roederer2010a_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    roederer2010a_df.loc[i, col] = e_logepsX
                else:
                    roederer2010a_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    roederer2010a_df.drop(columns=[col for col in roederer2010a_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        roederer2010a_df = roederer2010a_df[roederer2010a_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return roederer2010a_df

def load_roederer2019(io=None):
    """
    Load the Roederer et al. 2019 data for the Sylgr stellar stream.

    Table 1 - Observations & Stellar Parameters
    Table 3 - Abundance Table
    """

    ## Read in the data tables
    obs_param_df = pd.read_csv(data_dir + "abundance_tables/roederer2019/table1.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
    abund_df = pd.read_csv(data_dir + "abundance_tables/roederer2019/table3.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])

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
    roederer2019_df = pd.DataFrame(
                    columns=['I/O','Name','Simbad_Identifier','Reference','Ref','Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                    'Teff','logg','M/H','Vmic'] + epscols + ulcols + XHcols + ulXHcols + XFecols 
                    + ulXFecols + errcols)
    for i, name in enumerate(abund_df['Name'].unique()):
        roederer2019_df.loc[i,'Name'] = name
        roederer2019_df.loc[i,'Simbad_Identifier'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Simbad_Identifier'].values[0]
        roederer2019_df.loc[i,'Reference'] = 'Roederer+2010a'
        roederer2019_df.loc[i,'Ref'] = 'ROE10a'
        roederer2019_df.loc[i,'I/O'] = 1
        roederer2019_df.loc[i,'Loc'] = 'SS'
        roederer2019_df.loc[i,'System'] = obs_param_df.loc[obs_param_df['Name'] == name, 'System'].values[0]
        roederer2019_df.loc[i,'RA_hms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'RA_hms'].values[0]
        roederer2019_df.loc[i,'RA_deg'] = scoord.ra_hms_to_deg(roederer2019_df.loc[i,'RA_hms'], precision=6)
        roederer2019_df.loc[i,'DEC_dms'] = obs_param_df.loc[obs_param_df['Name'] == name, 'DEC_dms'].values[0]
        roederer2019_df.loc[i,'DEC_deg'] = scoord.dec_dms_to_deg(roederer2019_df.loc[i,'DEC_dms'], precision=2)
        roederer2019_df.loc[i,'Teff'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Teff'].values[0]
        roederer2019_df.loc[i,'logg'] = obs_param_df.loc[obs_param_df['Name'] == name, 'logg'].values[0]
        roederer2019_df.loc[i,'M/H'] = obs_param_df.loc[obs_param_df['Name'] == name, 'M/H'].values[0]
        roederer2019_df.loc[i,'Vmic'] = obs_param_df.loc[obs_param_df['Name'] == name, 'Vmic'].values[0]

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
                roederer2019_df.loc[i, col] = row["logepsX"] if pd.isna(row["l_logepsX"]) else np.nan

            ## Assign ulX values
            col = make_ulcol(species_i)
            if col in ulcols:
                roederer2019_df.loc[i, col] = row["logepsX"] if pd.notna(row["l_logepsX"]) else np.nan

            ## Assign [X/H] and ul[X/H]values
            col = make_XHcol(species_i).replace(" ", "")
            if col in XHcols:
                if pd.isna(row["l_logepsX"]):
                    roederer2019_df.loc[i, col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                    roederer2019_df.loc[i, 'ul'+col] = np.nan
                else:
                    roederer2019_df.loc[i, col] = np.nan
                    roederer2019_df.loc[i, 'ul'+col] = normal_round(row["logepsX"] - logepsX_sun_a09, 2)
                if 'e_[X/H]' in row.index:
                    roederer2019_df.loc[i, 'e_'+col] = row["e_[X/H]"]

            ## Assign [X/Fe] values
            col = make_XFecol(species_i).replace(" ", "")
            if col in XFecols:
                if pd.isna(row["l_logepsX"]):
                    roederer2019_df.loc[i, col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                    roederer2019_df.loc[i, 'ul'+col] = np.nan
                else:
                    roederer2019_df.loc[i, col] = np.nan
                    roederer2019_df.loc[i, 'ul'+col] = normal_round((row["logepsX"] - logepsX_sun_a09) - feh_a09, 2)
                if 'e_[X/Fe]' in row.index:
                    roederer2019_df.loc[i, 'e_'+col] = row["e_[X/Fe]"]

            ## Assign error values
            col = make_errcol(species_i)
            if col in errcols:
                e_logepsX = row.get("e_logepsX", np.nan)
                if pd.notna(e_logepsX):
                    roederer2019_df.loc[i, col] = e_logepsX
                else:
                    roederer2019_df.loc[i, col] = np.nan

    ## Drop the Fe/Fe columns
    roederer2019_df.drop(columns=[col for col in roederer2019_df.columns if 'Fe/Fe' in col or 'Fe2/Fe' in col], inplace=True, errors='ignore')

    ## Filter the DataFrame based on the I/O column
    if io == 0 or io == 1:
        roederer2019_df = roederer2019_df[roederer2019_df['I/O'] == io]
    elif io is None:
        pass
    else:
        raise ValueError("Invalid value for 'io'. It should be 0, 1, or None.")
    
    return roederer2019_df

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
    df['Name'] = df['Name'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df['Name'] = df['Name'].str.replace("2M", "2MASS J", regex=False).str.rstrip("'")
    for i, row in df.iterrows():
        if pd.isna(row['Name']):
            df.loc[i, 'Name'] = "2MASS J19044856-3107181"
    
    df['Simbad_Identifier'] = df['Name']

    df.rename(columns={
        'RA':'RA_deg',
        'DEC':'DEC_deg',
        'M_H_ERR': 'e_mh',
        'ALPHA_M': 'alpha_m',
        'ALPHA_M_ERR': 'e_alpha_m',
        'TEFF_ERR': 'e_Teff',
        'LOGG_ERR': 'e_logg',
        }, inplace=True)
    
    df["System"] = "Sagittarius"
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
            row['RA_deg'] = scoord.ra_hms_to_deg(row['RA_hms'], precision=6)
            df.at[idx, 'RA_deg'] = row['RA_deg']

        if pd.isna(row['DEC_deg']) and pd.notna(row['DEC_dms']):
            row['DEC_deg'] = scoord.dec_dms_to_deg(row['DEC_dms'], precision=2)
            df.at[idx, 'DEC_deg'] = row['DEC_deg']

        if pd.isna(row['RA_hms']) and pd.notna(row['RA_deg']):
            row['RA_hms'] = scoord.ra_deg_to_hms(float(row['RA_deg']), precision=2)
            df.at[idx, 'RA_hms'] = row['RA_hms']

        if pd.isna(row['DEC_dms']) and pd.notna(row['DEC_deg']):
            row['DEC_dms'] = scoord.dec_deg_to_dms(float(row['DEC_deg']), precision=2)
            df.at[idx, 'DEC_dms'] = row['DEC_dms']

    XHcol_from_XFecol(df)
    epscol_from_XHcol(df)
    ulXHcol_from_ulcol(df)
    ulXFecol_from_ulcol(df)

    # Categorize columns & reorder dataFrame
    columns = list(df.columns)
    aux_cols = [
        'Reference','Ref','Name','Simbad_Identifier','RA_hms','DEC_dms','RA_deg','DEC_deg',
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