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
data_dir = script_dir+"../../data/"

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
        load_lmc(), #jinabase=jinabase_df),
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
        # load_lmc(jinabase=jinabase_df),
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
    epsfe_sun_a09 = get_solar('Fe', version='asplund2009').values[0]
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
        X_name = elem.title().replace('1','I').replace('2','II')
        el = elem.title().replace('1','').replace('2','')

        try:
            epsX_sun_09 = get_solar(el, version='asplund2009').values[0]
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
        load_roederer2019d() ## Sylgr
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
    epsfe_sun_a09 = get_solar('Fe', version='asplund2009').values[0]
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
        X_name = elem.title().replace('1','I').replace('2','II')
        el = elem.title().replace('1','').replace('2','')

        try:
            epsX_sun_09 = get_solar(el, version='asplund2009').values[0]
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
    ji2026_df = load_ji2026()
    limberg2025a_df = load_limberg2025a()
    lucey2026_df = load_lucey2026()

    ## Add filters for specific references
    reggiani2021_df = reggiani2021_df[reggiani2021_df['System'] == 'Large Magellanic Cloud']

    ## Combine the DataFrames
    lmc_df = pd.concat([
            chiti2024_df,
            reggiani2021_df,
            ji2026_df,
            limberg2025a_df,
            lucey2026_df
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

    ## Removing Duplicate stars
    dups = [
        ('Limberg+2025a', 'GDR3_526285')
    ]
    for ref, name in dups:
        lmc_df.loc[(lmc_df['Name'] == name) & (lmc_df['Reference'] == ref), 'I/O'] = 0
    lmc_df = lmc_df[lmc_df['I/O'] == 1].reset_index(drop=True)
    
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
    roederer2023a_df = load_roederer2023a()
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
            roederer2023a_df,
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
    hughes2026_df = load_hughes2026()
    francois2007_df = load_francois2007()
    nordlander2019_df = load_nordlander2019()
    mardini2022b_df = load_mardini2022b()


    ## Selects only halo stars (or more like everything unclassified in JINAbase)
    halo_df = jinabase_df[(jinabase_df['Loc'] == 'HA') | (jinabase_df['Loc'].isin(['', 'nan', np.nan]))]
    halo_df = pd.concat([halo_df, francois2007_df, nordlander2019_df, mardini2022b_df], ignore_index=True, sort=False)

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
    sass_df = pd.concat([jinabase_sass_df, hughes2026_df], ignore_index=True, sort=False)
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
        
        ('Melendez+2016a', '2MASSJ18082002-5104378'), # using Mardini+2022b
        
        ('Roederer+2014c', 'CS22952-015'), # we have measurements from Francois+2007 that don't make the cut (Sr too high), so we cut this star here
        ('Roederer+2014c', 'CS22189-009'), # we have measurements from Francois+2007 that don't make the cut (Sr too high), so we cut this star here
        
        ## not a duplicate, but sometimes removed due to upper limit in iron (has carbon)
        # ('Keller+2014', 'SMSSJ031300.36-670839.3')
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
    data = pd.read_csv(data_dir+"abundances/JINAbase-4-yelland/JINAbase-yelland25.csv", header=0, na_values=["*"]) #index_col=0
    uls  = pd.read_csv(data_dir+"abundances/JINAbase-4-yelland/JINAbase-yelland25-ulimits.csv", header=0, na_values=["*"]) #index_col=0
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
    uls_values = uls_values.map(lambda x: float(x.strip("<")) if isinstance(x, str) and x.strip().startswith("<") else np.nan)

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
    data.to_csv(data_dir+"abundances/JINAbase-4-yelland/JINAbase-yelland25-processed.csv", index=False)

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

def load_mardini2022a(io=None):
    """
    Atari Disk (Atr) Stars

    Load the data from Mardini et al. (2022), Table 5, for stars in the Atari Disk (Atr) region.
    """

    mardini2022a_df = pd.read_csv(data_dir+'abundances/mardini2022a/tab5_yelland.csv', comment='#')

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
    mardini2022a_df.loc[mardini2022a_df['Name'] == '2MASS J12450268-0738469', 'Ncap_key'] = 'S' # halo reference star
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'HE 0017-4346', 'Ncap_key'] = 'S' # [C/Fe] = 3.02
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'HE 1413-1954', 'C_key'] = 'NO' # [C/Fe] = 1.44
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'HE 1300+0157', 'C_key'] = 'NO' # (HE 1300+0157, https://www.aanda.org/articles/aa/pdf/2019/03/aa34601-18.pdf)
    mardini2022a_df.loc[mardini2022a_df['Reference'] == 'Aguado+2017', 'logg'] = 4.9
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'SDSS J124719.46-034152.4', 'logg'] = 4.0
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'SDSS J105519.28+232234.0', 'logg'] = 4.9
    mardini2022a_df.loc[mardini2022a_df['Name'] == 'UCAC3 215-112497', 'Name'] = 'SDSS J102915.14+172927.9' # update name to common name
    
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
    mardini2022a_df.to_csv(data_dir+'abundances/mardini2022a/tab5_processed.csv', index=False)

    return mardini2022a_df

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

    placco2014c_df = pd.read_csv(data_dir+"abundances/placco2014c/cds_files/table3_mod.csv") # using modified table for correct reference labeling

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
    mask_N = placco2014c_df['ul[N/Fe]'] == '{<=}'
    placco2014c_df.loc[mask_N, 'ul[N/Fe]'] = placco2014c_df.loc[mask_N, '[N/Fe]'].astype(str)
    placco2014c_df.loc[mask_N, '[N/Fe]'] = np.nan

    placco2014c_df.rename(columns={"l_[Sr/Fe]": "ul[Sr/Fe]"}, inplace=True)
    mask_Sr = placco2014c_df['ul[Sr/Fe]'] == '{<=}'
    placco2014c_df.loc[mask_Sr, 'ul[Sr/Fe]'] = placco2014c_df.loc[mask_Sr, '[Sr/Fe]'].astype(str)
    placco2014c_df.loc[mask_Sr, '[Sr/Fe]'] = np.nan

    placco2014c_df.rename(columns={"l_[Ba/Fe]": "ul[Ba/Fe]"}, inplace=True)
    mask_Ba = placco2014c_df['ul[Ba/Fe]'] == '{<=}'
    placco2014c_df.loc[mask_Ba, 'ul[Ba/Fe]'] = placco2014c_df.loc[mask_Ba, '[Ba/Fe]'].astype(str)
    placco2014c_df.loc[mask_Ba, '[Ba/Fe]'] = np.nan

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
    simbad_df = pd.read_csv(data_dir+'abundances/placco2014c/simbad_data.csv')
    for name in simbad_df['Name']:
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'Simbad_Identifier'] = simbad_df.loc[simbad_df['Name'] == name, 'MAIN_ID'].values[0]
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'RA_hms'] = simbad_df.loc[simbad_df['Name'] == name, 'RA'].values[0].replace(' ', ':')
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'DEC_dms'] = simbad_df.loc[simbad_df['Name'] == name, 'DEC'].values[0].replace(' ', ':')
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'RA_deg'] = scoord.ra_hms_to_deg(placco2014c_df.loc[placco2014c_df['Name'] == name, 'RA_hms'], precision=4)
        placco2014c_df.loc[placco2014c_df['Name'] == name, 'DEC_deg'] = scoord.dec_dms_to_deg(placco2014c_df.loc[placco2014c_df['Name'] == name, 'DEC_dms'], precision=2)
    new_columns = ['Simbad_Identifier', 'RA_hms', 'DEC_dms', 'RA_deg', 'DEC_deg']
    placco2014c_df = placco2014c_df[[placco2014c_df.columns[0]] + new_columns + list(placco2014c_df.columns[1:-len(new_columns)])]

    ## Save the pre-filtered DataFrame
    placco2014c_df.to_csv(data_dir+'abundances/placco2014c/placco2014c.csv', index=False)

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
    placco2014c_df.to_csv(data_dir+'abundances/placco2014c/placco2014c-processed.csv', index=False)
    
    return placco2014c_df


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
    tab = Table.read(data_dir+"abundances/APOGEE/apogee_sgr.fits")
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