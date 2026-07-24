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
        load_frebel2013c()[load_frebel2013c()['System'] == 'Segue 1'], ## only include Segue 1 stars from Frebel+2013c
        load_frebel2014(),
        load_frebel2016(),
        load_gilmore2013a(),
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
        ('Gilmore+2013a' , 'BooI-127'),
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
    lucey2026_df = load_lucey2026b()

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

### Base read-in functions...

def transpose_abund_df(
        abund_df: pd.DataFrame, 
        solar_init: str, 
        transpose_abund: str = '[X/Fe]',
        transpose_err: str = 'e_[X/Fe]',
    ) -> pd.DataFrame:

    ## Pre-compute element/ion metadata for each abundance column (done once, not once per row)
    fe_cols = [col for col in abund_df.columns if (col.startswith('[Fe') | col.startswith('e_[Fe')) & col.endswith('/H]')]
    _val_col_getters = {
        'logepsX': epscolnames,
        '[X/H]':   XHcolnames,
        '[X/Fe]':  XFecolnames,
    }
    _id_cols = {
        'logepsX': ['Name'],
        '[X/H]':   ['Name'],
        '[X/Fe]':  ['Name'] + fe_cols,
    }

    val_cols = _val_col_getters[transpose_abund](abund_df)
    err_cols = errcolnames(abund_df)
    id_cols  = _id_cols[transpose_abund]
    
    col_meta = {}
    for col in val_cols:
        elem = getelem(col)             # element (e.g. 'Fe')
        ion = element_to_ion(elem)      # ion (e.g. 'Fe II')
        species = ion_to_species(ion)   # species (e.g. 26.1)
        col_meta[col] = (species, ion, elem)

    ## Reshape from wide to long format: one row per (star, element, error) trio    
    abund_df_T = (
        abund_df[id_cols + val_cols]
        .melt(id_vars=id_cols, value_vars=val_cols, var_name='col', value_name='val')
        .dropna(subset=['val'])
        .reset_index(drop=True)
    )
    err_df_T = (
        abund_df[['Name'] + err_cols]
        .melt(id_vars=['Name'], value_vars=err_cols, var_name='err_col_name', value_name='err')
        .assign(col=lambda df: df['err_col_name'].map(dict(zip(err_cols, val_cols))))
        .drop(columns='err_col_name')
        .reset_index(drop=True)
    )
    abund_df_T = abund_df_T.merge(err_df_T, on=['Name', 'col'], how='left')
        
    ## Attach species/ion/element metadata to the transposed dataframe
    abund_df_T[['Species', 'Ion', 'Elem']] = pd.DataFrame(
        abund_df_T['col'].map(col_meta).tolist(), index=abund_df_T.index
    )
    
    ## Extract the upper-limit flags and numeric values from value column
    val_str = abund_df_T['val'].astype(str)
    abund_df_T['l_logepsX'] = val_str.str.extract(r'([<>])')[0]
    abund_df_T['init_abund'] = val_str.str.replace(r'[<>]', '', regex=True).astype(float)

    ## Extract limit flags and numeric values from Fe id columns
    ## (these are carried through the melt unchanged, so may still contain '<'/'>' symbols)
    if transpose_abund == '[X/Fe]':
        value_fe_cols = [col for col in fe_cols if not col.startswith('e_')]
        for col in value_fe_cols:
            col_str = abund_df_T[col].astype(str)
            abund_df_T[f'l_{col}'] = col_str.str.extract(r'([<>])')[0]   # '<', '>', or NaN
            abund_df_T[col] = col_str.str.replace(r'[<>]', '', regex=True).astype(float)
            
            for idx, row in abund_df_T.iterrows():
                if (row['l_logepsX'] == '<') & (row[f'l_{col}'] == '>'):
                    warnings.warn(f"Star '{row['Name']}' has a lower limit on [Fe/H] and an upper limit on [{row['Elem']}/Fe]. We assume the upper limit comes from the [Fe/H] measurement, and set l_logepsX to NaN.")
                    abund_df_T.loc[idx, 'l_logepsX'] = np.nan
                elif (row['l_logepsX'] == '>') & (row[f'l_{col}'] == '<'):
                    warnings.warn(f"Star '{row['Name']}' has an upper limit on [Fe/H] and a lower limit on [{row['Elem']}/Fe]. We assume the lower limit comes from the [Fe/H] measurement, and set l_logepsX to NaN.")
                    abund_df_T.loc[idx, 'l_logepsX'] = np.nan
                elif ((row['l_logepsX'] == '<') & (row[f'l_{col}'] == '<')) | ((row['l_logepsX'] == '>') & (row[f'l_{col}'] == '>')):
                    raise ValueError(f"Star '{row['Name']}' has the same limit type on both [Fe/H] and [{row['Elem']}/Fe]. This is not possible and indicates a problem with the input data.")

    ## Compute logepsX row-wise using the initial solar abundance scale
    def _convert_to_logepsX(row):
        if transpose_abund == 'logepsX': 
            return row['init_abund']
        elif transpose_abund == '[X/H]': 
            return eps_from_XH(row['init_abund'], row['Elem'], version=solar_init)
        elif transpose_abund == '[X/Fe]':
            feh_col = next((col for col in fe_cols if ((col in row.index) & ('II' not in col) & ('2' not in col))), None)
            if feh_col is None:
                raise ValueError(f"None of the Fe columns {fe_cols} were found in the row.")
            return eps_from_XFe(row['init_abund'], row[feh_col], row['Elem'], version=solar_init)
        
    abund_df_T['logepsX'] = abund_df_T.apply(lambda r: _convert_to_logepsX(r), axis=1)
    abund_df_T[transpose_err] = abund_df_T.apply(lambda r: r['err'], axis=1)
    
    if transpose_abund == '[X/Fe]':
        logepsFe_sun = get_solar('Fe', version=solar_init)
        fe_rows = abund_df[['Name'] + fe_cols].copy()

        for col in [col for col in fe_cols if 'e_' not in col]:
            fe_ion = element_to_ion(getelem(col))

            # Strip '<'/'>' from the original abund_df Fe column before arithmetic
            col_str             = fe_rows[col].astype(str)
            fe_rows[f'l_{col}'] = col_str.str.extract(r'([<>])')[0]
            fe_rows[f'l_logepsX'] = col_str.str.extract(r'([<>])')[0]
            fe_rows[col]        = col_str.str.replace(r'[<>]', '', regex=True).astype(float)

            fe_rows['Species'] = ion_to_species(fe_ion)
            fe_rows['Ion'] = fe_ion
            fe_rows['Elem']    = getelem(col)
            fe_rows['logepsX'] = fe_rows[col].apply(lambda feh: feh + logepsFe_sun).round(2)
            fe_rows['e_[X/H]'] = fe_rows['e_' + col] if ('e_' + col) in fe_rows.columns else np.nan

            abund_df_T = pd.concat([abund_df_T, fe_rows], ignore_index=True, sort=False)

        abund_df_T = abund_df_T.drop(columns=fe_cols+('l_' + np.array(value_fe_cols)).tolist())

    ## Sort by star name and atomic number, and drop intermediate columns
    abund_df_T = (
        abund_df_T
            .sort_values(['Name', 'Species'])
            .drop(columns=['col', 'val', 'err', 'Elem', 'init_abund'])
            .reset_index(drop=True)
    )

    return abund_df_T

def load_authorYYYYx(
        author: str,
        year: str,
        shortname: str,
        loc: str,
        obs_table: str,
        param_table: str,
        abund_table: str,
        solar_init: str = 'asplund2009',
        solar_final: str = 'asplund2009',
        init_abund: str = 'logepsX', # logepsX,[X/H],[X/Fe]
        transpose: bool = False,
        transpose_abund: str = '[X/Fe]',
        transpose_err: str = 'e_[X/Fe]',
    ) -> pd.DataFrame:
    """
    Load data for a given author and year,and return a dataframe in the standard format.
    
    Parameters:
        author: str
            The author's name,with the first letter capitalized (e.g.,'Yelland').
        year: str
            The year of the study,with a letter suffix if needed (e.g.,'2026b').
        shortname: str
            The combined short name for the reference using the first three letters 
            of the author's last name captialized (and occationally the first letter of the 
            author's first name lowercase) and the last two digits of the year and suffix (e.g.,'YEL26b' or 'YELa26b').
        loc: str
            The location of the collection of stars: ['HA','BU','DS','DW','UF','GC',''].
             - 'HA' = halo
             - 'BU' = bulge
             - 'DS' = disk
             - 'DW' = dwarf galaxy
             - 'UF' = ultra-faint dwarf galaxy
             - 'GC' = globular cluster
             - 'MX' = mixed location (see obs_table)
             - ''   = unknown
        obs_table: str
            The name of the observations table file. This table should contain the columns: Name,Simbad_Identifier,System,RA_hms,DEC_dms.
        param_table: str
            The name of the stellar parameters table file. This table should contain the columns: Name,Teff,logg,Fe/H,Vmic.
        abund_table: str
            The name of the abundance table file. This table should contain the columns: Name,Species,[abundance columns].
            
    Returns:
        authorYYYYx_df: pandas.DataFrame
            A dataframe containing the data for the given author and year,formatted according to the standard format.
    """

    # ---------------------------------------- #
    # Read in the data tables
    
    _na = [""," ","nan","NaN","N/A","n/a"]
    base_path = f"{data_dir}abundances/{author.lower()}{year}/"
    obs_df   = pd.read_csv(f"{base_path}{obs_table}.csv",  comment="#",na_values=_na)
    param_df = pd.read_csv(f"{base_path}{param_table}.csv",comment="#",na_values=_na)
    abund_df = pd.read_csv(f"{base_path}{abund_table}.csv",comment="#",na_values=_na)

    if transpose:
        abund_df = transpose_abund_df(abund_df, solar_init, transpose_abund, transpose_err)
    
    # ---------------------------------------- #
    # Prepare the abundance table for pivoting
    
    ## Map ions (e.g. 'Fe I') to species (e.g. 26.0) and elements (e.g. 'Fe')
    if 'Ion' in abund_df.columns:
        abund_df['species_i'] = abund_df['Ion'].map(ion_to_species)
        abund_df['elem_i']    = abund_df['Ion'].map(ion_to_element)
    elif 'Species' in abund_df.columns:
        abund_df['Ion'] = abund_df['Species'].map(species_to_ion)
        abund_df['species_i'] = abund_df['Species']
        abund_df['elem_i']    = abund_df['Species'].map(species_to_element)
    else:
        raise ValueError("Abundance dataframe must contain either an 'Ion' or 'Species' column.")
    species = list(dict.fromkeys(abund_df['species_i']))
    
    def _add_solar_col(_df: pd.DataFrame, _solar: str):
        """
        Add a column of solar abundances (logepsX_sun) for each element in the abundance dataframe.
        
        Args:
            _df (pd.DataFrame): The abundance dataframe to which the solar abundance column will be added. Must contain an 'elem_i' column with element symbols.
            _solar (str): The version of solar abundances to use. This will be passed to the get_solar function to retrieve the solar abundance values. Options include...
                - 'anders1989'
                - 'grevesse1998'
                - 'asplund2005'
                - 'asplund2009' (default)
                - 'asplund2021'
                - 'lodders2025'
        """
        solar_abund_map = {
                elem: get_solar(elem, version=_solar).values[0]
                for elem in _df['elem_i'].unique()
            }
        _df['logepsX_sun'] = _df['elem_i'].map(solar_abund_map)
    
    def _add_FeH_col(_df: pd.DataFrame, _solar: str):
        """
        Add a column of iron abundance ([Fe/H]) for each star in the abundance dataframe, calculated from the specified abundance form and solar abundances.

        Args:
            _df (pd.DataFrame): The abundance dataframe to which the [Fe/H] column will be added. Must contain 'Name', 'Ion', and the specified abundance form columns.
            _abund_form (str): The form of abundance to use for calculating [Fe/H]. Options include:
                - 'logepsX' (default)
                - '[X/H]'
                - '[X/Fe]'
            _solar (str): The version of solar abundances to use for calculating [Fe/H]. This will be passed to the get_solar function to retrieve the solar abundance of Fe. Options include...
                - 'anders1989'
                - 'grevesse1998'
                - 'asplund2005'
                - 'asplund2009' (default)
                - 'asplund2021'
                - 'lodders2025'
        """
        logepsFe_sun = get_solar('Fe', version=_solar).values[0]
        feh_map = {
                name: _df.loc[(_df['Name'] == name) & (_df['Ion'] == 'Fe I'), 'logepsX'].values[0] - logepsFe_sun
                for name in _df['Name'].unique()
            }
        _df['[Fe/H]'] = _df['Name'].map(feh_map)

    def _find_logepsX_from_XH(df: pd.DataFrame, _solar_init: str, rows=slice(None)):
        """
        If the initial abundance form is in [X/H], calculate logepsX using the solar abundances.
        """
        if 'logepsX_sun' not in df.columns:
            _add_solar_col(df, _solar_init) 
        df.loc[rows, 'logepsX']   = df.loc[rows, '[X/H]'] + df.loc[rows, 'logepsX_sun']
        df.loc[rows, 'l_logepsX'] = df.loc[rows, 'l_[X/H]']
        return df
            
    def _find_logepsX_from_XFe(df: pd.DataFrame, _solar_init: str):
        """
        If the initial abundance form is in [X/Fe], calculate logepsX and [X/H] using the solar abundances and the [Fe/H] values.
        """
        ## For the Fe species
        fe_rows = (df['Ion'] == 'Fe I') | (df['Ion'] == 'Fe II')
        _find_logepsX_from_XH(df, _solar_init, rows=fe_rows)

        ## For non-Fe species
        ### If the [FeII/H] column was empty and abundances for [FeII/FeI] were provided instead, then we modify the `non_fe_rows` mask 
        ### below to include the FeII rows in the [X/Fe] tp logepsX conversion.
        ### This is checked by confirming that all logepsX values for Fe II rows are NaN, which would be the case if the [FeII/H] column 
        ### was left empty and the Fe II abundances were represented as [FeII/FeI] in the [X/Fe] column.
        feII_rows = (df['Ion'] == 'Fe II')
        if any(feII_rows) and (df.loc[feII_rows, 'logepsX'].isna().sum() == len(df.loc[feII_rows, 'logepsX'])):
            # print(f"WARNING: Check the [FeII/H] abundances in {author}+{year}; they are all NaN. Data is loaded with " \
            #        "the assumption that the FeII abundances are represented in the [X/Fe] column (e.g. [FeII/FeI]).")
            non_fe_rows = (df['Ion'] != 'Fe I')
        else:
            non_fe_rows = (df['Ion'] != 'Fe I') & (df['Ion'] != 'Fe II')
        _add_FeH_col(df, _solar_init)
        
        if 'logepsX_sun' not in abund_df.columns:
            _add_solar_col(abund_df[non_fe_rows], _solar_init) # use solar abundances referenced in literature source
        abund_df.loc[non_fe_rows,'logepsX'] = abund_df.loc[non_fe_rows,'[X/Fe]'] + abund_df.loc[non_fe_rows,'[Fe/H]'] + abund_df.loc[non_fe_rows,'logepsX_sun']
        abund_df['l_logepsX'] = abund_df['l_logepsX'].astype(object)
        abund_df.loc[non_fe_rows,'l_logepsX'] = abund_df.loc[non_fe_rows,'l_[X/Fe]']
        return df

    ## Calculate the 'logepsX' column based on the initial abundance form provided (logepsX, [X/H], or [X/Fe])    
    match init_abund:
        case 'logepsX': pass
        case '[X/H]':   abund_df = _find_logepsX_from_XH(abund_df, solar_init)
        case '[X/Fe]':  abund_df = _find_logepsX_from_XFe(abund_df, solar_init)
    
    # update the solar abundance column and [Fe/H] column for desired version of the solar abundances
    _add_solar_col(abund_df, solar_final) 
    _add_FeH_col(abund_df, solar_final)
    
    ## Calculate [X/H] and [X/Fe] for each row
    abund_df['XH_raw']  = abund_df['logepsX'] - abund_df['logepsX_sun']
    abund_df['XFe_raw'] = abund_df['XH_raw']  - abund_df['[Fe/H]']
    
    ## Separate limits from measured abundances & round to 2 decimal places (rounding safe for NaN)
    def _r2(series):
        def round_or_nan(v):
            if pd.notna(v):
                return normal_round(v,2)
            else:
                return np.nan
        return series.apply(round_or_nan)

    ulmask_eps = abund_df['l_logepsX'].eq('<')
    ulmask_XH  = abund_df['l_logepsX'].eq('<')
    ulmask_XFe = abund_df['l_[X/Fe]'].eq('<') if 'l_[X/Fe]' in abund_df.columns else ulmask_eps
    
    llmask_eps = abund_df['l_logepsX'].eq('>')
    llmask_XH  = abund_df['l_logepsX'].eq('>')
    llmask_XFe = abund_df['l_[X/Fe]'].eq('>') if 'l_[X/Fe]' in abund_df.columns else llmask_eps
    
    ### Update the ulmask and create a NaN mask for [X/Fe] by propagating Fe I lower limits to non-Fe [X/Fe] abundances
    ### - [X/H] is measured       --> [X/Fe] becomes an upper limit
    ### - [X/H] is itself a limit --> [X/Fe] is undefined (NaN)

    llFe_stars    = set(abund_df.loc[(abund_df['Ion'] == 'Fe I') & llmask_eps, 'Name'])
    llFe_stars_mask  = abund_df['Name'].isin(llFe_stars)
    ulFe_stars    = set(abund_df.loc[(abund_df['Ion'] == 'Fe I') & ulmask_eps, 'Name'])
    ulFe_stars_mask  = abund_df['Name'].isin(ulFe_stars)
    nonFe_abund_mask = ~abund_df['Ion'].isin(['Fe I', 'Fe II'])
    
    XH_measured_mask = ~ulmask_XH & ~llmask_XH
    llFe_xh_measured = llFe_stars_mask & nonFe_abund_mask & XH_measured_mask  # [Fe/H] is a LL + [X/H] measured   --> [X/Fe] is UL
    llFe_xh_limited  = llFe_stars_mask & nonFe_abund_mask & ~XH_measured_mask # [Fe/H] is a LL + [X/H] is a limit --> [X/Fe] is NaN
    ulFe_xh_measured = ulFe_stars_mask & nonFe_abund_mask & XH_measured_mask  # [Fe/H] is a UL + [X/H] measured   --> [X/Fe] is LL
    ulFe_xh_limited  = ulFe_stars_mask & nonFe_abund_mask & ~XH_measured_mask # [Fe/H] is a UL + [X/H] is a limit --> [X/Fe] is NaN
    
    ulmask_XFe = ulmask_XFe | llFe_xh_measured
    llmask_XFe = llmask_XFe | ulFe_xh_measured
    xfe_is_nan = llFe_xh_limited | ulFe_xh_limited

    abund_df['epsX_ll']  = _r2(abund_df['logepsX'].where(llmask_eps))
    abund_df['epsX_val'] = _r2(abund_df['logepsX'].where(~llmask_eps & ~ulmask_eps))
    abund_df['epsX_ul']  = _r2(abund_df['logepsX'].where(ulmask_eps))
    
    abund_df['XH_ll']    = _r2(abund_df['XH_raw'] .where(llmask_XH))
    abund_df['XH_val']   = _r2(abund_df['XH_raw'] .where(~llmask_XH & ~ulmask_XH))
    abund_df['XH_ul']    = _r2(abund_df['XH_raw'] .where(ulmask_XH))
    
    abund_df['XFe_ll']  = _r2(abund_df['XFe_raw'].where(llmask_XFe))
    abund_df['XFe_val'] = _r2(abund_df['XFe_raw'].where(~llmask_XFe & ~ulmask_XFe & ~xfe_is_nan))
    abund_df['XFe_ul']  = _r2(abund_df['XFe_raw'].where(ulmask_XFe  & ~xfe_is_nan))

    # ---------------------------------------- #
    # Pivot the abundance table to wide format
    
    def _pivot_column(col):
        """
        Return a Name-indexed wide table (dataframe) for a single derived abundance column.
        - Uses the numerical species_i for column labels,which will be renamed later via the standard column naming functions.
        - If the column doesn't exist in the abundance table,returns an empty dataframe with the correct shape and column labels.
        """
        if col not in abund_df.columns:
            return pd.DataFrame(index=abund_df['Name'].unique(),columns=species,dtype=float)
        return abund_df.pivot_table(index='Name',columns='species_i',values=col,aggfunc='last').reindex(columns=species)

    ## Create a dictionary of dataframes for each pivoted abundance column,using the helper function above
    pivots = { 
        'll':    _pivot_column('epsX_ll'),
        'eps':   _pivot_column('epsX_val'),
        'ul':    _pivot_column('epsX_ul'),
        'llXH':  _pivot_column('XH_ll'),
        'XH':    _pivot_column('XH_val'),
        'ulXH':  _pivot_column('XH_ul'),
        'llXFe': _pivot_column('XFe_ll'),
        'XFe':   _pivot_column('XFe_val'),
        'ulXFe': _pivot_column('XFe_ul'),
        'e_eps': _pivot_column('e_logepsX'),
        'e_XH':  _pivot_column('e_[X/H]'),
        'e_XFe': _pivot_column('e_[X/Fe]'),
    }

    ## Create a dictionary of column naming functions for each pivoted abundance column
    colname_functions = {
        'eps':   make_epscol,
        'll':    make_llcol,
        'ul':    make_ulcol,
        'XH':    make_XHcol,
        'llXH':  make_llXHcol,
        'ulXH':  make_ulXHcol,
        'XFe':   make_XFecol,
        'llXFe': make_llXFecol,
        'ulXFe': make_ulXFecol,
        'e_eps': lambda s: 'e_' + make_epscol(s),
        'e_XH':  lambda s: 'e_' + make_XHcol(s),
        'e_XFe': lambda s: 'e_' + make_XFecol(s),
    }
    
    ## Apply the column naming functions to rename the columns of each pivoted dataframe
    for key,piv_df in pivots.items():
        piv_df.columns = [colname_functions[key](s) for s in species]
    
    # ---------------------------------------- #
    # Prepare the final dataframe with standard SPAG format
    
    ## `obs_table`` and `param_table`` data merged into `base_df` with additional columns
    base_df = obs_df.merge(param_df)
    base_df['Reference'] = f'{author}+{year}'
    base_df['Ref']       = shortname
    base_df['I/O']       = 1
    if loc in ['HA','BU','DS','DW','UF','GC','SS','']:
        base_df['Loc']   = loc 
    elif loc == 'MX':
        base_df['Loc']   = base_df.get('Loc',pd.Series([np.nan]*len(base_df)))
    else:
        raise ValueError(
            f"Invalid loc value: {loc}. Must be one of ['HA','BU','DS','DW','UF','GC','MX','']. \n" + 
            " - 'HA' = halo \n" +
            " - 'BU' = bulge \n" +
            " - 'DS' = disk \n" +
            " - 'DW' = dwarf galaxy \n" +
            " - 'UF' = ultra-faint dwarf galaxy \n" +
            " - 'GC' = globular cluster \n" +
            " - 'SS' = stellar stream \n" +
            " - 'MX' = mixed location (see obs_table) \n" +
            " - ''   = unknown" 
        )
    if 'RA_hms' in base_df.columns and 'DEC_dms' in base_df.columns:
        base_df['RA_deg']    = base_df['RA_hms'].apply(lambda x: scoord.ra_hms_to_deg(x,precision=6))
        base_df['DEC_deg']   = base_df['DEC_dms'].apply(lambda x: scoord.dec_dms_to_deg(x,precision=6))
    elif 'RA_deg' in base_df.columns and 'DEC_deg' in base_df.columns:
        base_df['RA_hms']    = base_df['RA_deg'].apply(lambda x: scoord.ra_deg_to_hms(x,precision=2))
        base_df['DEC_dms']   = base_df['DEC_deg'].apply(lambda x: scoord.dec_deg_to_dms(x,precision=2))

    ## Join the base dataframe with the abundance dataframes
    base_df = base_df.set_index('Name')
    result_df = base_df.join(list(pivots.values())).reset_index()
    
    ## Retain only stars that appear in the abundance table
    result_df = result_df[result_df['Name'].isin(abund_df['Name'].unique())]

    ## Filter the columns to ensure a consistent order: fixed columns first,then abundance columns
    fixed_cols = ['I/O','Name','Simbad_Identifier','Reference','Ref',
                'Loc','System','RA_hms','RA_deg','DEC_dms','DEC_deg',
                'Teff','logg','M/H','Vmic']
    abund_cols = [col for piv in pivots.values() for col in piv.columns]
    result_df  = result_df.reindex(columns=fixed_cols + abund_cols)
    result_df  = result_df.reset_index(drop=True)
    

    ## Drop Fe/Fe artefact columns
    result_df.drop(
        columns=[c for c in result_df.columns if 'Fe/Fe' in c or 'Fe2/Fe' in c],
        inplace=True,errors='ignore'
    )

    return result_df

### To be fixed...

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

    mardini2022a_df = pd.read_csv(data_dir+'abundances/_incompleted/mardini2022a/table5_corrected.csv', comment='#')

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
    mardini2022a_df.to_csv(data_dir+'abundances/_incompleted/mardini2022a/table5_corrected_processed.csv', index=False)

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

### New reference functions...

def load_cayrel2004() -> pd.DataFrame:
    """
    Loads the Cayrel et al. (2004) & Francois et al. (2007) data for Milky Way halo data. 
    
    Cayrel et al. (2004) contains the light element abundances for the "First Stars." 
    series (First Stars. V.).
      --> [published the logepsX abbundances]
    
    Francois et al. (2007) contains the heavy element abundances for the "First Stars." 
    series (First Stars. VIII.).
      --> [published the [X/Fe] abbundances, using 'grevesse1998' solar abundances]

    Table 2 - Cayrel+2004 Observation Table
    Table 4 - Cayrel+2004 Stellar Parameters
    Table 8 - Cayrel+2004 Abundance Table (The [Fe/H] ([Fe/H]_c) values are the avg of Fe I and Fe II)
    Table 3,4,5 - Francois+2007 Abundance Table
    
    Note: I combined the Cayrel+2004 and Francois+2007 abundance tables into a single table.
    """
    df = load_authorYYYYx(
        author      = 'Cayrel',
        year        = '2004',
        shortname   = 'CAY04',
        loc         = 'HA',
        obs_table   = 'table2',
        param_table = 'table4',
        abund_table = 'abundances-c04_f07_combined'
    )
    return df

def load_chiti2018a(data_subset='merged') -> pd.DataFrame:
    """
    Load the Chiti et al. (2018a) data for the Sculptor Dwarf Galaxy, for both the MagE and M2FS measurements.
    
    Table 5 - MagE Abundance and Observations Table
    Table 6 - M2FS Abundance and Observations Table
    
    Note: Pending on the data_subset parameter, this function can load either the MagE data, the M2FS data, some merged version, or both combined into a single DataFrame.
    Note: See information regarding an error in the solar abundances in the data storage location. (`important_note.txt`)
    
    data_subset: str, optional
        Specifies which subset of the data to load. Options are:
        - 'merged' (default): Load both MagE and M2FS data and combine them into a single DataFrame.
        - 'm2fs': Load only the M2FS data (table6).
        - 'mage': Load only the MagE data (table5).
        - 'all': Load both MagE and M2FS data as separate DataFrames and concatenate them into a single DataFrame.
    """
    
    match data_subset:
        case 'merged':
            obs_param_table = 'chiti2018a_merged_param'
            abund_table     = 'chiti2018a_merged_abund'
        case 'm2fs':
            obs_param_table = 'chiti2018a_m2fs_param'
            abund_table     = 'chiti2018a_m2fs_abund'
        case 'mage':
            obs_param_table = 'chiti2018a_mage_param'
            abund_table     = 'chiti2018a_mage_abund'
        case 'all':
            df = pd.read_csv(data_dir + "abundances/chiti2018a/chiti2018a_alldata.csv", comment="#", na_values=["", " ", "nan", "NaN", "N/A", "n/a"])
            return df
        case _:
            raise ValueError(f"Invalid data_subset value: {data_subset}. Must be one of ['merged', 'm2fs', 'mage', 'all'].")
    
    df = load_authorYYYYx(
        author      = 'Chiti',
        year        = '2018a',
        shortname   = 'CHI18a',
        loc         = 'DW',
        obs_table   = obs_param_table,
        param_table = obs_param_table,
        abund_table = abund_table,
        transpose   = True,
        transpose_abund = '[X/Fe]',
        transpose_err   = 'e_[X/Fe]'
    )
    return df

def load_chiti2018b() -> pd.DataFrame:
    """
    Load the Chiti et al. (2018b) data for the Tucana II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 3 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Chiti',
        year        = '2018b',
        shortname   = 'CHI18b',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table2',
        abund_table = 'table3',
        init_abund  = '[X/H]'
    )
    return df

def load_chiti2023() -> pd.DataFrame:
    """
    Load the Chiti et al. (2023) data for the Tucana II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 3 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Chiti',
        year        = '2023',
        shortname   = 'CHI23',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table2',
        abund_table = 'table3',
        init_abund  = '[X/H]'
    )
    return df

def load_chiti2024() -> pd.DataFrame:
    """
    Load the Chiti et al. (2024) data for the Large Magellanic Cloud (LMC).

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 3 - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Chiti',
        year        = '2024',
        shortname   = 'CHI24',
        loc         = 'DW',
        obs_table   = 'obs_et1',
        param_table = 'param_t1_et2',
        abund_table = 'abund_merged',
    )
    return df

def load_chiti2025a() -> pd.DataFrame:
    """
    Load the Chiti et al. (2025) data for the Pictor II Ultra-Faint Dwarf Galaxy.

    Table 0 - Observation Table & Stellar Parameters
    Table 2 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Chiti',
        year        = '2025a',
        shortname   = 'CHI25a',
        loc         = 'DW',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table2',
        init_abund  = '[X/H]',
    )
    return df

def load_cowan2002() -> pd.DataFrame:
    """
    Load the data from Cowan et al. (2002) for BD +17 3248.

    Table 0 - Observation Table & Stellar Parameters Table
    Table 1,3 - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Cowan',
        year        = '2002',
        shortname   = 'COW02',
        loc         = '',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table1_3',
    )
    return df

def load_feltzing2009() -> pd.DataFrame:
    """
    Load the Feltzing et al. (2009) data for the Bootes I Ultra-Faint Dwarf Galaxies.

    Table 1a - Observations & Stellar Parameters
    Table 1b - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Feltzing',
        year        = '2009',
        shortname   = 'FEL09',
        loc         = 'UF',
        obs_table   = 'table1a',
        param_table = 'table1a',
        abund_table = 'table1b',
    )
    return df

def load_francois2007() -> pd.DataFrame:
    """
    Loads the Cayrel et al. (2004) & Francois et al. (2007) data for Milky Way halo data. 
    
    Cayrel et al. (2004) contains the light element abundances for the "First Stars." 
    series (First Stars. V.).
      --> [published the logepsX abbundances]
    
    Francois et al. (2007) contains the heavy element abundances for the "First Stars." 
    series (First Stars. VIII.).
      --> [published the [X/Fe] abbundances, using 'grevesse1998' solar abundances]

    Table 2 - Cayrel+2004 Observation Table
    Table 4 - Cayrel+2004 Stellar Parameters
    Table 8 - Cayrel+2004 Abundance Table (The [Fe/H] ([Fe/H]_c) values are the avg of Fe I and Fe II)
    Table 3,4,5 - Francois+2007 Abundance Table
    
    Note: I combined the Cayrel+2004 and Francois+2007 abundance tables into a single table.
    """
    df = load_authorYYYYx(
        author      = 'Francois',
        year        = '2007',
        shortname   = 'FRA07',
        loc         = 'HA',
        obs_table   = 'table0',
        param_table = 'table1',
        abund_table = 'abundances-c04_f07_combined'
    )
    return df

def load_francois2016() -> pd.DataFrame:
    """
    Load the Francois et al. (2016) data for the Bootes II, Leo IV, Cane Venatici I, Cane Venatici II, and Hercules Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table

    Note: The paper used the Grevesse & Sauval (1998) solar abundances. -- Published in a book chapter in 2000.
    """
    df = load_authorYYYYx(
        author      = 'Francois',
        year        = '2016',
        shortname   = 'FRA16',
        loc         = 'HA',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table6',
        solar_init  = 'grevesse1998',
        transpose   = True,
        transpose_abund = '[X/Fe]',
        transpose_err   = 'e_[X/Fe]'
    )
    return df

def load_frebel2010a() -> pd.DataFrame:
    """
    Load the Frebel et al. (2010a) data for the Ursa Major II and Coma Berenices Ultra-Faint Dwarf Galaxies.

    Table 1,2,5 - Observation and Stellar Parameters
    Table 6,7 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Frebel',
        year        = '2010a',
        shortname   = 'FRE10a',
        loc         = 'UF',
        obs_table   = 'table1_2_5',
        param_table = 'table1_2_5',
        abund_table = 'table6_7',
    )
    return df

def load_frebel2010b() -> pd.DataFrame:
    """
    Load the Frebel et al. (2010b) data for a Sculptor Classical Dwarf Galaxy star, S1020549.

    Table 0 - Observations
    Table 0 - Stellar Parameters
    Table 1 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Frebel',
        year        = '2010b',
        shortname   = 'FRE10b',
        loc         = 'DW',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table1',
    )
    return df

def load_frebel2013c() -> pd.DataFrame:
    """
    Load the Frebel et al. (2013c) data for a stars in the Segue 1 Ultra-Faint Dwarf Galaxy (300 km s^-1 stream),
    in addition to three comparison stars.

    Table 2 - Observations
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Frebel',
        year        = '2013c',
        shortname   = 'FRE13c',
        loc         = 'MX',
        obs_table   = 'table2',
        param_table = 'table3',
        abund_table = 'table4_6',
    )
    return df

def load_frebel2014() -> pd.DataFrame:
    """
    Load the Frebel et al. (2014) data for the Segue 1 Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table

    Note: J100714+160154 is an s-process star.
    """
    df = load_authorYYYYx(
        author      = 'Frebel',
        year        = '2014',
        shortname   = 'FRE14',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table4',
    )
    return df

def load_frebel2016() -> pd.DataFrame:
    """
    Load the Frebel et al. (2016) data for the Bootes I Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations & Stellar Parameters
    Table 3 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Frebel',
        year        = '2016',
        shortname   = 'FRE16',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table1',
        abund_table = 'table3',
    )
    return df

def load_gilmore2013a() -> pd.DataFrame:
    """
    Load the Gilmore et al. (2013) data for the Bootes I Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Gilmore',
        year        = '2013a',
        shortname   = 'GIL13a',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table3_mod',
        abund_table = 'table6_mod',
    )
    return df

def load_gull2021() -> pd.DataFrame:
    """
    Load the Gull et al. (2021) data for the Helmi debris stream,Helmi trail stream,and omega Centauri stream.
    
    Helmi debris stream (Helmi et al. 1999)
        Helmi & White (1999) found 13 members of the now so-called debris stream.
        Roederer et al. (2010) performed a detailed abundance analysis of 12 of those 13 members.
        The Helmi debris stars manifest themselves in a well-defined stream,
         with prominent negative vz motion (Myeong et al. 2019).
    
    Helmi trail stream (Helmi et al. 1999)
        Chiba & Beers (2000) 9 stars apart of a secondary stream associated with the Helmi debris stream trail stream.
        The Helmi trail stream distinguishes itself from the Helmi debris stream kinematically (Yuan et al. 2020). 
         by displaying a positive vz (vertical velocity) motions,slightly higher energy,larger radial motions,
         and are more diffuse without clear features on kinematic diagrams

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 5 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Gull',
        year        = '2021',
        shortname   = 'GUL21',
        loc         = 'SS',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table5'
    )
    return df

def load_hansent2017() -> pd.DataFrame:
    """
    Load the Hansen et al. (2017) data for the Tucana III Ultra-Faint Dwarf Galaxy.

    Table 3a - Observations and Stellar Parameters
    Table 3b - Abundance Table

    Note: The paper used the Grevesse & Sauval (1998) solar abundances. -- Published in a book chapter in 2000.
    """
    df = load_authorYYYYx(
        author      = 'HansenT',
        year        = '2017',
        shortname   = 'HANt17',
        loc         = 'UF',
        obs_table   = 'table3_a',
        param_table = 'table3_a',
        abund_table = 'table3_b',
        transpose = True,
        transpose_abund = '[X/Fe]',
        transpose_err   = 'e_[X/Fe]',
    )
    return df

def load_hansent2020a() -> pd.DataFrame:
    """
    Load the Hansen T. et al. (2020a) data for the Grus II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 5 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'HansenT',
        year        = '2020a',
        shortname   = 'HANt20a',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table2',
        abund_table = 'table5',
    )
    return df

def load_hansent2024() -> pd.DataFrame:
    """
    Load the Hansen T. et al. (2024) data for the Tucana V Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation Table
    Table 2 - Stellar Parameters
    Table 4 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'HansenT',
        year        = '2024',
        shortname   = 'HANt24',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table2',
        abund_table = 'table4',
    )
    return df

def load_hughes2026() -> pd.DataFrame:
    """
    Load the Hughes et al. (2025) data for the 10 SASS stars.

    table_obs - Observations
    table_param - Stellar Parameters
    table_abund - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Hughes',
        year        = '2026',
        shortname   = 'HUG26',
        loc         = 'HA',
        obs_table   = 'table_obs',
        param_table = 'table_param',
        abund_table = 'table_abund',
        init_abund  = '[X/H]'
    )
    return df

def load_ishigaki2014() -> pd.DataFrame:
    """
    Load the Ishigaki et al. (2014) data for the Bootes I Ultra-Faint Dwarf Galaxy and two halo reference stars.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 5 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ishigaki',
        year        = '2014',
        shortname   = 'ISH14',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table5'
    )
    return df

def load_ito2013() -> pd.DataFrame:
    """
    Load the data from Ito et al. (2013) for the star: BD+44 493

    Table 0 - Observation Table
    Table 9 - Stellar Parameters Table
    Table 4 - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Ito',
        year        = '2013',
        shortname   = 'ITO13',
        loc         = '',
        obs_table   = 'table0',
        param_table = 'table9',
        abund_table = 'table10',
    )
    return df

def load_ji2016a() -> pd.DataFrame:
    """
    Load the Ji et al. (2016a) data for the Bootes II Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ji',
        year        = '2016a',
        shortname   = 'JI16a',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table4',
    )
    return df

def load_ji2016b() -> pd.DataFrame:
    """
    Load the Ji et al. (2016b) data for the Reticulum II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observation and Stellar Parameters
    Table 3 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ji',
        year        = '2016b',
        shortname   = 'JI16b',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table1',
        abund_table = 'table3',
    )
    return df

def load_ji2018() -> pd.DataFrame:
    """
    Load the Ji et al. (2018) data for the brightest star in Reticulum II.

    Table 0 - Observations & Stellar Parameters
    Table 2 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ji',
        year        = '2018',
        shortname   = 'JI18',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table2_modified',
    )
    return df

def load_ji2019a() -> pd.DataFrame:
    """
    Load the Ji et al. (2019a) data for the Grus I & Triangulum II Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ji',
        year        = '2019a',
        shortname   = 'JI19a',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table4',
    )
    return df

def load_ji2020a() -> pd.DataFrame:
    """
    Load the Ji et al. (2020a) data for the Carina II and Carina III Ultra-Faint Dwarf Galaxies.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 6 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ji',
        year        = '2020a',
        shortname   = 'JI20a',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table6',
    )
    return df

def load_ji2020b() -> pd.DataFrame:
    """
    Load the Ji et al. (2020) data for the 7 stellar streams in the Milky Way.
    These streams include: ATLAS,Aliqa Uma,Chenab,Elqui,Indus,Jhelum,and Phoenix

    Table 1 - Observations
    Table 2 - Stellar Parameters
    Table 6 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ji',
        year        = '2020b',
        shortname   = 'JI20b',
        loc         = 'SS',
        obs_table   = 'table1',
        param_table = 'table2',
        abund_table = 'table6'
    )
    return df

def load_ji2026() -> pd.DataFrame:
    """
    Load the Ji et al. (2026) data for a Large Magellanic Cloud star.

    Table 0 - Observations
    Table 0 - Stellar Parameters
    Table 1 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ji',
        year        = '2026',
        shortname   = 'JI26',
        loc         = 'DW',
        obs_table   = 'table0_mod',
        param_table = 'table0_mod',
        abund_table = 'table1_mod',
    )
    df.loc[df['Name'] == 'J0715-7334_NLTE','I/O'] = 0
    return df

def load_kirby2017b() -> pd.DataFrame:
    """
    Load the Kirby et al. (2017b) data for the Triangulum II Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters
    Table 6 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Kirby',
        year        = '2017b',
        shortname   = 'KIR17b',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table6',
    )
    return df

def load_koch2008c() -> pd.DataFrame:
    """
    Load the Koch et al. (2008c) data for the Hercules Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters
    Table 1 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Koch',
        year        = '2008c',
        shortname   = 'KOC08c',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table1',
        init_abund  = '[X/Fe]',
        solar_init = 'asplund2005'
    )
    return df

def load_koch2013b() -> pd.DataFrame:
    """
    Load the Koch et al. 2013b data for the Hercules Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters (custom made table from the text and Aden+2011)
    Table 1, modified - Abundance Table (chose to use the 3-sigma upper limits for Ba)
    
    Note: [Fe/H] and [Ca/H] are taken from Adén et al. (2011).
    """
    df = load_authorYYYYx(
        author      = 'Koch',
        year        = '2013b',
        shortname   = 'KOC13b',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table1_mod'
    )
    return df

def load_lai2011b() -> pd.DataFrame:
    """
    Load the Lai et al. (2011b) data for the Bootes I Ultra-Faint Dwarf Galaxy.

    Table 1a - Observations Table & Stellar Parameters
    Table 1b - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Lai',
        year        = '2011b',
        shortname   = 'LAI11b',
        loc         = 'UF',
        obs_table   = 'table1a',
        param_table = 'table1a',
        abund_table = 'table1b',
        init_abund  = '[X/Fe]'
    )
    return df

def load_lemasle2012() -> pd.DataFrame:
    """
    Load the Lemasle et al. (2012) data for the Carina Classical Dwarf Galaxy.

    Table 3 - Observations
    Table 5 - Stellar Parameters
    Table 7 & 8 - Abundance Tables
    """
    df = load_authorYYYYx(
        author      = 'Lemasle',
        year        = '2012',
        shortname   = 'LEM12',
        loc         = 'DW',
        obs_table   = 'table3',
        param_table = 'table5',
        abund_table = 'table7_8',
        transpose   = True,
        transpose_abund = '[X/H]',
        transpose_err   = 'e_[X/H]'
    )
    return df

def load_lemasle2014() -> pd.DataFrame:
    """
    Load the Lemasle et al. (2014) data for the Fornax dwarf spheroidal galaxy.

    Table A.3 - Observation Parameters
    Table 3 - Stellar Parameters
    Table A.5 - Abundance Table

    Note: Which solar abundances used in this dataset is unclear/not mentioned in the paper. Assuming they follow Aslpund et al. 2009
    """
    df = load_authorYYYYx(
        author      = 'Lemasle',
        year        = '2014',
        shortname   = 'LEM14',
        loc         = 'DW',
        obs_table   = 'tablea3',
        param_table = 'table3',
        abund_table = 'tablea5',
        transpose   = True,
        transpose_abund = '[X/H]',
        transpose_err   = 'e_[X/H]'
    )
    return df

def load_letarte2010() -> pd.DataFrame:
    """
    Load the Letarte et al. (2010) data for the Fornax Classical Dwarf Galaxy.

    Table A.2 & A.3 - Observations & Stellar Parameters
    Table A.5 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Letarte',
        year        = '2010',
        shortname   = 'LET10',
        loc         = 'DW',
        obs_table   = 'tablea3',
        param_table = 'tablea2',
        abund_table = 'tablea5',
        solar_init  = 'letarte2010',
        transpose   = True,
        transpose_abund = '[X/Fe]',
        transpose_err   = 'e_[X/Fe]'
    )
    return df

def load_limberg2025a() -> pd.DataFrame:
    """
    Load the Limberg et al. (2025a) data for a Large Magellanic Cloud star

    Table 0 - Observations
    Table 0 - Stellar Parameters
    Table 2 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Limberg',
        year        = '2025a',
        shortname   = 'LIM25a',
        loc         = 'DW',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table2',
    )
    return df

def load_lucchesi2024() -> pd.DataFrame:
    """
    Load the Lucchesi et al. (2024) data for the Carina & Fornax dSph galaxies. There
    are 4 stars in Carina and 2 stars in Fornax.

    Table 0 - Observations & Stellar Parameters (created from Table 1,2,3)
    Table A.4 - Abundance Table (restructured from the original Table A.4)
    """
    df = load_authorYYYYx(
        author      = 'Lucchesi',
        year        = '2024',
        shortname   = 'LUC24',
        loc         = 'DW',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'tablea4'
    )
    return df

def load_lucey2026b() -> pd.DataFrame:
    """
    Load the Lucey et al. 2026 data for the first five CEMP stars in the Large Magellanic Cloud.
    
    Table 1, obs_param - Observations Table & Stellar Parameters
    Table 1, abund - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Lucey',
        year        = '2026b',
        shortname   = 'LUCm26b',
        loc         = 'DW',
        obs_table   = 'table1_obs_param',
        param_table = 'table1_obs_param',
        abund_table = 'table1_abund',
        transpose   = True,
        transpose_abund = '[X/Fe]',
        transpose_err   = 'e_[X/Fe]'
    )
    return df

def load_mardini2022b() -> pd.DataFrame:
    """
    Load the Mardini et al. (2022b) data for a SASS star.

    Table 0 - Observations & Stellar Parameters
    Table 2 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Mardini',
        year        = '2022b',
        shortname   = 'MARm22b',
        loc         = 'HA',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table2'
    )
    return df

def load_mardini2024b() -> pd.DataFrame:
    """
    Load the Mardini et al. (2024b) data for a single Atari star.

    Table 1 - Observations & Stellar Parameters
    Table 2 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Mardini',
        year        = '2024b',
        shortname   = 'MARm24b',
        loc         = 'DS',
        obs_table   = 'table1',
        param_table = 'table1',
        abund_table = 'table2'
    )
    return df

def load_marshall2019() -> pd.DataFrame:
    """
    Load the Marshall et al. (2019) data for the Tucana III Ultra-Faint Dwarf Galaxy.

    Table 1 - Observations
    Table 2 - Stellar Parameters
    Table 4 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Marshall',
        year        = '2019',
        shortname   = 'MARj19',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table2',
        abund_table = 'table4'
    )
    return df

def load_martin2022a() -> pd.DataFrame:
    """
    Load the Martin et al. (2022a) data for the C-19 Stream.

    Table 2 - Observations Table
    Table 3 - Abundance Table 1, for Gemini/GRACES observations
    Table 5 - Abundance Table 2, for OSIRIS observations
    """
    df = load_authorYYYYx(
        author      = 'Martin',
        year        = '2022a',
        shortname   = 'MARn22a',
        loc         = 'SS',
        obs_table   = 'table2',
        param_table = 'table3_5_param',
        abund_table = 'table3_5_abund_xh',
        transpose   = True,
        transpose_abund = '[X/H]',
        transpose_err   = 'e_[X/H]'
    )
    return df

def load_nagasawa2018() -> pd.DataFrame:
    """
    Load the Nagasawa et al. (2018) data for the Horologium I Ultra-Faint Dwarf Galaxy.

    Table 1 & Table 2 - Observations
    Table 4 - Stellar Parameters
    Table 5 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Nagasawa',
        year        = '2018',
        shortname   = 'NAG18',
        loc         = 'UF',
        obs_table   = 'table1_2',
        param_table = 'table4',
        abund_table = 'table5'
    )
    return df

def load_nordlander2019() -> pd.DataFrame:
    """
    Load the Nordlander et al. (2019) data for the halo/SASS star SMSS J160540.18-144323.1 (SMSS 1605-1443).

    Table 0 - Observations
    Table 0 - Stellar Parameters
    Table 1 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Nordlander',
        year        = '2019',
        shortname   = 'NORt19',
        loc         = 'HA',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table1'
    )
    return df

def load_norris2010a() -> pd.DataFrame:
    """
    Load the Norris et al. (2010a) data for the Bootes I (Boo-1137) Ultra-Faint Dwarf Galaxy.

    Table 0 - Observation and Stellar Parameters
    Table 2 - Abundance Table

    Note: The abundance ratios are using the Asplund+2005 solar abundances, not the Asplund+2009 solar abundances.
    """
    df = load_authorYYYYx(
        author      = 'Norris',
        year        = '2010a',
        shortname   = 'NOR10a',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table2',
        solar_init = 'asplund2005'
    )
    return df

def load_norris2010b() -> pd.DataFrame:
    """
    Load the Norris et al. (2010b) data for the Segue 1 (Seg 1-7) Ultra-Faint Dwarf Galaxy.

    Table 0 - Observation and Stellar Parameters
    Table 2 - Abundance Table

    Note: Which solar abundances are used is not stated in the text, although I assume they are the Asplund+2005 solar abundances. Not the Asplund+2009 solar abundances.
    """
    df = load_authorYYYYx(
        author      = 'Norris',
        year        = '2010b',
        shortname   = 'NOR10b',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table2',
        solar_init = 'asplund2005'
    )
    return df

def load_norris2010c() -> pd.DataFrame:
    """
    Load the Norris et al. (2010c) data for the Bootes I and Segue 1 Ultra-Faint Dwarf Galaxies.

    Table 1,2 - Observations Table
    Table 3,4 - Stellar Parameters
    Table 3,4 - Abundance Table

    Note: Which solar abundances are used is not stated in the text, although I assume they are the Asplund+2005 solar 
          abundances. Not the Asplund+2009 solar abundances.
    """
    df = load_authorYYYYx(
        author      = 'Norris',
        year        = '2010c',
        shortname   = 'NOR10c',
        loc         = 'UF',
        obs_table   = 'table1_2',
        param_table = 'table3_4_param',
        abund_table = 'table3_4_abund',
        solar_init  = 'asplund2005',
        transpose   = True,
        transpose_abund = '[X/H]',
        transpose_err   = 'e_[X/H]'
    )
    return df

def load_norris2017b() -> pd.DataFrame:
    """
    Load the Norris et al. (2017b) data for the Carina Classical Dwarf Spheroidal Galaxies.
    Paper reports on 63 stars, but only 32 new stars are from this study. The other 31
    stars are from Venn+2012, Shetrone+2003, and Lemasle+2012.

    Table 1 - Observations
    Table 5 - Stellar Parameters
    Table 6 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Norris',
        year        = '2017b',
        shortname   = 'NOR17b',
        loc         = 'DW',
        obs_table   = 'table1',
        param_table = 'table5',
        abund_table = 'table6',
    )
    return df

def load_ou2024c() -> pd.DataFrame:
    """
    Load the Ou et al. (2024c) data for the Gaia Sausage Enceladus (GSE) Dwarf Galaxy star.
    
    obs_param - Observations Table & Stellar Parameters
    xh_abund - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ou',
        year        = '2024c',
        shortname   = 'OUx24c',
        loc         = 'DW',
        obs_table   = 'obs_param',
        param_table = 'obs_param',
        abund_table = 'xh_abund',
        transpose   = True,
        transpose_abund = '[X/H]',
        transpose_err   = 'e_[X/H]'
    )
    return df

def load_ou2025() -> pd.DataFrame:
    """
    Load the Ou et al. (2025) data for the Sagittarius Dwarf Galaxy.
    
    obs_param - Observations Table & Stellar Parameters
    xh_abund - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Ou',
        year        = '2025',
        shortname   = 'OUx25',
        loc         = 'DW',
        obs_table   = 'obs_param',
        param_table = 'obs_param',
        abund_table = 'xh_abund',
        transpose   = True,
        transpose_abund = '[X/H]',
        transpose_err   = 'e_[X/H]'
    )
    return df

def load_reggiani2021() -> pd.DataFrame:
    """
    Load the Reggiani et al. (2021) data for the Small and Large Magellanic Clouds.

    Table 1 - Observation Table
    Table 3 - Stellar Parameters
    Table 5 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Reggiani',
        year        = '2021',
        shortname   = 'REG21',
        loc         = 'DW',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table5',
    )
    return df

def load_roederer2010a() -> pd.DataFrame:
    """
    Load the Roederer et al. (2010a) data for the Helmi stellar stream.

    Table 2 - Observations
    Table 5 - Stellar Parameters
    Table 7,8,9,10 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2010a',
        shortname   = 'ROE10a',
        loc         = 'SS',
        obs_table   = 'table2',
        param_table = 'table5',
        abund_table = 'table7-8-9-10'
    )
    return df

def load_roederer2010b() -> pd.DataFrame:
    """
    Load the Roederer et al. (2010b) for the star: BD +17 3248

    Table 0 - Observation Table
    Table 0 - Stellar Parameters
    Table 1b - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2010b',
        shortname   = 'ROE10b',
        loc         = '',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table1b',
    )
    return df

def load_roederer2012a() -> pd.DataFrame:
    """
    Load the data from Roederer et al. (2012d) for three stars will Tellurium abundances.

    Table 0 - Observation Table & Stellar Parameters Table
    abund - Abundance Table, extracted from the text and concatenated with abundance tables from Roederer et al. (2012d) and Cowan et al. (2002).
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2012a',
        shortname   = 'ROE12a',
        loc         = '',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'abund',
    )
    return df

def load_roederer2012b() -> pd.DataFrame:
    """
    Load the data from Roederer et al. (2012b) for HD 160617.

    Table 0 - Observation Table & Stellar Parameters Table
    Table 15 - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2012b',
        shortname   = 'ROE12b',
        loc         = '',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table15',
    )
    return df

def load_roederer2012c() -> pd.DataFrame:
    """
    Load the data from Roederer et al. (2012c) for six stars with germanium, arsenic, and selenium.

    Table 0 - Observation Table
    Table 2 - Stellar Parameters Table
    Table 4,5 - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2012c',
        shortname   = 'ROE12c',
        loc         = '',
        obs_table   = 'table0',
        param_table = 'table2',
        abund_table = 'table4_5',
    )
    return df

def load_roederer2012d() -> pd.DataFrame:
    """
    Load the data from Roederer et al. (2012d) for four stars with heavy-element abundances.

    Table 0 - Observation Table
    Table 6 - Stellar Parameters Table
    Table 7,8 - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2012d',
        shortname   = 'ROE12d',
        loc         = '',
        obs_table   = 'table0',
        param_table = 'table6',
        abund_table = 'table7_8',
    )
    return df

def load_roederer2014a() -> pd.DataFrame:
    """
    Load the data from Roederer et al. (2014a) for 16 stars, including neutron-capture elements.
    
    Table 2 - Observation Table
    Table 3 - Stellar Parameters Table
    Table 5 through Table 20 - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2014a',
        shortname   = 'ROE14a',
        loc         = 'MX',
        obs_table   = 'table2_obs',
        param_table = 'table3',
        abund_table = 'table5-20',
    )
    return df

def load_roederer2014b() -> pd.DataFrame:
    """
    Load the Roederer et al. (2014b) data for the Segue 2 Ultra-Faint Dwarf Galaxy.

    Table 0 - Observations & Stellar Parameters
    Table 3 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2014b',
        shortname   = 'ROE14b',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table3'
    )
    return df

def load_roederer2014c() -> pd.DataFrame:
    """
    Load the data from Roederer et al. (2014c) for 313 stars.

    Table 3 - Observation Table
    Table 7 - Stellar Parameters Table
    Table 12 - Abundance Table
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2014c',
        shortname   = 'ROE14c',
        loc         = 'MX',
        obs_table   = 'table3_obs',
        param_table = 'table7',
        abund_table = 'table12',
    )
    return df

def load_roederer2014d() -> pd.DataFrame:
    """
    Load the data from Roederer et al. (2014d) for 2 stars: HD 108317 & HD 128279

    Table 0 - Observation Table & Stellar Parameters Table
    Table 4 - Abundance Table, in addition to concatenated abundances from Roederer et al. (2012d)
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2014d',
        shortname   = 'ROE14d',
        loc         = 'MX',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table4_all_abund',
    )
    return df

def load_roederer2014e() -> pd.DataFrame:
    """
    Load the data from Roederer et al. (2014e) for 14 stars with phosphorus abundances.

    Table 0 - Observation Table
    Table 0 - Stellar Parameters Table
    Table 4 - Abundance Table, in addition to concatenated abundances from Roederer et al. (2012d)
    """
    
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2014e',
        shortname   = 'ROE14e',
        loc         = 'MX',
        obs_table   = 'table0',
        param_table = 'table4',
        abund_table = 'table6_7_T',
    )
    return df

def load_roederer2016b() -> pd.DataFrame:
    """
    Load the Roederer et al. (2016b) data for stars in Reticulum II.

    Table 1 - Observations
    Table 4 - Stellar Parameters
    Table 6,7 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2016b',
        shortname   = 'ROE16b',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table4',
        abund_table = 'table67',
    )
    return df

def load_roederer2019d() -> pd.DataFrame:
    """
    Load the Roederer et al. (2019d) data for the Sylgr stellar stream.

    Table 1 - Observations & Stellar Parameters
    Table 3 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2019d',
        shortname   = 'ROE19d',
        loc         = 'SS',
        obs_table   = 'table1',
        param_table = 'table1',
        abund_table = 'table3'
    )
    return df

def load_roederer2023a() -> pd.DataFrame:
    """
    Load the Roederer et al. (2023a) data for Sextans stars.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 5,6,7 - Abundance Table (There are two versions of the table. One has LTE abundances and the other has NLTE abundances.
                  We use the LTE abundances for consistency with the rest of the literature.
    """
    df = load_authorYYYYx(
        author      = 'Roederer',
        year        = '2023a',
        shortname   = 'ROE23a',
        loc         = 'DW',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table5_6_7_lte',
    )
    return df

def load_ruchti2011a() -> pd.DataFrame:
    """
    Load the data from Ruchti et al. (2011a).
    
    Table 1 - Observation Table
    Table 2 - Stellar Parameters Table
    Table 2 - Abundance Table
    
    Note: To load the available data, we have made the assumption that [FeI/H] == [FeII/H], though we know this is not true.
          The original paper does not report the [FeI/H] abundances, only the [FeII/H] abundances. However, the [X/Fe] abundances 
          are reported for both FeI and FeII species, so we have assumed that the [FeI/H] abundances are equal to the [FeII/H] 
          abundances in order to calculate the [X/Fe] abundances for all elements.
    """
    
    df = load_authorYYYYx(
        author      = 'Ruchti',
        year        = '2011a',
        shortname   = 'RUC11a',
        loc         = 'MX',
        obs_table   = 'table1',
        param_table = 'table2_param',
        abund_table = 'table2_abund_assumed_T',
    )
    return df

def load_sakari2018b() -> pd.DataFrame:
    """
    Load the data from Sakari et al. (2018b) from the RPA paper: "The R-Process Alliance: First 
    Release from the Northern Search for r-process-enhanced Metal-poor Stars in the Galactic Halo"
    
    Table 1 - Observation Table
    Table 9 - Stellar Parameters Table
    Table 3,5,6,7 - Abundance Table
    
    Note: The carbon abundances have already been corrected, following Placco et al. (2014c).
    """
    
    df = load_authorYYYYx(
        author      = 'Sakari',
        year        = '2018b',
        shortname   = 'SAK18b',
        loc         = 'MX',
        obs_table   = 'table1',
        param_table = 'table9',
        abund_table = 'table3_5_6_7_T',
    )
    return df

def load_sbordone2007() -> pd.DataFrame:
    """
    Load the Sbordone et al. (2007) data for the Sagittarius dSph galaxy and Terzan 7 (globular cluster).

    Table 1 - Observation and Stellar Parameters
    Table 4,5,6 - Abundance Tables (merged into one table)
    """
    df = load_authorYYYYx(
        author      = 'Sbordone',
        year        = '2007',
        shortname   = 'SBD07',
        loc         = 'MX',
        obs_table   = 'table1',
        param_table = 'table1',
        abund_table = 'table456a_long'
    )
    return df

def load_sestito2024b() -> pd.DataFrame:
    """
    Load the data from Sestito et al. (2024b) for stars in the Sagittarius dwarf galaxy. 
    This is from the PIGS IX survey.
    
    Table 1 - Observations Table
    Table 2 - Stellar Parameters
    XH Table - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Sestito',
        year        = '2024b',
        shortname   = 'SES24b',
        loc         = 'DW',
        obs_table   = 'table1',
        param_table = 'table2',
        abund_table = 'xh_abund',
        transpose   = True,
        transpose_abund = '[X/H]',
        transpose_err   = 'e_[X/H]'
    )
    return df

def load_sestito2024d() -> pd.DataFrame:
    """
    Load the data from Sestito et al. (2024d) for stars in the Sagittarius dwarf galaxy. 
    This is low/med-resolution photometry from the PIGS X survey.
    
    obs_param_table - Observations Table
    obs_param_table - Stellar Parameters
    abund_table - Abundance Table
    
    Note: Using Asplund et al. (2009) solar abundances, following their usage in Sestito et al. (2024b).
    """
    df = load_authorYYYYx(
        author      = 'Sestito',
        year        = '2024d',
        shortname   = 'SES24d',
        loc         = 'DW',
        obs_table   = 'obs_param_table',
        param_table = 'obs_param_table',
        abund_table = 'abund_table',
        transpose   = True,
        transpose_abund = '[X/Fe]',
        transpose_err   = 'e_[X/Fe]'
    )
    return df

def load_shetrone2003() -> pd.DataFrame:
    """
    Load the Shetrone et al. (2003) data for Carina, Fornax, Leo I, Sculptor, M30, M55, and M68.
    
    Note: M55-283 and M55-76 do not have coordinates in any of the referenced work. Thus, I
    compared the image of M55 (NGC 6809) from Alcaino, G. 1975, "The Globular Cluster NGC 6809" 
    (Figure 2) to the Aladin Sky Atlas to identify the Gaia and 2MASS observations and extract
    the appropriate coordinates and identifiers for these two stars.

    Table 0 - Observations
    Table 5 - Stellar Parameters
    Table 7,8,9,10 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Shetrone',
        year        = '2003',
        shortname   = 'SHE03',
        loc         = 'DW',
        obs_table   = 'table0',
        param_table = 'table5',
        abund_table = 'table7_8_9_10',
        init_abund  = '[X/Fe]'
    )
    return df

def load_simon2010() -> pd.DataFrame:
    """
    Load the Simon et al. (2010) data for the Leo IV Ultra-Faint Dwarf Galaxies.

    Table 0 - Observations & Stellar Parameters
    Table 2 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Simon',
        year        = '2010',
        shortname   = 'SIM10',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table2'
    )
    return df

def load_spite2018() -> pd.DataFrame:
    """
    Load the Spite et al. (2018) data for the Pisces II Ultra-Faint Dwarf Galaxy.

    Table 0 - Observations & Stellar Parameters
    Table 2 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Spite',
        year        = '2018',
        shortname   = 'SPT18',
        loc         = 'UF',
        obs_table   = 'table0',
        param_table = 'table0',
        abund_table = 'table2'
    )
    return df

def load_venn2012() -> pd.DataFrame:
    """
    Load the Venn et al. (2012) data for the Carina Classical Dwarf Spheroidal Galaxies.

    Table 1 - Observations
    Table 6 - Stellar Parameters
    Table 10,11,12,13 - Abundance Table
    
    Note: [FeII/FeI] abundances are provided in the table, instead of [FeII/H].
    """
    df = load_authorYYYYx(
        author      = 'Venn',
        year        = '2012',
        shortname   = 'VEN12',
        loc         = 'DW',
        obs_table   = 'table1',
        param_table = 'table6',
        abund_table = 'table10_11_12_13',
        init_abund  = '[X/Fe]'
    )
    return df

def load_waller2023() -> pd.DataFrame:
    """
    Load the Waller et al. (2023) data for the Segue 1 Ultra-Faint Dwarf Galaxy.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 7 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Waller',
        year        = '2023',
        shortname   = 'WAL23',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table7',
        init_abund  = '[X/Fe]',
    )
    return df

def load_webber2023() -> pd.DataFrame:
    """
    Load the Webber et al. (2023) data for the Cetus II Ultra-Faint Dwarf Galaxy.

    Table 1 - Observations
    Table 3 - Stellar Parameters
    Table 4 - Abundance Table
    """
    df = load_authorYYYYx(
        author      = 'Webber',
        year        = '2023',
        shortname   = 'WEB23',
        loc         = 'UF',
        obs_table   = 'table1',
        param_table = 'table3',
        abund_table = 'table4'
    )
    return df

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