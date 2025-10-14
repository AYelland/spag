# coding: utf-8

""" Utility functions from extracting and manipulating data """

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
import pandas as pd


################################################################################
## Stellar Abundance Classification functions

def classify_metallicity(FeH):
    """
    Classify the star by its metallicity, based on Frebel et al. 2018 (Table 1).
    """
    metallicity_str = ''

    if FeH > 0.0:
        metallicity_str = 'MR'
    elif FeH <= 0.0 and FeH > -1.0:
        metallicity_str = 'SUN'
    elif FeH <= -1.0 and FeH > -2.0:
        metallicity_str = 'MP'
    elif FeH <= -2.0 and FeH > -3.0:
        metallicity_str = 'VMP'
    elif FeH <= -3.0 and FeH > -4.0:
        metallicity_str = 'EMP'
    elif FeH <= -4.0 and FeH > -5.0:
        metallicity_str = 'UMP'
    elif FeH <= -5.0 and FeH > -6.0:
        metallicity_str = 'HMP'
    elif FeH <= -6.0 and FeH > -7.0:
        metallicity_str = 'MMP'
    elif FeH <= -7.0 and FeH > -8.0:
        metallicity_str = 'SMP'
    elif FeH <= -8.0 and FeH > -9.0:
        metallicity_str = 'OMP'
    elif FeH <= -9.0 and FeH > -10.0:
        metallicity_str = 'GMP'
    elif FeH <= -10.0:
        metallicity_str = 'RMP'
    else:
        metallicity_str = 'NaN'
    
    return metallicity_str

def classify_neutron_capture(EuFe = np.nan, BaFe = np.nan, SrFe = np.nan, PbFe = np.nan, LaFe = np.nan, HfFe = np.nan, IrFe = np.nan):
    """
    Classify the star by its neutron-capture abundance pattern, based on Frebel et al. 2018 (Table 1) & Holmbeck et al. 2020 (Section 4.1).
    """
    ncap_str = ''

    BaEu = BaFe - EuFe
    BaPb = BaFe - PbFe
    SrBa = SrFe - BaFe
    SrEu = SrFe - EuFe
    LaEu = LaFe - EuFe
    HfIr = HfFe - IrFe

    # if (EuFe < 0.4):
    #     ncap_str += ', ' if ncap_str else ''
    #     ncap_str += 'R0' #if (EuFe <= 0.3) else '~R0'
    if (EuFe > 0.3 and EuFe <= 0.7) and (BaEu < 0.0):
        ncap_str += ', ' if ncap_str else ''
        ncap_str += 'R1'
    if (EuFe > 0.7) and (BaEu < 0.0):
        ncap_str += ', ' if ncap_str else ''
        ncap_str += 'R2'
    if (EuFe < 0.3) and (SrBa > 0.5 and SrEu > 0.0):
        ncap_str += ', ' if ncap_str else ''
        ncap_str += 'RL'
    if (BaFe > 1.0) and (BaEu > 0.5):
        if ~pd.isna(BaPb):
            if (BaPb > -1.5):
                ncap_str += ', ' if ncap_str else ''
                ncap_str += 'S'
        else:
            ncap_str += ', ' if ncap_str else ''
            ncap_str += 'S'
    if (BaEu > 0.0 and BaEu < 0.5) and (BaPb > -1.0 and BaPb < -0.5):
        ncap_str += ', ' if ncap_str else ''
        ncap_str += 'RS'
    if (LaEu > 0.0 and LaEu < 0.6) and (HfIr > 0.7 and HfIr < 1.3):
        ncap_str += ', ' if ncap_str else ''
        ncap_str += 'I'

    return ncap_str

def classify_carbon_enhancement(CFe=np.nan, BaFe=np.nan, ulCFe=False, llCFe=False):
    """
    Classify the star by its carbon enhancement, based on Frebel et al. 2018 (Table 1).
    """
    cemp_str = ''
    
    # if CFe is NaN, it will not contribute to the classification
    if pd.isna(CFe) or CFe == '':
        return cemp_str
    
    threshold = 0.7
    if CFe > threshold:
        if (BaFe < 0.0) and cemp_str == '':
            cemp_str += 'NO' # Neutron-capture-normal
        else:
            cemp_str += 'CE' # Carbon-enhanced
    # else:
    #     cemp_str = 'C' # Carbon-poor
    
    return cemp_str

def classify_alpha_enhancement(MgFe, SiFe, CaFe, TiFe):
    """
    Classify the star by its alpha-enhancement, based on Frebel et al. 2018 (Table 1).
    """

    assert not any(pd.isna(val) for val in [MgFe, SiFe, CaFe, TiFe]), "One or more input values are NaN"

    alpha_str = ''

    alphaFe = np.nanmean([MgFe, SiFe, CaFe, TiFe])
    if (alphaFe > 0.35 and alphaFe < 0.45):
        alpha_str = 'alpha'

    return alpha_str

def combine_classification(df, c_key_col='C_key', ncap_key_col='Ncap_key', output_col='Class'):
    """
    Combine carbon and neutron-capture classifications into a unified CEMP classification.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        c_key_col (str): Column name for carbon classification (e.g., 'C_key').
        ncap_key_col (str): Column name for neutron-capture classification (e.g., 'Ncap_key').
        output_col (str): Name of the new combined classification column.

    Returns:
        pd.DataFrame: DataFrame with a new column `output_col`.
    """

    def classify(c_key, ncap_key):
        c_key = np.nan if c_key == '' else c_key
        ncap_key = np.nan if ncap_key == '' else ncap_key
        if pd.notna(c_key) and pd.notna(ncap_key):
            combo = c_key + '+' + ncap_key
            mapping = {
                'CE+RS': 'CEMP-r/s',
                'CE+S': 'CEMP-s',
                'CE+I': 'CEMP-i',
                'CE+R1': 'CEMP-rI',
                'CE+R2': 'CEMP-rII',
                'CE+RL': 'CEMP-r-lim'
            }
            return mapping.get(combo, combo)
        elif pd.notna(c_key):
            if c_key == 'CE':
                return 'CEMP'
            elif c_key == 'NO':
                return 'CEMP-no'
            else:
                return c_key
        elif pd.notna(ncap_key):
            return {
                'R1': 'rI',
                'R2': 'rII',
                'S': 's',
                'RS': 'r/s',
                'I': 'i',
                'RL': 'r-lim'
            }.get(ncap_key, ncap_key)
        else:
            return ''

    df[output_col] = df.apply(lambda row: classify(row[c_key_col], row[ncap_key_col]), axis=1)
    
    return df