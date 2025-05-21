# coding: utf-8

""" Utility functions from extracting and manipulating data """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from astropy import table
from astropy import coordinates as coord
from astropy import units as u

import warnings

from six import string_types

import os
basepath = os.path.dirname(__file__)
datapath = os.path.join(basepath,"data")

## Regular expressions
import re
m_XH = re.compile('\[(\D+)/H\]')
m_XFe= re.compile('\[(\D+)/Fe\]')

# SPAG imports
from spag.convert import *
import spag.periodic_table  as pt
import spag.read_data as rd


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

def classify_carbon_enhancement(CFe=np.nan, BaFe=np.nan):
    """
    Classify the star by its carbon enhancement, based on Frebel et al. 2018 (Table 1).
    """
    cemp_str = ''
    
    # if CFe is NaN, it will not contribute to the classification
    if pd.isna(CFe) or CFe == '':
        return cemp_str
    
    if CFe > 0.7:
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
            return 'CEMP' if c_key == 'CE' else c_key
        elif pd.notna(ncap_key):
            return {
                'R1': 'rI',
                'R2': 'rII',
                'S': 's',
                'RS': 'r/s',
                'I': 'i',
                'RL': 'r-lim'
            }.get(ncap_key, ncap_key)
        return ''

    df[output_col] = df.apply(lambda row: classify(row[c_key_col], row[ncap_key_col]), axis=1)
    
    return df
    
################################################################################
## Formatting and converting datafiles

def align_ampersands(filename, start_line, end_line):
    # Read the file and split lines into a list
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Process only the lines in the specified range
    selected_lines = lines[start_line-1:end_line]
    
    # Split each line by "&" and calculate max widths for each column
    split_lines = [line.split('&') for line in selected_lines]
    max_col_widths = [max(len(col.strip()) for col in column) for column in zip(*split_lines)]
    
    # Reformat each line with aligned "&"
    aligned_lines = []
    for line_parts in split_lines:
        aligned_line = ' & '.join(col.strip().ljust(width) for col, width in zip(line_parts, max_col_widths))
        aligned_lines.append(aligned_line + '\n')
    
    # Replace original lines in range with aligned lines
    lines[start_line-1:end_line] = aligned_lines
    
    # Write the modified lines back to the file
    with open(filename, 'w') as file:
        file.writelines(lines)


################################################################################
## Rounding functions

from decimal import Decimal, ROUND_HALF_UP

def normal_round(value, precision=2):
    """
    value: float
        The value to round.
    precision: int
        The number of decimal places to round to.
    
    Rounds a value to a specified number of decimal places using half-up rounding.

    Returns:
        float: The rounded value.
    """
    value = Decimal(str(value)) # Convert to string first to avoid floating point precision issues
    multiplier = Decimal('1.' + '0' * precision)  # Decimal precision
    return float(value.quantize(multiplier, rounding=ROUND_HALF_UP))

def round_to_nearest(x, base=0.5, how="normal"):
    """
    x: float or array-like
        The number or numbers to be rounded.
    base: float (default: 0.5)
        The base value to which the number should be rounded.
    how: str (default: "normal")
        The method by which the number should be rounded. Options are:
        - "normal": round to the nearest multiple of the base value.
        - "ceiling": round up to the nearest multiple of the base value.
        - "floor": round down to the nearest multiple of the base value.
        
    Rounds a number or array-like set of numbers to the nearest multiple of the
    given `base` value. The `how` parameter can be used to specify whether the 
    number(s) should be rounded up, down, or to the nearest multiple.
    """
    
    if how == "normal":
        if isinstance(x, (list, np.ndarray)):
            return np.array([base * round(xi / base) for xi in x])
        else:
            return base * round(x / base)
    elif how == "ceiling":
        if isinstance(x, (list, np.ndarray)):
            return np.array([base * np.ceil(xi / base) for xi in x])
        else:
            return base * np.ceil(x / base)
    elif how == "floor":
        if isinstance(x, (list, np.ndarray)):
            return np.array([base * np.floor(xi / base) for xi in x])
        else:
            return base * np.floor(x / base)
    else:
        raise ValueError(f"Invalid rounding method: {how}")


def pad_and_round(value, precision):
    """
    value: float
        The value to round.
    precision: int
        The number of decimal places to round to.
    
    Rounds a value to a specified number of decimal places using half-up rounding,
    and pads the resulting string with zeros to the specified number of decimal places
    (if the rounded value has fewer decimal places).

    Returns:
        str: The rounded and padded value as a string.
    """
    return f"{normal_round(value, precision):.{precision}f}"

################################################################################
## String manipulation functions

from functools import reduce
from six import string_types

def get_common_letters(strlist):
    """
    strlist: list
        A list of strings (e.g., ['str1', 'str2',' 'str3'])
    
    Returns the common letters in the strings in the list, each in the same 
    position of the string. If there are no common letters, an empty string is
    returned. (e.g., ['Horse', 'House',' Harse'] -> 'Hse')
    """
    return "".join([x[0] for x in zip(*strlist) \
        if reduce(lambda a,b:(a == b) and a or None,x)])

def find_common_start(strlist):
    """
    strlist: list
        A list of strings (e.g., ['str1', 'str2',' 'str3'])
        
    Returns the common letters at the start of the strings in the list. If there
    are no common letters, an empty string is returned. 
    (e.g., ['Horse', 'House', 'Harse'] -> 'H')
    """
    strlist = strlist[:]
    prev = None
    while True:
        common = get_common_letters(strlist)
        if common == prev:
            break
        strlist.append(common)
        prev = common
    return get_common_letters(strlist)


################################################################################
## Plotting functions

def extend_limits(values, fraction=0.10, tolerance=1e-2, round_to_val=None):
    """
    values: list
        A list of data values, of which the minimum and maximum values will be
        used to extend the range covered by the values.
    fraction: float (default: 0.10)
        The fraction of the range by which to extend the values.
    tolerance: float (default: 1e-2)
        The tolerance for the difference between the new limits.
    round_to_val: float (default: None)
        If provided, the new limits will be rounded to the nearest multiple of
        this value.
        
    Extend viewable range covered by a list of values. This is done by finding
    the minimum and maximum values in the list, and extending the range by a
    fraction of the peak-to-peak (ptp) value. If the difference between the new
    limits is less than the tolerance, the values are extended by the tolerance.
    
    This is useful for plotting data, where the limits of the plot are extended
    slightly to ensure that all data points are visible.
    """
    values = np.array(values)
    finite_indices = np.isfinite(values)

    if np.sum(finite_indices) == 0:
        raise ValueError("no finite values provided")

    lower_limit, upper_limit = np.min(values[finite_indices]), np.max(values[finite_indices])
    ptp_value = np.ptp([lower_limit, upper_limit])

    if round_to_val is None:
        new_limits = lower_limit - fraction * ptp_value, upper_limit + fraction * ptp_value
    else:
        new_limits = round_to_nearest(lower_limit - fraction * ptp_value, round_to_val, how="floor"), \
                     round_to_nearest(upper_limit + fraction * ptp_value, round_to_val, how="ceiling")

    if np.abs(new_limits[0] - new_limits[1]) < tolerance:
        print("Extending limits by tolerance, instead of fraction.")
        if np.abs(new_limits[0]) < tolerance:
            # Arbitrary limits, since we"ve just been passed zeros
            offset = 1
        else:
            offset = np.abs(new_limits[0]) * fraction
        
        if round_to_val is None:
            new_limits = new_limits[0] - offset, new_limits[0] + offset
        else:
            new_limits = round_to_nearest(new_limits[0] - offset, round_to_val, how="floor"), \
                         round_to_nearest(new_limits[0] + offset, round_to_val, how="ceiling")

    return np.array(new_limits)