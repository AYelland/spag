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
# Utility functions for column names

def _getcolnames(df,prefix):
    """
    Returns a list of all columns with a specific prefix
    """
    allnames = []
    for col in df:
        try:
            this_prefix,elem = identify_prefix(col)
        except ValueError:
            continue
        if this_prefix==prefix: allnames.append(col)
    return allnames

def getelem(elem, lower=False, keep_species=False):
    """
    Converts an element's common name to a standard formatted chemical symbol
    """
    common_molecules = {'CH':'C','NH':'N'}
    special_ions = ['Ti I','Cr II']
    
    if isinstance(elem, string_types):
        prefix = None
        try:
            prefix,elem_ = identify_prefix(elem)
            elem = elem_
        except ValueError:
            pass

        if pt.element_query(elem) != None: # No ionization, e.g. Ti
            elem = pt.element_query(elem).symbol
        elif elem in common_molecules:
            elem = common_molecules[elem]
        elif prefix != None and '.' in elem:
            elem,ion = elem.split('.')
            elem = format_elemstr(elem)
        #elif '.' in elem: #Not sure if this works correctly yet
        #    elem,ion = elem.split('.')
        #    elem = format_elemstr(elem)
        elif elem[-1]=='I': #Check for ionization
            # TODO account for ionization
            if ' ' in elem: #of the form 'Ti II' or 'Y I'
                species = element_to_species(elem)
                elem = species_to_element(species)
                elem = elem.split()[0]
            else: #of the form 'TiII' or 'YI'
                if elem[0]=='I':
                    assert elem=='I'*len(elem)
                    elem = 'I'
                else:
                    while elem[-1] == 'I': elem = elem[:-1]
        else:
            # Use smh to check for whether element is in periodic table
            species = element_to_species(elem)
            elem = species_to_element(species)
            elem = elem.split()[0]
            
    elif isinstance(elem, (int, np.integer)):
        elem = int(elem)
        elem = pt.element_query(elem)
        ## TODO common molecules
        assert elem != None
        elem = elem.symbol
        if keep_species: raise NotImplementedError()
    
    elif isinstance(elem, float):
        species = elem
        elem = species_to_element(species)
        if not keep_species: elem = elem.split()[0]

    if lower: elem = elem.lower()
    return elem

def epscolnames(df):
    """
    Returns a list of all epsilon columns
    """
    return _getcolnames(df,'eps')

def errcolnames(df):
    """
    Returns a list of all error columns
    """
    return _getcolnames(df,'e_')

def ulcolnames(df):
    """
    Returns a list of all upper limit columns
    """
    return _getcolnames(df,'ul')

def XHcolnames(df):
    """
    Returns a list of all [X/H] columns
    """
    return _getcolnames(df,'XH')

def XFecolnames(df):
    """
    Returns a list of all [X/Fe] columns
    """
    return _getcolnames(df,'XFe')

def epscol(elem):
    """
    Returns the epsilon column name for an element
    """
    return 'eps'+getelem(elem,lower=True)

def errcol(elem):
    """
    Returns the error column name for an element
    """
    try:
        return 'e_'+getelem(elem,lower=True)
    except ValueError:
        if elem=="alpha": return "e_alpha"
        else: raise
    
def eABcol(elems):
    """
    Input a tuple of elements, returns the error column name for the pair
    """
    A,B = elems
    return f"eAB_{getelem(A)}/{getelem(B)}"

def ulcol(elem):
    """
    Returns the upper limit column name for an element
    """
    try:
        return 'ul'+getelem(elem,lower=True)
    except ValueError:
        if elem=="alpha": return "ulalpha"
        else: raise
    
def XHcol(elem,keep_species=False):
    """
    Returns the [X/H] column name for an element
    """
    try:
        return '['+getelem(elem,keep_species=keep_species)+'/H]'
    except ValueError:
        if elem=="alpha": return "[alpha/H]"
        else: raise
    
def XFecol(elem,keep_species=False):
    """
    Returns the [X/Fe] column name for an element
    """
    try:
        return '['+getelem(elem,keep_species=keep_species)+'/Fe]'
    except ValueError:
        if elem=="alpha": return "[alpha/Fe]"
        else: raise
    
def ABcol(elems):
    """
    Input a tuple of elements, returns the column name for the pair
    Note: by default the data does not have [A/B]
    """
    A,B = elems
    return '['+getelem(A)+'/'+getelem(B)+']'

def make_XHcol(species):
    """
    Converts species to a formatted [X/H] column name
    """
    if species==22.0: return "[Ti I/H]"
    if species==23.1: return "[V II/H]"
    if species==26.1: return "[Fe II/H]"
    if species==24.1: return "[Cr II/H]"
    if species==38.0: return "[Sr I/H]"
    if species==106.0: return "[C/H]"
    if species==607.0: return "[N/H]"
    return XHcol(species)

def make_XFecol(species):
    """
    Converts species to a formatted [X/Fe] column name
    """
    if species==22.0: return "[Ti I/Fe]"
    if species==23.1: return "[V II/Fe]"
    if species==26.1: return "[Fe II/Fe]"
    if species==24.1: return "[Cr II/Fe]"
    if species==38.0: return "[Sr I/Fe]"
    if species==106.0: return "[C/Fe]"
    if species==607.0: return "[N/Fe]"
    return XFecol(species)

def make_epscol(species):
    """
    Converts species to a formatted epsilon column name
    """
    if species==22.0: return "epsti1"
    if species==23.1: return "epsv2"
    if species==26.1: return "epsfe2"
    if species==24.1: return "epscr2"
    if species==38.0: return "epssr1"
    if species==106.0: return "epsc"
    if species==607.0: return "epsn"
    return epscol(species)

def make_errcol(species):
    """
    Converts species to a formatted error column name
    """
    if species==22.0: return "e_ti1"
    if species==23.1: return "e_v2"
    if species==26.1: return "e_fe2"
    if species==24.1: return "e_cr2"
    if species==38.0: return "e_sr1"
    if species==106.0: return "e_c"
    if species==607.0: return "e_n"
    return errcol(species)

def make_ulcol(species):
    """
    Converts species to a formatted upper limit column name
    """
    if species==22.0: return "ulti1"
    if species==23.1: return "ulv2"
    if species==26.1: return "ulfe2"
    if species==24.1: return "ulcr2"
    if species==38.0: return "ulsr1"
    if species==106.0: return "ulc"
    if species==607.0: return "uln"
    return ulcol(species)

def format_elemstr(elem):
    """
    Capitalizes the first letter of an element string
    """
    assert len(elem) <= 2 and len(elem) >= 1
    return elem[0].upper() + elem[1:].lower()

def getcolion(col):
    """
    Returns the ionization state of an element column
    """
    prefix,elem = identify_prefix(col)
    if '.' in elem: int(ion = elem.split('.')[1])
    else: ion = get_default_ion(elem)
    ionstr = 'I'
    for i in range(ion): ionstr += 'I'
    return ionstr

def identify_prefix(col):
    """
    Identifies the prefix of a column name
    """
    for prefix in ['eps','e_','ul','XH','XFe']:
        if prefix in col:
            return prefix, col[len(prefix):]
        if prefix=='XH':
            matches = m_XH.findall(col)
            if len(matches)==1: return prefix,matches[0]
        if prefix=='XFe':
            matches = m_XFe.findall(col)
            if len(matches)==1: return prefix,matches[0]
    raise ValueError("Invalid column:"+str(col))

def get_default_ion(elem):
    """
    Returns the default ionization state for an element
    """
    default_to_1 = ['Na','Mg','Al','Si','Ca','Cr','Mn','Fe','Co','Ni']
    default_to_2 = ['Sc','Ti','Sr','Y','Zr','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Dy']
    elem = getelem(elem)
    if elem in default_to_1:
        return 1
    elif elem in default_to_2:
        return 2
    else:
        warnings.warn("get_default_ion: {} not in defaults, returning 2".format(elem))
        return 2


################################################################################
# Utility functions operating on standard DataFrame(s)
################################################################################

def get_star_abunds(starname,data,type):
    """
    Input: starname, DataFrame, and type of abundance to extract ('eps', 'XH', 'XFe', 'e_', 'ul')
    Returns: a pandas Series of abundances for a star by extracting the columns of the specified type
    """
    assert type in ['eps','XH','XFe','e_','ul']
    star = data.ix[starname]
    colnames = _getcolnames(data,type)
    if len(colnames)==0: raise ValueError("{} not in data".format(type))
    abunds = np.array(star[colnames])
    elems = [getelem(elem) for elem in colnames]
    return pd.Series(abunds,index=elems)

def XH_from_eps(df):
    """
    Converts epsilon columns to [X/H] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        asplund = rd.get_solar(epscols)
        for col in epscols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            df[XHcol(col)] = df[col].astype(float) - float(asplund[col])

def XFe_from_eps(df):
    """
    Converts epsilon columns to [X/Fe] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        assert 'epsfe' in epscols
        asplund = rd.get_solar(epscols)
        feh = df['epsfe'].astype(float) - float(asplund['epsfe'])
        for col in epscols:
            if col=='epsfe': continue
            if XFecol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XFecol(col)))
            XH = df[col].astype(float) - float(asplund[col])
            df[XFecol(col)] = XH - feh

def eps_from_XH(df):
    """
    Converts [X/H] columns to epsilon columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XHcols = XHcolnames(df)
        asplund = rd.get_solar(XHcols)
        for col in XHcols:
            if epscol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(epscol(col)))
            df[epscol(col)] = df[col] + float(asplund[col])

def XFe_from_XH(df):
    """
    Converts [X/H] columns to [X/Fe] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XHcols = XHcolnames(df)
        assert '[Fe/H]' in XHcols
        feh = df['[Fe/H]']
        for col in XHcols:
            if col=='[Fe/H]': continue
            if XFecol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XFecol(col)))
            df[XFecol(col)] = df[col] - feh

def eps_from_XFe(df):
    """
    Converts [X/Fe] columns to epsilon columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XFecols = XFecolnames(df)
        assert '[Fe/H]' in df
        asplund = rd.get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            df[epscol(col)] = df[col] + feh + float(asplund[col])

def XH_from_XFe(df):
    """
    Converts [X/Fe] columns to [X/H] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XFecols = XFecolnames(df)
        assert '[Fe/H]' in df
        asplund = rd.get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            df[XHcol(col)] = df[col] + feh


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