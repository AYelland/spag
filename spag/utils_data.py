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
        asplund = get_solar(epscols)
        for col in epscols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            df[XHcol(col)] = df[col] - float(asplund[col])

def XFe_from_eps(df):
    """
    Converts epsilon columns to [X/Fe] columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epscols = epscolnames(df)
        assert 'epsfe' in epscols
        asplund = get_solar(epscols)
        feh = df['epsfe'] - float(asplund['epsfe'])
        for col in epscols:
            if col=='epsfe': continue
            if XFecol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XFecol(col)))
            XH = df[col]-float(asplund[col])
            df[XFecol(col)] = XH - feh

def eps_from_XH(df):
    """
    Converts [X/H] columns to epsilon columns
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        XHcols = XHcolnames(df)
        asplund = get_solar(XHcols)
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
        asplund = get_solar(XFecols)
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
        asplund = get_solar(XFecols)
        feh = df['[Fe/H]']
        for col in XFecols:
            if XHcol(col) in df: warnings.warn("{} already in DataFrame, replacing".format(XHcol(col)))
            df[XHcol(col)] = df[col] + feh
