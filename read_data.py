#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import  sys, os, glob, time, IPython
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

import seaborn as sns
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
## Utility Functions

def element_matches_atomic_number(elem, Z):
    """
    elem: str
        Element symbol.
    Z: int
        Atomic number.
        
    Returns True if the element symbol matches the atomic number, and False
    otherwise.    
    """
    
    element_dict = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B",
        6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
        16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
        21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
        26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
        31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br",
        36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
        41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh",
        46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
        51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs",
        56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
        61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb",
        66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
        71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re",
        76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
        81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
        86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
        91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am",
        96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
        101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db",
        106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
        111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc",
        116: "Lv", 117: "Ts", 118: "Og"
    }
    
    if elem != element_dict[Z]:
        return False
    else:
        return True
    
    
################################################################################
## Solar Composition Functions

# Asplund 2009 
# https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract

def solar_logepsX_asplund09(Z=None, elem=None, return_error=False):
    """
    Z: int or None
        Atomic number of the element.
    elem: str or None
        Element symbol of the element.
    return_error: bool
        If True, the function will return the abundance and error of the solar
        composition in a tuple.
        
    Reads the datafile for Table 1 in "Asplund et al. 2009", and returns a 
    pandas table of the solar composition. 
    
    If 'Z' or 'elem' is provided (not None), it will return the photospheric 
    abundances if available, otherwise it will use the meteoritic 
    abundances. 
    
    If the element is not found in the table, the function will return None.
    """
    asplund09 = pd.read_csv(data_dir+"solar/asplund2009_solarcomposition.csv", skipinitialspace=True)
    print("Loading Datafile: ", data_dir+"solar/asplund2009_solarcomposition.csv")
    
    if (elem is None) and (Z is None):
        return asplund09
    
    elif (elem is not None) and (Z is None):
        if elem not in asplund09['element'].values:
            raise ValueError("Element symbol (elem) is invalid.")
        abund = asplund09[asplund09['element'] == elem]['photosphere_abundance'].values[0]
        error = asplund09[asplund09['element'] == elem]['photosphere_abundance_error'].values[0]
        if np.isnan(abund):
            abund = asplund09[asplund09['element'] == elem]['meteorite_abundance'].values[0]
            error = asplund09[asplund09['element'] == elem]['meteorite_abundance_error'].values[0]
        
    elif (elem is None) and (Z is not None):
        if Z not in asplund09['Z'].values:
            raise ValueError("Atomic number (Z) is invalid.")
        abund = asplund09[asplund09['Z'] == Z]['photosphere_abundance'].values[0]
        error = asplund09[asplund09['Z'] == Z]['photosphere_abundance_error'].values[0]
        if np.isnan(abund):
            abund = asplund09[asplund09['Z'] == Z]['meteorite_abundance'].values[0]
            error = asplund09[asplund09['Z'] == Z]['meteorite_abundance_error'].values[0]
        
    elif (elem is not None) and (Z is not None):
        if not element_matches_atomic_number(elem, Z):
            raise ValueError("The provided element symbol (elem) and atomic number (Z) do not match.")
        abund = asplund09[asplund09['element'] == elem]['photosphere_abundance'].values[0]
        error = asplund09[asplund09['element'] == elem]['photosphere_abundance_error'].values[0]
        if np.isnan(abund):
            abund = asplund09[asplund09['element'] == elem]['meteorite_abundance'].values[0]
            error = asplund09[asplund09['element'] == elem]['meteorite_abundance_error'].values[0]
    else:
        raise AttributeError("Please check the function's input parameters.")
    
    if return_error:
        return (abund, error)
    else:
        return abund

# Asplund 2021
# https://ui.adsabs.harvard.edu/abs/2021A%26A...653A.141A/abstract

def solar_logepsX_asplund21(Z=None, elem=None, return_error=False):
    """
    Z: int or None
        Atomic number of the element.
    elem: str or None
        Element symbol of the element.
    return_error: bool
        If True, the function will return the abundance and error of the solar
        composition in a tuple.
        
    Reads the datafile for Table B.1 in "Asplund et al. 2021", and returns a 
    pandas table of the solar composition. 
    
    If 'Z' or 'elem' is provided (not None), it will return the photospheric 
    abundances if available, otherwise it will use the meteoritic 
    abundances. 
    
    If the element is not found in the table, the function will return None.
    """
    asplund21 = pd.read_csv(data_dir+"solar/asplund2021_solarcomposition.csv", skipinitialspace=True)
    print("Loading Datafile: ", data_dir+"solar/asplund2021_solarcomposition.csv")
    
    if (elem is None) and (Z is None):
        return asplund21
    
    elif (elem is not None) and (Z is None):
        if elem not in asplund21['element'].values:
            raise ValueError("Element symbol (elem) is invalid.")
        abund = asplund21[asplund21['element'] == elem]['photosphere_abundance'].values[0]
        error = asplund21[asplund21['element'] == elem]['photosphere_abundance_error'].values[0]
        if np.isnan(abund):
            abund = asplund21[asplund21['element'] == elem]['meteorite_abundance'].values[0]
            error = asplund21[asplund21['element'] == elem]['meteorite_abundance_error'].values[0]
        
    elif (elem is None) and (Z is not None):
        if Z not in asplund21['Z'].values:
            raise ValueError("Atomic number (Z) is invalid.")
        abund = asplund21[asplund21['Z'] == Z]['photosphere_abundance'].values[0]
        error = asplund21[asplund21['Z'] == Z]['photosphere_abundance_error'].values[0]
        if np.isnan(abund):
            abund = asplund21[asplund21['Z'] == Z]['meteorite_abundance'].values[0]
            error = asplund21[asplund21['Z'] == Z]['meteorite_abundance_error'].values[0]
            