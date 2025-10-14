#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)


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
# plotting_dir = script_dir+"plots/"
# if not os.path.exists(plotting_dir):
#     os.makedirs(plotting_dir)


################################################################################
## Read-in the solar abundnace table for r- and s-process elements

solar = pd.read_csv(data_dir+"solar/solar_r_s_fractions.csv", comment='#')

solar[['logeps_r','logeps_s']] = np.log10(
    solar[['rproc','sproc']].where(solar[['rproc','sproc']] > 0)
)

solar_Z = np.array(solar['Z'])
solar_logeps_r = np.array(solar['logeps_r'])
solar_logeps_s = np.array(solar['logeps_s'])

## Replace any NaN or Inf values with np.nan (does not change the pandas dataframe)
for i in range(len(solar_Z)):
    if np.isnan(solar_logeps_r[i]) or np.isinf(solar_logeps_r[i]):
        solar_logeps_r[i] = np.nan
    if np.isnan(solar_logeps_s[i]) or np.isinf(solar_logeps_s[i]):
        solar_logeps_s[i] = np.nan

################################################################################
## Dictionary of the atomic number (Z) and the corresponding element symbol

element_symbols = {
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

################################################################################
## Plot the solar r- and s-process abundances

def plot_solar_r_s_process():
    """
    Plot the solar r- and s-process abundances.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # ax.set_title("solar r- and s-process abundances")
    ax.set_title("Solar s- and r- process patterns", fontsize=12, fontweight='bold')
    ax.set_xlabel(r'Atomic Number, $Z$')
    ax.set_ylabel(r'log $\epsilon$ (X)')

    ## solar abundances, r-process and s-process 
    solar_rcolor = 'black'
    solar_scolor = 'gray'

    # solar abundances, r-process & s-process 
    line_r = ax.plot(solar_Z, solar_logeps_r, label='solar r-process', color=solar_rcolor, linestyle='-', marker='.', zorder=1, alpha=0.5)
    line_s = ax.plot(solar_Z, solar_logeps_s, label='solar s-process', color=solar_scolor, linestyle='-', marker='.', zorder=1, alpha=0.5)

    for j in range(len(solar_Z)):
        offset = 0.1 if j % 2 == 0 else -0.1
        vertical_align = 'bottom' if j % 2 == 0 else 'top'
        if 44 <= solar_Z[j] < 62:
            vertical_align = 'top' if j % 2 == 0 else 'bottom'
            offset = -offset  # Flip the offset direction for elements in range(44, 62)
                    
        if solar_Z[j] in [33, 35, 36, 38, 40, 42, 45, 47, 49, 50, 51, 53, 55, 56, 58, 59, 65, 67, 69, 71, 73, 74, 75, 77, 79, 80, 82, 83]:
            ax.text(solar_Z[j], solar_logeps_s[j] + offset, element_symbols[int(solar_Z[j])], fontsize=10, color=solar_rcolor, ha='center', va=vertical_align, alpha=0.8)
        else:
            ax.text(solar_Z[j], solar_logeps_r[j] + offset, element_symbols[int(solar_Z[j])], fontsize=10, color=solar_rcolor, ha='center', va=vertical_align, alpha=0.8)

    # for i in range(len(solar_Z)):
    #     offset = 0.1 if i%2 == 0 else -0.1
    #     vertical_align = 'bottom' if i%2 == 0 else 'top'
    #     if 44 <= solar_Z[i] < 62:
    #         vertical_align = 'top' if i%2 == 0 else 'bottom'
    #         offset = -offset  # Flip the offset direction for elements in range(44, 62)
    #     ax.text(solar_Z[i], solar_logeps_s[i] + offset, element_symbols[int(solar_Z[i])], fontsize=10, color=solar_scolor, ha='center', va=vertical_align)


    ## Axe attributes
    ax.set_axisbelow(True)
    ax.set_xticks(np.arange(np.min(solar_Z)-5, np.max(solar_Z)+5, 5)) # major x-ticks every 5th number
    ax.set_xticks(np.arange(np.min(solar_Z)-5, np.max(solar_Z)+5), minor=True) # minor x-ticks at every number
    ax.set_xticklabels([int(i) for i in ax.get_xticks()]) # only label the major x-ticks

    ax.set_yticks(np.arange(-3, 4, 1)) # major y-ticks every 1
    ax.set_yticks(np.arange(-3, 3, 0.5), minor=True) # minor y-ticks at every 0.5

    ax.set_xlim(np.min(solar_Z)-2, np.max(solar_Z)+2)
    ax.set_ylim(-3, 3)

    ## Make the plot features white
    # ax.spines['bottom'].set_color('white')
    # ax.spines['top'].set_color('white') 
    # ax.spines['right'].set_color('white')
    # ax.spines['left'].set_color('white')
    # ax.tick_params(axis='x', which="both", colors='white')
    # ax.tick_params(axis='y', which="both", colors='white')
    # ax.yaxis.label.set_color('white')
    # ax.xaxis.label.set_color('white')
    # ax.title.set_color('white')

    ax.grid(True, which='major', linestyle='-', linewidth=1)
    ax.grid(True, which='minor', linestyle='-', linewidth=0.3)
    ax.legend()

    ## Save the plot
    # if not os.path.exists(plotting_dir):
    #     os.makedirs(plotting_dir)
    # plt.savefig(plotting_dir+"solar_rproc.png", dpi=300, bbox_inches='tight') #, transparent=True)
