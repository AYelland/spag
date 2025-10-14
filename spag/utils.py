# coding: utf-8

""" Utility functions from extracting and manipulating data """

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np
import pandas as pd
import re

# import os
# basepath = os.path.dirname(__file__)
# datapath = os.path.join(basepath,"data")

from spag.periodic_table import pt_dict

################################################################################
## Roman numeral conversion functions

def int_to_roman(n):
    """ Convert an integer to Roman numerals. """
    roman_int_dict = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, \
                      'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, \
                      'CM': 900, 'M': 1000}
    roman_numeral = ''
    for numeral, value in sorted(roman_int_dict.items(), key=lambda x: x[1], reverse=True):
        while n >= value:
            roman_numeral += numeral
            n -= value
    
    return roman_numeral

def roman_to_int(roman):
    """ Convert a Roman numeral to an integer, up to several thousand. """
    roman = roman.upper()
    roman_int_dict = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, \
                      'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, \
                      'CM': 900, 'M': 1000}
    value = 0
    for i in range(len(roman)):
        if i > 0 and roman_int_dict[roman[i]] > roman_int_dict[roman[i - 1]]:
            value += roman_int_dict[roman[i]] - 2 * roman_int_dict[roman[i - 1]]
        else:
            value += roman_int_dict[roman[i]]
    return value

################################################################################
## Column name manipulation functions

def identify_prefix(col):
    """
    Identifies the prefix of a column name
    """
    m_XH = re.compile(r'\[(\D+)/H\]')
    m_XFe= re.compile(r'\[(\D+)/Fe\]')
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

################################################################################
## Validation/Testing Functions

def element_matches_atomic_number(elem, Z):
    """
    elem: str
        Element symbol.
    Z: int
        Atomic number.
        
    Returns True if the element symbol matches the atomic number, and False
    otherwise.    
    """
    
    if elem != pt_dict[Z]:
        return False
    else:
        return True


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
        
        ## These two lines work, but they pad the last column with white space up to the maximum width
        # aligned_line = ' & '.join(col.strip().ljust(width) for col, width in zip(line_parts, max_col_widths))
        # aligned_lines.append(aligned_line + '\n')

        ## This works as well, without the padding on the end of each line (the last column)
        aligned_cols = [
            col.strip().ljust(width) if i < len(max_col_widths)-1 else col.strip()
            for i, (col, width) in enumerate(zip(line_parts, max_col_widths))
        ]
        aligned_line = ' & '.join(aligned_cols)
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
        str: The rounded and padded value as a string, or an empty string if the value is NaN.
    """
    if pd.isna(value):
        return ''
    else:
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