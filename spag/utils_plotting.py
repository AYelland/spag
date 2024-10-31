# coding: utf-8

""" Utility functions for plotting data """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

# Standard library
import os
import logging
import platform
import string
import sys
import traceback
import tempfile
from functools import reduce

from collections import Counter

from six import string_types

from hashlib import sha1 as sha
from random import choice
from socket import gethostname, gethostbyname

# Third party imports
import numpy as np
import astropy.table

# Functions to import when using 'from spag.utils_smh import *'
__all__ =  ["round_to_nearest", "extend_limits"]

################################################################################
## Utility functions for plotting data

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


