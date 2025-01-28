# coding: utf-8

""" Utility functions for string manipulation """

from functools import reduce
from six import string_types

# Functions to import when using 'from spag.utils_smh import *'
__all__ =  ["get_common_letters", "find_common_start"]

################################################################################
## String manipulation functions

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