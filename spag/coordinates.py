
# coding: utf-8

""" Astronomy coordinate conversion functions. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from six import string_types

################################################################################
## Coordinate Conversion Functions

def ra_hms2deg(ra_str, precision=5):
    """
    ra_str: str
        Right ascension in the form 'hh:mm:ss.ss'
    
    Converts a right ascension string to degrees.
    """
    
    ra_str = ra_str.split(':')
    ra_deg = round(15.0 * (float(ra_str[0]) + float(ra_str[1])/60.0 + float(ra_str[2])/3600.0), precision)
    return ra_deg

def ra_deg2hms(ra_deg, precision=2):
    """
    ra_deg: float
        Right ascension in degrees.
    
    Converts right ascension in degrees to a string.
    """
    
    ra_h = int(ra_deg / 15.0)
    ra_m = int((ra_deg - ra_h * 15.0) * 4.0)
    ra_s = round((ra_deg - ra_h * 15.0 - ra_m / 4.0) * 240.0, precision)
    return "{:02d}:{:02d}:{:05.2f}".format(ra_h, ra_m, ra_s)

def decl_deg2dms(dec_deg, precision=2):
    """
    dec_deg: float
        Declination in degrees.
    
    Converts declination in degrees to a string.
    """
    
    if dec_deg < 0:
        sign = '-'
        dec_deg = -dec_deg
    else:
        sign = '+'
    
    dec_d = int(dec_deg)
    dec_m = int((dec_deg - dec_d) * 60.0)
    dec_s = round((dec_deg - dec_d - dec_m / 60.0) * 3600.0, precision)
    return "{}{:02d}:{:02d}:{:05.2f}".format(sign, dec_d, dec_m, dec_s)

def decl_dms2deg(dec_str, precision=5):
    """
    dec_str: str
        Declination in the form '+dd:mm:ss.ss'
    
    Converts a declination string to degrees.
    """
    
    sign = -1 if dec_str[0] == '-' else 1
    dec_str = dec_str[1:].split(':')
    dec_deg = sign * round((float(dec_str[0]) + float(dec_str[1])/60.0 + float(dec_str[2])/3600.0), precision)
    return dec_deg