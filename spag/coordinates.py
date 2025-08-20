
# coding: utf-8

""" Astronomy coordinate conversion functions. """

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from six import string_types
from astropy.coordinates import SkyCoord
import astropy.units as u
from gala.coordinates import SagittariusLaw10

# Ignore AstropyDeprecationWarning for SagittariusLaw10
from astropy.utils.exceptions import AstropyDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=AstropyDeprecationWarning)

from spag.utils import normal_round

################################################################################
## Utility Functions

def replace_non_ascii(text):
    """
    Replaces non-ASCII characters using a given replacement dictionary.
    """
    # Build a dictionary of problematic Unicode characters
    replacement_map = {
        "\u2212": "-",  # Unicode minus (−) → ASCII hyphen (-)
        "\u2013": "-",  # En dash (–) → ASCII hyphen (-)
        "\u2014": "-",  # Em dash (—) → ASCII hyphen (-)
    }
    return ''.join(replacement_map.get(char, char) for char in text)

################################################################################
## RA & Decl Coordinate Functions

def ra_hms_to_deg(ra_str, precision=None):
    """
    ra_str: str
        Right ascension in the form 'hh:mm:ss.ss'
    precision: int (default=5)
        Number of decimal places to round to.
    
    Converts a right ascension string to degrees.
    """
    ra_str = replace_non_ascii(ra_str)
    ra_str = ra_str.strip()
    ra_str = ra_str.split(':')
    if type(precision) == int:
        ra_deg = round(15.0 * (float(ra_str[0]) + float(ra_str[1])/60.0 + float(ra_str[2])/3600.0), precision)
    elif type(precision) == type(None):
        ra_deg = 15.0 * (float(ra_str[0]) + float(ra_str[1])/60.0 + float(ra_str[2])/3600.0)
    else:
        raise TypeError("precision must be an int or None")
    return ra_deg

def ra_deg_to_hms(ra_deg, precision=2):
    """
    ra_deg: float
        Right ascension in degrees.
    precision: int (default=2)
        Number of decimal places to round to.

    Converts right ascension in degrees to a string in the form 'hh:mm:ss.ss'.
    """
    ra_h = int(ra_deg / 15.0)
    ra_m = int((ra_deg - ra_h * 15.0) * 4.0)
    ra_s = (ra_deg - ra_h * 15.0 - ra_m / 4.0) * 240.0
    if isinstance(precision, int):
        output_str = f"{ra_h:02d}:{ra_m:02d}:{ra_s:0{3+precision}.{precision}f}"
    elif precision is None:
        output_str = f"{ra_h:02d}:{ra_m:02d}:{int(ra_s):02d}"
    else:
        raise TypeError("precision must be an int or None")
    return output_str

def dec_dms_to_deg(dec_str, precision=None):
    """
    dec_str: str
        Declination in the form '+dd:mm:ss.ss'
    precision: int (default=5)
        Number of decimal places to round to.
    
    Converts a declination string to degrees.
    """
    dec_str = replace_non_ascii(dec_str)
    dec_str = dec_str.strip()
    if dec_str[0] == '-':
        sign = -1
        dec_str = dec_str[1:].split(':')
    else:
        sign = 1
        dec_str = dec_str.split(':')
    if type(precision) == int:
        dec_deg = sign * round((float(dec_str[0]) + float(dec_str[1])/60.0 + float(dec_str[2])/3600.0), precision)
    elif type(precision) == type(None):
        dec_deg = sign * (float(dec_str[0]) + float(dec_str[1])/60.0 + float(dec_str[2])/3600.0)
    else:
        raise TypeError("precision must be an int or None")
    return dec_deg

def dec_deg_to_dms(dec_deg, precision=2):
    """
    dec_deg: float
        Declination in degrees.
    precision: int (default=2)
        Number of decimal places to round to.

    Converts declination in degrees to a string in the form '+dd:mm:ss.ss'
    """
    if dec_deg < 0:
        sign = '-'
        dec_deg = -dec_deg
    else:
        sign = '+'
    dec_d = int(dec_deg)
    dec_m = int((dec_deg - dec_d) * 60.0)
    dec_s = (dec_deg - dec_d - dec_m / 60.0) * 3600.0
    if isinstance(precision, int):
        output_str = f"{sign}{dec_d:02d}:{dec_m:02d}:{dec_s:0{3+precision}.{precision}f}"
    elif precision is None:
        output_str = f"{sign}{dec_d:02d}:{dec_m:02d}:{int(dec_s):02d}"
    else:
        raise TypeError("precision must be an int or None")
    return output_str

def round_hms(hms_str, precision=1):
    """
    hms_str: str ('hh:mm:ss.ss')
        Right ascension in the form 'hh:mm:ss.ss'
    precision: int (default=1)
        Number of decimal places to round to.
    
    Rounds the seconds component of a right ascension string.
    """
    h, m, s = hms_str.split(':')
    s = f"{float(s):.{precision}f}"  # Convert to float and round to the specified number of decimals
    return f"{h}:{m}:{s.zfill(4 if precision > 0 else 2)}"  # Ensure consistent padding

def round_dms(dms_str, precision=1):
    """
    dms_str: str ('+dd:mm:ss.ss')
        Declination in the form '+dd:mm:ss.ss'
    precision: int (default=1)
        Number of decimal places to round to.
        
    Rounds the seconds component of a declination string.
    """
    d, m, s = dms_str.split(':')
    s = f"{float(s):.{precision}f}"  # Convert to float and round to the specified number of decimals
    return f"{d}:{m}:{s.zfill(4 if precision > 0 else 2)}"  # Ensure consistent padding

################################################################################
## Coordinate Comparison Functions

def coords_equal(ra1, dec1, ra2, dec2, precision=5):
    """
    ra1: float or str
        Right ascension in degrees or in the form 'hh:mm:ss.ss'
    dec1: float or str
        Declination in degrees or in the form '+dd:mm:ss.ss'
    ra2: float or str
        Right ascension in degrees or in the form 'hh:mm:ss.ss'
    dec2: float or str
        Declination in degrees or in the form '+dd:mm:ss.ss'
    precision: int (default=5)
        Number of decimal places to round to.
    
    Compares two sets of coordinates and returns True if they are equal.
    """
    if isinstance(ra1, string_types) and isinstance(dec1, string_types):
        ra1 = ra_hms_to_deg(ra1, precision)
        dec1 = dec_dms_to_deg(dec1, precision)
    if isinstance(ra2, string_types) and isinstance(dec2, string_types):
        ra2 = ra_hms_to_deg(ra2, precision)
        dec2 = dec_dms_to_deg(dec2, precision)
    
    ra1 = float(normal_round(ra1, precision))
    dec1 = float(normal_round(dec1, precision))
    ra2 = float(normal_round(ra2, precision))
    dec2 = float(normal_round(dec2, precision))

    return ra1 == ra2 and dec1 == dec2

################################################################################
## Special Coordinate System Conversions

def sgr_LambdaBeta(ra, dec):
    """
    ra_deg: float or str
        Right Ascension in degrees or in the form 'hh:mm:ss.ss'
    dec_deg: float or str
        Declination in degrees or in the form '+dd:mm:ss.ss'
        
    Converts from RA, Dec coordinates into the Sagittarius (Sgr) heliocentric 
    spherical coordinates (Lambda, Beta), where Lambda is the longitude along 
    the Sgr stream and Beta is the latitude. Lambda is defined to be 0 degrees 
    at the center Sgr core and increases along the stream.
    
    Majewski et al. 2003: http://adsabs.harvard.edu/abs/2003ApJ...599.1082M
    Law & Majewski 2010: http://adsabs.harvard.edu/abs/2010ApJ...714..229L
    
    https://gala.adrian.pw/en/v1.9.1/api/gala.coordinates.SagittariusLaw10.html
    """
   
    ra_deg = ra_hms2deg(ra) if isinstance(ra, string_types) else ra
    dec_deg = dec_dms2deg(dec) if isinstance(dec, string_types) else dec 
    
    ## RA, Dec to ICRS Coordinates
    icrs_coord = SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
    
    ## ICRS to Galactic Coordinates
    galactic_coord = icrs_coord.transform_to('galactic')
    
    ## Galactic to Sgr Coordinates (l, b) by Law & Majewski 2010
    sgr_coord = galactic_coord.transform_to(SagittariusLaw10)
    sgr_l = sgr_coord.Lambda.wrap_at(360 * u.deg) # wrap at 360 degrees
    sgr_b = sgr_coord.Beta

    return sgr_l, sgr_b