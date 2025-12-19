
# coding: utf-8

""" Astronomy quick observing scaling functions. """

from __future__ import (division, print_function, absolute_import, unicode_literals)

import numpy as np


## Calculate the required exposure time for a desired S/N
def snr_scaling_for_exptime(snr_new, t_exp_ref, snr_ref):
    return t_exp_ref * (snr_new / snr_ref) ** 2


## Calculate expected S/N for different exposure times
def exptime_scaling_for_snr(t_exp_new, t_exp_ref, snr_ref):
    return snr_ref * np.sqrt(t_exp_new / t_exp_ref)

def mag_scaling_for_exptime(m, m_ref, t_ref):
    """
    m: int or float
        Magnitude of the star
    m_ref: int or float
        Magnitude of the reference star
    t_ref: int or float
        Exposure time of the reference star
    
    Calculates the exposure time of the star using the reference star.
    """
    t = t_ref * 10**((m - m_ref) / 2.5)
    return t