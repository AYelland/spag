#!/usr/bin/env python
# -*- coding: utf-8 -*-
# add to top of read_data.py temporarily

from __future__ import (division, print_function, absolute_import, unicode_literals)


import  sys, os, glob, time

import numpy as np
from scipy.stats import norm, ks_2samp

# from spag.utils import identify_prefix, element_matches_atomic_number

################################################################################
## Directory Variables

# script_dir = "/".join(IPython.extract_module_locals()[1]["__vsc_ipynb_file__"].split("/")[:-1]) + "/" # use this if in ipython
script_dir = os.path.dirname(os.path.realpath(__file__))+"/" # use this if not in ipython (i.e. terminal script)
data_dir = script_dir+"data/"
plots_dir = script_dir+"plots/"
linelist_dir = script_dir+"linelists/"

################################################################################
## Hypothesis Testing Functions

def calculate_p_value(observed, expected, sigma):
    """
    Calculate the p-value for the observed value given the expected value and standard deviation.
    """
    z_score = (observed - expected) / sigma
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
    return p_value

def calculate_z_score(observed, expected, sigma):
    """
    Calculate the z-score for the observed value given the expected value and standard deviation.
    """
    z_score = (observed - expected) / sigma
    return z_score

def calculate_confidence_interval(mean, sigma, confidence_level=0.95):
    """
    Calculate the confidence interval for a given mean, standard deviation, and confidence level.
    """
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    lower_bound = mean - z_score * sigma
    upper_bound = mean + z_score * sigma
    return lower_bound, upper_bound

def calculate_standard_error(sigma, n):
    """
    Calculate the standard error of the mean given the standard deviation and sample size.
    """
    return sigma / np.sqrt(n)

def calculate_t_statistic(sample_mean, population_mean, standard_error):
    """
    Calculate the t-statistic for a sample mean given the population mean and standard error.
    """
    t_statistic = (sample_mean - population_mean) / standard_error
    return t_statistic

#student t test for two independent samples
def calculate_t_statistic_independent(sample_mean1, sample_mean2, standard_error1, standard_error2):
    """
    Calculate the t-statistic for two independent samples given their means and standard errors.
    """
    pooled_standard_error = np.sqrt(standard_error1**2 + standard_error2**2)
    t_statistic = (sample_mean1 - sample_mean2) / pooled_standard_error
    return t_statistic

def ks_test(sample1, sample2):
    """
    Perform the Kolmogorov-Smirnov test for two samples.
    Returns the KS statistic and p-value.
    """
    ks_statistic, p_value = ks_2samp(sample1, sample2)
    return ks_statistic, p_value

