#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)


import  sys, os, glob, time
import corner
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.io import fits

from spag.convert import *
from spag.utils import *
import spag.coordinates as coord

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
## Calculating the Carbon Corrections=, using Placco et al. 2014 website
# https://vplacco.pythonanywhere.com/

import requests
from bs4 import BeautifulSoup

def calc_carbon_correction(logg, feh, cfe, e_logg=0.1, e_feh=0.15, e_cfe=0.15, return_e_correction=False):
    payload = {
        'lgg': str(logg),
        'e_lgg': str(e_logg),
        'feh': str(feh),
        'e_feh': str(e_feh),
        'cfe': str(cfe),
        'e_cfe': str(e_cfe),
        'n_samples': '1000',
    }

    URL = 'https://vplacco.pythonanywhere.com'  # example: 'http://vmplacco.pythonanywhere.com'
    session = requests.Session()
    response = session.post(URL, data=payload)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    correction_tag = soup.find('pre', style=lambda s: s and 'font-size: 30px' in s)
    if correction_tag:
        correction_tag_list = str(correction_tag.text.strip()).split(' ')
        correction = float(correction_tag_list[4])
        try:
            e_correction = float(correction_tag_list[6])
        except:
            e_correction = np.nan
    else:
        correction = None
        e_correction = None
    
    if return_e_correction:
        return correction, e_correction
    else:
        return correction

## Calculate carbon corrections
def calc_carbon_correction_for_df(df, ulim_shift=0.3, ll_cfe_exist=True):

    ## Identify entries with missing values for logg, feh, cfe
    missing_logg_mask = df['logg'].isna()
    missing_feh_mask  = df['[Fe/H]'].isna() & df['ul[Fe/H]'].isna()
    if ll_cfe_exist:
        missing_cfe_mask = (df['[C/Fe]'].isna() & df['ll[C/Fe]'].isna() & df['ul[C/Fe]'].isna())
    else:
        missing_cfe_mask = df['[C/Fe]'].isna() & df['ul[C/Fe]'].isna()
        
    missing_values_df = df.loc[missing_logg_mask | missing_feh_mask | missing_cfe_mask].copy()
    print("Entries with missing values (logg, feh, cfe): ", missing_values_df.shape[0])
    for name, id in zip(missing_values_df['Name'].values, missing_values_df['Simbad_Identifier'].values):
        print(f"   {name} ({id})")

    ## Calculate correction values
    print("Number of Entries in Datatable: ", len(df))
    for i, row in df.iterrows():
        ### logg
        logg = row['logg']
        
        ### feh
        if pd.notna(row['[Fe/H]']):
            feh = row['[Fe/H]']
        else:
            feh = float(row['ul[Fe/H]']) - ulim_shift

        ### cfe
        if ll_cfe_exist:
            if pd.notna(row['[C/Fe]']) and pd.isna(row['ll[C/Fe]']) and pd.isna(row['ul[C/Fe]']):
                cfe = float(row['[C/Fe]'])
            elif pd.isna(row['[C/Fe]']) and pd.notna(row['ll[C/Fe]']) and pd.isna(row['ul[C/Fe]']):
                cfe = float(row['ll[C/Fe]'])
            elif pd.isna(row['[C/Fe]']) and pd.isna(row['ll[C/Fe]']) and pd.notna(row['ul[C/Fe]']):
                cfe = float(row['ul[C/Fe]']) - ulim_shift
            else:
                cfe = np.nan
                print(f"No [C/Fe] value found for star {row['Name']}, {row['Reference']}, {row['System']}")
        else:
            if pd.notna(row['[C/Fe]']) and pd.isna(row['ul[C/Fe]']):
                cfe = float(row['[C/Fe]'])
            elif pd.isna(row['[C/Fe]']) and pd.notna(row['ul[C/Fe]']):
                cfe = float(row['ul[C/Fe]']) - ulim_shift
            else:
                cfe = np.nan
                print(f"No [C/Fe] value found for star {row['Name']}, {row['Reference']}, {row['System']}")

        ### correction (epsc_c)
        if pd.isna(logg) or pd.isna(feh) or pd.isna(cfe):
            df.at[i, 'epsc_c'] = np.nan
        else:
            correction = calc_carbon_correction(logg, feh, cfe)
            df.at[i, 'epsc_c'] = correction

    print("Number of Entries in Datatable: ", len(df))
    print("Number of Entries in Datatable, without correction: ", len(df.loc[(df['epsc_c'].isna())]))

    ## Applying the correction to create [C/H]f and [C/Fe]f columns
    df['ulc_f'] = np.nan
    df['epsc_f'] = np.nan

    for i, row in df.iterrows():

        if pd.notna(row['epsc']) and pd.isna(row['ulc']):
            if isinstance(row['[C/H]'], str):
                df.at[i, 'epsc_f'] = float(row['epsc']) + row['epsc_c']
                df.at[i, '[C/H]f'] = float(row['[C/H]']) + row['epsc_c']
                df.at[i, '[C/Fe]f'] = float(row['[C/Fe]']) + row['epsc_c']
                if ll_cfe_exist: 
                    df.at[i, 'll[C/Fe]f'] = row['ll[C/Fe]'] + row['epsc_c']
            elif isinstance(row['[C/H]'], (int, float)):
                df.at[i, 'epsc_f'] = float(row['epsc']) + row['epsc_c']
                df.at[i, '[C/H]f'] = float(row['[C/H]']) + row['epsc_c']
                df.at[i, '[C/Fe]f'] = float(row['[C/Fe]']) + row['epsc_c']
                if ll_cfe_exist:
                    df.at[i, 'll[C/Fe]f'] = row['ll[C/Fe]'] + row['epsc_c']
            else:
                print("Error: [C/H] is not a correct value type.", i)

        elif pd.isna(row['epsc']) and pd.notna(row['ulc']):
            if isinstance(row['ul[C/H]'], str):
                df.at[i, 'ulc_f'] = float(row['ulc']) + row['epsc_c']
                df.at[i, 'ul[C/H]f'] = float(row['ul[C/H]']) + row['epsc_c']
                df.at[i, 'ul[C/Fe]f'] = float(row['ul[C/Fe]']) + row['epsc_c']
            elif isinstance(row['ul[C/H]'], (int, float)):
                df.at[i, 'ulc_f'] = float(row['ulc']) + row['epsc_c']
                df.at[i, 'ul[C/H]f'] = float(row['ul[C/H]']) + row['epsc_c']
                df.at[i, 'ul[C/Fe]f'] = float(row['ul[C/Fe]']) + row['epsc_c']
            else:
                print("Error: ul[C/H] is not a correct value type.", i)
                    
    ## Apply spag.utils.normal_round() to new columns
    new_cols = ['epsc_c', 'epsc_f', 'ulc_f', '[C/H]f', 'ul[C/H]f', 'ul[C/Fe]f', '[C/Fe]f']
    new_cols += ['ll[C/Fe]f'] if ll_cfe_exist else []
    for col in new_cols:
        for i, row in df.iterrows():
            if isinstance(row[col], str) and row[col] != '':  # Check for non-empty string
                df.at[i, col] = normal_round(float(row[col]), 2)
            elif isinstance(row[col], (int, float)):
                df.at[i, col] = normal_round(row[col], 2)
            elif row[col] == '':
                df.at[i, col] = np.nan  # Convert the empty string to np.nan (or keep as empty string)
            else:
                print("Error: {} is not a correct value type.".format(col), i)

            ## Remove 'nan' strings from the data
            if 'nan' in str(row[col]):
                df.at[i, col] = np.nan
                
    return df

################################################################################
## Calculating the CEMP fraction

def calc_cempfrac(df, feh_limit=-2.0, cfe_limit=0.7):
    """
    Calculate the carbon fraction for a given DataFrame and [Fe/H] limit.

    Returns: (cempfrac, n_cemp, n_tot)
        cempfrac (int): CEMP fraction, rounded to the nearest integer percentage (0-100), or -1 if undefined
        n_cemp (int): number of CEMP stars in numerator
        n_tot (int): total number of stars in denominator
    """
    df_filtered = df[(df['[Fe/H]'] <= feh_limit) | (df['ul[Fe/H]'] <= feh_limit)]
    
    ## n_cemp = (all measured values) + (lower limits above the cfe threshold)
    n_cemp = len(df_filtered[df_filtered['[C/Fe]f'] >= cfe_limit])
    n_cemp += len(df_filtered[(df_filtered['ll[C/Fe]f'].notna()) & (df_filtered['ll[C/Fe]f'] >= cfe_limit-0.2)]) # lower limits
    
    ## n_tot = (all measured values) + (lower limits above the cfe threshold) + (upper limits below the cfe threshold)
    n_tot = len(df_filtered[df_filtered['[C/Fe]f'].notna()]) # real data values
    n_tot += len(df_filtered[(df_filtered['ll[C/Fe]f'].notna()) & (df_filtered['ll[C/Fe]f'] >= cfe_limit-0.2)]) # lower limits
    n_tot += len(df_filtered[(df_filtered['ul[C/Fe]f'].notna()) & (df_filtered['ul[C/Fe]f'] <= cfe_limit+0.2)]) # upper limits
    
    if n_tot > 0:
        cempfrac = (n_cemp / n_tot) * 100.
    elif n_tot == 0:
        if n_cemp == 0: cempfrac = np.nan #-100
        if n_cemp != 0: raise ValueError("n_tot is 0 but n_cemp is not 0, which should not be possible.")
    
    if cempfrac > 100.0: 
        raise ValueError("CEMP fraction is greater than 1.0, which should not be possible.")

    # if not np.isnan(cempfrac):
    #     cempfrac = int(normal_round(cempfrac * 100, 0))
    # else:
    #     cempfrac = -1

    # print(n_cemp, n_tot, feh_limit, cfe_limit)
    # print(f"[Fe/H] <= {feh_limit}, [C/Fe] >= {cfe_limit}: {n_cemp}/{n_tot} = {cempfrac:.2f}")

    return cempfrac, n_cemp, n_tot

def calc_cempfrac_mc(
        df,
        feh_limit=-2.0,
        cfe_limit=0.7,
        cfe_stddev=0.15,
        n_iterations=10000,
        print_stats=False,
        plot_distribution=False
    ):
    """
    Calculate the CEMP fraction using a Monte Carlo approach to estimate uncertainties,
    with integrated Wilson Score Interval to account for both classification and 
    sampling uncertainties.
    
    1) Convert the [C/Fe]f (and associated limits) into numpy arrays
    2) Define standard deviation values for the data points (+-0.15 dex)
    3) Assume each data point has a Gaussian/normal distribution centered on the measured 
       value (or limit) with the defined standard deviation
    4) Iterate through a large number of MC iterations (e.g., 10,000), and for each iteration:
        a) Randomly sample [C/Fe]f values for each star from their respective distributions
           (Classification uncertainty propagation)
        b) Sum the number of CEMP stars (n_cemp) and total stars (n_tot) for that iteration
        c) Calculate Wilson Score Interval bounds for the true CEMP fraction
           (Sampling uncertainty from binomial proportion)
        d) Store point estimates and interval bounds
    5) After all iterations, analyze the distributions of CEMP fraction, n_cemp, n_tot,
       and Wilson Score bounds
    6) (Optional) Plot the distributions with Wilson bounds integrated
    7) (Optional) Create corner plots and uncertainty visualizations

    Parameters:
    -----------
    df (DataFrame): Input DataFrame containing stellar data.
    feh_limit (float): [Fe/H] threshold for selecting stars.
    cfe_limit (float): [C/Fe] threshold for classifying CEMP stars.
    n_iterations (int): Number of Monte Carlo iterations to perform.
    cfe_stddev (float): Typical standard deviation to use for sampling [C/Fe] values.
    print_stats (bool): Whether to print detailed statistics.
    plot_distribution (bool): Whether to plot the distribution of CEMP fractions from the simulations.
    
    Returns:
    --------
    stats_dict (dict): A dictionary containing statistics for the CEMP fraction, n_cemp, n_tot, and Wilson Score bounds, as well as the calculated uncertainties.
    stats_dict = {
            'cempfrac': stats_cempfrac,
            'n_cemp': stats_n_cemp,
            'n_tot': stats_n_tot,
            'wilson_lower': stats_wilson_lower,
            'wilson_upper': stats_wilson_upper,
            'unc': {
                'classification': uncertainty_classification,
                'sampling_lower': uncertainty_sampling_lower,
                'sampling_upper': uncertainty_sampling_upper,
                'total_lower': uncertainty_total_lower, 
                'total_upper': uncertainty_total_upper
        }
    """
    
    ## Filter the DataFrame based on [Fe/H] limits
    df_filtered = df[(df['[Fe/H]'] <= feh_limit) | (df['ul[Fe/H]'] <= feh_limit)].copy()
    
    ## Extract relevant columns and convert to numpy arrays
    cfe = df_filtered['[C/Fe]f'].values
    cfe_ll = df_filtered['ll[C/Fe]f'].values
    cfe_ul = df_filtered['ul[C/Fe]f'].values
    
    ## Drop NaN values
    cfe = cfe[~np.isnan(cfe)]
    cfe_ll = cfe_ll[~np.isnan(cfe_ll)]
    cfe_ul = cfe_ul[~np.isnan(cfe_ul)]
    
    ## Initialize arrays to store results from each iteration
    n_iterations = int(n_iterations)  # ensure n_iterations is an integer
    n_cemps = np.zeros(n_iterations)
    n_tots = np.zeros(n_iterations)
    cempfracs = np.zeros(n_iterations)
    
    # Arrays for Wilson Score Interval bounds
    wilson_lowers = np.zeros(n_iterations)
    wilson_uppers = np.zeros(n_iterations)
    
    for i in range(n_iterations):
        
        ## Sample [C/Fe] values for each star (Classification uncertainty)
        sampled_cfe = np.random.normal(cfe, cfe_stddev)
        sampled_cfe_ll = np.random.normal(cfe_ll, cfe_stddev)
        sampled_cfe_ul = np.random.normal(cfe_ul, cfe_stddev)
        
        ## Calculate CEMP counts for this iteration
        ### n_cemp = (all measured values) + (lower limits above the cfe threshold)
        n_cemp = len(sampled_cfe[sampled_cfe >= cfe_limit])  # real data values
        n_cemp += len(sampled_cfe_ll[sampled_cfe_ll >= cfe_limit - 0.2])  # lower limits
        
        ### n_tot = (all measured values) + (all lower limits) + (upper limits below the cfe threshold)
        n_tot = len(sampled_cfe)  # real data values
        n_tot += len(sampled_cfe_ll)  # lower limits
        n_tot += len(sampled_cfe_ul[sampled_cfe_ul <= cfe_limit + 0.2])  # upper limits

        ## Calculate CEMP fraction
        if n_tot > 0:
            cempfrac = (n_cemp / n_tot) * 100.0
        elif n_tot == 0:
            if n_cemp == 0:
                cempfrac = np.nan
            else:
                raise ValueError("n_tot is 0 but n_cemp is not 0, which should not be possible.")
        
        if not np.isnan(cempfrac) and cempfrac > 100.0:
            raise ValueError("CEMP fraction is greater than 100%, which should not be possible.")
        
        ## Calculate Wilson Score Interval bounds (Sampling uncertainty)
        ### Convert to fraction (0-1) for Wilson calculation
        if n_tot > 0:
            lower_frac, upper_frac, _ = wilson_ci(n_cemp, n_tot) # returns lower and upper bounds for the 16th and 84th percentiles (1 sigma) by default
            wilson_lowers[i] = lower_frac * 100.0  # Convert back to percentage
            wilson_uppers[i] = upper_frac * 100.0
        else:
            wilson_lowers[i] = np.nan
            wilson_uppers[i] = np.nan
        
        ## Store results
        n_cemps[i] = n_cemp
        n_tots[i] = n_tot
        cempfracs[i] = cempfrac
    
    ## Calculate statistics for all distributions
    def calc_stats_dict(data, label=""):
        """Calculate statistics for a data array"""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            return {
                'label': label,
                'distribution': data,
                'mean': np.nan,
                'stddev': np.nan,
                'stderr': np.nan,
                'median': np.nan,
                'percentiles': {
                    '5%': np.nan,
                    '16%': np.nan,
                    '50%': np.nan,
                    '84%': np.nan,
                    '95%': np.nan,
                },
                'min': np.nan,
                'max': np.nan,
                'n_valid': 0,
            }
        else:
            stats_dict = {
                'label': label,
                'distribution': valid_data,
                'mean': np.mean(valid_data),
                'stddev': np.std(valid_data),
                'stderr': np.std(valid_data) / np.sqrt(len(valid_data)),
                'median': np.median(valid_data),
                'percentiles': {
                    '5%': np.percentile(valid_data, 5),
                    '16%': np.percentile(valid_data, 16),
                    '50%': np.percentile(valid_data, 50),
                    '84%': np.percentile(valid_data, 84),
                    '95%': np.percentile(valid_data, 95),
                },
                'min': np.min(valid_data),
                'max': np.max(valid_data),
                'n_valid': len(valid_data),
            }
        
        return stats_dict
    
    stats_n_cemp = calc_stats_dict(n_cemps, "n_cemp")
    stats_n_tot = calc_stats_dict(n_tots, "n_tot")
    stats_cempfrac = calc_stats_dict(cempfracs, "CEMP Fraction (%)")
    stats_wilson_lower = calc_stats_dict(wilson_lowers, "Wilson Lower Bounds")
    stats_wilson_upper = calc_stats_dict(wilson_uppers, "Wilson Upper Bounds")

    ## Classification Uncertainty: (symmetric)
    ### - Accounts for the variability in CEMP fraction due to the CEMP classification process and [C/Fe] resampling in the MC
    ### - Standard deviation of the CEMP fraction distribution from the MC iterations
    ### - Assumed symmetric due to Gaussian sampling of [C/Fe] values
    uncertainty_classification = stats_cempfrac['stddev']
    
    ## Sampling Uncertainty: (asymmetric)
    ### - Accounts for the variability in CEMP fraction due to the finite sample size and binomial nature of the problem
    ### - Wilson Score Intervals are naturally asymmetric, especially near boundaries
    ### - Split into lower and upper components, through the distance from point estimate (cempfrac value) 
    ###   to lower Wilson bound averaged across iterations
    uncertainty_sampling_lower = stats_cempfrac['median'] - stats_wilson_lower['median']
    uncertainty_sampling_upper = stats_wilson_upper['median'] - stats_cempfrac['median']

    ## Combined Total Uncertainty:
    ### - Linear addition for approximation of correlated uncertainties.
    uncertainty_total_lower = uncertainty_classification + uncertainty_sampling_lower 
    uncertainty_total_upper = uncertainty_classification + uncertainty_sampling_upper 
    
    ## Bounding the total uncertainty to ensure the CEMP fraction remains within [0, 100]%
    if stats_cempfrac['median'] - uncertainty_total_lower < 0:
        uncertainty_total_lower = np.abs(stats_cempfrac['median'] - 0)
    if stats_cempfrac['median'] + uncertainty_total_upper > 100:
        uncertainty_total_upper = np.abs(100.0 - stats_cempfrac['median'])

    ## Consolidate Dictionaries
    stats_dict = {
        'cempfrac': stats_cempfrac,
        'n_cemp': stats_n_cemp,
        'n_tot': stats_n_tot,
        'wilson_lower': stats_wilson_lower,
        'wilson_upper': stats_wilson_upper,
        'unc': {
            'classification': uncertainty_classification,
            'sampling_lower': uncertainty_sampling_lower,
            'sampling_upper': uncertainty_sampling_upper,
            'total_lower': uncertainty_total_lower, 
            'total_upper': uncertainty_total_upper
        }
    }
    
    ## Optional: Print detailed statistics
    if print_stats:
        print(f"Data summary:")
        print(f"  {len(cfe):5} [C/Fe]f measured values")
        print(f"  {len(cfe_ll):5} ll[C/Fe]f lower limits")
        print(f"  {len(cfe_ul):5} ul[C/Fe]f upper limits")
        
        print("\n" + "="*70)
        print("CEMP FRACTION ANALYSIS WITH WILSON SCORE INTERVAL")
        print("="*70)
        
        print(f"\ncempfrac Statistics")
        print("-" * 70)
        print(f"  Mean:                {stats_cempfrac['mean']:.2f}%")
        print(f"  Median:              {stats_cempfrac['median']:.2f}%")
        print(f"  Std Dev:             {stats_cempfrac['stddev']:.2f}%")
        print(f"  Std Err:             {stats_cempfrac['stderr']:.2f}%")

        print(f"\nn_cemp Statistics")
        print("-" * 70)
        print(f"  Mean:                {stats_n_cemp['mean']:.1f}")
        print(f"  Median:              {stats_n_cemp['median']:.1f}")
        print(f"  Std Dev:             {stats_n_cemp['stddev']:.2f}")
        
        print(f"\nn_tot Statistics")
        print("-" * 70)
        print(f"  Mean:                {stats_n_tot['mean']:.1f}")
        print(f"  Median:              {stats_n_tot['median']:.1f}")
        print(f"  Std Dev:             {stats_n_tot['stddev']:.2f}")
         
        print(f"\nWilson Score Interval")
        print("-" * 70)
        print(f"  Lower Bound (median):  {stats_wilson_lower['median']:.2f}%")
        print(f"  Upper Bound (median):  {stats_wilson_upper['median']:.2f}%")
        print(f"  Interval Width (68%):  {stats_wilson_upper['median'] - stats_wilson_lower['median']:.2f}%")
              
        print(f"\nUncertainty Breakdown")
        print("-" * 70)
        print(f"  Classification (MC):    {uncertainty_classification:.2f}%")
        print(f"  Sampling (Wilson C.I.): [{uncertainty_sampling_lower:.2f}%, {uncertainty_sampling_upper:.2f}%]")
        print(f"  Combined Uncertainty:   [{uncertainty_total_lower:.2f}%, {uncertainty_total_upper:.2f}%]")

    # Optional: Plot distributions
    if plot_distribution: 
        plot_cempfrac_distributions(stats_dict, n_iterations)
    
    return stats_dict

def wilson_ci(successes, n, confidence=0.6827):
    """
    Calculate Wilson Score Interval for an advanced binomial proportion.
    - Stays within [0, 1] with good coverage.
    - Decent approximation for small sample size `n` and where `p=0` or `p=1`.
    - Formed by adding 2 successes and 2 failures to the `n_x` and `n` values in `p=n_x/n`

https://www.statisticshowto.com/wilson-ci/

    Parameters:
    -----------
    successes : int
        Number of successes (e.g., CEMP stars)
    n : int
        Total number of trials
    confidence : float
        Confidence level (default 0.6827 for 68.27% [1 sigma or 16th-84th percentiles])
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound, point_estimate) of the confidence interval
    """
    if n == 0 or successes < 0:
        return np.nan, np.nan, np.nan
    
    z = sp.stats.norm.ppf((1 + confidence) / 2) # z-score for the confidence level
    p_hat = successes / n # sample proportion
    
    denominator = 1 + (z**2 / n)
    center = (1/denominator) * (p_hat + z**2 / (2*n))  
    margin = (1/denominator) * (z/(2*n)) * np.sqrt(4*n*p_hat*(1-p_hat) + z**2) 
    
    lower = center - margin
    upper = center + margin
    
    # Clamp to [0, 1] # Wilson Score Interval can sometimes produce bounds outside [0, 1], so we clamp them
    lower = max(0, lower)
    upper = min(1, upper)
    
    return lower, upper, p_hat

def plot_cempfrac_distributions(stats_dict, n_iterations):
    """
    Plot distributions for classification uncertainty (3 panels) and
    Wilson Score sampling uncertainty (1 panel)
    """
    
    def _plot_distribution(ax, data, mean, stddev, bin_width,
                         sigma_levels=[1, 2, 3], 
                         ilabel='', imarker='.', icolor='k', 
                         xlabel='', show_label_in_legend=True):
        
        """Generic distribution plotting function"""
        
        # Remove NaN values
        valid_data = data[~np.isnan(data)]

        # Determine bin edges based on the data range and specified bin width
        start = np.floor(np.min(valid_data) / bin_width) * bin_width - bin_width/2
        end   = np.ceil(np.max(valid_data) / bin_width) * bin_width + bin_width/2
        bin_edges = np.arange(start, end + bin_width, bin_width)
        
        # Create and normalize histogram
        counts, _ = np.histogram(valid_data, bins=bin_edges) #, density=True)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_range = np.linspace(bin_edges[0] - bin_width/2, bin_edges[-1] + bin_width/2, 100)
        
        norm_counts = 1 / (np.sum(counts * bin_width))
        histy = counts * norm_counts
        histy_err = np.sqrt(counts) * norm_counts

        # Plot histogram with error bars
        x = np.concatenate(([bin_centers[0] - bin_width], bin_centers, [bin_centers[-1] + bin_width]))
        y = np.concatenate(([0], histy, [0]))
        ax.plot(x, y, drawstyle='steps-mid', color=icolor, zorder=3)
        
        label_prefix = '' if show_label_in_legend else '_'
        ax.errorbar(bin_centers, histy, yerr=histy_err, marker=imarker, markersize=8, 
                   drawstyle='steps-mid', color=icolor, label=ilabel, alpha=0.6, zorder=3)

        # Fit Gaussian or Binomial to the distribution
        if not np.isnan(stddev) and stddev > 0:
            gaussian_fit = sp.stats.norm.pdf(bin_range, mean, stddev)
            ax.plot(bin_range, gaussian_fit, color=icolor, label=f'{label_prefix}Gaussian Fit', zorder=4)
            ylim = np.max(histy) * 1.3 if np.max(gaussian_fit) < np.max(histy) * 1.1 else np.max(gaussian_fit) * 1.1
        else:
            ylim = np.max(histy) * 1.3
            
        # Plot mean line
        ax.vlines(mean, -1, 1, linestyle='dashed', color=icolor, label=fr'{label_prefix}$\mu =$ {mean:.2f}', zorder=1)
        
        # Plot sigma regions
        sigma_colors = [icolor if icolor != 'k' else 'cadetblue' for _ in sigma_levels]
        sigma_alphas = np.linspace(0.3, 0.1, len(sigma_levels))
        for i, sigma_level in enumerate(sigma_levels):
            ax.axvspan(
                mean - sigma_level*stddev, mean + sigma_level*stddev, 
                alpha=sigma_alphas[i], color=sigma_colors[i],
                label=fr'{label_prefix}{sigma_level}$\sigma = \pm$ {sigma_level*stddev:.2f}', zorder=0
            )
        
        # Set aesthetics
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized Counts")
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_ylim(0, ylim if not np.isnan(ylim) else 1.0)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=1)
        
        return ax
    
    def bin_width_from_data(xdata, max_bins=25):
        """Calculate bin width based on data range and desired maximum number of bins"""
        valid_data = xdata[~np.isnan(xdata)]
        if len(valid_data) == 0:
            return 1.0  # default bin width if no valid data
        data_range = np.max(valid_data) - np.min(valid_data)
        max_bins = min(max_bins, len(np.unique(valid_data)))
        bin_width = data_range / max_bins if data_range > 0 else 1.0
        return bin_width
    
    # Extract stats
    stats_cempfrac = stats_dict['cempfrac']
    stats_n_cemp = stats_dict['n_cemp']
    stats_n_tot = stats_dict['n_tot']
    stats_wilson_lower = stats_dict['wilson_lower']
    stats_wilson_upper = stats_dict['wilson_upper']
    stats_unc = stats_dict['unc']
    
    # Figure 1: MC Classification (3 panels, all sigma levels)
    fig1, ax1 = plt.subplots(1, 3, figsize=(16, 4))
    fig1.suptitle(f"Monte Carlo Classification Uncertainty: {n_iterations:.1e} Iterations", fontsize=14)
    
    _plot_distribution(
            ax1[0], stats_n_tot['distribution'], stats_n_tot['mean'], 
            stats_n_tot['stddev'], ilabel='n_tot',
            bin_width=bin_width_from_data(stats_n_tot['distribution']),
            xlabel='Total Number of Stars'
        )
    _plot_distribution(
            ax1[1], stats_n_cemp['distribution'], stats_n_cemp['mean'], 
            stats_n_cemp['stddev'], ilabel='n_cemp',
            bin_width=bin_width_from_data(stats_n_cemp['distribution']),
            xlabel='Number of CEMP Stars'
        )
    _plot_distribution(
            ax1[2], stats_cempfrac['distribution'], stats_cempfrac['mean'], 
            stats_cempfrac['stddev'], ilabel='cempfrac',
            bin_width=bin_width_from_data(stats_cempfrac['distribution']),
            xlabel='CEMP Fraction (%)'
        )
    plt.tight_layout()
    
    # Figure 2: Wilson Score (1 panel, only 1 sigma level)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    fig2.suptitle(f"Wilson Score Interval Sampling Uncertainty:", fontsize=14)
    
    ax2 = _plot_distribution(
            ax2, stats_wilson_upper['distribution'], stats_wilson_upper['mean'], 
            stats_wilson_upper['stddev'], sigma_levels=[1],
            bin_width=bin_width_from_data(stats_wilson_upper['distribution']),
            ilabel='Upper Bounds', imarker='^', icolor='b', 
            xlabel='CEMP Fraction (%)', show_label_in_legend=False
        )
    ax2 = _plot_distribution(
            ax2, stats_cempfrac['distribution'], stats_cempfrac['mean'], 
            stats_cempfrac['stddev'], sigma_levels=[1],
            bin_width=bin_width_from_data(stats_cempfrac['distribution']),
            ilabel='CEMP Fraction Estimates', imarker='o', icolor='k',
            xlabel='CEMP Fraction (%)', show_label_in_legend=False
        )
    ax2 = _plot_distribution(
            ax2, stats_wilson_lower['distribution'], stats_wilson_lower['mean'], 
            stats_wilson_lower['stddev'], sigma_levels=[1],
            bin_width=bin_width_from_data(stats_wilson_lower['distribution']),
            ilabel='Lower Bounds', imarker='s', icolor='r',
            xlabel='CEMP Fraction (%)', show_label_in_legend=False
        )
    
    ## Set y-axis limits to encompass all distributions
    all_y_values = []
    for line in ax2.get_lines():
        y_data = line.get_ydata()
        all_y_values.extend(y_data)
    max_y = np.max(all_y_values)
    ax2.set_ylim(0, max_y * 1.2)
    
    ## Plot the value and uncertainty estimates for the CEMP fraction with error bars
    ax2.errorbar(stats_cempfrac['mean'], max_y*1.15, xerr=[[stats_unc['total_lower']], [stats_unc['total_upper']]], label='total_uncertainty', 
                 fmt='o', color='k', capsize=5, zorder=5)
    ax2.errorbar(stats_cempfrac['mean'], max_y*1.05, xerr=stats_unc['classification'], label='classification_mc', 
                 fmt='o', color='c', capsize=5, zorder=5)
    ax2.errorbar(stats_cempfrac['mean'], max_y*0.95, xerr=[[stats_unc['sampling_lower']], [stats_unc['sampling_upper']]], label='sampling_wilson', 
                 fmt='o', color='m', capsize=5, zorder=5)
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    
    plt.show()
    
    return fig1, fig2

################################################################################
## Calculating Dtrans 

def calc_dtrans(ch, oh=None):
    """
    Calculate Dtrans values for the given [C/H] value.
    
        Dtrans = log10(10^[C/H] + 0.9 * 10^[O/H])
    
    If [O/H] is not provided, this function assumes carbon and 
    oxygen production are correlated within -0.6 <= [C/O] <= 0.0, 
    where [O/H] = [C/H] - [C/O].
    --> Returns: (Dtrans_l, Dtrans_u)
    
    If [O/H] is provided, this function returns a single Dtrans value.
    --> Returns: Dtrans
    """
    
    assert isinstance(ch, (float, int)), "Input ch must be a float or int"

    if oh is None:
        # [C/O] values, representing the delta (dex) between C and O abundances
        co_l = 0.0 # use the higher [C/O] value for lower [O/H] ==> lower Dtrans value
        co_u = -0.6 # use the lower [C/O] value for higher [O/H] ==> upper Dtrans value

        oh_l = ch - co_l
        oh_u = ch - co_u

        Dtrans_l = np.log10(10**ch + (0.9 * 10**oh_l))
        Dtrans_u = np.log10(10**ch + (0.9 * 10**oh_u))

        return (Dtrans_l, Dtrans_u)

    else:
        assert isinstance(oh, (float, int)), "Input oh must be a float or int"
        
        Dtrans = np.log10(10**ch + (0.9 * 10**oh))
        
        return Dtrans

def calc_dtrans_line(feh):
    """
    Calculate Dtrans values for the solar abundances, scaled by metallicity 
    from a given [Fe/H] value (or array of values).
    This function creates the diagonal line in the Dtrans vs [Fe/H] plot.
    """
    ch = 0.0 # [C/H] = 0.0, the solar ratio for carbon

    # [C/O] values, representing the delta (dex) between C and O abundances
    co_lower = 0.0 
    co_upper = -0.6

    oh_l = ch - co_lower # [C/H] = 0.0
    oh_u = ch - co_upper # [C/H] = 0.6

    Dtrans_l = np.log10(10**(ch+feh) + (0.9 * 10**(oh_l+feh)))
    Dtrans_u = np.log10(10**(ch+feh) + (0.9 * 10**(oh_u+feh)))

    return Dtrans_l , Dtrans_u
    
def calc_dtrans_columns(df, precision=2):
    """
    Calculate Dtrans values for the given dataframe. The dataframe must have
    either a [C/H]f or ul[C/H]f column. 
    
    The function will add four new columns to the dataframe:
        Dtrans_l: lower [O/H] value used when calculating Dtrans (using co_lower)
        Dtrans_llim: lower [O/H] value used when calculating Dtrans (using co_lower) for upper limits
        Dtrans_u: upper [O/H] value used when calculating Dtrans (using co_upper)
        Dtrans_ulim: upper [O/H] value used when calculating Dtrans (using co_upper) for upper limits
    """
    
    # [C/O] values, representing the delta (dex) between C and O abundances
    co_lower = 0.0 
    co_upper = -0.6

    def dtrans(ch, oh, precision=precision):
        return normal_round(np.log10(10**ch + (0.9 * 10**oh)), precision)
    
    for i, row in df.iterrows():
        if not pd.isna(row['[C/H]f']):
            ch = float(row['[C/H]f'])

            oh_l = ch - co_lower
            Dtrans_l = dtrans(ch, oh_l)
            df.loc[i, 'Dtrans_l'] = Dtrans_l
            df.loc[i, 'Dtrans_llim'] = np.nan

            oh_u = ch - co_upper
            Dtrans_u = dtrans(ch, oh_u)
            df.loc[i, 'Dtrans_u'] = Dtrans_u
            df.loc[i, 'Dtrans_ulim'] = np.nan

        elif not pd.isna(row['ul[C/H]f']):
            ch = float(row['ul[C/H]f'])

            oh_l = ch - co_lower
            Dtrans_l = dtrans(ch, oh_l)
            df.loc[i, 'Dtrans_l'] = np.nan
            df.loc[i, 'Dtrans_llim'] = Dtrans_l
            
            oh_u = ch - co_upper
            Dtrans_u = dtrans(ch, oh_u)
            df.loc[i, 'Dtrans_u'] = np.nan
            df.loc[i, 'Dtrans_ulim'] = Dtrans_u

        else:
            print(f"Row {i} does not have [C/H]f or ul[C/H]f: {row['Name']}")

    ## Add the columns to the dataframe, if they do not exist already
    for col in ['Dtrans_u', 'Dtrans_ulim', 'Dtrans_l', 'Dtrans_llim']:
        if col not in df.columns:
            df[col] = np.nan

    ## Round the calculated values
    for i, row in df.iterrows():
        for col in ['Dtrans_u', 'Dtrans_ulim', 'Dtrans_l', 'Dtrans_llim']:
            df.at[i, col] = normal_round(row[col], 2)

    return df