#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import, unicode_literals)


import  sys, os, glob, time
import numpy as np
from scipy import odr

################################################################################

def polyfit_with_asym_errors(x, y, unc_lower, unc_upper, n_iterations=1000, x_range=None, deg=1, semilog=False):
    """
    Perform bootstrap/Monte Carlo fitting accounting for asymmetric uncertainties.
     - For each iteration, randomly sample from the asymmetric uncertainties to create a new y dataset.
     - Fit a polynomial to this new dataset and store the coefficients.
     - After all iterations, calculate the median fit and the 16th/84th percentiles to represent the uncertainty band.
     
    Returns:
        fit_params: Array of fitted coefficients for each bootstrap iteration
        (x_fit, y_fit, y_fits, y_lower, y_upper): Tuple containing:
            x_fit: x-values for the fit line
            y_fit: Best fit line without uncertainties
            y_fits: Array of fit lines for each bootstrap iteration
            y_lower: 16th percentile of the fit lines (lower bound of uncertainty)
            y_upper: 84th percentile of the fit lines (upper bound of uncertainty)
    """
    n_points = len(x)
    fit_params = []
    
    # Transform x to log space
    if semilog:
        x_fitspace = np.log10(x)
    else:
        x_fitspace = x.copy()
    
    for _ in range(int(n_iterations)):
        # Randomly sample from asymmetric uncertainties
        y_samples = np.zeros(n_points)
        for i in range(n_points):
            # Sample from appropriate side based on sign
            if np.random.rand() < 0.5:
                y_samples[i] = y[i] - np.abs(np.random.normal(0, unc_lower[i]))
            else:
                y_samples[i] = y[i] + np.abs(np.random.normal(0, unc_upper[i]))
        
        # Fit this bootstrap sample
        coeffs_boot = np.polyfit(x_fitspace, y_samples, deg=deg)
        fit_params.append(coeffs_boot)
    
    fit_params = np.array(fit_params)
    
    # Create x_fit in linear/log space
    if semilog:
        if x_range is None:
            x_fit = np.logspace(np.log10(x.min()) - 0.2, np.log10(x.max()) + 0.2, 1000)
        else:
            x_fit = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), 1000)
        x_eval = np.log10(x_fit)
    else:
        if x_range is None:
            x_fit = np.linspace(x.min() - 0.5, x.max() + 0.5, 1000)
        else:
            x_fit = np.linspace(x_range[0], x_range[1], 1000)
        x_eval = x_fit

    # Evaluate polynomial
    y_fit = np.polyval(np.median(fit_params, axis=0), x_eval)
    y_fits = np.array([np.polyval(params, x_eval) for params in fit_params])
    y_lower = np.percentile(y_fits, 16, axis=0)
    y_upper = np.percentile(y_fits, 84, axis=0)
    
    return fit_params, (x_fit, y_fit, y_fits, y_lower, y_upper)

def polyfit_with_asym_errors_odr(
        x, x_unc_lower, x_unc_upper, 
        y, y_unc_lower, y_unc_upper, 
        n_iterations=1000, x_range=None, deg=1, semilog=False
    ):
    """
    Perform bootstrap/Monte Carlo ODR fitting accounting for asymmetric uncertainties in both x and y.
    Uses Orthogonal Distance Regression to fit in both directions.
    
    Parameters:
    -----------
    x : array
        x data
    y : array
        y data
    x_unc_lower : array
        Lower uncertainties in x
    x_unc_upper : array
        Upper uncertainties in x
    y_unc_lower : array
        Lower uncertainties in y
    y_unc_upper : array
        Upper uncertainties in y
    n_iterations : int
        Number of bootstrap iterations
    x_range : tuple or None
        Range for fit evaluation [x_min, x_max]
    deg : int
        Degree of polynomial fit
    semilog : bool
        If True, fit in log(x) space
    
    Returns:
    --------
    fit_params : array
        Array of fitted coefficients for each bootstrap iteration
    (x_fit, y_fit, y_fits, y_lower, y_upper) : tuple
        Fit results and uncertainty bands
    """
    n_points = len(x)
    fit_params = []
    
    # Define polynomial model for ODR
    def poly_func(p, x):
        """Polynomial function: p[0] + p[1]*x + p[2]*x^2 + ..."""
        return np.polyval(p, x)
    
    # Transform x to log space if needed
    if semilog:
        x_fitspace = np.log10(x)
        x_unc_lower_fitspace = x_unc_lower / (x * np.log(10)) # Error propagation for log
        x_unc_upper_fitspace = x_unc_upper / (x * np.log(10)) # Error propagation for log
    else:
        x_fitspace = x.copy()
        x_unc_lower_fitspace = x_unc_lower.copy()
        x_unc_upper_fitspace = x_unc_upper.copy()
    
    for _ in range(int(n_iterations)):
        # Randomly sample from asymmetric uncertainties in both x and y
        x_samples = np.zeros(n_points)
        y_samples = np.zeros(n_points)
        x_samples_unc = np.zeros(n_points)
        y_samples_unc = np.zeros(n_points)
        
        for i in range(n_points):
            # Sample x from asymmetric uncertainties -- the sample uncertainties would be considered as symmetric for the ODR fit, but the sampling accounts for the asymmetry
            if np.random.rand() < 0.5:
                x_samples[i] = x_fitspace[i] - np.abs(np.random.normal(0, x_unc_lower_fitspace[i]))
                x_samples_unc[i] = np.sqrt(x_unc_lower_fitspace[i]**2)
            else:
                x_samples[i] = x_fitspace[i] + np.abs(np.random.normal(0, x_unc_upper_fitspace[i]))
                x_samples_unc[i] = np.sqrt(x_unc_upper_fitspace[i]**2)
            
            # Sample y from asymmetric uncertainties -- the sample uncertainties would be considered as symmetric for the ODR fit, but the sampling accounts for the asymmetry
            if np.random.rand() < 0.5:
                y_samples[i] = y[i] - np.abs(np.random.normal(0, y_unc_lower[i]))
                y_samples_unc[i] = np.sqrt(y_unc_lower[i]**2)
            else:
                y_samples[i] = y[i] + np.abs(np.random.normal(0, y_unc_upper[i]))
                y_samples_unc[i] = np.sqrt(y_unc_upper[i]**2)
        
        # Initial guess using standard polyfit
        guess_params = np.polyfit(x_samples, y_samples, deg=deg)
        
        # Create ODR model
        model = odr.Model(poly_func)
        data = odr.RealData(x_samples, y_samples, sx=x_samples_unc, sy=y_samples_unc)
        odr_obj = odr.ODR(data, model, beta0=guess_params)
        output = odr_obj.run()
        # print(output.pprint())
        fit_params.append(output.beta)
        
    fit_params = np.array(fit_params)
    # print("Fit parameters shape:", fit_params.shape)
    
    # Create x_fit in linear/log space
    if semilog:
        if x_range is None:
            x_fit = np.logspace(np.log10(x.min()) - 0.2, np.log10(x.max()) + 0.2, 1000)
        else:
            x_fit = np.logspace(np.log10(x_range[0]), np.log10(x_range[1]), 1000)
        x_eval = np.log10(x_fit)
    else:
        if x_range is None:
            x_fit = np.linspace(x.min() - 0.5, x.max() + 0.5, 1000)
        else:
            x_fit = np.linspace(x_range[0], x_range[1], 1000)
        x_eval = x_fit

    # Evaluate polynomial
    y_median_fit = np.polyval(np.median(fit_params, axis=0), x_eval)
    y_fits = np.array([np.polyval(params, x_eval) for params in fit_params])
    y_lower = np.percentile(y_fits, 16, axis=0)
    y_upper = np.percentile(y_fits, 84, axis=0)
    
    return fit_params, (x_fit, y_median_fit, y_fits, y_lower, y_upper)