#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:23:58 2020

@author: idchiang
"""
import numpy as np
from scipy.stats import median_absolute_deviation


def RMSE(y_true, y_pred, sample_weight=None):
    """
    Calculate the root-mean-square error.

    Parameters
    ----------
    y_true : array-like
        Ground true values

    y_pred : array-like (same shape as y_true)
        Estimated values

    sample_weight : array-like (same shape as y_true), default=None
        Sample weights

    Returns
    -------
    rmse : float
        The root-mean-square error

    """
    # Sanity checks
    if type(y_true) is not np.ndarray:
        y_true = np.array(y_true)
    if type(y_pred) is not np.ndarray:
        y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        print('### RMSE alert: y_true.shape != y_pred.shape. Returning -1.')
        return -1
    if sample_weight is not None:
        if type(sample_weight) is not np.ndarray:
            sample_weight = np.array(sample_weight)
        if y_true.shape != sample_weight.shape:
            print('### RMSE alert: y_true.shape != sample_weight.shape.',
                  'Returning -1.')
            return -1
    # Masking nan
    if sample_weight is None:
        mask = np.isfinite(y_true + y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
    else:
        mask = np.isfinite(y_true + y_pred + sample_weight)
        y_true, y_pred, sample_weight[mask] = \
            y_true[mask], y_pred[mask], sample_weight[mask]
    # Return result
    rmse = np.sqrt(np.average((y_pred - y_true)**2, weights=sample_weight))
    return rmse


def rms_from_mad(data):
    """
    Estimate the root-mean-square (rms) error from median absolute deviation
    (MAD). Useful for data with non-negligible outliers.

    Parameters
    ----------
    data : ndarray
        Input data.

    Returns
    -------
    rms : scalar
        rms error converted from MAD.
    """
    rms = median_absolute_deviation(data) / 0.6745
    return rms


def multiband_bkgs(data_arr, bkgmask0):
    """
    Calculate the background level with outlier rejection from multibands

    Parameters
    ----------
    data_arr : list
        A list containing all data with the same shape
    bkgmask0 : array_like
        An initial mask marking the background region. True=background.

    Returns
    -------
    bkgs : list
        A list containing the background level of each band
    """
    bkgmask = bkgmask0.copy()
    outliermask = np.zeros_like(bkgmask0, dtype=bool)
    for data in data_arr:
        curbkgmask = bkgmask0 & np.isfinite(data)
        bkgmask &= np.isfinite(data)
        AD = np.abs(data - np.median(data[curbkgmask]))
        MAD = np.median(AD[curbkgmask])
        outliermask |= AD > 3 * MAD
    bkgmask &= (~outliermask)
    bkgs = [np.nanstd(data[bkgmask]) for data in data_arr]
    return bkgs
