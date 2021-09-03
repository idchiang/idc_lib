#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:37:24 2020

@author: idchiang
"""
import numpy as np
import astropy.units as u
from astropy.constants import c, h, k_B

hkB_KHz = (h / k_B).to(u.K / u.Hz).value
B_const = 2e20 * (h / c**2).to(u.J * u.s**3 / u.m**2).value
c_ums = c.to(u.um / u.s).value
MBB_const = 0.00020884262122368297


def B_fast(T, freq):
    """
    Generate the blackbody SED in MJy/sr

    Parameters
    ----------
    T : scalar or numpy.ndarray
        Blackbody temperature in K.
    freq : scalar or numpy.ndarray
        Frequency in Hz.

    Returns
    -------
    Blackbody SED in MJy/sr : scalar or ndarray
        If both T and freq are scalar --> return a scalar
        Elif only one of them is scalar --> return a ndarray with the same
        shape as the other one.
        Elif both of them are ndarray with matching shape --> return a ndarray
        with the same shape
        Else --> error

    """
    with np.errstate(over='ignore'):
        return B_const * freq**3 / (np.exp(hkB_KHz * freq / T) - 1)


def SE(lambda_, Sigma_d, T_d, beta=2.0, kappa_160=10.10):
    """
    Generate Simple-Emissivity MBB in MJy/sr

    Parameters
    ----------
    lambda_ : scalar or numpy.ndarray
        Wavelength in um.
    Sigma_d : scalar or numpy.ndarray
        Dust surface density in Msun/pc^2
    T_d : scalar or numpy.ndarray
        Dust temperature in K
    beta : scalar or numpy.ndarray
        The power index in emissivity. The default is 2.0.
    kappa160 : float, optional
        The emissivity at 160um in cm^2/g. The default is 10.10.
        (Chiang+18)

    Returns
    -------
    Dust emission SED : scalar or numpy.ndarray
        The SE MBB in MJy/sr

    """
    emissivity = kappa_160 * (160.0 / lambda_)**beta
    freq = c_ums / lambda_
    return MBB_const * Sigma_d * emissivity * B_fast(T_d, freq)


def BE(lambda_, Sigma_d, T_d, beta_2, beta=2.0, lambda_b=300.0,
       kappa_160=20.73):
    """
    Generate Broken-Emissivity MBB in MJy/sr

    Parameters
    ----------
    lambda_ : scalar or numpy.ndarray
        Wavelength in um.
        If ndarray, all other inputs must be scalar.
    Sigma_d : scalar or numpy.ndarray
        Dust surface density in Msun/pc^2.
        If ndarray, all other ndarray must have the same shape.
    T_d : scalar or numpy.ndarray
        Dust temperature in K
        If ndarray, all other ndarray must have the same shape.
    beta_2 : scalar or numpy.ndarray
        The power index in emissivity at long wavelength.
        If ndarray, all other ndarray must have the same shape.
    beta : scalar or numpy.ndarray, optional
        The power index in emissivity at short wavelength. The default is 2.0.
        If ndarray, all other ndarray must have the same shape.
    lambda_b : scalar or numpy.ndarray, optional
        The break wavelength in um. The default is 300.0.
        If ndarray, all other ndarray must have the same shape.
    kappa160 : float, optional
        The emissivity at 160um in cm^2/g. The default is 20.73.
        (Chiang+18)

    Returns
    -------
    Dust emission SED : scalar or numpy.ndarray
        The BE MBB in MJy/sr

    """
    freq = c_ums / lambda_
    if np.isscalar(lambda_):
        # Case 1: only 1 wavelength; multi-dim parameter allowed.
        emissivity = kappa_160
        if lambda_ < lambda_b:
            emissivity *= (1.0 / lambda_)**beta
        else:
            emissivity *= ((1.0 / lambda_b)**beta) * \
                ((lambda_b / lambda_)**beta_2)
    else:
        # Case 2: ndarray wavelength; parameters must be scalar.
        emissivity = np.full_like(lambda_, kappa_160)
        short_lambda_ = lambda_ < lambda_b
        emissivity[short_lambda_] *= (1.0 / lambda_[short_lambda_])**beta
        emissivity[~short_lambda_] *= ((1.0 / lambda_b)**beta) * \
            ((lambda_b / lambda_[~short_lambda_])**beta_2)
    #
    freq = c_ums / lambda_
    return MBB_const * Sigma_d * emissivity * B_fast(T_d, freq)
