#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 16:43:20 2020

@author: idchiang
"""
import numpy as np


def acoMW(metal=8.69):
    """
    Generate the Milky Way alpha_CO

    Parameters
    ----------
    metal : scalar or ndarray, optional
        A scalar or an ndarray matching the output shape

    Returns
    -------
    aco : scalar or ndarray
        The Milky Way alpha_CO
    """
    aco = np.full_like(metal, 4.35)
    if np.isscalar(metal):
        aco = float(aco)
    return aco


def acoS12(metal):
    """
    Generate the Schruba+12 Table 7: HERACLES-All galaxies alpha_CO

    Parameters
    ----------
    metal : scalar or ndarray
        12 + log(O/H)

    Returns
    -------
    aco : scalar or ndarray
        The Schruba+12 alpha_CO
    """
    aco = 8.0 * 10**(-2.0 * (metal - 8.7))
    return aco


def acoB13(metal, mstar, mhi, ico, sigma100_gmc=1.0, gamma=0.5):
    """
    Generate the Bolatto+13 Eq. (31) alpha_CO.
    metal, mstar, mhi, and ico should have the same shape.

    Parameters
    ----------
    metal : scalar or ndarray
        12 + log(O/H)
    mstar : scalar or ndarray
        Stellar mass surface density in [Msun/pc^2]
    mhi : scalar or ndarray
        HI mass surface density in [Msun/pc^2], excluding the 1.36 factor for
        helium and heavy elements
    ico : scalar or ndarray
        Integrated CO J=1-0 intensity in [K*km/s]
    sigma100_gmc : scalar, optional
        The mass surface density for GMC in [100 Msun/pc^2]
        The default is 1.0.
    gamma : scalar, optional
        The exponential index for dense regions

    Returns
    -------
    aco : scalar or ndarray
        The Bolatto+13 alpha_CO
    """
    if np.isscalar(metal):
        assert np.isscalar(mstar) and np.isscalar(mhi) and np.isscalar(ico)
    else:
        assert metal.shape == mstar.shape == mhi.shape == ico.shape
    #
    with np.errstate(invalid='ignore'):
        sigma100_tot0 = (mstar + mhi * 1.36) / 100.0
        ico100 = ico / 100.0
        z_rel = 10**(metal - 8.69)
        aco0 = 2.9 * np.exp(0.4 / z_rel / sigma100_gmc)
    #
    aco = acoMW(metal)
    max_AD = 10.0
    if np.isscalar(metal):
        while(max_AD > 0.1):
            prev_aco = aco
            aco = aco0
            sigma100_tot = sigma100_tot0 + ico100 * aco
            with np.errstate(invalid='ignore'):
                if sigma100_tot > 1.0:
                    aco *= sigma100_tot**(-gamma)
                max_AD = np.nanmax(np.abs(aco - prev_aco))
    else:
        while(max_AD > 0.1):
            prev_aco = aco.copy()
            aco = aco0.copy()
            sigma100_tot = sigma100_tot0 + ico100 * aco
            with np.errstate(invalid='ignore'):
                mask = sigma100_tot > 1.0
                if np.any(mask):
                    aco[mask] *= sigma100_tot[mask]**(-gamma)
                max_AD = np.nanmax(np.abs(aco - prev_aco))
    return aco


def acoH15(metal):
    """
    Generate the Hunt+15 Sect. 5.1 alpha_CO

    Parameters
    ----------
    metal : scalar or ndarray
        12 + log(O/H)

    Returns
    -------
    aco : scalar or ndarray
        The Hunt+15alpha_CO
    """
    aco = 4.35 * 10**(-1.96 * (metal - 8.69))
    if np.isscalar(metal):
        if metal >= 8.69:
            aco = 4.35
    else:
        with np.errstate(invalid='ignore'):
            aco[metal >= 8.69] = 4.35
    return aco
