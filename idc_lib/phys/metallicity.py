#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:27:57 2020

@author: idchiang
"""
import os
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
import astropy.units as u


rng0 = np.random.default_rng()


def metal_generator(gal, radius_arcsec, doError=False, rng=None):
    """
    Generate 12+log(O/H) from references with radius in arcsec.

    Parameters
    ----------
    gal : str
        Name of the galaxy
    radius_arcsec : scalar or ndarray
        Radius in arcsec
    doError : bool
        Include Gaussian errors in radial gradient coefficients or not
    rng : numpy.random.Generator, optional
        External numpy random generator

    Returns
    -------
    metal : scalar or ndarray
        12+log(O/H)
    """
    #
    if doError and (rng is None):
        rng = rng0
    # Import data table
    module_dir, _ = os.path.split(__file__)
    metalgradient_dir = \
        os.path.join(module_dir, "data/metallicity_gradients.csv")
    df = pd.read_csv(metalgradient_dir)
    df = df.set_index(df['gal'])
    if gal not in df.index:
        print('Supported galaxies:', df['gal'].values)
        raise ValueError('12+log(O/H) generator: galaxy ' + gal +
                         ' not supported!')
    # Generate proper radius for calculation
    if df.loc[gal, 'method'] == 'kpc':
        # old "radius_kpc"
        radius_cal = \
            radius_arcsec * df.loc[gal, 'dist_kpc'] * Angle(1 * u.arcsec).rad
    elif df.loc[gal, 'method'] == 'r25':
        # old "radius_r25"
        radius_cal = radius_arcsec / df.loc[gal, 'R25_arcsec']
    else:
        print('Supported methods: r25, kpc')
        raise ValueError('12+log(O/H) generator: method ' +
                         df.loc[gal, 'method'] + ' not supported!')
    # Calculate 12+log(O/H)
    assert df.loc[gal, 'slope'] >= 0
    metal = df.loc[gal, 'metal0'] - df.loc[gal, 'slope'] * radius_cal
    if doError:
        metal += rng.normal(0., df.loc[gal, 'emetal0']) - \
            rng.normal(0., df.loc[gal, 'eslope']) * radius_cal
    return metal


def metal2Z(metal):
    """
    Convert 12+log(O/H) to Z

    Parameters
    ----------
    metal : scalar or ndarray
        12 + log(O/H)

    Returns
    -------
    Z : scalar or ndarray
        metallicity

    """
    Z = 10**(metal - 12.0) * 16.0 / 1.008 / 0.51 / 1.36
    return Z
