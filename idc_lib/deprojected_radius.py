#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 13:35:16 2020

@author: idchiang
"""
import warnings
import numpy as np
import astropy
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import Angle


warnings.filterwarnings("ignore", category=astropy.wcs.FITSFixedWarning)


def minor_axis_mask(shape, hdr, ra_rad, dec_rad, pa_rad, theta0_rad=np.pi / 4):
    """
    Return a mask selecting pixels within an angle (theta0) spread around the
    minor axis.
    Not tested for data with ndim != 2

    Parameters
    ----------
    shape : tuple
        shape of fits data
    hdr : header
        header of fits data
    ra_rad : scalar
        RA in rad
    dec_rad : scalar
        DEC in rad
    pa_rad : scalar
        Position angle in rad
    theta0_rad : scalar, optional
        The spread angle around minor axis. The default is np.pi / 4.

    Returns
    -------
    mask : ndarray
        True = within theta0 of the minor axis. False = outside.

    """
    cosPA, sinPA = np.cos(pa_rad), np.sin(pa_rad)
    xcm, ycm = ra_rad, dec_rad
    dp_coords = np.zeros(list(shape) + [2])
    wcs = WCS(hdr, naxis=2)
    # Original coordinate is (y, x)
    # :1 --> x, RA --> the one needed to be divided by cos(incl)
    # :0 --> y, Dec
    dp_coords[:, :, 0], dp_coords[:, :, 1] = \
        np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    dp_coords += 1.0  # offset issue 2019/08/20
    # Now, value inside dp_coords is (x, y)
    # :0 --> x, RA --> the one needed to be divided by cos(incl)
    # :1 --> y, Dec
    for i in range(shape[0]):
        dp_coords[i] = Angle(wcs.wcs_pix2world(dp_coords[i], 1) * u.deg).rad
    dp_coords[:, :, 0] = 0.5 * (dp_coords[:, :, 0] - xcm) * \
        (np.cos(dp_coords[:, :, 1]) + np.cos(ycm))
    dp_coords[:, :, 1] -= ycm
    # Now, dp_coords is (dx, dy) in the original coordinate
    # cosPA*dy-sinPA*dx is new y
    # cosPA*dx+sinPA*dy is new x
    dy = cosPA * dp_coords[:, :, 1] + sinPA * dp_coords[:, :, 0]
    dx = cosPA * dp_coords[:, :, 0] - sinPA * dp_coords[:, :, 1]
    cottheta = np.abs(dy / dx)
    maxCottheta = 1 / np.tan(theta0_rad)
    mask = (cottheta <= maxCottheta)
    return mask


def radius_arcsec(shape, w, ra_rad, dec_rad, pa_rad, incl_rad,
                  incl_correction=False, cosINCL_limit=0.5):
    if incl_correction and (np.isnan(pa_rad + incl_rad)):
        pa_rad = 0.0
        incl_rad = 0.0
        # Not written to the header
        msg = '\n::z0mgs:: PA or INCL is NaN in ' + \
            'radius calculation \n' + \
            '::z0mgs:: Setting both to zero.'
        # Warning message ends
        warnings.warn(msg, UserWarning)
        # Warning ends
    cosPA, sinPA = np.cos(pa_rad), np.sin(pa_rad)
    cosINCL = np.cos(incl_rad)
    if incl_correction and (cosINCL < cosINCL_limit):
        cosINCL = cosINCL_limit
        # Not written to the header
        msg = '\n::z0mgs:: Large inclination encountered in ' + \
            'radius calculation \n' + \
            '::z0mgs:: Input inclination: ' + str(incl_rad) + \
            ' rads. \n' + \
            '::z0mgs:: cos(incl) is set to ' + str(cosINCL_limit)
        # Warning message ends
        warnings.warn(msg, UserWarning)
        # Warning ends
    xcm, ycm = ra_rad, dec_rad

    dp_coords = np.zeros(list(shape) + [2])
    # Original coordinate is (y, x)
    # :1 --> x, RA --> the one needed to be divided by cos(incl)
    # :0 --> y, Dec
    dp_coords[:, :, 0], dp_coords[:, :, 1] = \
        np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    dp_coords += 1.0  # offset issue 2019/08/20
    # Now, value inside dp_coords is (x, y)
    # :0 --> x, RA --> the one needed to be divided by cos(incl)
    # :1 --> y, Dec
    for i in range(shape[0]):
        dp_coords[i] = Angle(w.wcs_pix2world(dp_coords[i], 1) * u.deg).rad
    dp_coords[:, :, 0] = 0.5 * (dp_coords[:, :, 0] - xcm) * \
        (np.cos(dp_coords[:, :, 1]) + np.cos(ycm))
    dp_coords[:, :, 1] -= ycm
    # Now, dp_coords is (dx, dy) in the original coordinate
    # cosPA*dy-sinPA*dx is new y
    # cosPA*dx+sinPA*dy is new x
    radius = np.sqrt((cosPA * dp_coords[:, :, 1] +
                      sinPA * dp_coords[:, :, 0])**2 +
                     ((cosPA * dp_coords[:, :, 0] -
                       sinPA * dp_coords[:, :, 1]) / cosINCL)**2)
    radius = Angle(radius * u.rad).arcsec
    return radius
