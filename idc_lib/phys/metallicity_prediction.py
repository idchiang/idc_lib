#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 01:35:40 2021

@author: idchiang
"""
import warnings
import numpy as np


def pred_logoh_at_re(log_mstar,
                     radius_re=None,
                     p0=8.647, p1=-0.718, p2=0.682, p3=-0.133,
                     offset=0.46, blank_re=False, blank_mstar=False,
                     fit2022jun24=False):
    """
    Predicting 12+log(O/H) from stellar mass and galacto-centric radius in
    effective radius, using the method described in Sun et al. (2020) and
    Chiang et al. (2021)

    The default offset is calibrated with z0MGS stellar mass and PHANGS-ALMA
    `MASS' effective radius (r_e). The "gold standard" is the
    PG16 S-calibration data measured in DustPedia and PHANGS-MUSE.

    Known limits:
        1) log_mstar > 11.0 and log_mstar < 9.0 are not recommended
        2) radius_re > 2.0 and radius_re < 0.3 are not recommended.

    2021.10.07 update: offset 0.45 --> 0.46

    Parameters
    ----------
    log_mstar : float or array-like
        log(M_* / M_sun).
    radius_re : None, float or array-like, optional.
        Galacto-centric radius (r_g) in r_e (r_g/r_e). The default is None.
        We set the representative 12+log(O/H) calculated with log_mstar at
        r_g/r_e == 1.0. For the other parts of the galaxy, we extrapolate
        their values with a gradient of -0.1 dex/r_e (Sanchez et al. 2014).

        If None, radius_re=1.0 will be used.
        If array-like, the shape should be the same as log_mstar
    p0 : float, optional
        0th-order coefficent in the pMZR M-Z relation. The default is 8.647.
        See Sanchez et al. (2019) Table 1, 'pyqz' calibration.
    p1 : float, optional
        1st-order coefficent in the pMZR M-Z relation. The default is -0.718.
        See Sanchez et al. (2019) Table 1, 'pyqz' calibration.
    p2 : float, optional
        2nd-order coefficent in the pMZR M-Z relation. The default is 0.682.
        See Sanchez et al. (2019) Table 1, 'pyqz' calibration.
    p3 : float, optional
        3rd-order coefficent in the pMZR M-Z relation. The default is -0.133.
        See Sanchez et al. (2019) Table 1, 'pyqz' calibration.
    offset : float, optional
        Empirical offset applied to the pMZR formula. The default is 0.45.
        See Chiang et al. in prep. (2021b?)

    Returns
    -------
    pred_logoh : float or np.ndarray
        The predicted 12+log(O/H).

    """
    # check if inputs make sense
    if radius_re is None:
        radius_re = 1.0
    log_mstar = np.array(log_mstar)
    if np.any(log_mstar > 11.0):
        if blank_mstar:
            log_mstar[log_mstar > 11.0] = np.nan
            warnings.warn('12+log(O/H) Prediction: log(mstar) > 11.0 found. ' +
                          'This is not suggested by our extrapolation method.')
            warnings.warn("They're blanked since blank_mstar==True")
    if np.any(log_mstar < 9.0):
        if blank_mstar:
            log_mstar[log_mstar < 9.0] = np.nan
            warnings.warn('12+log(O/H) Prediction: log(mstar) < 9.0 found. ' +
                          'This is not suggested by our extrapolation method.')
            warnings.warn("They're blanked since blank_mstar==True")
    radius_re = np.array(radius_re)
    if np.any(radius_re < 0.0):
        assert False, '12+log(O/H) Prediction: radius < 0 found.'
    if np.any(radius_re > 2.0):
        if blank_re:
            radius_re[radius_re > 2.0] = np.nan
            warnings.warn('12+log(O/H) Prediction: r_g/r_e > 5.0 found. ' +
                          'This is not suggested by our extrapolation method.')
            warnings.warn("They're blanked since blank_re==True")
    if np.any(radius_re < 0.3):
        if blank_re:
            radius_re[radius_re < 0.3] = np.nan
            warnings.warn('12+log(O/H) Prediction: r_g/r_e > 0.3 found. ' +
                          'This is not suggested by our extrapolation method.')
            warnings.warn("They're blanked since blank_re==True")
    # calculate reference metallicity at 1 * Re
    if fit2022jun24:
        x = log_mstar - 8.0 - 3.5
        a = 8.56  # +- 0.02
        b = 0.010  # +- 0.002
        ref = a + b * x * np.exp(-x)
        offset = 0
    else:
        x = log_mstar - 8.0
        ref = p0 + p1 * x + p2 * x**2 + p3 * x**3
        # calculate extrapolation threshold
        xp = (-2 * p2 - np.sqrt(4 * p2**2 - 12 * p1 * p3)) / 6 / p3
        refp = p0 + p1 * xp + p2 * xp**2 + p3 * xp**3
        mask = x > xp
        if type(mask) is np.bool_:
            if mask:
                ref = refp
        else:
            ref[mask] = p0 + p1 * xp + p2 * xp**2 + p3 * xp**3
    # calculate gradient
    gradient = -0.1 * (radius_re - 1)
    #
    pred_logoh = ref + gradient - offset
    #
    return pred_logoh
