#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 17:40:56 2021

@author: idchiang
"""
from string import digits
import numpy as np
from scipy import ndimage
from astropy.convolution import convolve_fft
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from idc_lib.const import FWHM2sigma

FWHM_TO_AREA = 2 * np.pi / (8 * np.log(2))


def gauss_to_gauss_kernel(ps, bpa_in, bmaj_in, bmin_in, fwhm_out):
    """
    Generate a Gaussian-to-Gaussian kernel for convolution

    Parameters
    ----------
    ps : float
        Pixel scale of the Kernel in arcsec
    bpa_in : float
        Position angle of the input Gaussian in degrees.
    bmaj_in : float
        Major axis of the input beam in arcsec.
    bmin_in : float
        Minor axis of the input beam in arcsec.
    fwhm_out : float
        Beam size of the output beam in arcsec.

    Returns
    -------
    kernel : ndarray, shape (m, n)
        The generated kernel

    """
    # Converting scales
    bpa_in *= np.pi / 180
    sigma_x_sq = (fwhm_out**2 - bmin_in**2) * FWHM2sigma**2 / ps**2
    sigma_y_sq = (fwhm_out**2 - bmaj_in**2) * FWHM2sigma**2 / ps**2
    # Generating grid points. Ref Anaino total dimension ~729", half 364.5"
    lx, ly = 364.5 // ps, 364.5 // ps
    x, y = np.meshgrid(np.arange(-lx, lx + 1), np.arange(-ly, ly + 1))
    cosbpa, sinbpa = np.cos(bpa_in), np.sin(bpa_in)
    xp, yp = cosbpa * x + sinbpa * y, cosbpa * y - sinbpa * x
    kernel = np.exp(-0.5 * (xp**2 / sigma_x_sq + yp**2 / sigma_y_sq))
    kernel /= np.sum(kernel)
    return kernel


def conv_repr_wrapper(data_list, kernel_list, hdr_in, hdr_out,
                      nan2zero=False, threshold=0.9,
                      skip_convolution=False, skip_reproject=False):
    """
    My main wrapper for convolution and reproject

    Parameters
    ----------
    data_list : TYPE
        DESCRIPTION.
    kernel_list : TYPE
        DESCRIPTION.
    hdr_in : TYPE
        DESCRIPTION.
    hdr_out : TYPE
        DESCRIPTION.
    nan2zero : TYPE, optional
        DESCRIPTION. The default is False.
    threshold : TYPE, optional
        DESCRIPTION. The default is 0.9.
    skip_convolution : TYPE, optional
        DESCRIPTION. The default is False.
    skip_reproject : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    rdata_list : TYPE
        DESCRIPTION.

    """
    if type(data_list) is not list:
        data_list = list(data_list)
    if (not skip_convolution) and (type(kernel_list) is not list):
        kernel_list = list(kernel_list)
    #
    # Append good points map
    good_pts = np.ones_like(data_list[0], dtype=bool)
    for i in range(len(data_list)):
        good_pts *= np.isfinite(data_list[i])
        data_list[i][~np.isfinite(data_list[i])] = 0.0
    good_pts = good_pts.astype(float)
    data_list.append(good_pts)
    #
    # Convolve map
    if skip_convolution:
        cdata_list = data_list
    else:
        cdata_list = []
        with np.errstate(invalid='ignore', divide='ignore'):
            for i, data in enumerate(data_list):
                for kernel in kernel_list:
                    data = convolve_fft(data, kernel, allow_huge=True)
                cdata_list.append(data)
        del data_list
    #
    # Reproject map
    if skip_reproject:
        rdata_list = cdata_list
    else:
        w_in = WCS(hdr_in, naxis=2)
        w_out = WCS(hdr_out, naxis=2)
        s_out = w_out.array_shape
        bitpix = abs(int(hdr_in['BITPIX']))
        rdata_list = []
        with np.errstate(invalid='ignore', divide='ignore'):
            for i, cdata in enumerate(cdata_list):
                rdata, _ = reproject_interp((cdata, w_in), w_out, s_out)
                rdata_list.append(rdata)
        del cdata_list
    #
    # Pop good points map
    good_pts = rdata_list.pop(-1)
    good_pts[np.isnan(good_pts)] = 0.0
    nan_pts = good_pts < threshold
    #
    # Do nan and bitpix
    for i in range(len(rdata_list)):
        if nan2zero:
            rdata_list[i][nan_pts] = 0.0
        else:
            rdata_list[i][nan_pts] = np.nan
        if bitpix == 32:
            rdata_list[i] = rdata_list[i].astype(np.float32)
        elif bitpix == 16:
            rdata_list[i] = rdata_list[i].astype(np.float16)
    #
    return rdata_list


def header_generator(beam_size_arcsec, ra_deg, dec_deg, r25_arcsec,
                     ps_arcsec=None):
    # Default: L = 8 * r25
    # Default: pixel size = resolution / 3
    # ps_arcsec input will overwrite beam_size_arcsec
    if ps_arcsec is None:
        ps_arcsec = beam_size_arcsec / 3.0
    ps = ps_arcsec / 3600.0
    L_arcsec = r25_arcsec * 8
    L = int(L_arcsec / ps_arcsec)
    hdu = fits.PrimaryHDU()
    hdr_out = hdu.header
    hdr_out['BITPIX'] = 64
    hdr_out['NAXIS'] = 2
    hdr_out['NAXIS1'] = L
    hdr_out['NAXIS2'] = L
    hdr_out['CTYPE1'] = 'RA---TAN'
    hdr_out['CRVAL1'] = ra_deg
    hdr_out['CDELT1'] = -ps
    hdr_out['CRPIX1'] = L / 2
    hdr_out['CUNIT1'] = 'deg'
    hdr_out['CTYPE2'] = 'DEC--TAN'
    hdr_out['CRVAL2'] = dec_deg
    hdr_out['CDELT2'] = ps
    hdr_out['CRPIX2'] = L / 2
    hdr_out['CUNIT2'] = 'deg'
    hdr_out['EPOCH'] = 2000
    return hdr_out


def reject_small_regions(image, reject_area_in_beams=1.0,
                         bmaj_pix=3, bmin_pix=3):
    """
    Take patchy data, remove small regions and return
    non-data pixels should be np.nan

    Parameters
    ----------
    image : ndarray
        Data to process. non-data pixels should be np.nan.

    reject_area_in_beams : float
        The threshold of area we want to remove (in beams)

    bmaj_pix : float
        Major beam size in pix

    bmin_pix : float
        Minor beam size in pix

    Returns
    -------
    processed_image : ndarray
        Data after small regions removed

    """
    mask = (~np.isnan(image)).astype(int)
    beam_area_pix = bmaj_pix * bmin_pix * FWHM_TO_AREA
    regions, n_regions = ndimage.label(mask)
    myhistogram = ndimage.measurements.histogram(regions, 0, n_regions + 1,
                                                 n_regions + 1)
    object_slices = ndimage.find_objects(regions)
    for i in range(n_regions):
        if myhistogram[i + 1] < reject_area_in_beams * beam_area_pix:
            mask[object_slices[i]] = 0
    mask = mask.astype(bool)
    processed_image = image.copy()
    processed_image[~mask] = np.nan
    return processed_image


def physical2angular_resolution(resolution_str='', dist_mpc=0.0):
    # Given target physical resolution + distance,
    # return beam size in arcsec (round to 1st decimal)
    s = resolution_str.lstrip('_').lower()
    if s[-2:] != 'pc':
        raise ValueError("Invalid expression in resolution_str " +
                         resolution_str + ': not ending in pc.')
    dist_kpc = dist_mpc * 1000
    # arcsec_to_kpc = 1. / 3600 / 180 * np.pi * dist_kpc
    kpc_to_arcsec = 3600. * 180 / np.pi / dist_kpc
    if s[-3] == 'k':
        beamsize_arcsec = float(s[:-3]) * kpc_to_arcsec
    elif s[-3] in digits:
        beamsize_arcsec = float(s[:-2]) / 1000.0 * kpc_to_arcsec
    else:
        raise ValueError("Invalid expression in resolution_str " +
                         resolution_str + ': unit not implemented.')
    if s[0] == '0':
        beamsize_arcsec /= 10
    return round(beamsize_arcsec, 1)


def str2resolution(resolution_str='', dist_mpc=0.0,
                   round_digit=None):
    # Given target physical resolution + distance,
    # return beam size in arcsec (round to 1st decimal)
    s = resolution_str.lstrip('_').lower()
    if s[-2:] == 'pc':
        assert dist_mpc > 0, "Need input distance!"
        dist_kpc = dist_mpc * 1000
        # arcsec_to_kpc = 1. / 3600 / 180 * np.pi * dist_kpc
        kpc_to_arcsec = 3600. * 180 / np.pi / dist_kpc
        if s[-3] == 'k':
            beamsize_arcsec = float(s[:-3]) * kpc_to_arcsec
        elif s[-3] in digits:
            beamsize_arcsec = float(s[:-2]) / 1000.0 * kpc_to_arcsec
        else:
            raise ValueError("Invalid expression in resolution_str " +
                             resolution_str + ': unit not implemented.')
        if s[0] == '0':
            beamsize_arcsec /= 10
    elif s[:5] == 'gauss':
        beamsize_arcsec = float(s[5:])
    else:
        raise ValueError("Invalid expression in resolution_str " +
                         resolution_str)
    if round_digit is not None:
        return round(beamsize_arcsec, round_digit)
    else:
        return beamsize_arcsec


"""
To be deprecated.
"""


def convolve_wrapper(data, kernel, threshold=0.9):
    """
    Convolve image with kernel, with additional work dealing w/ nan

    Parameters
    ----------
    data : ndarray, shape (m, n)
        Input image
    kernel : ndarray, shape (p, q)
        Kernel for convolution. Should match the pixel size
    threshold : float, optional
        The threshold to remove nan regions. 0 <= threshold <= 1.
        The smaller the stricter. The default is 0.9.

    Returns
    -------
    cdata : ndarray, shape (m, n)
        Convolved image.

    """
    # Adjust the default threshold to 0.9 on 06/25/2019
    bad_pts = (~np.isfinite(data)).astype(float)
    # Convolve map
    with np.errstate(invalid='ignore', divide='ignore'):
        cdata = convolve_fft(data, kernel, allow_huge=True)
        cbad_pts = convolve_fft(bad_pts, kernel, allow_huge=True)
        #
        cdata[~np.isfinite(data)] = np.nan
        cdata[cbad_pts > threshold] = np.nan
    f1, f2 = np.nansum(data), np.nansum(cdata)
    if (f1 == 0) and (f2 == 0):
        print("-- Done. 0-to-0.")
    else:
        print("-- Done. Flux variation (%):", round(100 * (f2 - f1) / f1, 2))
    return cdata


def reproject_wrapper(data, hdr_in, hdr_out, threshold=0.9):
    # Adjust the default exact=False on 06/25/2019
    # Add threshold on 06/25/2019
    w_in = WCS(hdr_in, naxis=2)
    w_out = WCS(hdr_out, naxis=2)
    s_out = w_out.array_shape
    #
    bad_pts = (~np.isfinite(data)).astype(float)
    boundary = np.ones_like(data, dtype=float)
    data[~np.isfinite(data)] = 0.0
    #
    rdata, _ = reproject_interp((data, w_in), w_out, s_out)
    rbabp, _ = reproject_interp((bad_pts, w_in), w_out, s_out)
    rboundary, _ = reproject_interp((boundary, w_in), w_out, s_out)
    with np.errstate(invalid='ignore'):
        rdata[rbabp > threshold] = np.nan
        rdata[rboundary < threshold] = np.nan
    bitpix = abs(int(hdr_in['BITPIX']))
    if bitpix == 32:
        rdata = rdata.astype(np.float32)
    elif bitpix == 16:
        rdata = rdata.astype(np.float16)
    return rdata


def remove_axis(hdr, target_axis):
    naxis_in = hdr['NAXIS']
    hdr['NAXIS'] -= 1
    #
    target_axis_str = str(target_axis)
    for key in hdr.copy():
        if target_axis_str in key:
            hdr.remove(key)
    #
    for next_axis in range(target_axis + 1, naxis_in + 1):
        next_axis_str = str(next_axis)
        next_axis_m1_str = str(next_axis - 1)
        for key in hdr.copy():
            if next_axis_str in key:
                hdr[key] = hdr[key].replace(next_axis_str, next_axis_m1_str)
    #
    return hdr
