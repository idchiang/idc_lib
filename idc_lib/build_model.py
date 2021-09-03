#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:22:44 2020

@author: idchiang
"""
import os
import numpy as np
from astropy.io import fits

default_bands = ['PACS_100', 'PACS_160', 'SPIRE_250', 'SPIRE_350',
                 'SPIRE_500']
grid_col_def = {
    'SE': ['logSigma_d', 'T_d', 'beta'],
    'FB': ['logSigma_d', 'T_d'],
    'BE': ['logSigma_d', 'T_d', 'beta_2']}
modnames = {
    'SE': 'MBB with free beta',
    'FB': 'MBB with fixed beta',
    'BE': 'MBB with Broken Power Law Emissivity',
    }
params = {
    'logSigma_d': 'log(Sigma_dust)',
    'T_d': 'T_dust',
    'beta': 'beta',
    'beta_2': 'beta_2'}
param_comments = {
    'logSigma_d': 'Log of dust surface density [Msun/pc^2]',
    'T_d': 'Dust temperature [K]',
    'beta': 'The power-law index',
    'beta_2': 'The power-law index at long-wavelength'}


def build_model(model_str, grid_def={}, bands=default_bands,
                kappa_160, beta_f=2.0, lambda_c_f=300.0):
    output_dir = '/home/idchiang/processed/dust_sed_models/'
    # 1. build grid
    grid_def = load_defaults(grid_def)
    grid, grid_cols = grid_gen(model_str, grid_def)
    # 2. build grid hdr
    modname_comment = 'kappa_160 = ' + str(round(kappa_160, 2)) + ' cm^2/g'
    if model_str == 'BE':
        modname_comment += '; beta =' + str(round(beta, 2)) + \
            '; lambda_c =' + str(round(lambda_c_f, 1)) + ' um'
    elif model_str == 'FB':
        modname_comment += '; beta =' + str(round(beta, 2))
    #
    hdr1 = fits.Header()
    hdr1['MODNAME'] = modnames[model_str]
    hdr1.comments['MODNAME'] = modname_comment
    for i, p in enumerate(grid_cols):
        hdr1['PARAM' + str(i + 1)] = params[p]
        hdr1.comments['PARAM' + str(i + 1)] = param_comments[p]
    # dat1 = grid
    hdu1 = fits.ImageHDU(data=grid, header=hdr1)
    # 3. generate model 
    dat2 = np.zeros([len(bands), grid.shape[1]])
    lambda_ = 
    for j in range(grid.shape[1]):
        
        generate model
        call integration
        save integrated model

    # 4. save model
    hdr2 = fits.Header()
    hdr2['MODNAME'] = modnames[model_str]
    hdr2.comments['MODNAME'] = modname_comment
    for i, band in enumerate(bands):
        hdr2['BAND' + str(i + 1)] = band
    hdu2 = fits.ImageHDU(data=dat2, header=hdr2)
    #
    fn = output_dir + 'model_' + model_str + '.fits'
    if os.path.isfile(fn):
        os.remove(fn)
    hdul = fits.HDUList(hdus=[fits.PrimaryHDU(), hdu1, hdu2])
    hdul.writeto(fn)


def load_defaults(grid_def):
    if 'logSigma_d' not in grid_def:
        grid_def['Sigma_d'] = [-4.,  1., 0.025]
    if 'T_d' not in grid_def:
        grid_def['T_d'] = [5., 50., 0.5]
    if 'beta' not in grid_def:
        grid_def['beta'] = [-1.0, 4.0, 0.1]
    if 'beta_2' not in grid_def:
        grid_def['beta_2'] = [-1.0, 4.0, 0.1]


def grid_gen(model_str, grid_def):
    if model_str in grid_col_def:
        grid_cols = grid_col_def[model_str]
    else:
        print('Model', model_str, 'not recognized!')
    arrs = {p: np.arange(grid_def[p][0], grid_def[p][1], grid_def[p][2])
            for p in grid_cols}
    grid = np.array(np.meshgrid(
        *tuple(arrs[p] for p in grid_cols))).T.reshape(-1, len(grid_cols))
    return grid, grid_cols
