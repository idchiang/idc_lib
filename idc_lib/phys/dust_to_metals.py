#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:53:33 2020

@author: idchiang
"""
import os
import numpy as np
import pandas as pd


def aniano20(metal, taua2d=0.1, taua2star=1e-2,
             Zm_frac=0.45498, reproduce=False):
    """
    Generate D/M from Aniano+20 toy model

    Parameters
    ----------
    metal : scalar or ndarray
        12 + log(O/H)
    taua2d : scalar, optional
        Dust grow-to-dust destruction time scale. The default is 0.1.
    taua2star : TYPE, optional
        Dust grow-to-stellar injection time scale. The default is 1e-5.
    Zm_frac : TYPE, optional
        Fraction of dust-forming metals. The default is 0.45498.
    reproduce : TYPE, optional
        If True, return M_dust/M_H instead. The default is False.

    Returns
    -------
    If reproduce:
        M_dust/M_H : scalar or ndarray
    else:
        D/M : scalar or ndarray
    """
    metal_sol = 8.69
    Z_sol = 10**(metal_sol-8.69) * 0.0134  # ~0.0154
    Zm_sol = Z_sol * Zm_frac
    #
    Z = Z_sol * 10**(metal - metal_sol)
    Zm = Z * Zm_frac
    b = Zm - Zm_sol * taua2d
    Zd = b / 2 + \
        np.sqrt(b**2 + (4 * Zm_sol) * Zm * taua2star) / 2
    if reproduce:
        return Zd * 1.4
    else:
        return Zd / Z


def read_prev_obs(prev):
    """
    Read literature data into DataFrame

    Parameters
    ----------
    prev : str
        R14, D19 or PH20

    Returns
    -------
    df : pandas.DataFrame
        Loaded data
    df_p20_lim : pandas.DataFrame
        PH20 data with upper limit only
    """
    module_dir, _ = os.path.split(__file__)
    if prev == 'R14':
        # load Remy-Ruyer+14
        df_r14 = pd.read_csv(
            os.path.join(module_dir, "data/remyruyer/Remy-Ruyer_2014.csv"))
        df_r14['metal'] = df_r14['12+log(O/H)']
        df_r14['metal_z'] = 10**(df_r14['metal'] - 12.0) * \
            16.0 / 1.008 / 0.51 / 1.36
        df_r14['gas'] = df_r14['MU_GAL'] * \
            (df_r14['MHI_MSUN'] + df_r14['MH2_Z_MSUN'])
        df_r14['fh2'] = df_r14['MH2_Z_MSUN'] / \
            (df_r14['MHI_MSUN'] + df_r14['MH2_Z_MSUN'])
        df_r14['dtm'] = df_r14['MDUST_MSUN'] / df_r14['metal_z'] / \
            df_r14['gas']
        return df_r14, 0
    elif prev == 'D19':
        # load De Vis+19
        path_dp = 'data/dustpedia/'
        df_d19 = pd.read_csv(os.path.join(module_dir,
            path_dp + 'dustpedia_cigale_results_final_version.csv'))
        df_d19 = df_d19.rename(columns={'id': 'name'})
        # dp metal
        
        df_d19_temp = pd.read_csv(os.path.join(module_dir,
            path_dp + 'DP_metallicities_global.csv'))
        df_d19 = df_d19.merge(df_d19_temp, on='name')
        # dp hi
        df_d19_temp = pd.read_csv(os.path.join(module_dir,
            path_dp + 'DP_HI.csv'))
        df_d19 = df_d19.merge(df_d19_temp, on='name')
        # dp h2
        df_d19_temp = pd.read_excel(os.path.join(module_dir,
            path_dp + 'DP_H2.xlsx'))
        df_d19_temp = df_d19_temp.rename(columns={'Name': 'name'})
        df_d19 = df_d19.merge(df_d19_temp, on='name')
        # renames
        del df_d19_temp
        df_d19 = \
            df_d19.rename(columns={'SFR__Msol_per_yr': 'sfr',
                                   'Mstar__Msol': 'star',
                                   'MHI': 'hi',
                                   '12+logOH_PG16_S': 'metal'})
        df_d19['h2'] = df_d19['MH2-r25'] * 1.36
        df_d19['gas'] = df_d19['hi'] * 1.36 + df_d19['h2']
        df_d19['metal_z'] = 10**(df_d19['metal'] - 12.0) * \
            16.0 / 1.008 / 0.51 / 1.36
        df_d19['dtm'] = df_d19['Mdust__Msol'] / df_d19['metal_z'] / \
            df_d19['gas']
        # df_d19
        # used XCO (no 1.36) / a constanyt XCO is used
        # Need to check merging data frames with missing rows
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
        #
        return df_d19, 0
    elif prev == 'PH20':
        # load howk's review (Peroux+20)
        path_dp = 'data/howk/tableSupplement_howk.csv'
        df_p20 = pd.read_csv(os.path.join(module_dir, path_dp), comment='#')
        df_p20['metal'] = 8.69 + df_p20['[M/H]']
        df_p20['dtm'] = 10**df_p20['log_DTM']
        df_p20_lim = df_p20[df_p20['log_DTM'] <= -1.469]
        df_p20 = df_p20[df_p20['log_DTM'] > -1.469]
        return df_p20, df_p20_lim
