#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:57:14 2021

@author: idchiang
"""
import multiprocessing as mp
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
from idc_lib.phys.midplane_pressure import P_DE
from idc_lib.phys.metallicity import metal2Z
from idc_lib.alphaco_fit_models import dict_models


"""
plt basics
"""
plt.style.use('idcgrid')
plt.ioff()


def helper(func='metal_power'):
    """
    Explains the selected function
    Pause after execution

    Parameters
    ----------
    func : TYPE, optional
        Function name. The default is 'metal_power'.
    """
    print('## Explaining:', func)
    assert (func in dict_models), 'Function not defined!!'
    model = dict_models[func]
    print('##', model.get_description())
    param_description = model.get_param_description()
    k = len(param_description)
    print('## num of params:', k)
    for i, d in enumerate(param_description):
        print('## param ' + str(i) + ':', d)
    input('## Seems correct? Press any key to continue...')


def fitter(SigmaDust, SigmaHI, ICO, metal, SigmaMstar,
           r25_mpc,
           func='metal_power', params=np.zeros((0, 2)),
           nop=10):
    """
    Fitter for 1 target galaxy
    (n): shape of input quantities
    (m): length of input parameter space
    (k): number of parameters of the selected function

    Parameters
    ----------
    SigmaDust : list-like, shape (n)
        Dust surface density (Msun/pc2)
    SigmaHI : list-like, shape (n)
        HI surface density (Msun/pc2). NOT including He correction.
    ICO : list-like, shape (n)
        Integrated CO(1-0) intensity (K*km/s)
    metal : list-like, shape (n)
        12+log(O/H)
    SigmaMstar : list-like, shape (n)
        Stellar mass surface density (Msun/pc2)
    r25_mpc : float
        R25 in Mpc.
    func : string, optional
        The function used in modelling. The default is 'metal_power'.
        valid inputs:
            'metal_power': a power-law with metallicity as single input
    params : array-like floats, shape (m, k)
        The input parameters. The default is [].

    Returns
    -------
    res : np.ndarray, 4 arrays with shape (m)
        Arrays containing D/M-metall correlation, D/M-PDE correlation,
        max D/M, median D/M, respectively

    """
    # Import function
    assert (func in dict_models), 'Function not defined!!'
    model = dict_models[func]
    # Sanity check: n
    n = len(SigmaDust)
    for q in [SigmaHI, ICO, metal, SigmaMstar]:
        assert len(q) == n
    metal_z = metal2Z(metal)
    const_metal = np.nanmax(metal) - np.nanmin(metal) < 0.01
    # Sanity check: m, k
    m, k = params.shape
    assert k == len(model.get_param_description())

    def mp_wrapper(mpid, list_corr_metal, list_corr_pde, list_max_dm,
                   list_med_dm):
        begin = int(m * mpid / nop)
        end = int(m * (mpid + 1) / nop)
        for i in range(begin, end):
            alphaCO = model.aco_generator(
                params=params[i],
                SigmaHI=SigmaHI, ICO=ICO, metal=metal, SigmaMstar=SigmaMstar)
            mask = np.isfinite(alphaCO)
            if np.sum(mask) < 2:
                list_corr_metal[i] = -1.0
                list_corr_pde[i] = -1.0
                list_max_dm[i] = 10
                list_med_dm[i] = 10
            else:
                SigmaGas = 1.36 * SigmaHI + ICO * alphaCO
                logDM = np.log10(SigmaDust / SigmaGas / metal_z)
                logPDE = np.log10(P_DE(SigmaGas, SigmaMstar, r25_mpc))
                # metal
                if const_metal:
                    list_corr_metal[i] = 0.0
                else:
                    list_corr_metal[i] = pearsonr(logDM[mask], metal[mask])[0]
                # pressure
                list_corr_pde[i] = pearsonr(logDM[mask], logPDE[mask])[0]
                # max D/M
                list_max_dm[i] = 10**np.nanmax(logDM)
                # median D/M
                list_med_dm[i] = 10**np.nanmedian(logDM)

    list_corr_metal = mp.Manager().list([0] * m)
    list_corr_pde = mp.Manager().list([0] * m)
    list_max_dm = mp.Manager().list([0] * m)
    list_med_dm = mp.Manager().list([0] * m)
    processes = [
        mp.Process(target=mp_wrapper,
                   args=(mpid, list_corr_metal, list_corr_pde,
                         list_max_dm, list_med_dm))
        for mpid in range(nop)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    return np.array(list_corr_metal), np.array(list_corr_pde), \
        np.array(list_max_dm), np.array(list_med_dm)


def plotter(params, pspace_shape, param_1ds,
            ax, images1d, mode=0,
            aco_func='metal_power'):
    titles = ['D/M-12+log(O/H)', r'D/M-P$_{DE}$', 'Max D/M',
              'Overall score']
    cmaps = ['Spectral', 'Spectral', 'Spectral_r', 'cool_r']
    vmaxs = [None] * 4
    vmins = [None] * 4
    if mode == 0:
        # plot one galaxy
        vmaxs = [1.0, 1.0, 2.0, 1.0]
        vmins = [-1.0, -1.0, 0.0, 0.0]
    elif mode == 1:
        # plot all galaxies: count
        cmaps = ['inferno'] * 4
        titles = ['D/M-12+log(O/H) Score',
                  r'D/M-P$_{DE}$ Score', 'Max D/M Score',
                  'Overall score']
        vmins = [0.0, 0.0, 0.0, 0.0]
    elif mode == 2:
        # test coordinates
        if aco_func == 'metal_power':
            titles = ['a (normalization)', 'b (slope)']
        elif aco_func == 'b13':
            titles = ['a (exponential factor)',
                      r'$\gamma$ (high-density correction)']
        elif aco_func == 'b13_3param':
            titles = ['a (exponential factor)',
                      r'$\gamma$ (high-density correction)',
                      'b (log normalization)']
    elif mode == 3:
        # high-score points
        vmins = [0] * 4
        titles = ['Score top 1%',
                  'Score top 2%',
                  'Score top 3%',
                  'Score top 5%']
        cmaps = ['inferno'] * 4
    #
    if aco_func == 'metal_power':
        xtick_vals = [np.log10(0.25 * 4.35), np.log10(0.5 * 4.35),
                      np.log10(4.35), np.log10(2 * 4.35),
                      np.log10(4 * 4.35)]
        xticks = np.interp(xtick_vals, param_1ds[0],
                           np.arange(len(param_1ds[0])))
        xticklabels = [r'$\alpha_{CO}^{MW}/4$',
                       r'$\alpha_{CO}^{MW}/2$',
                       r'$\alpha_{CO}^{MW}$',
                       r'$2\alpha_{CO}^{MW}$',
                       r'$4\alpha_{CO}^{MW}$']
        yticklabels = [-4.0, -2.0, 0.0]
        yticks = np.interp(yticklabels, param_1ds[1],
                           np.arange(len(param_1ds[1])))
        xlabel = 'a (normalization)'
        ylabel = 'b (slope)'
    elif aco_func == 'b13':
        xtick_vals = [-0.2, 0.1, 0.4, 0.7, 1.0]
        xticks = np.interp(xtick_vals, param_1ds[0],
                           np.arange(len(param_1ds[0])))
        xticklabels = xtick_vals
        yticklabels = [0.0, 0.5, 1.0]
        yticks = np.interp(yticklabels, param_1ds[1],
                           np.arange(len(param_1ds[1])))
        xlabel = 'a (exponential factor)'
        ylabel = r'$\gamma$ (high-density correction)'
    elif aco_func == 'b13_3param':
        qtick_vals = {
            0: [-1.0, -0.5, 0.0, 0.5, 1.0],
            1: [0.0, 0.7, 1.4],
            2: [np.log10(2.9 / 4), np.log10(2.9), np.log10(2.9 * 4)]}
        qticks = {}
        qticklabels = {}
        for i in range(3):
            qticks[i] = np.interp(qtick_vals[i], param_1ds[i],
                                  np.arange(len(param_1ds[i])))
            qticklabels[i] = [str(round(num, 1)) for num in qtick_vals[i]]
        qlabel = {
            0: 'b (exponential factor)',
            1: r'$\gamma$ (high-density correction)',
            2: 'a (log-scale normalization)'}
        combs = [[2, 0], [2, 1], [0, 1]]
        # shape: (p01, p00, p02)
        sum_axis = [2, 0, 1]
    if len(ax.shape) == 1:
        for i in range(len(ax)):
            im = ax[i].imshow(images1d[i].reshape(pspace_shape),
                              origin='lower',
                              cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i],
                              interpolation='hanning')
            plt.colorbar(im, ax=ax[i])
            # xlim = ax[i].get_xlim()
            ax[i].set_xlabel(xlabel)
            ax[i].set_xticks(xticks)
            # ax[i].set_xticklabels([round(num, 2) for num in xticklabels])
            ax[i].set_xticklabels(xticklabels)
            ax[i].set_yticks(yticks)
            ax[i].set_yticklabels([round(num, 1) for num in yticklabels])
            ax[i].set_title(titles[i])
        ax[0].set_ylabel(ylabel)
    elif len(ax.shape) == 2:
        for j in range(ax.shape[0]):
            x, y = combs[j]
            for i in range(ax.shape[1]):
                image3d = images1d[i].reshape(pspace_shape)
                if mode == 0:  # One object
                    if i < 3:  # conditions
                        image2d = np.nanmedian(image3d, axis=sum_axis[j])
                    else:  # Overall score
                        image2d = np.nanmean(image3d, axis=sum_axis[j])
                elif mode in {1, 2}:  # Overall, test
                    image2d = np.nanmean(image3d, axis=sum_axis[j])
                elif mode == 3:
                    image2d = np.nansum(image3d, axis=sum_axis[j])
                im = ax[j, i].imshow(image2d,
                                     origin='lower',
                                     cmap=cmaps[i],
                                     vmin=vmins[i], vmax=vmaxs[i],
                                     interpolation='hanning')
                plt.colorbar(im, ax=ax[j, i])
                ax[j, i].set_xlabel(qlabel[x], size=12)
                ax[j, i].set_xticks(qticks[x])
                ax[j, i].set_xticklabels(qticklabels[x])
                ax[j, i].set_yticks(qticks[y])
                ax[j, i].set_yticklabels(qticklabels[y])
                if j == 0:
                    ax[j, i].set_title(titles[i], size=12)
            ax[j, 0].set_ylabel(qlabel[y], size=12)
