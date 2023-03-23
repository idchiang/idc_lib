#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 09:57:14 2021

@author: idchiang
"""
import multiprocessing as mp
import warnings
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl
from idc_lib.phys.midplane_pressure import P_DE
from idc_lib.phys.metallicity import metal2Z
from idc_lib.alphaco_fit_models import dict_models


"""
plt basics
"""
plt.style.use('idcgrid')
plt.ioff()


def w_mean(x, w):
    return np.sum(x * w) / np.sum(w)


def w_cov(x, y, w):
    return np.sum((x - w_mean(x, w)) * (y - w_mean(y, w)) * w) / np.sum(w)


def w_pearsonr(x, y, w):
    return w_cov(x, y, w) / np.sqrt(w_cov(x, x, w) * w_cov(y, y, w))


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


def mp_wrapper(func, param, SigmaHI, ICO, metal, SigmaMstar, SigmaDust, metal_z, one_over_rg, rg,
               r25_mpc, const_metal, pixel_by_pixel):
    """
    return: (corr_metal, corr_pde, max_dm, med_dm)
    """
    model = dict_models[func]
    alphaCO = model.aco_generator(
        params=param,
        SigmaHI=SigmaHI, ICO=ICO, metal=metal, SigmaMstar=SigmaMstar)
    mask = np.isfinite(alphaCO + one_over_rg)
    if np.sum(mask) / len(mask) < 0.95:
        # return (-1.0, -1.0, 10.0, 10.0)
        return (np.nan, np.nan, np.nan, np.nan)
    else:
        SigmaGas = 1.36 * SigmaHI + ICO * alphaCO
        DM = SigmaDust / SigmaGas / metal_z
        logPDE = np.log10(P_DE(SigmaGas, SigmaMstar, r25_mpc))
        # metal
        if const_metal:
            corr_metal = 0.0
        else:
            if pixel_by_pixel:
                corr_metal = w_pearsonr(
                    DM[mask], metal[mask], one_over_rg[mask])
            else:
                corr_metal = pearsonr(DM[mask], metal[mask])[0]
        # PDE
        const_pde = (np.nanmax(logPDE) - np.nanmin(logPDE)) < 0.2
        if const_pde:
            corr_pde = 0.0
        else:
            if pixel_by_pixel:
                corr_pde = w_pearsonr(
                    DM[mask], logPDE[mask], one_over_rg[mask])
            else:
                corr_pde = pearsonr(DM[mask], logPDE[mask])[0]
        # D/M > 1 percentage
        max_dm = np.nansum(DM[mask] > 1.0) / len(DM[mask]) * 100
        # f(H2) and radius
        fh2 = ICO * alphaCO / SigmaGas
        if pixel_by_pixel:
            corr_fh2 = w_pearsonr(
                fh2[mask], rg[mask], one_over_rg[mask])
        else:
            corr_fh2 = pearsonr(fh2[mask], rg[mask])[0]
        return (corr_metal, corr_pde, max_dm, corr_fh2)


def fitter(SigmaDust, SigmaHI, ICO, metal, SigmaMstar, one_over_rg, rg,
           r25_mpc,
           func='metal_power', params=np.zeros((0, 2)),
           pixel_by_pixel=True):
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
    const_metal = np.nanmax(metal) - np.nanmin(metal) < 0.05
    # Sanity check: m, k
    m, k = params.shape
    assert k == len(model.get_param_description())

    p = mp.Pool(mp.cpu_count())
    results = p.starmap(
        mp_wrapper,
        [(func,
          params[idx],
          SigmaHI,
          ICO,
          metal,
          SigmaMstar,
          SigmaDust,
          metal_z,
          one_over_rg,
          rg,
          r25_mpc,
          const_metal,
          pixel_by_pixel) for idx in range(m)]
    )
    return np.array([elem[0] for elem in results]), \
        np.array([elem[1] for elem in results]), \
        np.array([elem[2] for elem in results]), \
        np.array([elem[3] for elem in results])


def plotter(params, pspace_shape, param_1ds,
            ax, images1d, mode=0,
            aco_func='metal_power', objname=None):
    titles = [r'$\tilde{\rho}_{\rm f(H_2),~R_g}$',
              r'$\tilde{\rho}_{\rm D/M,~P_{DE}}$',
              'pct of D/M > 1 (%)',
              r'$<C_\theta>$']
    if objname is not None:
        titles[3] = r'$<C_\theta^{' + objname + '}>$'
    cmaps = ['bwr', 'bwr_r', 'bwr', 'bwr']
    vmaxs = [None] * 4
    vmins = [None] * 4
    if mode == 0:
        # plot one galaxy
        vmaxs = [1.0, 1.0, 10.0, 1.0]
        vmins = [-1.0, -1.0, 0.0, 0.0]
    elif mode == 1:
        # plot all galaxies: score
        cmaps = ['Greys'] * 4
        titles = ['D/M-12+log(O/H) Score',
                  r'D/M-P$_{DE}$ Score', 'Max D/M Score',
                  'Overall score']
        vmaxs = [1.0] * 4
        vmins = [-5.0] * 4
    elif mode == 2:
        # test coordinates
        if aco_func == 'metal_power':
            titles = ['p0 (log normalization)', 'p1 (slope)']
        # elif aco_func == 'b13':
        #     titles = ['a (exponential factor)',
        #               r'$\gamma$ (high-density correction)']
        elif aco_func == 'b13_3param':
            titles = ['q0 (log normalization)',
                      'q1 (exponential factor)',
                      r'$\gamma$ (high-density correction)'
                      ]
        elif aco_func == 'b13_no_gamma':
            titles = ['q0 (log normalization)',
                      'q1 (exponential factor)'
                      ]
        elif aco_func in {'double_power_law', 'double_power_law_no_cut'}:
            titles = ['p0 (log normalization)',
                      'p1 (slope)',
                      r'$\gamma$ (high-density correction)']
    elif mode == 3:
        # high-score points
        vmins = [0] * 4
        titles = ['Score top 1%',
                  'Score top 2%',
                  'Score top 3%',
                  'Score top 5%']
        cmaps = ['inferno'] * 4
    elif mode == 4:
        # likelihood sum
        vmins = [0] * 4
        titles = [None] * 4
        cmaps = ['Greys'] * 4
    #
    if aco_func == 'metal_power':
        xticklabels = [0.2, 0.6, 1.0]
        xticks = np.interp(xticklabels, param_1ds[0],
                           np.arange(len(param_1ds[0])))
        yticklabels = [-4.0, -2.0, 0.0]
        yticks = np.interp(yticklabels, param_1ds[1],
                           np.arange(len(param_1ds[1])))
        xlabel = r'$p_0$'  # (normalization)'
        ylabel = r'$p_1$'  # (slope)'
        qlabel = {0: r'$p_0$', 1: r'$p_1$'}
        qticks = {0: xticks, 1: yticks}
        qticklabels = {0: xticklabels, 1: yticklabels}
    # elif aco_func == 'b13':
    #     xtick_vals = [-0.2, 0.1, 0.4, 0.7, 1.0]
    #     xticks = np.interp(xtick_vals, param_1ds[0],
    #                        np.arange(len(param_1ds[0])))
    #     xticklabels = xtick_vals
    #     yticklabels = [0.0, 0.5, 1.0]
    #     yticks = np.interp(yticklabels, param_1ds[1],
    #                        np.arange(len(param_1ds[1])))
    #     xlabel = 'a'  # (exponential factor)'
    #     ylabel = r'$\gamma$'  # (high-density correction)'
    #     qlabel = {0: 'a', 1: r'$\gamma$'}
    #     qticks = {0: xticks, 1: yticks}
    #     qticklabels = {0: xticklabels, 1: yticklabels}
    elif aco_func == 'b13_no_gamma':
        qtick_vals = {
            0: [-0.5, 0.0, 0.5, 1.0, 1.5],
            1: [0.0, 0.5, 1.0, 1.5]}
        qticks = {}
        qticklabels = {}
        for i in range(2):
            qticks[i] = np.interp(qtick_vals[i], param_1ds[i],
                                  np.arange(len(param_1ds[i])))
            qticklabels[i] = [str(round(num, 1)) for num in qtick_vals[i]]
        qlabel = {
            0: r'$q_0$',  # (log-scale normalization)'}
            1: r'$q_1$'}  # (exponential factor)',
        xticklabels = qticklabels[0]
        xticks = qticks[0]
        yticklabels = qticklabels[1]
        yticks = qticks[1]
        xlabel = qlabel[0]
        ylabel = qlabel[1]
    elif aco_func == 'b13_3param':
        qtick_vals = {
            0: [-0.5, 0.0, 0.5, 1.0, 1.5],
            1: [0.0, 0.5, 1.0, 1.5],
            2: [0.0, 0.5, 1.0]}
        qticks = {}
        qticklabels = {}
        for i in range(3):
            qticks[i] = np.interp(qtick_vals[i], param_1ds[i],
                                  np.arange(len(param_1ds[i])))
            qticklabels[i] = [str(round(num, 1)) for num in qtick_vals[i]]
        qlabel = {
            0: r'$q_0$',  # (log-scale normalization)'}
            1: r'$q_1$',  # (exponential factor)',
            2: r'$\gamma$'}  # (high-density correction)',
        combs = [[1, 0], [2, 0], [2, 1]]
        # shape: (p00, p01, p02)
        sum_axis = [2, 1, 0]
    elif aco_func in {'double_power_law', 'double_power_law_no_cut'}:
        qtick_vals = {
            0: [0.2, 0.6, 1.0],
            1: [-4.0, -2.0, 0.0],
            2: [0.0, 0.5, 1.0]}
        qticks = {}
        qticklabels = {}
        for i in range(3):
            qticks[i] = np.interp(qtick_vals[i], param_1ds[i],
                                  np.arange(len(param_1ds[i])))
            qticklabels[i] = [str(round(num, 1)) for num in qtick_vals[i]]
        qlabel = {
            0: r'$p_0$',  # (normalization)',
            1: r'$p_1$',  # (slope)',
            2: r'$\gamma$'}  # (high-density correction)'}
        combs = [[1, 0], [2, 0], [2, 1]]
        # shape: (p00, p01, p02)
        sum_axis = [2, 1, 0]
    if mode == 4:
        # corner plot
        image3d = images1d[0].reshape(pspace_shape)
        for i in range(len(ax)):
            for j in range(len(ax)):
                if j > i:
                    pass
                elif j == i:
                    sum_axis_1d = tuple([k for k in range(len(ax)) if k != i])
                    w = param_1ds[i][1] - param_1ds[i][0]
                    ax[i, j].bar(param_1ds[i], np.nanmean(
                        image3d, axis=sum_axis_1d), color='white', edgecolor='black', width=w)
                    # ax[i, j].bar(param_1ds[i], np.percentile(
                    #     image3d, 16, axis=sum_axis_1d), color='white', edgecolor='black', width=w)
                    ax[i, j].set_xlim(
                        [np.min(param_1ds[i]), np.max(param_1ds[i])])
                    ax[i, j].set_ylim([0, 1])
                    ax[i, j].set_xlabel(qlabel[i], size=16)
                    # ax[i, j].set_xticks(qticks[i])
                    # ax[i, j].set_xticklabels(qticklabels[i], size=9)
                    # ax[i, j].tick_params(axis='y', which='both',
                    #                      left=False, right=False,
                    #                      labelleft=False, labelright=False)
                    ax[i, j].tick_params(axis='x', which='both',
                                         top=False, labeltop=False)
                    ax[i, j].tick_params(axis='x', which='minor',
                                         bottom=False)
                    ax[i, j].tick_params(
                        axis='x', which='major', direction='out')
                    # ax[i, j].set_title(qlabel[i], size=20)
                else:  # 2-d distribution
                    if len(image3d.shape) == 3:
                        sum_axis_1d = tuple(
                            [k for k in range(len(ax)) if (k != i) and (k != j)])
                        image2d = np.nanmean(image3d, axis=sum_axis_1d[0]).T
                        # image2d = np.percentile(image3d, 16, axis=sum_axis_1d[0]).T
                        im = ax[i, j].contourf(image2d,
                                               origin='lower',
                                               cmap=cmaps[i],
                                               vmin=0, vmax=None)
                    else:
                        image2d = image3d.T
                        im = ax[i, j].contourf(image2d,
                                               origin='lower',
                                               cmap=cmaps[i],
                                               vmin=0, vmax=None)
                    ax[i, j].set_xlabel(qlabel[j], size=16)
                    ax[i, j].set_xticks(qticks[j])
                    ax[i, j].set_xticklabels(qticklabels[j], size=9)
                    ax[i, j].set_yticks(qticks[i])
                    ax[i, j].set_yticklabels(qticklabels[i], size=9)
                    ax[i, j].set_ylabel(qlabel[i], size=16)
    else:
        if aco_func in {'metal_power', 'b13', 'b13_no_gamma'}:
            for i in range(len(ax)):
                im = ax[i].contourf(images1d[i].reshape(pspace_shape).T,
                                    origin='lower',
                                    cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i])
                plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
                    vmin=vmins[i], vmax=vmaxs[i]), cmap=cmaps[i]), ax=ax[i])
                # xlim = ax[i].get_xlim()
                ax[i].set_xlabel(xlabel, size=12)
                ax[i].set_xticks(xticks)
                # ax[i].set_xticklabels([round(num, 2) for num in xticklabels])
                ax[i].set_xticklabels(xticklabels, size=10)
                ax[i].set_yticks(yticks)
                ax[i].set_yticklabels([round(float(num), 1)
                                       for num in yticklabels], size=10)
                ax[i].set_title(titles[i], size=16)
            ax[0].set_ylabel(ylabel, size=12)
        else:
            for j in range(min(3, ax.shape[0])):
                x, y = combs[j]
                for i in range(ax.shape[1]):
                    image3d = images1d[i].reshape(pspace_shape)
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            "ignore", category=RuntimeWarning)
                        if mode == 0:  # One object
                            if i < 3:  # conditions
                                image2d = np.nanmedian(
                                    image3d, axis=sum_axis[j])
                            else:  # Overall cost
                                image2d = np.nanmean(
                                    image3d, axis=sum_axis[j])
                        elif mode == 1:  # Overall scores
                            image2d = np.log10(np.nanmean(
                                10**image3d, axis=sum_axis[j]))
                        elif mode == 2:  # test
                            image2d = np.nanmedian(image3d, axis=sum_axis[j])
                        elif mode == 3:  # count
                            image2d = np.nansum(image3d, axis=sum_axis[j])
                        if (vmins[i] is not None) and (np.nanmax(image2d) < vmins[i]):
                            image2d = np.full_like(image2d, vmins[i])
                    im = ax[j + 1, i].contourf(image2d,
                                               origin='lower',
                                               cmap=cmaps[i],
                                               vmin=vmins[i], vmax=vmaxs[i])
                    # im = ax[j, i].imshow(image2d,
                    #                      origin='lower',
                    #                      cmap=cmaps[i],
                    #                      vmin=vmins[i], vmax=vmaxs[i])
                    if (j == 0) and ax.shape[0] in {2, 4}:
                        dummyfig = plt.figure()
                        if vmins[i] is not None:
                            im = plt.imshow(np.array([[vmins[i], vmaxs[i]], [vmins[i], vmaxs[i]]]),
                                            cmap=cmaps[i],
                                            vmin=vmins[i], vmax=vmaxs[i])
                        plt.colorbar(im,
                                     cax=ax[0, i],
                                     orientation='horizontal')
                    ax[j + 1, i].set_xlabel(qlabel[x], size=12)
                    ax[j + 1, i].set_xticks(qticks[x])
                    ax[j + 1, i].set_xticklabels(qticklabels[x], size=10)
                    ax[j + 1, i].set_yticks(qticks[y])
                    ax[j + 1, i].set_yticklabels(qticklabels[y], size=10)
                    if j == 0:
                        ax[j, i].set_title(titles[i], size=16)
                ax[j + 1, 0].set_ylabel(qlabel[y], size=12)
