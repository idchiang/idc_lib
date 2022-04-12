#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:46:15 2021

@author: idchiang
"""
import warnings
import numpy as np

max_loop = 100
max_aco = 10000


class PowerLawMetallicity():
    def __init__(self):
        pass

    def get_description(self):
        description = "A power-law with metallicity as single input"
        return description

    def get_param_description(self):
        description = [
            "log offset. suggested range: 0.0~1.2",
            "Slope for 12+log(O/H). suggested range: -4.0~0.5"]
        return description

    def aco_generator(self, params,
                      SigmaHI=None, ICO=None, metal=None, SigmaMstar=None):
        p0, p1 = params
        aco = 10**(p0 + p1 * (metal - 8.69))
        return aco


class DoublePowerLawMetallicity():
    def __init__(self):
        pass

    def get_description(self):
        description = "Metallicity Power Law + high-surface-desity correction"
        return description

    def get_param_description(self):
        description = [
            "log offset. suggested range: 0.0~1.2",
            "Slope for 12+log(O/H). suggested range: -4.0~0.5",
            "gamma: 0.2~0.8"]
        return description

    def aco_generator(self, params,
                      SigmaHI=None, ICO=None, metal=None, SigmaMstar=None):
        # 10**(p0 + p1 * (metal - 8.69)) * (SigmaTotal100)**(-gamma)
        p0, p1, gamma = params
        aco_prev = 10**(p0 + p1 * (metal - 8.69))
        SigmaH2_temp = ICO * aco_prev
        SigmaAtomStar = SigmaHI * 1.36 + SigmaMstar
        for i in range(max_loop):
            SigmaTot100 = (SigmaAtomStar + SigmaH2_temp) / 100.0
            aco = 10**(p0 + p1 * (metal - 8.69))
            aco[SigmaTot100 > 1] *= SigmaTot100[SigmaTot100 > 1]**(-gamma)
            aco[aco > max_aco] = np.nan
            if np.sum(np.isnan(aco)) == len(aco):
                break
            SigmaH2_temp = ICO * aco
            #
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                diff = np.abs(aco - aco_prev) / \
                    np.nanmax(np.array([aco, aco_prev]), axis=0)
                if np.nanmax(diff) <= 0.01:
                    break
            aco_prev = aco
        if i == max_loop - 1:
            mask = diff > 0.01
            print('WARNING: Double Metal LOOP LIMIT REACHED IN', params,
                  ICO[mask], SigmaAtomStar[mask], metal[mask])
        return aco


class DoublePowerLawMetallicityNoCut():
    def __init__(self):
        pass

    def get_description(self):
        description = "Metallicity Power Law + high-surface-desity correction"
        return description

    def get_param_description(self):
        description = [
            "log offset. suggested range: 0.0~1.2",
            "Slope for 12+log(O/H). suggested range: -4.0~0.5",
            "gamma: 0.2~0.8"]
        return description

    def aco_generator(self, params,
                      SigmaHI=None, ICO=None, metal=None, SigmaMstar=None):
        # 10**(p0 + p1 * (metal - 8.69)) * (SigmaTotal100)**(-gamma)
        p0, p1, gamma = params
        aco_prev = 10**(p0 + p1 * (metal - 8.69))
        SigmaH2_temp = ICO * aco_prev
        SigmaAtomStar = SigmaHI * 1.36 + SigmaMstar
        for i in range(max_loop):
            SigmaTot100 = (SigmaAtomStar + SigmaH2_temp) / 100.0
            aco = 10**(p0 + p1 * (metal - 8.69)) * SigmaTot100**(-gamma)
            aco[aco > max_aco] = np.nan
            if np.sum(np.isnan(aco)) == len(aco):
                break
            SigmaH2_temp = ICO * aco
            #
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                diff = np.abs(aco - aco_prev) / \
                    np.nanmax(np.array([aco, aco_prev]), axis=0)
                if np.nanmax(diff) <= 0.01:
                    break
            aco_prev = aco
        if i == max_loop - 1:
            mask = diff > 0.01
            print('WARNING: Double Metal LOOP LIMIT REACHED IN', params,
                  ICO[mask], SigmaAtomStar[mask], metal[mask])
        return aco
