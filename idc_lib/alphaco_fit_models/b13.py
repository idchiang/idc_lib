#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:46:15 2021

@author: idchiang
"""
import warnings
import numpy as np
from idc_lib.phys.metallicity import metal2Z


max_loop = 100
max_aco = 10000


class B13Exponential():
    def __init__(self):
        pass

    def get_description(self):
        description = "BWL13 metallicity exponential + high-desity correction"
        return description

    def get_param_description(self):
        description = [
            "Exponential index (and SigmaGMC): -0.3~1.1",
            "gamma: 0.2~0.8"]
        return description

    def aco_generator(self, params,
                      SigmaHI=None, ICO=None, metal=None, SigmaMstar=None):
        # 2.9 * np.exp(0.4 / Z' / SigmaGMC100) * (SigmaTotal100)**(-gamma)
        q1, gamma = params
        SigmaH2_temp = ICO * 4.35
        SigmaAtomStar = SigmaHI * 1.36 + SigmaMstar
        metal_rel = metal2Z(metal) / metal2Z(8.69)
        aco_prev = np.full_like(ICO, 4.35)
        for i in range(max_loop):
            SigmaTot100 = (SigmaAtomStar + SigmaH2_temp) / 100.0
            aco = 2.9 * np.exp(q1 / metal_rel)
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
            print('WARNING: B13 LOOP LIMIT REACHED IN', params,
                  ICO[mask], SigmaAtomStar[mask], metal[mask])
        return aco


class B13All():
    def __init__(self):
        pass

    def get_description(self):
        description = "BWL13 metallicity exponential + " + \
            "high-desity correction + normalization"
        return description

    def get_param_description(self):
        description = [
            "log(normalization): log(2.9/2)~log(2.9*2)",
            "Exponential index (and SigmaGMC): -0.3~1.1",
            "gamma: 0.0~1.0"]
        return description

    def aco_generator(self, params,
                      SigmaHI=None, ICO=None, metal=None, SigmaMstar=None):
        # 2.9 * np.exp(0.4 / Z' / SigmaGMC100) * (SigmaTotal100)**(-gamma)
        q0, q1, gamma = params
        q0_exp = 10**q0
        SigmaH2_temp = ICO * 4.35
        SigmaAtomStar = SigmaHI * 1.36 + SigmaMstar
        metal_rel = metal2Z(metal) / metal2Z(8.69)
        aco_prev = np.full_like(ICO, 4.35)
        for i in range(max_loop):
            SigmaTot100 = (SigmaAtomStar + SigmaH2_temp) / 100.0
            aco = q0_exp * np.exp(q1 / metal_rel)
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
            print('WARNING: B13 LOOP LIMIT REACHED IN', params,
                  ICO[mask], SigmaAtomStar[mask], metal[mask])
        return aco


class B13nogamma():
    def __init__(self):
        pass

    def get_description(self):
        description = "BWL13 metallicity exponential"
        return description

    def get_param_description(self):
        description = [
            "log(normalization): log(2.9/2)~log(2.9*2)",
            "Exponential index (and SigmaGMC): -0.3~1.1"]
        return description

    def aco_generator(self, params,
                      SigmaHI=None, ICO=None, metal=None, SigmaMstar=None):
        # 2.9 * np.exp(0.4 / Z' / SigmaGMC100) * (SigmaTotal100)**(-gamma)
        q0, q1 = params
        q0_exp = 10**q0
        metal_rel = metal2Z(metal) / metal2Z(8.69)
        aco = q0_exp * np.exp(q1 / metal_rel)
        return aco
