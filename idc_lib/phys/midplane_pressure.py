#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 17:18:08 2020

@author: idchiang
"""
import numpy as np
import astropy.units as u
import astropy.constants as const


P_DE_self_coef = np.pi / 2 * \
    (const.G / const.k_B * (u.solMass / u.pc**2)**2).to(u.K / u.cm**3).value
P_DE_star_coef = u.solMass / u.pc**2 * u.km / u.s / const.k_B * \
        np.sqrt(const.G * u.solMass / u.pc**2 / u.Mpc)
P_DE_star_coef = P_DE_star_coef.to(u.K / u.cm**3).value
P_DE_star_coef *= np.sqrt(2 / 0.54 * 4.6) * 11.0


def P_DE_self(gas):
    return P_DE_self_coef * gas**2


def P_DE_star(gas, star, R25_mpc):
    return P_DE_star_coef * gas * np.sqrt(star) / R25_mpc


def P_DE(gas, star, R25_mpc):
    return P_DE_self(gas) + P_DE_star(gas, star, R25_mpc)
