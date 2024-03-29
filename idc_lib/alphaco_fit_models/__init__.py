#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:12:44 2020

@author: idchiang
"""
from .metal_power import PowerLawMetallicity, DoublePowerLawMetallicity, DoublePowerLawMetallicityNoCut
from .b13 import B13Exponential, B13All, B13nogamma


dict_models = {'metal_power': PowerLawMetallicity(),
               'double_power_law': DoublePowerLawMetallicity(),
               'double_power_law_no_cut': DoublePowerLawMetallicityNoCut(),
               'b13': B13Exponential(),
               'b13_3param': B13All(),
               'b13_no_gamma': B13nogamma()}
