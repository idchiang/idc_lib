#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 15:12:44 2020

@author: idchiang
"""
from .metal_power import PowerLawMetallicity, DoublePowerLawMetallicity
from .b13 import B13Exponential, B13All


dict_models = {'metal_power': PowerLawMetallicity(),
               'double_power_law': DoublePowerLawMetallicity(),
               'b13': B13Exponential(),
               'b13_3param': B13All()}
