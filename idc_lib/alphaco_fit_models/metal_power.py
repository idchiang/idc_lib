#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:46:15 2021

@author: idchiang
"""


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
