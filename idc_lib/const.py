#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:03:31 2020

@author: idchiang
"""
import numpy as np
import astropy.units as u

col2sur = (1.0*u.M_p/u.cm**2).to(u.M_sun/u.pc**2).value
FWHM2sigma = 0.5 / np.sqrt(2*np.log(2))
