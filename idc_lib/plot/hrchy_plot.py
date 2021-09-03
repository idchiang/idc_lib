#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:21:03 2020

@author: idchiang
"""
from matplotlib.backends.backend_pdf import PdfPages


def savefig(fig, figname, tight_layout=False):
    """
    Save a PNG + a PDF file of the input figure

    Parameters
    ----------
    fig : matplotlib.figure.Figure() object
        The figure to save.
    figname : str
        file path + file name to save. excluding extension.

    Returns
    -------
    None.

    """
    bbox_inches = 'tight' if tight_layout else None
    fig.savefig(figname + '.png', bbox_inches=bbox_inches)
    with PdfPages(figname + '.pdf') as pp:
        pp.savefig(fig, bbox_inches=bbox_inches)
