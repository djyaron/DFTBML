# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:43:18 2021

@author: fhu14

In the future, can expose more models for use
"""
from .spline_model_backend import fit_linear_model, SplineModel, JoinedSplineModel
from .spline_backend import Bcond, spline_linear_model, construct_joined_splines,\
    spline_new_xvals, plot_spline_basis, plot_spline, spline_vals, maxabs,\
        merge_splines, merge_splines_new_xvals
from .spline_util import get_dftb_vals