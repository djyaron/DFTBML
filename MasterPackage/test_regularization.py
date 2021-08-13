#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:34:30 2021

@author: yaron
"""
import numpy as np
import scipy.optimize
from Spline import spline_linear_model, spline_vals
from Spline import SplineModel, fit_linear_model
from MasterConstants import Bcond
import matplotlib.pyplot as plt


def bend(xvals, slope1 = 2.0, slope2 = 2.0):
   yvals = [slope1*x**2 if x<=0.0 else slope2*x**2 for x in xvals]
   return yvals
 
npts = 21
xk1 = np.linspace(-1.0,1.0,npts)
xstep = xk1[1] - xk1[0]
xk2 = np.hstack([[-1.0],xk1[:-1]+xstep/2.0,[1.0]])

bconds = (Bcond(0,2,0.0), Bcond(-1,2,0.0))

ntarget = 100
xtarget = np.linspace(-1.0,1.0, ntarget)
ytarget = bend(xtarget)

#%%
# sp1 = spline_linear_model(xk1,xtarget, (xtarget,ytarget), bconds)
# y1 = spline_vals(sp1)

# sp2 = spline_linear_model(xk2,xtarget, (xtarget,ytarget), bconds)
# y2 = spline_vals(sp2)

sm1 = SplineModel({'xknots': xk1})
c1,_,_ = fit_linear_model(sm1, xtarget, ytarget)
y1 = sm1.vals(c1, xtarget)

sm2 = SplineModel({'xknots': xk2})
c2,_,_ = fit_linear_model(sm2, xtarget, ytarget)
y2 = sm2.vals(c2, xtarget)

plt.figure(1)
plt.plot(xk1,     bend(xk1),'kx')
plt.plot(xtarget, ytarget,  'k-')
plt.plot(xtarget, y1,       'b.')
plt.plot(xtarget, y2,       'r.')
plt.figure(2)
plt.plot(xtarget, y1-ytarget,'b.')
plt.plot(xtarget, y2-ytarget,'r.')

#%%      
class Loss1:
    def __init__(self, splineModel, xtarget, ytarget):
        self.splineModel = splineModel
        self.xtarget = xtarget
        self.ytarget = ytarget
        self.A, self.b = splineModel.linear_model(xtarget)
        self.wconvex = None
    def add_convex(self,ngrid,weight):
        self.wconvex = weight
        self.xgrid = np.linspace(np.min(self.xtarget), np.max(self.xtarget),
                                 ngrid)
        self.A2, self.b2 = self.splineModel.linear_model(xtarget,2)
    def nvars(self):
        return self.A.shape[1]
    def errs(self, coefs):
        yvals = np.dot(self.A, coefs) + self.b
        diffs = yvals - self.ytarget
        if self.wconvex is not None:
            y2 = np.dot(self.A2, coefs) + self.b2
            cost2 = self.wconvex * np.where(y2<0, y2,0)
            diffs = np.concatenate([diffs,cost2],0)
        return diffs
    
l1 = Loss1(sm1,xtarget,ytarget)
c1,_,_ = fit_linear_model(sm1, xtarget, ytarget)

cguess = c1 + 0.2 * np.random.randn(c1.shape[0])

lsq1 = scipy.optimize.least_squares(l1.errs, cguess, method='lm')
cfit1 = lsq1['x']

l1.add_convex(100, 10.0)
lsq2 = scipy.optimize.least_squares(l1.errs, cguess, method='lm')
cfit2 = lsq2['x']

plt.figure(3)
plt.plot(xk1,     bend(xk1),'kx')
plt.plot(xtarget, ytarget,  'k-')
plt.plot(xtarget, sm1.vals(cfit1,xtarget),       'b.')
plt.plot(xtarget, sm1.vals(cfit2,xtarget),       'r.')
plt.figure(4)
plt.plot(xtarget, sm1.vals(cfit1,xtarget)-ytarget,'b.')
plt.plot(xtarget, sm1.vals(cfit2,xtarget)-ytarget,'r.')
plt.figure(5)
plt.plot(xtarget, sm1.vals(cfit1,xtarget,2),'b.')
plt.plot(xtarget, sm1.vals(cfit2,xtarget,2),'r.')
