#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 17:34:30 2021

Figures work better with this on command line in spyder:
     %matplotlib qt


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

sm1 = SplineModel({'xknots': xk1, 'deg': 3})
c1,_,_ = fit_linear_model(sm1, xtarget, ytarget)
y1 = sm1.vals(c1, xtarget)

sm2 = SplineModel({'xknots': xk2, 'deg': 3})
c2,_,_ = fit_linear_model(sm2, xtarget, ytarget)
y2 = sm2.vals(c2, xtarget)

plt.figure(1)
plt.subplot(2,1,1)
plt.plot(xk1,     bend(xk1),'kx', label='knots 1')
plt.plot(xk2,     bend(xk2),'ko', label='knots 2')
plt.plot(xtarget, bend(xtarget),  'k-', label='target')
plt.plot(xtarget, y1,       'b.', label='fit spline 1')
plt.plot(xtarget, y2,       'r.',label='fit spline 2')
plt.legend()
plt.title('fitting target to each spline')

plt.subplot(2,1,2)
plt.title('fit bend() to spline 1 and 2')
plt.plot(xtarget, y1-ytarget,'b.', label='err spline 1')
plt.plot(xtarget, y2-ytarget,'r.', label='err spline 2')
plt.legend()

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

#%% Compare fitting of spline 1 and spline 2, with and without convex constraint
#%matplotlib qt
for imod,smod in enumerate([sm1,sm2]):
    l1 = Loss1(smod,xtarget,ytarget)
    c1,_,_ = fit_linear_model(smod, xtarget, ytarget)
    
    cguess = c1 + 0.2 * np.random.randn(c1.shape[0])
    
    lsq1 = scipy.optimize.least_squares(l1.errs, cguess, method='lm')
    cfit1 = lsq1['x']
    
    cpoints = 100
    cweight = 10.0
    l1.add_convex(cpoints, cweight)
    lsq2 = scipy.optimize.least_squares(l1.errs, cguess, method='lm')
    cfit2 = lsq2['x']
    
    plt.figure(11)
    plt.subplot(3,2, imod + 1)
    plt.plot(xk1,     bend(xk1),'kx', label='knots')
    plt.plot(xtarget, ytarget,  'k-', label='target')
    plt.plot(xtarget, smod.vals(cfit1,xtarget),       'b-', label='no constraint')
    plt.plot(xtarget, smod.vals(cfit2,xtarget),       'r-', label='convex constraint')
    plt.title("fit "+str(imod+1)+"convex ngrid: " + str(cpoints) +
              " weight: " + str(cweight))
    plt.legend()
    
    plt.subplot(3,2, 2+imod +1)
    plt.plot(xtarget, smod.vals(cfit1,xtarget)-ytarget,'b-', label='err no constraint')
    plt.plot(xtarget, smod.vals(cfit2,xtarget)-ytarget,'r-', label='err convex')
    plt.legend()
    
    plt.subplot(3,2, 4+imod + 1)
    plt.plot(xtarget, smod.vals(cfit1,xtarget,2),'b-', label='2nd der no constaint')
    plt.plot(xtarget, smod.vals(cfit2,xtarget,2),'r-',label='2nd der convex')
    plt.legend()

#%% disagreement between spline 1 and spline 2
# First, will do this in a step-by-step manner
# Get some coefficients for spline 1
c1,_,_ = fit_linear_model(sm1, xtarget, ytarget)
# Get predicted values from spline 1 on some grid
ngrid = 500
xgrid = np.linspace(np.min(xtarget), np.max(xtarget),ngrid)
pred1 = sm1.vals(c1,xgrid)
# Fit spline 2 to these predicions
c2,_,_ =  fit_linear_model(sm2, xgrid, pred1)
pred2 = sm2.vals(c2,xgrid)
# calculate loss for disagreement of sm1 and sm2
rloss = np.sqrt(np.mean(np.square(pred1-pred2)))

plt.figure(10)
plt.title("splines 1 - spline 2: "+str(rloss))
plt.plot(xgrid, pred2-pred1)

