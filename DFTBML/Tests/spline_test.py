# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 17:17:15 2021

@author: fhu14

This file runs the test_join function originally implemented in tfspline
"""
#%% Imports, definitions

import matplotlib.pyplot as plt
import numpy as np
from Spline import (construct_joined_splines, maxabs, merge_splines,
                    merge_splines_new_xvals, plot_spline, plot_spline_basis,
                    spline_new_xvals, spline_vals)

#%% Code behind

def test_join():
    mod_c1 = 0.5
    mod_c2 = 5.0
    funcs     = [lambda x: np.exp(-mod_c1*x) * np.cos(mod_c2*x),
                 lambda x: np.exp(-mod_c1*x) * ( -mod_c2 * np.sin(mod_c2*x) 
                                               - mod_c1 * np.cos(mod_c2*x) )]

    xdense = np.linspace(1.2, 5, 500)
    ydense = [np.apply_along_axis(funcs[ider],0,xdense) for ider in range(2)]
    
    # including both internal and external knots
    xk = np.linspace(1.2,5,30)
    yk = [np.apply_along_axis(funcs[ider],0,xk) for ider in range(2)]

    xknots = [xk[:15], xk[14:]]
    
    # make sure things still work with nonordered data
    iorder = np.arange(len(xdense))
    np.random.shuffle(iorder)
    xvals = xdense[iorder]
    yvals = ydense[0][iorder]   
    
    xevals = np.linspace(1.2, 5.0, 100)
    np.random.shuffle(xevals)
    
    #boundaries = ['natural',('fixed', yk[-1], yk_der[-1])]
    boundaries = [('fixed',yk[0][0],yk[1][0]),('fixed', yk[0][-1], yk[1][-1])]

    res = construct_joined_splines(xknots, xvals, yvals, boundaries, xevals)
    sps = [x['spline'] for x in res]
    #plot basis sets
    spdense = [spline_new_xvals(sp, xdense) for sp in sps]
    for ider in range(2):
        plt.figure(10+ider)
        for isp, sp in enumerate(spdense):
            plt.subplot(3,1,isp+1)
            plot_spline_basis(sp,ider)
            
    # plot comparisons
    for ider in range(2):
        plt.figure(100 + ider)
        plt.plot(xdense,ydense[ider],'y--')
        plot_spline(sps[2],ider, sym='g-' , sortx = True)
        plot_spline(sps[0],ider, sym='b--', sortx = True)
        plot_spline(sps[1],ider, sym='r--', sortx = True)

    # numerical comparisons
    print('Approximation error for the full spline (all comparisons are maxabs)')
    st = ''
    for ider in range(2):
        spx = spline_new_xvals(sps[2], xdense)
        diffs = spline_vals(spx,ider) - ydense[ider] 
        st += ' der '+str(ider)+'='+str(maxabs(diffs))
    print(st)      
    
    # comparison of full spline and approximate splines
    for idict,rdict in enumerate(res[0:2]):
        st = 'segment ' + str(ider) + ' diff from full'
        for ider in range(2):
            sp = rdict['spline']
            ievals = rdict['ievals']
            xx = xevals[ievals]
            yy = spline_vals(sp, ider)
            spfull = spline_new_xvals(res[2]['spline'],xx)
            yyf = spline_vals(spfull, ider)
            st += ' ider ' + str(ider) + ' err ' + str(maxabs(yyf-yy))
        print(st)

    print('Testing merge_splines')
    Xm, constm = merge_splines(res)
    coefm = np.concatenate([rdict['spline']['coefs'] for rdict in res[0:2]])
    st = 'ymerged versus yfull'
    for ider in range(2):
        ym = np.dot(Xm[ider], coefm) + constm[ider]
        yfull = spline_vals(res[2]['spline'],ider=ider)
        st += ' ider = ' + str(ider) + ' ' + str(maxabs(ym-yfull))
    print(st)

    print('Testing merge_splines_new_xvals')
    nv = 150
    iorder = np.arange(nv)
    np.random.shuffle(iorder)
    xv= np.linspace(1.2, 5.0, nv)
    xv = xv[iorder]
    Xm, constm = merge_splines_new_xvals(res, xv)
    coefm = np.concatenate([rdict['spline']['coefs'] for rdict in res[0:2]])
    st = 'ymerged versus yfull'
    for ider in range(2):        
        ym = np.dot(Xm[ider],coefm) + constm[ider]
        spnew = spline_new_xvals(res[2]['spline'], xv)
        ytest = spline_vals(spnew,ider)
        st += ' ider = ' + str(ider) + ' ' + str(maxabs(ym-ytest))
    print(st)
    
def run_spline_tests():
    test_join()

#%% Main block
if __name__ == "__main__":
    test_join()

