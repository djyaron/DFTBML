# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 09:04:37 2017

@author: yaron
"""

import math
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass

from auorg_1_1 import ParDict
from batch import Model,get_model_str


class PolyFunction(object):
    '''
    A simple polynomial basis function (rmax-r)^npow
    '''
    def __init__(self, npow, rmax):
        self.npow = npow
        self.rmax = rmax
    def values(self, rvals, derivative = 0):
        rtrunc = np.where(rvals > self.rmax, 0.0, self.rmax - rvals)
        if derivative == 0:
            res = rtrunc**self.npow
        elif derivative == 1:
            res = -1.0 * self.npow * rtrunc**(self.npow-1)
        return res

class BasisSet(object):
    '''
    values, at points r(ir) are linear combinations of the basis functions
      val[ir] = sum_ibasis  basisfunc[ibasis]( r[ir] ) * coef[ibasis]
      val = np.dot(X, coef)
    '''
    def __init__(self, basisfuncs = None):
        if basisfuncs is None:
            self.__basis = []
        else:
            self.__basis = basisfuncs
    def add_basisfunc(self, basisfunc):
        self.__basis.append(basisfunc)
    def nbasis(self):
        return len(self.__basis)
    def get_X(self, rvals, derivative = 0):
        res = np.zeros([len(rvals),self.nbasis()])
        for ibasis in range( self.nbasis() ):
            res[:,ibasis] = self.__basis[ibasis].values(rvals,derivative)
        return res
    def fit(self, x, y):
        X = self.get_X(x)
        cfit = np.linalg.lstsq(X,y)
        return X,cfit[0]
        
        
def polybasis(rmax, npows):
    bset = BasisSet()
    for npow in npows:
        bset.add_basisfunc(PolyFunction(npow,rmax))
    return bset
    
   

#%%
if __name__ == "__main__":
    from modelspline import get_dftb_vals
    mods = []
    oper = 'H'
    mods.append( Model(oper,(1,1), 'ss'     ) )
    for Zs in [(1,6), (1,8)]:
        mods.append( Model(oper,Zs, 'ss'     ) )
        mods.append( Model(oper,Zs, 'sp'     ) )
    for Zs in [(6,6), (8,8), (6,8) ]:
        mods.append( Model(oper,Zs, 'ss'     ) )
        mods.append( Model(oper,Zs, 'sp'     ) )
        mods.append( Model(oper,Zs, 'pp_sigma'  ) )
        mods.append( Model(oper,Zs, 'pp_pi'  ) )

    rvalsH = np.arange(0.65,12,0.1)
    rvalsC = np.arange(1.0,12,0.1)
    #yvals = np.exp(-alpha * rvals)
    
    pardict = ParDict()
    #Model = namedtuple('Model',['oper', 'Zs', 'orb'])
    if False:
        for nplot in range(1,5):
            plt.figure(nplot)
            plt.clf()
    
    for rmax_byhand in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
    #for cutoff in [10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]:
        errs = []
        errs_max = []
        errs_der = []
        rmax_save = {}
        for mod in mods:
            if 1 in mod.Zs:
                rvals = rvalsH
            else:
                rvals = rvalsC
            yvals = get_dftb_vals(mod, pardict, rvals)
            yvals_der = np.gradient(yvals, rvals[1]-rvals[0],edge_order = 2)
    
            #find point where yvals drops below a cutoff
            #temp1 = [rvals[i] for i in range(yvals.shape[0]) if (abs(yvals[i]) < cutoff/627.509)]
            #rmax = np.min(temp1)
            #rmax_save[mod] = rmax
            
            rmax = rmax_byhand            
            npows = list(range(4,13))
            bset = polybasis(rmax,npows)
                
            X,cfit = bset.fit(rvals, yvals)
            
            yfit = np.dot(X,cfit)
            
            err = np.sqrt( np.mean( (yvals-yfit)**2 ) ) * 627.509
            err_max = np.max(np.abs(yvals-yfit)) * 627.509
            errs.append(err)
            errs_max.append(err_max)    
        
            Xder = bset.get_X(rvals, derivative = 1)
            yfit_der = np.dot(Xder, cfit)
            
            err_der = np.sqrt( np.mean( (yvals_der[1:-1]-yfit_der[1:-1])**2 ) ) * 627.509
            errs_der.append(err_der)
            if False:
                print('model = ', get_model_str(mod), ' rms error = ', err, \
                    'max error = ',err_max, ' der err = ', err_der)
            
            if False: 
                plt.figure(1)
                plt.plot(rvals,yvals,'b-')
                plt.plot(rvals,yfit,'r.')
                plt.figure(2)
                plt.plot(rvals,yvals_der,'g-')
                plt.plot(rvals,yfit_der, 'k.')
                plt.figure(3)
                plt.plot(rvals,(yfit - yvals) * 627.509,'r-')
                plt.figure(4)
                plt.plot(rvals,(yfit_der - yvals_der) * 627.509,'r-')
        #print 'cutoff ', cutoff, ' r cutoff range ',np.min(rmax_save.values()), \
        #   ' to ', np.max(rmax_save.values())
        print('rmax = ', rmax_byhand,' powers = ', npows)
        print('mean error ',np.mean(errs),' max mean error ',np.max(errs), \
              'max_error ', np.max(errs_max))
        #print 'der  error ',np.mean(errs_der),' max error ',np.max(errs_der)

