#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 20:42:55 2020

@author: yaron
"""

import numpy as np
#import torch
from sccparam import _Gamma12 #Gamma func for computing off-diagonal elements
from functools import partial
from typing import Union, List, Optional, Dict, Any, Literal
#Tensor = torch.Tensor
Array = np.ndarray


def get_feed(xeval: Array) -> Dict:
    r"""Gets information for the feed dictionary for the OffDiagModel
    
    Arguments:
        mod_raw (List[RawData]): A list of raw data used to extract the distances
            at which to evaluate the model
    
    Returns:
        distances (Dict): A dictionary containing an array of all the distances
            at which to evaluate the model
    
    Notes: The only thing that needs to be added to the feed dictionary for the 
        OffDiagModel are the distances at which to evaluate them.
    """
    #xeval = np.arragety([elem.rdist for elem in mod_raw])
    zero_indices = np.where(xeval < 1.0e-5)[0]
    nonzero_indices = np.where(xeval >= 1.0e-5)[0]
    assert( len(zero_indices) + len(nonzero_indices) == len(xeval))
    nonzero_distances = xeval[nonzero_indices]
    return {'zero_indices'      : zero_indices,
            'nonzero_indices'   : nonzero_indices,
            'nonzero_distances' : nonzero_distances}

def _Expr(r12, tau1, tau2):
    sq1, sq2 = tau1**2, tau2**2
    sq1msq2 = sq1 - sq2
    quad2 = sq2**2
    termExp = np.exp(-tau1 * r12)
    term1 = 0.5 * quad2 * tau1 / sq1msq2**2
    term2 = sq2**3 - 3.0 * quad2 * sq1
    term3 = r12 * sq1msq2**3
    return termExp * (term1 - term2 / term3)
    
def get_values(feed: Dict, variables) -> Array:
    r"""Obtains the predicted values from the model
    
    Arguments:
        feed (Dict): Dictionary containing the distances to evaluate the model at
    
    Returns:
        results (Tensor): The predicted values for the model at the necessary 
            distances
    
    Notes: Computed by a map of _Gamma12() across all the distances with
        the given variables. Switching variable order does not affect
        computed result, i.e. _Gamma12(r, x, y) == _Gamma12(r, y, x).
    """
    zero_indices = feed['zero_indices']
    nonzero_indices = feed['nonzero_indices']
    r12 = feed['nonzero_distances']
    nelements = len(zero_indices) + len(nonzero_indices)
    results = np.zeros([nelements])
    
    hub1, hub2 = variables
    smallHubDiff = abs(hub1-hub2) < 0.3125e-5
    tau1 = 3.2 * hub1
    tau2 = 3.2 * hub2
    
    # G between shells on the same atom
    if len(zero_indices) > 0:
        if smallHubDiff:
            onatom = 0.5 * (hub1 + hub2)
        else:
            p12 = tau1 * tau2
            s12 = tau1 + tau2
            pOverS = p12 / s12
            onatom = 0.5 * ( pOverS + pOverS**2 / s12 )
        results[zero_indices] = onatom
        
    # G between atoms
    if len(nonzero_indices) > 0:
        if smallHubDiff:
            tauMean = 0.5 * (tau1 + tau2)
            termExp = np.exp(-tauMean * r12)
            term1 = 1.0/r12 + 0.6875 * tauMean + 0.1875 * r12 * tauMean**2
            term2 = 0.02083333333333333333 * r12**2 * tauMean**3
            expr = termExp * (term1 + term2)        
        else:
            expr = _Expr(r12, tau1, tau2) + _Expr(r12, tau2, tau1)
        results[nonzero_indices] = 1.0 / r12 - expr                      
        
    return results
#%%
# Generate random differences, with interspersed zero elements
passed = True
ntest = 1000
for zeros in ['none', 'some', 'all']:
    for hubdiff in [0, 2.12, -3.45]:
        if zeros == 'none':
            r12 = np.random.uniform(0.1, 10.0, ntest)
        elif zeros == 'some':
            r12 = np.random.uniform(-3.0, 10.0, ntest)
            r12[r12 < 0.0] = 0.0
        elif zeros == 'all':
            r12 = np.zeros([ntest])            
        
        hub1 = 11.11
        hub2 = hub1 + hubdiff
        
        feed = get_feed(r12)
        vals = get_values(feed,[hub1, hub2])
        
        test = np.array([_Gamma12(x, hub1, hub2) for x in r12])
        
        max_diff = np.max(np.abs( vals- test))
        if max_diff > 0.0:
            passed = False
        
        print('hub1',hub1,'hub2',hub2, zeros, 'zeros', 
              '#zeros', np.count_nonzero(r12 < 1.0e-5),
              'max diff',max_diff)
print("test passed = ", passed)



