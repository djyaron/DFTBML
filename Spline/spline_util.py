# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:06:38 2021

@author: fhu14

This file will contain misc functions like get_dftb_vals and plotting 
utilities for the splines. 

Functions here should be exposable at the top level

TODO: The _Gamma12 functions and others from sccparam need to be packaged
first.
"""
#%% Imports, definitions
from Elements import ELEMENTS
import numpy as np
from constants import ANGSTROM2BOHR

#%% Code behind

def get_dftb_vals(mod, par_dict, rs = []):
    sym = [ELEMENTS[z].symbol for z in mod.Zs]
    skinfo = par_dict[ sym[0] + '-' + sym[-1] ]
    if mod.oper == 'G':
        hub1 = par_dict[sym[0]+'-'+sym[0]].GetAtomProp('U'+mod.orb[0])
        hub2 = par_dict[sym[-1]+'-'+sym[-1]].GetAtomProp('U'+mod.orb[-1])
        if len(mod.Zs) == 1:
            value = _Gamma12(0.0, hub1, hub2)
            return np.array(value), value, hub1, hub2
        else:
            rs_bohr = rs * ANGSTROM2BOHR
            gs = [_Gamma12(r,hub1,hub2) for r in rs_bohr]
            return np.array(gs)
    if mod.oper == 'R':
        #print 'R for '+get_model_str(mod)
        #for x in rs:
        #    print x*ANGSTROM2BOHR, skinfo.GetRep(x*ANGSTROM2BOHR)
        return np.array( [skinfo.GetRep(x*ANGSTROM2BOHR) for x in rs] )
    if len(mod.Zs) == 1:
        # no models for S oper on a single atom
        sk_key = 'E' + mod.orb
        return skinfo.GetAtomProp(sk_key)
    else:
        to_sk = {'ss' : 'ss0', 'sp': 'sp0', 'pp_sigma': 'pp0', 
                 'pp_pi': 'pp1', 'sd': 'sd0',
                 'pd_sigma': 'pd0', 'pd_pi': 'pd1', 
                 'dd_sigma': 'dd0', 'dd_pi': 'dd1', 'dd_delta':'dd2'}
        sk_key = mod.oper + to_sk[mod.orb]
        rs_bohr = rs * ANGSTROM2BOHR
        vals = [skinfo.GetSkInt(sk_key, x) for x in rs_bohr]
        return np.array(vals)