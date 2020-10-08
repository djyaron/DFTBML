# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:24:30 2017

@author: yaron

"""
import math
import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
try:
    import tensorflow as tf
except:
    pass

from auorg_1_1 import ParDict
from sccparam import _Gamma12
from elements import ELEMENTS
from dftb import ANGSTROM2BOHR

from tfspline import construct_joined_splines, merge_splines, \
          merge_splines_new_xvals, absorb_constant, spline_linear_model, \
          Bcond, spline_new_xvals
from batch import Model, get_model_str
from modelbasis import PolyFunction, BasisSet, polybasis
from util import maxabs, unit_print

def get_dftb_grid(mod,par_dict):
    ''' r values from the DFTB sk files'''
    # returns values in Angstroms
    sym = [ELEMENTS[z].symbol for z in mod.Zs]
    skinfo = par_dict[ sym[0] + '-' + sym[-1] ]
    return np.array( skinfo.GetSkGrid() ) / ANGSTROM2BOHR

def get_dftb_vals(mod, par_dict, rs = []):
    sym = [ELEMENTS[z].symbol for z in mod.Zs]
    skinfo = par_dict[ sym[0] + '-' + sym[-1] ]
    if mod.oper == 'G':
        hub1 = par_dict[sym[0]+'-'+sym[0]].GetAtomProp('U'+mod.orb[0])
        hub2 = par_dict[sym[-1]+'-'+sym[-1]].GetAtomProp('U'+mod.orb[-1])
        if len(mod.Zs) == 1:
            return np.array(_Gamma12(0.0, hub1, hub2))
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

def npflip(x,ix):
    nx = len(x)
    return [x[nx-i-1] for i in range(nx)]

def rcutoff(mod, pardict, cutoff, rlow, rmax, rprec = 0.01, plot=False):
    '''
      Find point beyond which DFTB values are less than cutoff fraction of the
      max value.
      Input:
         mod     - Model from batch.py
         pardict - DFTB matrix element object
         cutoff  - matrix elements beyond cutoff < (max element in range) 
                                                    * cutoff
         rlow,rmax - range for consideration (in angstroms)
         rprec   - searches on grid with this spacing between r values
                   rcut is from linear interpolation on this grid
      Output:
         rcut - distance for cutoff
    '''
    xvals = np.linspace(rlow,rmax,int((rmax-rlow)/rprec))
    # Model = namedtuple('Model',['oper', 'Zs', 'orb'])
    if mod.oper == 'G':
        mc_zs = mod.Zs
        mc_orb = mod.orb
        if mod.orb in ['pp']:
            mc_orb = 'pp_sigma'
        if mod.orb == 'ps':
            mc_orb = 'sp'
            mc_zs = (mod.Zs[1], mod.Zs[0])
        mod_cut = Model('H',mc_zs, mc_orb)
    else:
        mod_cut = mod
    yvals = get_dftb_vals(mod_cut, pardict, xvals)
    ycut = cutoff * np.max(np.abs(yvals))
    # takes advantage of argmax returning the first value for which 
    # the array is True. We reverse yvals, and walk back from large distances
    # until the value goes above ycut
    icut = (len(yvals) - 1) - np.argmax( npflip(np.abs(yvals),0) >= ycut )
    if icut == len(yvals) - 1:
        rcut = xvals[-1]
    else:
        # linear fit
        xs = xvals[icut:(icut+2)]
        ys = np.abs(yvals[icut:(icut+2)])
        slope = (ys[1]-ys[0])/(xs[1]-xs[0])
        # ycut = ys[0] + slope * (rcut-xs[0])
        rcut = xs[0] + (ycut-ys[0])/slope
    assert(rcut >= xs[0] and rcut <= xs[1])
    if plot:
        yvals = get_dftb_vals(mod, pardict, xvals)
        plt.plot(xvals,yvals*627.509,'k-')
        plt.plot(xvals[:icut+1],yvals[:icut+1]*627.509,'b.')
        plt.plot(xvals[icut+1:],yvals[icut+1:]*627.509,'r.')    
        plt.title(str(mod.oper) + ' ' + str(mod.Zs) + ' ' + str(mod.orb))
        if mod.oper != 'G':
            plt.plot([rlow,rmax],np.sign(yvals[icut]) * 627.509 * 
                                                np.array([ycut,ycut]),'r--')
            plt.plot([rcut],[np.sign(yvals[icut]) *ycut*627.509],'rx')
    return rcut

def spline_style(oper, spec):
    if oper not in ['H','G','R']:
        raise ValueError('unrecognized oper in spline_style()')

    if spec != '6':
        res =  {'type' : 'joined',
                'rmax' : 12.0,
                'knots' : (('nknots',20), ('nknots',20)) }

    if spec == '0':
        # From plots of distributions, these are a bit past the covalent range
        if oper in ['H','G']:
            res['rjoin'] = {(1, 1): 2.24, (1, 6): 2.4, (1, 8): 2.1, 
                             (6, 6): 2.7,  (8, 8): 2.3, (6, 8): 2.4}
        elif oper == 'R':
            res['rjoin'] = {(1, 1): 1.04, (1, 6): 0.96, (1, 8): 0.901, 
                             (6, 6): 1.33,  (8, 8): 1.87, (6, 8): 1.33}
        res['zeros'] = False
    elif spec  == '1':
        if oper in ['H','G']:
            res['rjoin'] = {zs: 3.5 for zs in Zs}
        elif oper == 'R':
            res['rjoin'] = {(1, 1): 1.04, (1, 6): 0.96, (1, 8): 0.901, 
             (6, 6): 1.33,  (8, 8): 1.87, (6, 8): 1.33}
        res['zeros'] = False
    elif spec  == '2':
        if oper in ['H','G']:
            res['rjoin'] =  {zs: 3.5 for zs in Zs}
            res['zeros']   = False
        elif oper == 'R':
            res['rjoin'] =  {(1, 1): 1.04, (1, 6): 1.75, (1, 8): 1.4, 
             (6, 6): 1.75,  (8, 8): 2.0, (6, 8): 1.8}
            res['zeros']  = True
    elif spec  == '3':
        if oper in ['H','G']:
            res['rjoin'] =  {(1, 1): 1.95, (1, 6): 2.25, (1, 8): 2.20, 
             (6, 6): 2.6,  (8, 8): 2.35, (6, 8): 2.45}
            res['zeros']   = False
        elif oper == 'R':
            res['rjoin'] =  {(1, 1): 1.04, (1, 6): 1.75, (1, 8): 1.4, 
             (6, 6): 1.75,  (8, 8): 2.0, (6, 8): 1.8}
            res['zeros']  = True
    elif spec  == '4':
        if oper in ['H','G']:
            res['rjoin'] =  {(1, 1): 1.95, (1, 6): 2.35, (1, 8): 2.20, 
             (6, 6): 2.7,  (8, 8): 2.35, (6, 8): 2.5,
             (1, 7): 2.35,  (6, 7): 2.5, 
             (1,79): 2.5,  (79,79): 3.5,  (6,79): 2.7 }
            res['zeros']   = False
        elif oper == 'R':
            res['rjoin'] =  {(1, 1): 1.04, (1, 6): 1.5, (1, 8): 1.4, 
             (6, 6): 1.75,  (8, 8): 1.6, (6, 8): 1.8,
             (1, 7): 1.5,  (6, 7): 1.75, 
             (1,79): 2.5,  (79,79): 3.5,  (6,79): 2.7 }
            res['zeros']  = True
    elif spec  == '5':

        # It looks like at some point 'rhighs' were changed to 'rjoin'. Hence
        # some confusion over 'rhighs'. Note that none of the following spline
        # values can be trusted. They will have to be double checked to make
        # sure that they are physically meaningful.

        # These values are the "lower end of the spline". The assumption is
        # that this sets the lower bound of the modifiable spline. That is to
        # say, no modifications to points of the spline below these values can
        # be made during training.
        rlows = {
            (1, 1):  0.62, (1, 6):  0.80, (1, 7):   0.80,
            (1, 8):  0.75, (6, 6):  1.00, (6, 7):   1.00,
            (6, 8):  1.00, (7, 7):  1.00, (7, 8):   1.00,
            (8, 8):  1.00, (1, 79): 0.80, (6, 79):  0.80, 
            (7, 79): 0.80, (8, 79): 0.80, (79, 79): 2.00}

        # The assumption is made that 'rjoin' values specify the point at which
        # the fixed splines take over. That is to say that the spline can only
        # be modified between the distance rlow > x > rjoin.
        # rjoin values for the repulsive term
        if oper == 'R':
            rjoin = {
                (1, 1):  1.04, (1, 6):  1.50, (1, 7):   1.60,
                (1, 8):  1.40, (6, 6):  1.75, (6, 7):   1.65,
                (6, 8):  1.80, (7, 7):  1.65, (7, 8):   1.65,
                (8, 8):  1.60, (1, 79): 2.06, (6, 79):  2.62,
                (7, 79): 2.57, (8, 79): 2.52, (79, 79): 3.00}

        # rjoin values for the H and G terms
        elif oper in ['H', 'G']:
            rjoin = {
                (1, 1):  2.10, (1, 6):  2.45, (1, 7):   2.35,
                (1, 8):  2.33, (6, 6):  2.80, (6, 7):   2.77,
                (6, 8):  2.65, (7, 7):  2.65, (7, 8):   2.46,
                (8, 8):  2.45, (1, 79): 3.00, (6, 79):  3.00,
                (7, 79): 3.00, (8, 79): 3.00, (79, 79): 3.50}

        res['rlows'] = rlows
        res['rjoin'] = rjoin

    elif spec == '6':

        res =  {'type' : 'joined',
                'rmax' : 12.0,
                'knots' : (('nknots',30), ('nknots',20)) }

        rlows = {
            (1, 1):  0.72, (1, 6):  0.80, (1, 7):   0.80,
            (1, 8):  0.75, (6, 6):  1.00, (6, 7):   1.00,
            (6, 8):  1.00, (7, 7):  1.00, (7, 8):   1.00,
            (8, 8):  1.00, (1, 79): 0.80, (6, 79):  0.80,
            (7, 79): 0.80, (8, 79): 0.80, (79, 79): 2.00}

        if oper == 'R':
            rjoin = {
                (1, 1):  1.25, (1, 6):  2.20, (1, 7):   2.25,
                (1, 8):  2.25, (6, 6):  2.25, (6, 7):   2.25,
                (6, 8):  2.25, (7, 7):  2.40, (7, 8):   2.30,
                (8, 8):  2.25, (1, 79): 3.00, (6, 79):  3.00,
                (7, 79): 3.00, (8, 79): 3.00, (79, 79): 3.75}


            res['zeros'] = True

        elif oper in ['H', 'G']:
            rjoin = {
                (1, 1):  5.0, (1, 6):  5.0, (1, 7):   5.0,
                (1, 8):  5.0, (6, 6):  5.0, (6, 7):   5.0,
                (6, 8):  5.0, (7, 7):  5.0, (7, 8):   5.0,
                (8, 8):  5.0, (1, 79): 5.0, (6, 79):  5.0,
                (7, 79): 5.0, (8, 79): 5.0, (79, 79): 5.0}

        res['rlows'] = rlows
        res['rjoin'] = rjoin

    elif spec == '7':
        # Same as spec 6 but uses the same cut-offs as the dftb+ Auorg set

        if oper == 'R':
            res = {'type': 'joined',
                   'rmax': 12.0,
                   'knots': (('nknots', 30), ('nknots', 20))}
            rlows = {
                (1, 1): 0.72, (1, 6): 0.80, (1, 7): 0.80,
                (1, 8): 0.75, (6, 6): 1.00, (6, 7): 1.00,
                (6, 8): 1.00, (7, 7): 1.00, (7, 8): 1.00,
                (8, 8): 1.00, (1, 79): 0.80, (6, 79): 0.80,
                (7, 79): 0.80, (8, 79): 0.80, (79, 79): 2.00}
            rjoin = {
                (1, 1):  1.25, (1, 6):  2.20, (1, 7):   2.25,
                (1, 8):  2.25, (6, 6):  2.25, (6, 7):   2.25,
                (6, 8):  2.25, (7, 7):  2.40, (7, 8):   2.30,
                (8, 8):  2.25, (1, 79): 3.00, (6, 79):  3.00,
                (7, 79): 3.00, (8, 79): 3.00, (79, 79): 3.75}


            res['zeros'] = True

        elif oper in ['H', 'G']:
            res = {'type': 'joined',
                   'rmax': 12.0,
                   'knots': (('nknots', 40), ('nknots', 20))}
            # Realistically these should be at 0.20, however such a value causes
            # dftbd to pick up the zeros that are used to pad the lower distance
            # components of the DFTB+ parameter set, causing issues.
            rlows = {
                (1, 1): 0.25, (1, 6): 0.25, (1, 7): 0.25,
                (1, 8): 0.25, (6, 6): 0.25, (6, 7): 0.25,
                (6, 8): 0.25, (7, 7): 0.25, (7, 8): 0.25,
                (8, 8): 0.25, (1, 79): 0.25, (6, 79): 0.25,
                (7, 79): 0.25, (8, 79): 0.25, (79, 79): 0.25}
            rjoin = {
                (1, 1):  5.0, (1, 6):  5.0, (1, 7):   5.0,
                (1, 8):  5.0, (6, 6):  5.0, (6, 7):   5.0,
                (6, 8):  5.0, (7, 7):  5.0, (7, 8):   5.0,
                (8, 8):  5.0, (1, 79): 5.0, (6, 79):  5.0,
                (7, 79): 5.0, (8, 79): 5.0, (79, 79): 5.0}

        res['rlows'] = rlows
        res['rjoin'] = rjoin

    elif spec == 'f':
        # set all rhighs to values below rlow, this will hold them const
        res['rjoin'] = {k:(v-0.1) for k,v in res['rlows'].items()}
        res['zeros'] = False
    else:
        raise ValueError('spline style specifier not recognized: '+spec)
    return res


def construct_joined_spline_model(mod, style, par_dict, plot = False):
    '''
    Input:
       mod     - Model from batch.py
       style   - dict with info on how to construct spline (see below)
       par_dict - DFTB matrix element object
       plot     - if true, add plots used to find rcut to currently active 
                  figure
    Returns:
      dictionary with fields:
       splines    - the object defining the splines, returned by
                    construct_joined_splines
       coefs      - vector of coefficients that should be varied, values
                    are from fit to DFTB
       cfixed     - vector of coefficients that should not be varied,
                    values are from fit to DFTB
       rvals,yvals - DFTB values from par_dict, used to fit coefs and cfixed
       X          - preds for zeroth and first derivative, ider, at rvals are
                    np.dot(X[ider],np.concatenate([coefs,cfixed]))
       rms_err    - rms error in the fit
    
    style dictionary keys;
      'rlows' = dict from (Z1,Z2) --> lower end of spline
                assumes Z1 <= Z2 
      'rmax'  = maximum distance for splines (does not depend on Z)
                defaults to 12.0 Angstroms
      boundaries - tuple with boundaries for construct_joined_splines()
                (defaults to ('natural','natural'))
      nknots  = tuple (nvar, nfixed) with number of knots to use for the 
               variable (short distances) and fixed (long distances) splines
      
      if 'cutoff' is in style, then it determines the join point between variable
      and constrained spline using (this is not recommended):
          'cutoff' : dictionary with keys being operator ('H','G','R')
                     and values being cutoff ratios
                     Switch occurs when magnitude of function drops below 
                     cutoff * max(function)
          'rprec'  : precision, in angstroms, for the search for the cutoff
                     point (defaults to 0.01)
      
      if 'cutoff' is not in style (recommended), the switch is obtained from:
          'rjoin' : dict from (Z1,Z2) --> join point between variable and
                    and constrained spline (assumes Z1 <= Z2)
      
      'knots' = required field with (spec_var,spec_fixed) where the specs are
          ('nknots', int) where the int is the number of evenly spaced knots
                          to include in that region of the spline
          ('zeros')       (must be list of len 1) force func and derivative
                          to be zero above rjoin
                         
    
    '''
    if 'rlows' in style:
        rlows = style['rlows']
    else:
        raise ValueError('construct_joined_spline_model: style[rlows] not present')
    if 'rmax' in style:
        rmax = style['rmax']
    else:
        rmax = 12.0 
    if 'boundaries' in style:
        boundaries = style['boundaries']
    else:
        boundaries = ('natural','natural')

    Zs = tuple(np.sort(list(mod.Zs)))
    rlow = rlows[Zs]
    
    # Determine the join point for the splines
    if 'cutoff' in style:
        cutoff_dict = style['cutoff']
        # Model = namedtuple('Model',['oper', 'Zs', 'orb'])
        cutoff = cutoff_dict[mod.oper]
        if 'rprec' in style:
            rprec = style['rprec']
        else:
            rprec = 0.01
        rjoin = rcutoff(mod,par_dict,cutoff,rlow,rmax,rprec = rprec, plot=plot)
    elif 'rjoin' in style:
        rjoin = style['rjoin'][Zs]
    else:
        raise ValueError('construct_joined_spline_model: style[cutoff or rhighs] not present')

    if (rjoin <= rlow) or (rjoin >= rmax):
        raise ValueError('construct_joined_spline_model: rjoin not between rlow and rhigh')

    # Determine the position of the knots
    if 'knots' not in style:
        raise ValueError('construct_joined_spline_model: style[knots] not present')
    if len(style['knots']) != 2:
        raise ValueError('construct_joined_spline_model: len(style[knots]) is not 2')
    xknots = []
    mid_boundary = (None,None)
    force_fixed_zeros = False
    for iseg,knot_spec in enumerate(style['knots']):
        if knot_spec[0] == 'nknots':
            if iseg == 0:
                xknots.append( np.linspace(rlow,rjoin,knot_spec[1]) )
            else:
                xknots.append( np.linspace(rjoin,rmax,knot_spec[1]) )
        elif knot_spec == 'zeros':
            if iseg == 0:
                raise ValueError('construct_joined_spline_model: zeros not allowed in variable spline')
            xknots.append( np.linspace(rjoin,rmax,3) )
            force_fixed_zeros = True
            # value and derivative are zero at the join
            mid_boundary = (0.0,0.0)
        else:
            raise ValueError('construct_joined_spline_model: unrecognized knot spec')
    
    
    xgrid = get_dftb_grid(mod, par_dict)
    ikeep = np.array([i for i,x in enumerate(xgrid) if x>=rlow and x<=rmax])
    xvals = xgrid[ikeep]
    # need to expand grid to rmax, or else fitting to spline raises errors
    # because there is no data between knots
    if xvals[-1] < rmax:
        xspacing = xvals[-1] - xvals[-2]
        xstart = xvals[-1] + xspacing
        nx  = np.max( [int( (rmax - xstart)/ xspacing ), 2] )
        xadd = np.linspace(xvals[-1]+xspacing, rmax,nx)
        xvals = np.concatenate([xvals,xadd])
    yvals = get_dftb_vals(mod, par_dict, xvals)
    if force_fixed_zeros:
        for ix,x in enumerate(xvals):
            if x >= rjoin:
                yvals[ix] = 0.0
    splines = construct_joined_splines(xknots,xvals,yvals,boundaries,xvals, mid_boundary)

    coefs  = splines[0]['spline']['coefs']
    cfixed = splines[1]['spline']['coefs']
    X,const = merge_splines(splines)
    ypreds = np.dot(X[0],np.concatenate([coefs,cfixed])) + const[0]
    rms_err = np.sqrt(np.mean(np.square(yvals - ypreds)))

    X_new,cfixed_new = absorb_constant(X,const,cfixed=cfixed, coefs = coefs)
        
    return {'splines' : splines,
            'X'       : X_new,
            'coefs'   : coefs,
            'cfixed'  : cfixed_new,
            'rvals'   : xvals,
            'yvals'   : yvals,
            'rms_err' : rms_err}

def basis_set_spline_model(mod, par_dict, style):
    if style == 11: # meant for H
        rmax = 4.5
        npows = list(range(2,13))
        nknots = 50
        # This is for plotting, and regularization of derivative
        if (mod.Zs == (1,1)) or (1 not in mod.Zs): 
            xknots = np.arange(1.0,  rmax, (rmax-1.0)/nknots)
        else:
            xknots = np.arange(0.65, rmax, (rmax-0.65)/nknots)
    elif style == 101: # meant for R
        # just in case some policy on sorting of Zs changes
        zs = tuple(np.sort(list(mod.Zs)))
        xdict = {(1,1): (0.65, 1.1), 
                 (1,6): (0.65, 1.85),
                 (1,8): (0.65, 1.83),
                 (6,8): (1.0,  2.22),
                 (6,6): (1.0,  2.28),
                 (8,8): (1.0,  2.22)}
        rlimits = xdict[zs]
        rmax = rlimits[1]
        npows = list(range(2,13))
        nknots = 50
        xknots = np.arange(rlimits[0],rlimits[1]+0.02, 
                           (rlimits[1]-rlimits[0])/nknots)
    else:
        raise ValueError('spline_model style '+str(style)+ ' not recognized')    
    yknots = get_dftb_vals(mod, par_dict, xknots)
    
    bset = polybasis(rmax,npows)
        
    X,cfit = bset.fit(xknots, yknots)
    
    return {'bset'     : bset,
            'xknots'   : xknots,
            'coefs'   : cfit,
            'X'        : X} 

class Spline_model:
    # all units are Angstromg, except when passing values to skinfo.py, in 
    # which case _bohr is appended to the variable name
    def __init__(self, mod, par_dict = ParDict() , style = 0):
        # style = 0, 
        if len(mod.Zs) != 2:
            raise ValueError('Spline_model is only for len(mod.Zs) == 2')
        self.name = 'S'+get_model_str(mod)
        self.mod = mod
        self.par_dict = par_dict
        self.vars = {}
        self.const = {}
        self.style = style
        self.spline = None
        self.rmax = None

    def other_datafields_for_get_feeds(self):
        ''' fields assumed to be in other by get_feeds'''
        return ['mod_raw']
    def initialize_to_dftb(self, npdata):
        ''' 
        '''
        if self.style['type'] == 'joined':
            self.spline = construct_joined_spline_model(self.mod, self.style,
                                                  self.par_dict)
            print('Spline: ' + str(self.mod) + ' err ' + \
               str(self.spline['rms_err'] * 627.0) + ' kcal/mol')
        elif self.style['type'] == 'bset':
            self.spline = basis_set_spline_model(self.mod, self.par_dict, 
                                                 self.style)
        else:
            raise ValueError('modelspline type ' + str(self.style['type']) +
                             ' not recognized')
        # spline coefficients will become variables
        if self.spline['coefs'] is not None:
            self.vars[self.name+'spc'] =  self.spline['coefs'] # +\
              #self.spline['coefs'] * \
              #np.random.normal(scale = 0.10, size = len(self.spline['coefs']))
        if 'cfixed' in self.spline:
            self.const[self.name+'spcf'] = self.spline['cfixed']

    
            
    def get_fixed_data(self):
        ''' For use by store_model_data() in fitting.py
            This returns a dictionary with all the data that does not
            get updated as the model trains. It can be called once before
            optimizing the model.'''
        feed_fields = [self.name+'spR', self.name+'Dftb']
        # save space, since X is easy to regenerate
        spline_save = {k:v for k,v in self.spline.items() if k != 'X'}
        res = {'name'     : self.name,
               'mod'      : self.mod,
               'style'    : self.style,
               'spline'   : spline_save,
               'rmax'     : self.rmax,
               'toeval_once' : list(self.const.keys()),
               'toeval_each' : feed_fields}
        return res
    def get_variable_data(self):
        ''' For use by store_model_data() in fitting.py
            This contains a list of tensorflow objects that need to be
            evaluated to get their current values during training. The
            results are passed back as strings, with tfdata[string]
            being the object to evaluate (via a session call)
            There are two types:
                  toeval_once means that the reults are the same for every batch
                       For a spline this is always the case
                  toeval_each means that the values depend on the batch
                       this is the case for neural nets
        '''
        res = {'toeval_once' : list(self.vars.keys()),
               'toeval_each' : []}
        return res
    @staticmethod
    def get_plot_data(data, rvals = None, der = 0, 
                      dftb_vals = False, cache_in = None):
        ''' get model predictions:
            data = dictionary containing results for get_fixed_data()
                   If dftb_vals is False, you also need:
                       data[name+'spc'] = current values of tfdata[name + 'spc']
                       data[name+'spcf'] = current values of tfdata[name + 'spcf']
                   where name is the model name, which is stored in
                   the dict returned by get_fixed_data(), under the key 'name'
                   A tensorflow session must be used to evalute the tfdata objecs

            rvals = list of bond lengths at which to evaluate the model
                    None assumes these are in data[mod.name + 'spR']
            dftb_vals: If True, returns values from the skf files used to
                      initialize the model.
            cache_in: If this is going to be called many times, save the cache
                       returned by the first call and reuse in further evaluations
        Returns:
            spline values at rvals
            cache
        '''
        name = data['name']
        style = data['style']
        spline = data['spline']
        if cache_in is None:
            cache = {}
        else:
            cache = cache_in
        if 'X' in cache:
            X = cache['X']
        else:
            if rvals is None:
                rvals = data[name+'spR']
            if style['type'] != 'joined':
                raise ValueError('plot data for rvals not implemented'+ 
                        ' for style type ' + str(style['type']))
            if data['spline']['coefs'] is not None:
                X0, const0 = merge_splines_new_xvals(data['spline']['splines'], 
                                                     rvals, ders = [der])
            else:
                newspline = spline_new_xvals(data['spline']['splines'], 
                                             rvals, nder = 1)    
                X0 = newspline['X']
                const0 = newspline['const']
            X,_ = absorb_constant(X0,const0)                    
            cache['X'] = X
        if data['spline']['coefs'] is None:
            coef_var = None
        elif dftb_vals:
            coef_var = data['spline']['coefs']
        else:
            coef_var = data[name+'spc']
        if 'cfixed' in spline:
            if coef_var is not None:
                coefs = np.concatenate([coef_var,
                                    data[name+'spcf']], 0)
            else:
                coefs = data[name+'spcf']
        else:
            coefs = coef_var
        assert(coefs is not None)
        pred = np.dot(X[0], coefs)
        return pred, cache
    def max_r(self):
        # intended use is to so values greater than max_r can be 
        # replaced with DFTB values
        return None
    def get_consts(self):
        return self.const
    def get_variables(self):
        return self.vars            
    def get_feeds(self,gdict,other):
        res = {}
        # only need feed data for the elements that depend on r
        raw = other['mod_raw'][self.mod]
        rs = np.array( [x.rdist for x in raw] )
        res[self.name + 'Dftb'] = np.array( [x.dftb for x in raw] )
        res[self.name + 'spR']  = rs
        if self.style['type'] == 'joined':
            if self.spline['coefs'] is not None:
                X0, const0 = merge_splines_new_xvals(self.spline['splines'], 
                                               rs, ders = [0])
            else:
                newspline = spline_new_xvals(self.spline['splines'], rs, nder = 1)    
                X0 = newspline['X']
                const0 = newspline['const']
            X,_ = absorb_constant(X0,const0)
            res[self.name + 'spX'] = X[0]
        else:
            res[self.name + 'spX'] = self.spline['bset'].get_X(rs)
            
        ystored = np.array( [x.dftb for x in raw] )            
        ydftb = get_dftb_vals(self.mod, self.par_dict, rs)  
        if np.max(np.abs(ystored-ydftb))*627.0 > 1.0:
            print(self.mod)
            print('   max error stored ', np.max(np.abs(ystored-ydftb))*627.0)
            print('   rms error stored ', np.sqrt(np.mean((ystored-ydftb)**2)) * 627.0)

        coefs = np.concatenate([self.spline['coefs'],self.spline['cfixed']])
        ytest = np.dot(X[0],coefs)
        if np.max(np.abs(ytest-ydftb))*627.0 > 1.0:
            print(self.mod)
            print('   max error pred ', np.max(np.abs(ytest-ydftb))*627.0)
            print('   rms error pred ', np.sqrt(np.mean((ytest-ydftb)**2)) * 627.0)
            X0, const0 = merge_splines_new_xvals(self.spline['splines'], 
                                               rs, ders = [0])
            X,_ = absorb_constant(X0,const0)
        
        return res
    def get_feed_fields(self):
        return [self.name+'spR', self.name+'spX', self.name+'Dftb']
    def get_rdists(self,tfdata):
        return tfdata[self.name + 'spR']
    def get_predictions(self, tfdata):
        key = self.name+'Val'
        if key in list(tfdata.keys()):
            return tfdata[key] 
        if 'cfixed' in self.spline:
            if self.spline['coefs'] is not None:
                coefs = tf.concat([tfdata[self.name+'spc'],
                                   tfdata[self.name+'spcf']],0)  
            else:
                coefs = tfdata[self.name+'spcf']
        else:
            coefs = tfdata[self.name+'spc']
        coef_mat = tf.expand_dims(coefs,-1)
        tfdata[key] = tf.squeeze(tf.matmul(tfdata[self.name+'spX'], 
                         coef_mat),axis=1, name = key)
        return tfdata[key]
    def get_dftb(self,tfdata):
        key = self.name+'Dftb'
        if key in list(tfdata.keys()):
            return tfdata[key]
        raise ValueError('modelspline.get_dftb() should not get here')
        return tfdata[key]
    def get_dftb_feed_name(self):
        return self.name+'Dftb'
        
def get_dftb_table(mod, par_dict):
    sym = [ELEMENTS[z].symbol for z in mod.Zs]
    skinfo = par_dict[ sym[0] + '-' + sym[-1] ]
    if mod.oper == 'G' or mod.oper == 'R':
        raise ValueError('get_dftb_table: no table for G or R, must interpolate')
    if len(mod.Zs) == 1:
        # no models for S oper on a single atom
        sk_key = 'E' + mod.orb
        return skinfo.GetAtomProp(sk_key)
    else:
        to_sk = {'ss' : 'ss0', 'sp': 'sp0', 'pp_sigma': 'pp0', 
                 'pp_pi': 'pp1'}
        sk_key = mod.oper + to_sk[mod.orb]
        (xvals,yvals) = skinfo.GetSkData(sk_key)
        return np.array(xvals/ANGSTROM2BOHR), np.array(yvals)

def analyze_rcutoff():
    '''
      Determines cutoff for all matrix elements and plots these
    '''
    par_dict = ParDict()
    style = {'type'   : 'joined',
             'cutoff' : {'H' : 0.1, 'G' : 0.1, 'R' : 0.1},
             'nknots' : (20,20)}
    # Model = namedtuple('Model',['oper', 'Zs', 'orb'])
    fignum = 900
    for zs in [(1, 1), (1, 6), (1, 8), (6, 6),  (8, 8), (6, 8)]:
        fignum += 1
        plt.figure(fignum)
        plt.clf()
        for ioper,oper in enumerate(['H','R','G']):
            if zs == (1,1):
                orbs = ['ss']
            elif 1 in zs:
                orbs = ['ss','sp']
            else:
                if oper == 'H':
                    orbs = ['ss','sp','pp_sigma','pp_pi']
                elif oper == 'G':
                    if zs[0] == zs[1]:
                        orbs = ['ss','sp']
                    else:
                        orbs = ['ss','sp','ps','pp']
                else:
                    orbs = ['ss']
            
            for iorb,orb in enumerate(orbs):
                 mod =  Model(oper, zs, orb)
                 plt.subplot(len(orbs),3, ioper+3*iorb+1)
                 res = construct_joined_spline_model(mod,style,par_dict, plot=True)
                 rcut = res['splines'][1]['spline']['tckb'][0][0][0]
                 print('spline rms error = ' + get_model_str(mod) + ' ' +\
                        unit_print('E',res['rms_err']) + ' rcut= ' + str(rcut))


if __name__ == "__main__":
    from matplotlib.backends.backend_pdf import PdfPages
    style_type = '7'
    Zlist = spline_style('H',style_type)['rjoin'].keys()
    #Zlist = [(6, 6)]
    rzoom = 4.0
    filename = 'Splines.pdf'
    with PdfPages(filename) as pdf:
        for oper in ['R']:
            for Zs in Zlist:
                if oper == 'H':
                    all_orbs = ['ss','sp','pp_sigma','pp_pi','sd','pd_sigma','pd_pi',
                             'dd_sigma','dd_pi','dd_delta']
                elif oper == 'G':
                    all_orbs = ['ss','pp','dd']
                else:
                    all_orbs = ['ss']
                for orbs in all_orbs:
                    if (max(Zs) < 2 and ('p' in orbs or 'd' in orbs)):
                        continue
                    if (max(Zs) < 19 and ('d' in orbs)):
                        continue
                    try:
                        style = {'name': 'S'+style_type}
                        style['model'] = 'spline'
                        style.update(spline_style(oper ,style_type))
                        mod = Model(oper,Zs, orbs)
                        sp = Spline_model(mod,ParDict(), style)
                        sp.initialize_to_dftb(None)
                    except BrokenPipeError:
                        print('skipping ', Zs, orbs)
                        continue

                    # Get the r,y values that were used to fit the joined spline
                    rs_fit_to = sp.spline['rvals']
                    y_fit_to = sp.spline['yvals'] * 627.0
                    # test that the model was fit to the correct dftb values
                    ydftb = get_dftb_vals(sp.mod, sp.par_dict, rs_fit_to) * 627.0
                    #assert(np.max(np.abs(y_fit_to - ydftb)) == 0)
                    # generate the predicted values at these fit points
                    coefs = np.concatenate([sp.spline['coefs'],sp.spline['cfixed']])
                    X0, const0 = merge_splines_new_xvals(sp.spline['splines'], rs_fit_to, ders = [0])
                    X,_ = absorb_constant(X0,const0)
                    ypred = np.dot(X[0],coefs) * 627.0
                    # get the knots and the number of knots that are variable
                    rknots = sp.spline['splines'][0]['spline']['xknots']
                    ivariable = len(rknots)
                    rknots = np.hstack([rknots,sp.spline['splines'][1]['spline']['xknots'] ])
                    # get the spline predictions at the knots
                    X0, const0 = merge_splines_new_xvals(sp.spline['splines'], rknots, ders = [0])
                    X,_ = absorb_constant(X0,const0)
                    yknots = np.dot(X[0],coefs) * 627.0
                    # print error summary to screen
                    print(sp.mod)
                    rms_error = np.sqrt(np.mean((ypred-ydftb)**2))
                    print('   max error at fit points ', np.max(np.abs(ypred-ydftb)))
                    print('   rms error at fit points ', np.sqrt(np.mean((ypred-ydftb)**2)))
                    # plot predicted values and knots in upper panel, with model as title
                    plt.figure(1)
                    plt.clf()
                    for column in range(2):
                        plt.subplot(2,2,1+column)
                        plt.plot(rs_fit_to,ypred,'r-', label='spline')
                        plt.plot(rs_fit_to,ydftb,'b-', label='DFTB')
                        plt.plot(rknots[:ivariable], yknots[:ivariable], 'rx', label = 'knots variable')
                        plt.plot(rknots[ivariable:], yknots[ivariable:], 'bx', label = 'knots fixed')
                        if column == 0:
                            plt.title(str(sp.mod))
                            plt.xlabel('r (Angstroms)')
                            plt.ylabel('kcal/mol')
                            if np.mean(ydftb) < 0:
                                # values go from negative to zero at long range
                                plt.legend(loc='lower right')
                            else:
                                # values go from positive to zero at long range
                                plt.legend(loc='upper right')
                        else:
                            plt.xlim(right = rzoom)
                        # plot error in lower panel
                        plt.subplot(2,2,3+column)
                        plt.plot(rs_fit_to,ydftb-ypred,'b-',label = 'RMS err = ' + str(rms_error))
                        if column == 0:
                            plt.xlabel('r (Angstroms)')
                            plt.ylabel('kcal/mol')
                        else:
                            plt.xlim(right = rzoom)
                            
                        if column == 1:
                            pdf.savefig(figure=1)

if False:
    oper = 'H'
    mod = Model(oper,[6,7], 'ss')
    style = {'name': 'S5'}
    style['model'] = 'spline'
    style.update(spline_style(oper,'5'))
    sp = Spline_model(mod,ParDict(), style)
    sp.initialize_to_dftb(None)

    rs = sp.spline['rvals']
    coefs = np.concatenate([sp.spline['coefs'],sp.spline['cfixed']])
    X0, const0 = merge_splines_new_xvals(sp.spline['splines'], rs, ders = [0])
    X,_ = absorb_constant(X0,const0)
    ytest = np.dot(X[0],coefs)
    ydftb = get_dftb_vals(sp.mod, sp.par_dict, rs)
    print(sp.mod)
    print('   max error at fit points ', np.max(np.abs(ytest-ydftb))*627.0)
    print('   rms error at fit points ', np.sqrt(np.mean((ytest-ydftb)**2)) * 627.0)
   
#def test_plot_data():
#    graph_data_fields = ['models','onames','basis_sizes','glabels','qneutral', 
#                     'qneutral','occ_rho_mask','occ_eorb_mask','mod_raw',
#                     'dQ','gather_for_rot', 'rot_tensors', 'gather_for_oper',
#                     'gather_for_rep', 'segsum_for_rep',
#                     'dftb_elements', 'dftb_r', 'Eelec', 'Erep','Etot',
#                     'unique_gtypes', 'gtype_indices']
#    ngeom = 10
#    gtypes = []
#    geoms1 = random_triatomics(ngeom, [1,6,7],[0.7,1.1],[0.7,1.1],
#                       [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0])
#    gtypes.extend(['HCN'] * ngeom)
#    geoms1.extend(random_triatomics(ngeom, [7,6,1],[0.7,1.1],[0.7,1.1],
#                       [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]))
#    gtypes.extend(['NCH'] * ngeom)
#    geoms1.extend(random_triatomics(ngeom, [1,8,1],[0.7,1.1],[0.7,1.1],
#                       [(104.7+20.0)*math.pi/180.0,(104.-20.0)*math.pi/180.0]))
#    gtypes.extend(['H2O'] * ngeom)
#
#    batch = create_batch(geoms1, gtypes = gtypes)
#    for mod in batch['models']:
#        spmod = Spline_model(mod, par_dict, style)
#        spmod.initialize_to_dftb(batch)
#        fdata = spmod.get_fixed_data()
#        vdata = spmod.get_variable_data()
        
if False: 
    # Not a full test. This was debugging code while building style 1
    from batch import Model
    mod = Model('H',[6,6], 'pp_pi')
    sp = Spline_model(mod)
    par_dict = sp.par_dict
    
    # Full = range from the DFTB paramterization files (e.g. mio-0-1)
    xFull = sp.get_dftb_grid()
    yFull = get_dftb_vals(mod, par_dict, xFull)
    plt.figure(1)
    plt.clf()
    plt.plot(xFull,yFull,'k-')
    
    # use knots that go over about the range expected 
    # for molecules with up to 4 heavy atoms
    xlow = 0.5
    xhigh = 5.2
    nknots = 10
    xknots = np.linspace(xlow, xhigh, nknots)  
    # extend these using style 1
    spline = construct_spline_style1(xknots, mod, par_dict)
    xknots = spline['xknots']
    yknots = np.dot(spline['X'], spline['coeffs'])
    
    plt.plot(xknots,yknots,'ko')
    
    # This is a least squares fit to the data from the DFTB files and the above
    # extensions
    xFit = np.array([xFull[i] for i,x in enumerate(xFull) if x >= xlow and x <= xhigh])
    yFit = np.array([yFull[i] for i,x in enumerate(xFull) if x >= xlow and x <= xhigh])
    xExtend = np.array([xknots[i] for i,x in enumerate(xknots) if x >= xhigh])
    yExtend = np.array([yknots[i] for i,x in enumerate(xknots) if x >= xhigh])
    xFit = np.concatenate([xFit,xExtend], axis = 0)
    yFit = np.concatenate([yFit,yExtend], axis = 0)
    
    (Xtemp,ctemp) = fit_spline_matrix( xknots, xFit, yFit)
    X0,c0 = remove_unused_coefs(Xtemp,ctemp)
    y0 = np.dot(X0,c0)
    plt.plot(xFit, y0, 'r-')
    
    # This is a fit to just the values at the knots, but evaluated at the much
    # finer grid of xFit
    (Xtemp,ctemp) = spline_matrix( xknots, yknots, xFit)
    X1,c1 = remove_unused_coefs(Xtemp,ctemp)
    print('diff in c coefs from refit ',np.max(np.abs(spline['coeffs'] - c1)))
    y1 = np.dot(X1,spline['coeffs'])
    
    plt.plot(xFit, y1,'b-')
    
    # Evaluate the derivative at the knots points
    Xtemp,ctemp = spline_matrix(xknots, yknots, xknots, der = 1)
    Xkder,ckder = remove_unused_coefs(Xtemp,ctemp)
    print('diff in c coefs from refit der ',np.max(np.abs(spline['coeffs'] - ckder)))
    plt.plot(xknots, np.dot(Xkder, ckder), 'go')
    
    # Evaluate the derivative at the xFit points
    Xtemp,ctemp = spline_matrix(xknots, yknots, xFit, der = 1)
    X1der,c1der = remove_unused_coefs(Xtemp,ctemp)
    print('diff in c coefs from refit der ',np.max(np.abs(spline['coeffs'] - c1der)))
    
    y1der = np.dot(X1der,spline['coeffs'])
    plt.plot(xFit,y1der,'g-')

if False:
    from batch import Model
    mod = Model('H',[6,6], 'pp_pi')
    sp = Spline_model(mod)
    par_dict = sp.par_dict
    
    #[  1.25191954,   1.40545615,   1.55899276,   1.71252937,
    #     1.86606598,   2.01960259,   2.1731392 ,   2.32667581,
    #     2.48021242,   2.63374903,  10.08916602,  17.54458301,  25.        ]    
    xknots_in = np.array([ 1.25191954,   1.40545615,   1.55899276,   1.71252937,
         1.86606598,   2.01960259,   2.1731392 ,   2.32667581,
         2.48021242,   2.63374903]) 
    # Full = range from the DFTB paramterization files (e.g. mio-0-1)
    sp.spline = construct_spline_style1(xknots_in, mod, par_dict, noise=0.0)
    pd = sp.get_plot_data(sp.spline['coeffs'])
    plt.plot(pd['x'],pd['dftb'],'r.')
    xx = pd['x']
    dftb = get_dftb_vals(mod, par_dict, xx)
    plt.plot(xx, dftb,'k.')
    xknots = sp.spline['xknots']
    yknots = get_dftb_vals(mod, par_dict, xknots)
    plt.plot(xknots,yknots,'ro')
    
if False:
    # code used to test the basisset splines
    from batch import Model
    mod = Model('R',[6,6], 'pp_pi')
    sp = Spline_model(mod, style = 11)
    par_dict = sp.par_dict
    sp.initialize_to_dftb(None)
    spline = sp.spline
    
    xknots = spline['xknots']
    dftb = get_dftb_vals(mod, par_dict, xknots)
    ypred = np.dot(spline['X'], spline['coeffs'])
    plot_data = sp.get_plot_data(sp.spline['coeffs'])
    
    print('dftb versus ypred ', maxabs(dftb-ypred) * 627.509)                    
    
    plt.plot(xknots, dftb,'ro')
    plt.plot(xknots, ypred,'bo')
    plt.plot(plot_data['x'], plot_data['dftb'],'rx')
    plt.plot(plot_data['x'], plot_data['val'],'bx')

if False:
    # code used to test styles 2 and 3
    from batch import Model
    #mod = Model('H',[6,6], 'pp_pi')
    mod = Model('G',[6,6], 'ss')
    sp = Spline_model(mod, style = 3)
    par_dict = sp.par_dict
    sp.initialize_to_dftb(None, np.zeros([10]))
    spline = sp.spline
    
    xknots = spline['xknots']
    dftb = get_dftb_vals(mod, par_dict, xknots)
    ypred = np.dot(spline['X'], spline['coeffs'])
    plot_data = sp.get_plot_data(sp.spline['coeffs'])
    
    print('dftb versus ypred ', maxabs(dftb-ypred) * 627.509)                    
    
    plt.plot(xknots, dftb,'ro')
    plt.plot(xknots, ypred,'bo')
    plt.plot(plot_data['x'], plot_data['dftb'],'rx')
    plt.plot(plot_data['x'], plot_data['val'],'bx')

if False:
    # code used to test styles 2 and 3
    from batch import Model
    #mod = Model('H',[6,6], 'pp_pi')
    oper = 'H'
    mod = Model(oper,[6,7], 'ss')
    style = {'name': 'S5'}
    style['model'] = 'spline'
    style.update(spline_style(oper,''))
    sp = Spline_model(mod,ParDict(), style)
    par_dict = sp.par_dict
    sp.initialize_to_dftb(None)
    spline = sp.spline
    
    xknots = np.concatenate([x['spline']['xknots']  for x in spline['splines'][:2]])
    dftb = get_dftb_vals(mod, par_dict, xknots)
    data = sp.get_fixed_data()
    data.update(sp.get_variable_data())
    ypred = sp.get_plot_data(data, xknots)
    
    print('dftb versus ypred ', maxabs(dftb-ypred) * 627.509)                    
    
    xvals = np.linspace(xknots[0],xknots[-1],200)
    spl = sp.get_info_for_spline(xvals, der = 0)
    X = spl['X']
    nvar = spline['coeffs'].shape[0]
    for ivar in range(len(ctot)):
        if ivar < nvar:
            plt.plot(xvals, X[:,ivar], 'b-')
        else:
            plt.plot(xvals, X[:,ivar], 'r-')
    
    nsamples = 10
    mult = 0.1
    czero = spline['coeffs']
    add = np.mean(czero) * 0.1
    nc = czero.shape[0]
    vzero = np.dot(X, np.concatenate([czero, spline['cfixed']], 0))
    plt.figure(2)
    plt.plot(xvals, vzero,'k-')
    for i in range(nsamples):
        cnew = spline['coeffs'] * (1.0 + mult * np.random.randn(nc)) + \
          add * np.random.randn(nc) * np.ones([nc])
        vnew = np.dot(X, np.concatenate([cnew, spline['cfixed']], 0))
        plt.plot(xvals, vnew,'r-')

def testing():
    import pickle as pickle
    oper = 'H'
    Zs = (6, 6)
    style_type = '6'
    orbs = 'sp'
    style = {'name': 'S' + style_type}
    style['model'] = 'spline'
    style.update(spline_style(oper, style_type))
    mod = Model(oper, Zs, orbs)
    sp = Spline_model(mod, ParDict(), style)
    sp.initialize_to_dftb(None)

    # data does not depend on current parameters being fit during training
    data = sp.get_fixed_data()

    # grid to save the spline on
    ngrid = 100
    rlow = style['rlows'][Zs]
    rhigh = style['rmax']
    rvals = np.linspace(rlow, rhigh, ngrid)

    # As a test, we can put the coefs for the starting DFTB into data
    # these should be replaced with the current values in tfdata[name+'spc'] and
    # tfdata[name+'spcf']
    mod_name = data['name']
    data[mod_name + 'spc'] = data['spline']['coefs']
    data[mod_name + 'spcf'] = data['spline']['cfixed']
    yvals, _ = Spline_model.get_plot_data(data, rvals)

    # %% Format for the repulsive potential
    # except the last line is a 6th order spline. We can just set this to 0.0
    # because we going to long range
    from scipy.interpolate import CubicSpline

    cs = CubicSpline(rvals, yvals)
    for ix in range(cs.x.shape[0] - 1):
        coefs = cs.c[:, ix]
        print(cs.x[ix], cs.x[ix + 1], coefs[0], coefs[1], coefs[2], coefs[3])