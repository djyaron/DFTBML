# -*- coding: utf-8 -*-
"""
@author: yaron

Created on Sat Dec 23 10:41:45 2017

Functions to create a B-spline basis with constraints. The book that was most
useful to figuring this out was:

Trevor Hastie, Robert Tibshirani, Jerome Friedman
The Elements of Statistical Learning: Data Mining, Inference, and Prediction
Second Edition

A particularly confusing aspect was the distinction between boundary and 
interior knots. Chapter 5 and the Appendix to Chapter 5 were the most useful. 
Results from the scipy CubicSpline, which is similar to the standard spline
routines, can be reproduced with a B-spline basis by having the first and last
knot be boundary knots and the remainder being internal knots. For N total
knots, this produces a N+2 basis functions. This is as expected, since a cubic
spline is specified by N values at the knots and two boundary conditions. 

The approach below applies four boundary conditions to the spline: the values
at the boundaries and either a value for the first derivative or the second
derivative (second derivative of zero corresponds to a natural spline). The 
difference from the usual two-boundary conditions of the cubic spline, is that
here the values at the first and last knot are applied as boundary conditions.

An odd feature of splrep is that the length of the coefficient vector (the c 
of the tck tuple) is not the number of actual basis functions. For a cubic 
spline, if you generate the basis functions, the last four functions come out 
as zero when you evaluate them, e.g. using tck_to_Xmatrix() on a dense grid. 
The function trim_bspline_basis() determines the null basis functions, 
returning a new tuple, tckb = (tck, basis), where basis is a numpy int array 
holding the indices of the nonzero basis functions. It is this object that is 
created by tckb_from_knots(). The scipy docs warn against modifying tck itself,
so this seemed like a reasonable approach.

The interpolate portion of scipy has some significant additions at version 1.0,
but have had issues with numpy on PSC Bridges that made me drop to past version 
of numpy, so decided to stick with only things available in the older version of
scipy.

test_join tests construct_joined_splines(), but since this uses most all
routines, it is a reasonable test of everything.
"""
#%% Imports, definitions
import numpy as np
from scipy.interpolate import  splrep, splev, CubicSpline
try:
    import matplotlib.pyplot as plt
except:
    pass
from MasterConstants import Bcond

#%% Code behind

def maxabs(x):
    return np.max(np.abs(x))
    
def tckb_to_Xmatrix(tckb, xvals, der=0):
    '''
    Given tckb for a spline, generates the B spline basis so that
        y = X c
    where y[i] is the value of the spline at xvals[i]. This is done in a 
    straightforward, but potentially inefficient manner, by determining
    the ith row of X by setting c[j] = delta(i,j) and using scipy's splev
    to evalue y. 
    
    Input: 
       tckb   - scipy's tck representation of a spline, as from splrep,
              plus a list of nonzero basis functions (see trim_bsplin_basis())
       xvals - points at which to evaluate the spline
       der   - order of the derivative of the spline
    Output:
       X     - such that  y[i] = sum_j X[i,j] c[j]
    '''
    (tck,basis) = tckb
    coeff = tck[1]
    ncoeff = coeff.shape[0]
    nbasis = basis.shape[0]
    npoints = np.array(xvals).shape[0]
    X = np.zeros([npoints, nbasis])
    for ibasis in range(nbasis):
        icoef = basis[ibasis] 
        ctrial = np.zeros([ncoeff])
        ctrial[icoef] = 1.0
        tck_trial = (tck[0], ctrial, tck[2])
        X[:,ibasis] = splev(xvals,tck_trial,der = der)
        
    return X

def trim_bspline_basis(tck, xeval):
    '''
    An odd feature of splrep is that the length of the coefficient vector 
    (the c of the tck tuple) is not the number of actual basis functions. 
    For a cubic spline, if you generate the basis functions, the last four 
    functions come out as zero when you evaluate them on a dense grid. 
    This function determines the null basis functions, returning a new tuple, 
    tckb = that is (tck, basis), where basis is a numpy int array holding 
    the indices of the nonzero basis functions. 
    '''
    ncoeff = len(tck[1])
    # make a temperary tckb that includes all basis functions
    tckb_all = (tck, np.arange(ncoeff))
    # evalute on a grid that is sufficiently dense that a zero basis vector
    # implies the basis is genuinely zero
    #xknots = tck[0]
    #nknots = len(xknots)
    #xdense = np.linspace(xknots[0],xknots[-1], nknots * 3)
    X = tckb_to_Xmatrix(tckb_all, xeval)
    basis = np.array([i for i in range(X.shape[1]) if maxabs(X[:,i]) > 0.0])
    return (tck,basis)
    
def tckb_from_knots(xknots,xyeval = None, k=3):
    '''
       Get tck for a B-spline basis corresponding to a typical cubic spline,
       i.e. as would be generated from scipy's CubicSpline. To do this,
       scipy's splrep is used, treating the first and last knot as boundary
       knots and the remaining as internal knots. The resuling basis has 
       #knots + 2 nonzero basis functions, where the two extra correspond to
       the boundary conditions. 
       
       For debugging purposes, if k=3, we assert that number of non-zero
       basis functions = len(xknots) + 2. No assertion is done if k is not 3.
       
       Input:
         xknots- knots for which to define the basis
         xyeval- tuple (xeval,yeval) for which to fit a spline using
                 the provided xknots. This is done by splrep as a least squares 
                 fit
       Output:
         tckb- Trimmed (using trim_spline_basis) representation of the 
               resulting spline. 
              If xyeval is None, the coefficients, c, of tck are meaningless. 
              Otherwise, c is fit to xyeval. The fit apparently corresponds to
              a cubic spline with not-a-knot boundary conditions 
              (see comment at the bottom of: https://docs.scipy.org/doc/
        scipy-0.18.1/reference/generated/scipy.interpolate.CubicSpline.html)
    '''
    if xyeval is None:
        # just need a set of x that will satisfy Schoenberg-Whitney condition
        # create new points between each of the knots
        x_to_add = xknots[:-1] + 0.5 * np.diff(xknots)
        xall = np.concatenate([xknots, x_to_add])
        xeval = np.sort(xall)
        yeval = np.zeros(xeval.shape)
    else:
        xeval,yeval = xyeval
    tck = splrep(xeval,yeval, xb = xknots[0], xe = xknots[-1], s=0, k=k, 
                 t = xknots[1:-1])
    tckb = trim_bspline_basis(tck, xeval)
    if k == 3:
        assert(len(tckb[1]) == len(xknots) + 2)
    return tckb

def apply_constraint(X, const, bc= None, wz = None):
    '''
    Applies a boundary condition to remove a degree of freedom from the linear
    form of the bspline. Constraints typically introduce a constant term to the
    linear predictor, which is included here as const.
    
    The first application of the boundary conditions is done by passing in bc
    and having the first dimension of X be the knots at which the B spline basis
    was created.
    
    To apply the constraint to a grid that differs from the original knots:
      1) Use apply_constraint with X = X for original knots, and desired Bcond
         Save the wz's that are returned in sequence.
      2) Create the X corresponding to the same B spline basis, but with the
         desired evaluation points (as opposed to knots)
         Call apply_constraint with bc = None for each of the wz's 
         obtained in step 1.
    
    Input:
       X, const- original prediction for the ider'th derivative of the spline 
                has form:
                 Y[ider][a] = sum_i X[ider][a,i] C[i] + const[ider][a]
                 If const is None, it assumed to be a zero vector.
     Either bc or wz should be specified, with bc taking precedence if supplied
       bc- boundary condition as a Bcond named tuple with:
                 ix  - index of the knot at which the constraint applies
                 der - order of the derivative for the constraint
                 val - value of the derivative
           E.g. Bcond(0,1,1.2) constrains the first derivative at the first
                               knot to have a value of 1.2                 
       wz- Data structure used to save and re-apply constraints. See above
           two step approach.
           
     Output:
       Xnew, const_new, inew: such that original model is rewritten
           Cnew = C[inew]  {inew is list of retained indices}
           Y[ider][a] = sum_i Xnew[ider][a,i] Cnew[i] + const_new[ider][a]
       (W, Zdiff) : wz data structure to be used to reapply constraint as in
           the above two step approach
           
    Implementation details:
       The original model has the form:
           Y[a] = sum_i X[a,i] C[i] + const[a]
       The constraint is of the form:
           bc.val = sum_i X[ider][bc.ix,i] C[i] + const[ider][bc.ix]
       which we rewrite as
           bc.val = sum_i W[i] C[i] + Zc
               W = X[ider][bc.ix,:]
               Zc = const[ider][bc.ix]
       and introduce Zdiff = bc.val - Zc so that
           Zdiff = sum_i W[i] C[i]
       The internal data structure wz is the tuple (W,Zdiff)
       
       We first find the basis function, imax, that has the largest weight
       in the constraint:
             imax = argmax abs(W);   wmax = W[imax]
       And solve for the corresponding coefficient, so we can eliminate that
       basis function:
           Zdiff = wmax C[imax] + sum_(i ne imax) W[i] C[i]
           Cmax = Zdiff/wmax - sum_(i ne imax) (W[i]/wmax) C[i]
       We then substitute this Cmax into the prediction (dropping the ider
       index for convenience)
           Y[a] = sum_i X[a,i] C[i] + const[a]
           Y[a] = X[a,imax] Cmax + sum_(i ne imax) X[a,i] C[i] + const[a]
           Y[a] = X[a,imax] Zdiff/wmax 
                 - X[a,imax] sum_(i ne imax) (W[i]/wmax) C[i]
                 + sum_(i ne imax) X[a,i] C[i]
                 - const[a]
       combining these into a sum_(i ne imax) and a const term:
           Y[a] = sum_(i ne imax) {X[a,i] - X[a,imax] (W[i]/wmax)} C[i]
                  + const[a] + X[a,imax] Zdiff/wmax
           Y[a] = sum_(i ne imax) Xnew[a,i] C[i] + const_new[a]
       with:
           Xnew[a,i] = X[a,i] - X[a,imax] (W[i]/wmax)
           Xnew      = X      - np.outer(X[a,imax], W[i]/wmax) {and imax removed}
           const_new[a] = const[a] + X[a,imax] Zdiff/wmax
    '''
    if bc is not None:
        W = X[bc.der][bc.ix,:]
        if const is None:
            Zc = 0.0
        else:
            Zc = const[bc.der][bc.ix]
        Zdiff = bc.val - Zc
    else:
        (W,Zdiff) = wz
    imax = np.argmax(np.abs(W))
    wmax = W[imax]
    Xnew = []
    const_new = []
    inew = list(range(len(W)))
    del inew[imax]
    for ider in range(len(X)):
        cnew = X[ider][:,imax] * Zdiff/wmax
        if const is None:
            const_new.append(cnew)
        else:
            const_new.append(const[ider] + cnew)
        Xnew1 = np.outer(X[ider][:,imax], np.divide(W,wmax))
        Xnew1 = X[ider] - Xnew1 
        Xnew.append(Xnew1[:,inew])
    
    return Xnew, const_new, inew, (W, Zdiff)

def spline_linear_model(xknots, xeval, xyfit, bconds, max_der=2, deg=3):
    '''
        ypred = np.dot(X,coefs) + const
    Input:
        xknots    - x position of knots, including boundary and internal knots
        xeval     - ypred is evaluated at these values for x
                    if xeval is none, then xeval is the internal knots,
                    i.e. xeval = xknots[1:-1]
        xyfit     - tuple of (x,y) values to be used to fit the basis
                    if None, no fit is performed and returned coef is None
        bconds   - Bcond namedtuples defining the boundary conditions
        max_der  - evaluate X,const for derivatives from 0 to max_der
        deg      - degree of spline model
    Output:
        returns dict, a spline_dict, with:
          X,const - predictions from model is:
             ypred = np.dot(X[ider],coefs) + const[ider]
             where ider is the order of the derivative to predict
          xvals   - spline predicts values at these points (xeval)
          coefs   - coefs from fit to xyfit, or None if xyfit = None
          tckb    - tck and basis, defining the spline basis
          wz      - data needed to reapply constraints
    '''
    # bconds may require derivatives higher than those requested
    maxd = np.max([max_der] + [bc.der for bc in bconds])
    # get spline basis for xknots and use this to apply constraints
    tckb = tckb_from_knots(xknots, None, k=deg)
    Xk = [tckb_to_Xmatrix(tckb,xknots,ider) for ider in range(maxd+1)] 
    constk = None
    wzs = []
    for bc in bconds:
        Xk,constk,_,wz1 = apply_constraint(Xk,constk,bc = bc)
        wzs.append(wz1)
    
    # Generate the basis for xeval and apply the constraints
    if xeval is not None:
        X = [tckb_to_Xmatrix(tckb,xeval,ider) for ider in range(max_der+1)]
        const = None
        for wz in wzs:
            X,const,_,_ = apply_constraint(X,const,wz = wz)
    else:
        X = Xk[0:(max_der+1)]
        const = constk[0:(max_der+1)]
    
    coefs = None
    if xyfit is not None:
        (xfit,yvals) = xyfit
        X_fit = [tckb_to_Xmatrix(tckb,xfit)]
        const_fit = None
        for wz in wzs:
            X_fit,const_fit,_,_ = apply_constraint(X_fit,const_fit,wz = wz)        
        # remove constants from the yvals, leaving the part to be fit
        yfit = yvals - const_fit[0]
        if X_fit[0].shape[0] == X_fit[0].shape[1]:
          # rcond=-1 has been added to ensure future compatibility 
            coefs = np.linalg.solve(X_fit[0],yfit
                                    # , rcond=-1
                                    )
        else:
            coefs,_,_,_ = np.linalg.lstsq(X_fit[0],yfit, rcond = None)
        
    if const is None:
        const = np.zeros(len(xeval))
    
    return {'xknots': xknots,
            'X'     : X,
            'const' : const,
            'xvals' : xeval,
            'coefs' : coefs,
            'tckb'  : tckb,
            'wzs'   : wzs}

def spline_new_xvals(spline_dict, xvals, nder = None):
    '''
        Construct spline_dict that evaluates a spline at a new set of xvals
    
    Input: 
       spline_dict - output of spline_linear_model
       xvals       - new xvalues are which to evaluate spline
    Output:
       spline_dict with new X, const and xvals. Remaining values of the dict
          point to the values in the input dict.
    '''
    if nder is None:
        nder = len(spline_dict['X'])
    X = [tckb_to_Xmatrix(spline_dict['tckb'],xvals,ider) for ider in range(nder)]
    const = None
    for wz in spline_dict['wzs']:
        X,const,_,_ = apply_constraint(X,const,wz = wz)
    return {'X'     : X,
            'const' : const,
            'xvals' : xvals,
            'coefs' : spline_dict['coefs'],
            'tckb'  : spline_dict['tckb'],
            'wzs'   : spline_dict['wzs']}

def spline_vals(spline_dict, ider = 0, coefs = None):
    '''
     Evaluate spline at the points stored in the spline_dict returned
     by spline_linear_model
     
     Input:
       spline_dict - output of spline_linear_model
       ider        - order of derivative to evaluate
       coefs       - overrides coefs in spline_data
     Return:
       numpy array with values of the spline
    '''
    if coefs is None:
        coefs = spline_dict['coefs']
    X = spline_dict['X'][ider]
    const = spline_dict['const'][ider]
        
    return np.dot(X,coefs) + const


def construct_joined_splines(xknots, xvals, yvals, boundaries, xevals,
                             mid_boundary = (None,None)):
    '''
    Construct two cubic splines, that meet in middle. The boundary conditions
    at the ends of the conjoined splines are specified by the boundaries input.
    At the mid point, the splines have fixed derivatives and first derivatives.
    The values for these derivatives is determined from fitting xvals and
    yvals to a single cubic splines. These values may be over-ridden through
    the mid_boundary input.
    
      xknots = array [xknots0, xknots1] holding xknots for the two segements
               xknots[0][-1] must equal xknots[1][0] or exception is raised.
      xvals,yvals = data used to determine boundary conditions and
          initialize the spline coefficients. The assumption is that xknots1
          will not be varied, and so xknots0 meets xknots1 by agreeing on
          value and derivative
      boundaries = array [boundaries0,boundaries1] specifying external 
          boundary conditions (applied at first knot of xknots0, and
          last knot of xknots1). Each entry can be either:
            ('fixed',val,deriv) : 
                       value and derivative as specified. This is two boundary 
                       conditions such that changing the basis
                       expansion coeffs does not alter either the value or the 
                       derivative at the boundary.
            'natural': second derivative is zero at the boundary. This is one
                       boundary condition, such that changes to the basis
                       expansion coeffs alters the value at the boundary,
                       keeping only the second derivative at zero.
      xeval = points at which to evaluate the results
      mid_boundary = (val,deriv) with the value and derivative at the mid-point
          if these are None, the value and/or derivative are obtained from the
          spline fit discussed above.
      
      Returns:
          array of length 3 holding results for:
            0: segment 0, 1: segment 1, 2: full cubic spline
          the results are dict with:
           'spline' : output of spline_linear_model 
           'ivals'  : indices of xvals,yvals used to fit this spline
           'ievals' : indices of xevals that lie within range of this spline
    '''
    
    if (xknots[0][-1] != xknots[1][0]):
        raise ValueError('construct joined slines, input knots dont join ' +
            str(xknots[0][-1]) + ' != ' + str(xknots[1][0]))
    xmid = xknots[0][-1]
    # first we fit a full cubic spline to xvals,yvals in order to get the
    # values and derivatives at the mid points and, if needed, at the end
    # points
    xk = np.concatenate([ xknots[0], xknots[1][1:] ])
    
    # bc_exterior = [[Bcond's at xk[0] ], [Bcond's at xk[1]] ]
    #  saving these this way, so they can be reused when joining splines
    bc_exterior = []    
    boundary_index = [0, -1]
    for ib,boundary in enumerate(boundaries):
        if boundary == 'natural':
            bc_exterior.append( [Bcond(boundary_index[ib], 2, 0.0)] )
        elif len(boundary) == 3 and boundary[0] == 'fixed': 
            (val,deriv) = boundary[1:]
            bc_exterior.append( [Bcond(boundary_index[ib], 0, val),
                                 Bcond(boundary_index[ib], 1, deriv) ] )
        else:
            raise ValueError('boundary condition not recognized ' + str(boundary))
    # will get values at the endpoints and the midpoint
    xeval = np.array([xk[0],xmid,xk[-1]])
    bcs = sum(bc_exterior, [])
    full3 = spline_linear_model(xk, xeval, (xvals,yvals), bcs)
    vals = [spline_vals(full3,ider) for ider in [0,1]]
    if mid_boundary[0] is None:
        ymid = vals[0][1]
    else:
        ymid = mid_boundary[0]
    if mid_boundary[1] is None:
        ymid_der = vals[1][1]
    else:
        ymid_der = mid_boundary[1]
    
    results = []
    mid_index = [-1,0]
    for iseg in [0,1]:
        xk = xknots[iseg]
        bcs = bc_exterior[iseg]
        bcs.extend([Bcond(mid_index[iseg], 0, ymid),
                    Bcond(mid_index[iseg], 1, ymid_der)])
        if iseg == 0:
            ivals = np.array([i for i,x in enumerate(xvals) if x>=xk[0] and x<=xk[-1]])
            ievals = np.array([i for i,x in enumerate(xevals) if x>=xk[0] and x<=xk[-1]])
        else:
            ivals = np.array([i for i,x in enumerate(xvals) if x>xk[0] and x<=xk[-1]])
            ievals = np.array([i for i,x in enumerate(xevals) if x>xk[0] and x<=xk[-1]])
        seg = spline_linear_model(xk, xevals[ievals], (xvals[ivals],yvals[ivals]), bcs)
        results.append({'spline': seg, 'ivals': ivals, 'ievals': ievals})

    # insure no xevals were lost
    assert(sum([len(results[iseg]['spline']['xvals']) for iseg in range(2)],0)
           == len(xevals))
    full = spline_new_xvals(full3, xevals)
    results.append({'spline' : full, 'ivals'  : np.arange(len(xvals)), 
                    'ievals' : np.arange(len(xevals))})

    return results

def merge_splines(rdicts, ders = [0,1]):
    '''
    Get X,const needed to evaluate a joined spline, at the xvals
     passed to construct_joined splines()
    
    Input:
        rdicts - output from construct_joined_splines()
        ders   - list of derivatives to evaluate
    Output:
        X, const such that
           vals_ider = np.dot(X[ider], np.concatenate([coefs,cfixed])
                        + const[ider]
           vals_ider[i] = value at xvals[i] where xvals is the xvals passed to
                          to construct_joined_splines()
    '''
    neval = sum([len(rd['ievals']) for rd in rdicts[0:2]], 0)
    ncoef = sum([len(rd['spline']['coefs']) for rd in rdicts[0:2]], 0)
    X = [np.zeros([neval,ncoef]) for ider in ders]
    const = [np.zeros(neval) for ider in ders]
    for ider in ders:
        cstart = 0
        for iseg, rd in enumerate(rdicts[0:2]):
            spl = rd['spline']
            X[ider][rd['ievals'],cstart:(cstart + len(spl['coefs']))] = spl['X'][ider]
            const[ider][rd['ievals']] = spl['const'][ider]
            cstart += len(spl['coefs'])
    return X,const

def merge_splines_new_xvals(rdicts, xvals, ders = [0,1]):
    '''
       Same as merge_splines, except the spline is evaluated at xvals
       instead of the xvals passed to construct_joined_splines
    '''
    neval = len(xvals)
    ncoef = sum([len(rd['spline']['coefs']) for rd in rdicts[0:2]], 0)
    X = [np.zeros([neval,ncoef]) for ider in ders]
    const = [np.zeros(neval) for ider in ders]
    for ider in ders:
        cstart = 0
        for iseg, rd in enumerate(rdicts[0:2]):
            coefs = rd['spline']['coefs']
            xk    = rd['spline']['xknots']
            if iseg == 0:
                ivals = np.array([i for i,x in enumerate(xvals) 
                                                  if x>=xk[0] and x<=xk[-1]])
            else:
                ivals = np.array([i for i,x in enumerate(xvals) 
                                                  if x>xk[0] and x<=xk[-1]])
            if len(ivals) > 0:
                xnew = xvals[ivals]
                splx = spline_new_xvals(rd['spline'], xnew)
                X[ider][ivals,cstart:(cstart + len(coefs))] = splx['X'][ider]
                const[ider][ivals] = splx['const'][ider]
            cstart += len(coefs)
    return X,const

def absorb_constant(X,const,cfixed=None, coefs=None):
    '''
       convert from y = X*c + consts to  y = X*c

       Input:
         evaluation of spline corresponds to:
            coefs_all = np.concatenate([coefs,cfixed])
            vals[ider] = np.dot(X[ider],coefs_all) + const[ider]
        if coefs is not None, at test is performed by comparing the new
           values to the old and asserting that maxabs(diff) < 1.0e-13
           
       Output:
         X_new, cfixed_new
         evaluation of spline corresponds to:
            coefs_all = np.concatenate([coefs,cfixed_new])
            vals[ider] = np.dot(X_new[ider],coefs)
        Note that len(cfixed_new) = len(cfixed+1)
    '''
    ders = list(range(len(X)))
    const_exp = [np.expand_dims(const[ider],1) for ider in ders]
    X_new = [np.hstack( [X[ider],const_exp[ider]] ) for ider in ders]
    
    if cfixed is not None:
        cfixed_new = np.concatenate( [cfixed,np.ones([1])] )
    else:
        cfixed_new = np.ones([1])
    if coefs is not None:
        yorig = [np.dot(X[ider],np.concatenate([coefs,cfixed])) + const[ider]
                                                      for ider in ders]
        ynew = [np.dot(X_new[ider], np.concatenate([coefs,cfixed_new])) 
                                             for ider in range(len(const))]
        ydiff = np.concatenate(yorig) -np.concatenate(ynew)
        assert(maxabs(ydiff) < 1.0e-13)

    return X_new,cfixed_new

def plot_spline_basis(spline_dict, ider):
    '''
      Input:
        spline_dict - output of spline_linear_model
      Output:
        adds a plot of the B-spline basis to currently active figure
    '''
    X = spline_dict['X'][ider]
    const = spline_dict['const'][ider]
    xgrid = spline_dict['xvals']
    for i in range(X.shape[1]):
        xp = X[:,i]
        if const is not None:
            xp += const
        plt.plot(xgrid, xp, 'k-')
        plt.show() 
    
def plot_spline(spline_dict, ider, sym='k-', sortx = False):
    
    xvals = spline_dict['xvals']
    yvals = spline_vals(spline_dict, ider)
    if sortx:
        isort = np.argsort(xvals)
        xvals = xvals[isort]
        yvals = yvals[isort]
    plt.plot(xvals,yvals,sym)
