# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 16:51:34 2021

@author: fhu14

Backend for the models derived from the models defined in spline_backend.py.
These are the models that will be exposed at the top level of the package
for use by the rest of the code.
"""
#%% Imports, definitions
import numpy as np
Array = np.ndarray
from typing import List, Dict
import matplotlib.pyplot as plt
from .spline_backend import spline_linear_model, spline_new_xvals,\
    merge_splines, merge_splines_new_xvals, construct_joined_splines
from functools import reduce
import scipy.sparse
import scipy.special
from MasterConstants import Bcond
from Elements import ELEMENTS

#%% Code behind

class PairwiseLinearModel: #abstract
    def r_range(self):
        """
        Range beyond which the potential can be assumed to be zero
    
        Returns
        -------
        (rlow, rhigh) : doubles
            lower and upper range in Angstroms
    
        """
        raise NotImplementedError

    def nvar(self):
        """
        Number of variables in the model. The assumption is that
        these variables have a domain of -infinity..infinity, and
        that this covers all relevant functions (i.e. maps to the
        full hypothesis space).

        Returns
        -------
        nvar: int
         Number of variables

        """
        raise NotImplementedError

    def linear_model(self, r_eval, ider=0):
        """
        predictions at the points r can be obtained from:
            potential = np.dot(A, c) + b

        Parameters
        ----------
        r_eval : 1-D np.array of doubles
            distances at which to evaluate the potential
        ider : int, non-negative
            ider-th derivative

        Returns
        -------
        (A, b) A: 2-D np.array, shape = [len(r), self.nvar()]
               b: 1-D np.array, shape = [len(r)]
        """
        raise NotImplementedError

    def vals(self, coefs, r_eval, ider=0):
        A, b = self.linear_model(r_eval, ider)
        res = np.dot(A, coefs) + b
        return res

    def plot(self, coefs, sym='b-', ider=0, rvals=None):
        if rvals is None:
            rlow, rhigh = self.r_range()
            rvals = np.linspace(rlow, rhigh, 100)
        y = self.vals(coefs, rvals)
        plt.plot(rvals, y, sym)
        
def fit_linear_model(model, xvals, yvals, ider=0):
    r"""Initializes the spline coefficients for a given model based off 
        given xvals and yvals
    
    Arguments:
        model (PairwiseLinearModel): The model whose coefficients need to be
            initialized
        xvals (Array): The x values used to initialize the coefficients
        yvals (Array): The y values used to initialize the coefficients
    
    Returns:
        coefs (Array): The array of coefficients
        A (Array): The spline basis matrix
        b (Array): The array of constants
    
    Notes: Initializes the model by solving the least squares problem of the form
        yvals = A @ x + b, where the vector x is coefficient vector that is 
        solved for.
    """
    A, b = model.linear_model(xvals, ider)
    # A c + b = y
    coefs, _, _, _ = np.linalg.lstsq(A, yvals - b, rcond=None)
    return coefs, A, b

def map_linear_models(source, target, xgrid=None, ngrid=None):
    """
    For models of the form:
        source:  np.dot(A1,c1) + b1
        target:  np.dot(A2,c2) + b2
    This routine calculates the relation:
        c2 = np.dot(X,c1) + y
    resulting from the process of:
        use source to generate values, ygrid, on xgrid
        fit the target model to these ygrid values

    Parameters
    ----------
    source : PairwiseLinearModel  (typically a sparse model)
    target : PairwiseLinearModel  (typically a dense model)
    xgrid : 1-D numpy array  (ignored if ngrid is provided)
            note that xgrid must cover the full range of the models and be
            sufficiently dense that there is sufficient information to map
            between the source and target models
    ngrid  : int
            creates xgrid wkth ngrid points, spanning union of the range of
            the two models

    Returns
    -------
    X : 2-D numpy array
    y : 1-D numpy array

    """
    # convenient to loop over the source and target to figure out the ranges etc
    models = [source, target]
    # rlows and rhighs are lists with  the ranges of the two models
    rlows = np.zeros(2)
    rhighs = np.zeros(2)
    for imod, mod in enumerate(models):
        rlows[imod], rhighs[imod] = mod.r_range()
    # rlow and rhigh is the intersection of the ranges of the models
    rlow = np.min(rlows)
    rhigh = np.max(rhighs)

    # if xgrid is not specified, we build a grid on the intersection of the
    # models
    if ngrid is not None:
        xgrid = np.linspace(rlow, rhigh, ngrid)

    # As and bs are A,b of the linear models
    # because the range of either model may go outside the intersection,
    # we create A and b so that:
    #    A coef + b = 0
    # for all values outside of the intersection. This is done by setting
    #    A(igrid,icoef) = 0   if igrid is out of the intersection
    #    b(igrid) = 0         if igrid is out of the intersection
    As = []
    bs = []
    tol = 1.0e-10
    for imod, mod in enumerate(models):
        ilow = np.where(xgrid < (rlows[imod] - tol))[0]
        ihigh = np.where(xgrid > (rhighs[imod] + tol))[0]
        i_outside = np.concatenate([ilow, ihigh], 0)
        if len(i_outside) == 0:
            Afull, bfull = mod.linear_model(xgrid)
        else:
            ngrid = len(xgrid)
            nvar = mod.nvar()
            Afull = np.zeros([ngrid, nvar])
            bfull = np.zeros([ngrid])
            i_inside = np.setdiff1d(np.arange(len(xgrid)), i_outside,
                                    assume_unique=True)
            Afull[np.ix_(i_inside, np.arange(nvar))], bfull[i_inside] = \
                mod.linear_model(xgrid[i_inside])
        As.append(Afull)
        bs.append(bfull)

    A1, b1 = As[0], bs[0]
    A2, b2 = As[1], bs[1]
    # A2 c2 + b2 = A1 c1 + b1 
    # A2 c2 = A1 c1 + (b1-b2)
    # A c = b --> c = (A.T A)^-1 A.T b
    # c2 = (A2.T A2)^-1 A2.T (A1 c1 + b1 - b2)
    # c2 = X c1 + y
    #   X = (A2.T A2)^-1 A2.T    A1
    #   y = (A2.T A2)^-1 A2.T   (b1 - b2)
    try:
        t1 = np.dot(np.linalg.inv(np.dot(A2.T, A2)), A2.T)
    except np.linalg.LinAlgError:
        raise Exception('singular matrix in map_linear_models')
    X = np.dot(t1, A1)
    y = np.dot(t1, b1 - b2)

    return X, y

class SplineModel(PairwiseLinearModel):
    def __init__(self, config, xeval=None):
        if isinstance(xeval, (int, float)):
            xeval = np.array([xeval])
        self.xknots = config['xknots']
        if 'deg' in config:
            self.deg = config['deg']
        else:
            self.deg = 3
        if 'bconds' in config:
            bconds = config['bconds']
            if isinstance(bconds, list):
                # list of Bcond or an empty list for no boundary conditions
                if all(isinstance(x, Bcond) for x in bconds):
                    self.bconds = bconds
                else:
                    raise ValueError('Spline bconds is list with unknown types')
            elif bconds == 'natural':
                # natural at start and end
                self.bconds = [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)]
            elif bconds == 'vanishing':
                # natural at start point, zero value and derivative at end point
                self.bconds = [Bcond(0, 2, 0.0),
                               Bcond(-1, 0, 0.0), Bcond(-1, 1, 0.0)]
            elif bconds == 'none':
                self.bconds = []
            elif bconds == 'last_only':
                #Zero value and derivative at the end point only
                self.beconds = [Bcond(-1, 0, 0.0), Bcond(-1, 1, 0.0)]
        else:
            # Natural boundary conditions
            self.bconds = [Bcond(0, 2, 0.0), Bcond(-1, 2, 0.0)]
        if 'max_der' in config:
            self.max_der = config['max_der']
        else:
            self.max_der = 0
        if xeval is not None:
            self.spline_dict = spline_linear_model(self.xknots, xeval, None,
                                                   self.bconds, self.max_der,
                                                   self.deg)
        else:
            self.spline_dict = None

    def r_range(self):
        return self.xknots[0], self.xknots[-1]

    def nvar(self):
        if self.spline_dict is None:
            # force initialization of spline_dict
            _ = self.linear_model(self.xknots)
        return self.spline_dict['X'][0].shape[1]

    def linear_model(self, xeval, ider=0):
        # TODO: This may not be optimal, especially when derivatives above 0 
        #       are being requested
        if isinstance(xeval, (int, float)):
            xeval = np.array([xeval])
        if self.spline_dict is None:
            # dictionary was never initialized
            self.spline_dict = spline_linear_model(self.xknots, xeval, None,
                                                   self.bconds, ider,
                                                   self.deg)
        elif (len(self.spline_dict['X']) > ider) and \
                (len(self.spline_dict['xvals']) == len(xeval)) and \
                all(x == y for x, y in zip(self.spline_dict['xvals'], xeval)):
            # dictionary was evaluated at the same points as requested
            # and up to the required derivative
            pass
        else:
            # need to evaluate at a new set of points
            # nder = ider + 1 because of use of range inside
            self.spline_dict = spline_new_xvals(self.spline_dict, xeval, nder=ider + 1)

        return self.spline_dict['X'][ider], self.spline_dict['const'][ider]

class JoinedSplineModel(PairwiseLinearModel):
    
    def __init__(self, config: Dict, xeval: Array = None) -> None:
        r"""Initializes the joined spline model with the given configurations
        
        Arguments:
            config (Dict): A dictionary that contains the xknots, the degree, and the boundary 
                conditions of the joined spline
            xeval (Array): x positions to evaluate the spline at. Defaults to None,
                ignore on initialization
        
        Returns: 
            None
        
        Notes: Config has to contain the following information:
            
            xknots: array with all the knots. They will be separated in the function internally
                such that xknots[0][-1] == xknots[1][0] (they meet in the middle)
            equal_knots: bool, indicates whether to balance the knots such that both 
                segments of the joined spline will have the same number of knots. Default is false
            cutoff: float, the cutoff distance in angstroms; anything after that will be
                treated as segment 1 in the joined spline (fixed, not variable). If
                no cutoff is specified, default cutoff in the code is 4 angstroms.
            bconds: 'natural'. In the future, experiment with different boundary conditions
                but right now, use the 'natural' boundary condition for both segments.
                TODO: in the future, think about incorporating different boundary conditions.
                Code right now defaults to an array of ['natural', 'natural']
        """
        default_cutoff = 4.0
        if 'xknots' not in config:
            raise ValueError("Configuration dictionary must have the knots!")
        self.xknots = config['xknots']
        cutoff = default_cutoff if 'cutoff' not in config else config['cutoff']
        equal_knots = False if 'equal_knots' not in config else config['equal_knots']
        self.xknots = self.segment_knots(self.xknots, cutoff, equal_knots)
        self.bconds = ['natural', 'natural'] #default
        #Now something to do with the spline_dicts (multiple, because we have segments)
        self.spline_dicts = None
        
    def segment_knots(self, xknots: Array, cutoff: float, equality: bool) -> List[List[float]]:
        r"""Segments the given knots of the spline into the two pieces
        
        Arguments:
            xknots (Array): Array of the knots
            cutoff (float): The cutoff distance for the two segments
            equality (bool): Boolean specifying whether the two halves should have
                the same number of knots
        
        Returns:
            segments (List[List[float]]): The knots for the two segments in a list of lists of floats.
                This return type is necessary for later functions in the joined spline workflow
        
        Notes: If equality is toggled, then the cutoff distance may not be achieved.
        """
        if equality:
            mid_ind = len(xknots) // 2
            #Intentional overlap to ensure that xknots[0][-1] == xknots[1][0]
            first_segment = xknots[: mid_ind + 1]
            second_segment = xknots[mid_ind :]
            assert(first_segment[-1] == second_segment[0])
            return [first_segment, second_segment]
        else:
            cutoff_diffs = map(lambda x : abs(x - cutoff), xknots)
            indices = enumerate(cutoff_diffs)
            minimum_index = reduce(lambda x, y : x if x[1] <= y[1] else y, indices)
            min_ind = minimum_index[0] #This finds the index of the value closest to the cutoff, not exactly there
            first_segment = xknots[: min_ind + 1]
            second_segment = xknots[min_ind :]
            assert(first_segment[-1] == second_segment[0])
            assert(len(first_segment) + len(second_segment) == len(xknots) + 1)
            return [first_segment, second_segment]
    
    def r_range(self) -> (float, float):
        r"""Returns the full distance range spanned by the spline (i.e. first knot, last knot)
        
        Arguments:
            None
        
        Returns:
            min, max (float, float): The minimum and maximum distance for the spline.
        
        Notes: None
        """
        return (self.xknots[0][0], self.xknots[1][-1])
    
    def n_var(self) -> int:
        r"""Returns the number of variables used by the spline (coefficients)
        
        Arguments:
            None
        
        Returns:
            num var (int): The number of coefficients for the spline. If the spline
                dictionary is not initialized, the value 0 is returned
        
        Notes: None
        """
        if self.spline_dicts is None:
            return 0
        else:
            return len(self.spline_dicts[0]['splines']['coefs']) + len(self.spline_dict[1]['spline']['coefs'])
    
    def fit_model(self, xvals: Array, yvals: Array) -> (Array, Array):
        r"""Fits the model to initialize the spline coefficients
        
        Arguments: 
            xvals (Array): The x-values for fitting the spline
            yvals (Array): The y-values for fitting the spline
        
        Returns:
            coeffs, c_fixed (Array, Array): A tuple where the first array contains the 
                coefficients that need to be optimized and the second array contains the
                coefficients that are fixed
        
        Notes: This method should only be called once at the start to initialize the model.
            The inclusion of this method is interface-breaking, but it is the best workaround 
            given that the internal fit for the joined spline can produce the coefficients,
            and that seems easier than trying to work around the external method
            fit_linear_model.
        """
        if self.spline_dicts is not None:
            return (self.spline_dicts[0]['spline']['coefs'], self.spline_dicts[1]['spline']['coefs'])
        elif self.spline_dicts is None:
            #Spline has xknots, and the knots are already segmented so that 
            # construct_joined_splines can take them. Also, will not have to call 
            # merge_spline_dicts just yet.
            #The spline_dicts returned from construct_joined_splines is an array of length 3
            # with the third term containing the information for the full cubic spline
            self.spline_dicts = construct_joined_splines(self.xknots, xvals, yvals, self.bconds, xvals)
            assert(len(self.spline_dicts) == 3)
            return (self.spline_dicts[0]['spline']['coefs'], self.spline_dicts[1]['spline']['coefs'])
        
    def linear_model(self, xeval: Array, ider: int = 0, max_deriv: int = 2) -> (Array, Array):
        r"""Method for getting the matrix A and vector b for the spline matrix multiply
        
        Arguments:
            xeval (Array): The distances to evaluate the spline at
            ider (int): The derivative value needed. Defaults to 0
            max_deriv (int): The maximum derivative to calculate. Defaults to 2
        
        Returns:
            X, const (Array, Array): Returns X and const such that the prediction
                from the spline can be generated as follows:
                
                y = X @ cat (coeffs, c_fixed) + const
                
                where the fixed and variable coefficients are concatenated together
                before being matrix multiplied by the matrix X.
        
        Notes: None
        """
        #Get the list of derivatives to evaluate
        deriv_lst = [i for i in range(max_deriv + 1)]
        #Check if we're dealing with old or new xvals using the full cubic spline
        full_xvals = self.spline_dicts[2]['spline']['xvals']
        if (len(full_xvals) == len(xeval)) and all(x == y for x, y in zip(full_xvals, xeval))\
            and (len(self.spline_dicts[0]['spline']['X']) > ider and len(self.spline_dicts[1]['spline']['X']) > ider):
            #If dealing with old xvals, just return the merged spline from the original:
            X, const = merge_splines(self.spline_dicts, deriv_lst)
            return X[ider], const[ider]
        else:
            #Dealing with new xvals
            X, const = merge_splines_new_xvals(self.spline_dicts, xeval, deriv_lst)
            return X[ider], const[ider]

class ExponentialModel(PairwiseLinearModel):
    def __init__(self, exponents, rlow=0.0, rhigh=5.0):
        """
        f(x) = sum_i^nbasis  c_i exp( -exponents[i], x)
        Parameters
        ----------
        exponents : 1-D np.array, exponents of the basis functions
        """
        self.exponents = exponents.copy()
        self.rlow = rlow
        self.rhigh = rhigh

    def r_range(self):
        return self.rlow, self.rhigh

    def nvar(self):
        return len(self.exponents)

    def linear_model(self, r, ider=0):
        # Want the form:
        #   f[j] = sum_i A[j,i] c[i] + b[j]
        # our basis expansion is:
        #  f[j] = sum_i exp(-exponent[i] r[j]) c[i]
        # and the nth derivative is
        #  fn[j] sum_i (-exponent[i])^n exp(-exponent[i] r[j]) c[i]
        # so
        #   A[j,i] = (-exponent[i])^n exp(-exponent[i] r[j]) 
        #   b[j] = 0.0
        if isinstance(r, (int, float)):
            r = np.array([r])
        A = np.zeros([len(r), len(self.exponents)])
        for iexp, exponent in enumerate(self.exponents):
            A[:, iexp] = (-exponent) ** ider * \
                         np.exp(-exponent * r)
        b = np.zeros(len(r))
        return A, b

class PolynomialModel(PairwiseLinearModel):
    def __init__(self, powers, rlow=0.0, rhigh=5.0):
        """
        f(x) = sum_{i} c[i] x^powers[i]
        in the range rlow,rhigh
        Parameters
        ----------
        exponents : 1-D np.array, exponents of the basis functions
        """
        if isinstance(powers, int):
            self.powers = np.array([powers])
        else:
            self.powers = powers.copy()
        self.rlow = rlow
        self.rhigh = rhigh

    def r_range(self):
        return self.rlow, self.rhigh

    def nvar(self):
        return len(self.powers)

    def linear_model(self, r, ider=0):
        # Want the form:
        #   f[j] = sum_i A[j,i] c[i] x^powers[i]
        # the nth derivative of a x^degree is
        if isinstance(r, (int, float)):
            r = np.array([r])

        A = np.zeros([len(r), len(self.powers)])
        for ipow, power in enumerate(self.powers):
            if ider > power:
                A[:, ipow] = np.zeros(len(r))
            else:
                if ider == 0:
                    prefix = 1.0
                else:
                    prefix = scipy.special.factorial(power) / \
                             scipy.special.factorial(power - ider)
                A[:, ipow] = prefix * np.power(r, power - ider)

        b = np.zeros(len(r))
        return A, b
    
class MIOFunction(PairwiseLinearModel):
    ANGSTROM2BOHR = 1.889725989

    def __init__(self, Z, par_dict):
        self.Z = Z
        self.par_dict = par_dict

    def r_range(self):
        return 0, 5.0

    def nvar(self):
        return 0

    def linear_model(self, r_eval, ider=0):
        if ider != 0:
            raise NotImplementedError('MIOFunction does not support derivatives')
        A = 0.0
        b = self._get_dftb_vals(r_eval)
        return A, b

    def vals(self, coefs, r_eval, ider=0):
        if ider != 0:
            raise NotImplementedError('MIOFunction does not support derivatives')
        return self._get_dftb_vals(r_eval)

    def _get_dftb_grid(self):
        """ r values from the DFTB sk files"""
        # returns values in Angstroms
        sym = [ELEMENTS[z].symbol for z in self.Z]
        skinfo = self.par_dict[sym[0] + '-' + sym[-1]]
        return np.array(skinfo.GetSkGrid()) / MIOFunction.ANGSTROM2BOHR

    def _get_dftb_vals(self, rs):
        sym = [ELEMENTS[z].symbol for z in self.Z]
        skinfo = self.par_dict[sym[0] + '-' + sym[-1]]
        return np.array([skinfo.GetRep(x * MIOFunction.ANGSTROM2BOHR) for x in rs])
    
class PiecewiseFunction(PairwiseLinearModel): #NOT FULLY TESTED!
    def __init__(self, function1, function2, rmatch, derivs_match):
        """
        Parameters
        ----------
        function1 : TYPE
            DESCRIPTION.
        function2 : TYPE
            DESCRIPTION.
        rmatch : TYPE
            DESCRIPTION.
        derivs_match : List of ider
            DESCRIPTION.
        """
        self.f1 = function1
        self.f2 = function2
        self.rmatch = rmatch
        self.derivs = derivs_match

        self.constraint = list()
        for ider in self.derivs:
            (A1, b1) = self.f1.linear_model(self.rmatch, ider)
            (A2, b2) = self.f2.linear_model(self.rmatch, ider)
            # loc1 = np.arange(0, self.f1.nvar())
            # loc2 = np.arange(self.f1.nvar(), self.f1.nvar() + self.f2.nvar())
            # A1 * c[loc1] + b1 = A2 * c[loc2] + b2
            # A1 * c[loc1] - A2 * c[loc2] = b2 - b1
            # use Ar and br to indicate values for the constraint 
            Ar = np.concatenate([A1.flatten(), -A2.flatten()])
            br = b2 - b1
            for cdata in self.constraint:
                ArMatrix = Ar.reshape([1, -1])
                (Ac, bc) = PiecewiseFunction.__apply_constraint(ArMatrix,
                                                                br, cdata)
                Ar = Ac.flatten()
                br = bc
            # Use Ar * c = br to eliminate a coefficient
            # choose coefficient with largest Ar
            jmax = np.argmax(np.abs(Ar))
            notj = np.arange(0, len(Ar))
            notj = notj[notj != jmax]
            # we can use j' to mean sum over all j except jmax
            #    Ar[j'] c[j'] + Ar[jmax] c[jmax] = br
            #    c[jmax] = (-Ar[j']/Ar[jmax]) c[j'] + br/Ar[jmax]
            #    c[jmax] = X[j'] c[j'] + br/Ar[jmax]
            #       with X[j'] = -Ar[j']/Ar[jmax]
            X = -Ar[notj] / Ar[jmax]
            bx = br / Ar[jmax]
            # So if before applying the constraint, we had a model of form
            #    y[i] = A0[i,j] c[j] + b0[i]
            # Pull jmax out of the first summation
            #    y[i] = A0[i,j'] c[j'] + A0[i,jmax] c[jmax] + b0[i]
            # Substitute in c[jmax] from above:
            #    y[i] = A0[i,j'] c[j'] + A0[i,jmax] (X[j'] c[j'] + br) +b0[i]
            # Arrange back into a linear form in space j'
            #    y[i] = (A0[i,j'] + A0[i,jmax] X[j']) c[j'] + (A0[i,jmax] bx +b0[i])
            #    y[i] = A1[i,j'] c[j'] + b1
            #       A1[i,j'] = A0[i,j'] + A0[i,jmax] X[j']
            #       b1[i] = A0[i,jmax] bx +b0[i]
            self.constraint.append((jmax, notj, X, bx))

    def r_range(self):
        return self.f1.r_range()[0], self.f2.r_range()[1]

    def nvar(self):
        return self.f1.nvar() + self.f2.nvar() - len(self.constraint)

    @staticmethod
    def __apply_constraint(A, b, constraint):
        (jmax, notj, X, bx) = constraint
        A1 = A[:, notj] + np.outer(A[:, jmax], X)
        b1 = A[:, jmax] * bx + b
        return A1, b1

    def original_coefs(self, coefs):
        cfull = coefs.copy()
        for cdata in reversed(self.constraint):
            (jmax, notj, X, bx) = cdata
            cmax = np.dot(X, cfull) + bx
            # insert cmax into jmax location of coefs
            cfull = np.insert(cfull, jmax, cmax)
        loc1 = np.arange(0, self.f1.nvar())
        loc2 = np.arange(self.f1.nvar(), self.f1.nvar() + self.f2.nvar())
        return cfull[loc1], cfull[loc2]

    def linear_model(self, r, ider=0):
        # call func1 for all r <= rmatch
        #      func2 for all r> rmatch
        if isinstance(r, (float, int)):
            r = [r]
        r = np.asarray(r)
        i1 = r <= self.rmatch
        i2 = ~i1
        loc1 = np.arange(0, self.f1.nvar())
        loc2 = np.arange(self.f1.nvar(), self.f1.nvar() + self.f2.nvar())

        A = np.zeros([len(r), self.f1.nvar() + self.f2.nvar()])
        b = np.zeros([len(r)])
        (A1, b1) = self.f1.linear_model(r[i1], ider)
        (A2, b2) = self.f2.linear_model(r[i2], ider)
        A[np.ix_(i1, loc1)] = A1
        A[np.ix_(i2, loc2)] = A2
        b[i1] = b1
        b[i2] = b2

        for cdata in self.constraint:
            (A, b) = PiecewiseFunction.__apply_constraint(A, b, cdata)

        return A, b

