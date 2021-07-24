import itertools
import numpy as np
import os
import pickle as pkl
import scipy.sparse
import scipy.special
from elements import ELEMENTS
from h5py import File
from matplotlib import pyplot as plt
from deprecated.slakos_deprecated.mio_0_1 import ParDict
from tfspline import spline_linear_model, Bcond, spline_new_xvals
from fold import Fold
from util import get_dataset_type
from consts import ALIAS2TARGET, HARTREE


class PairwiseLinearModel:
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
    A, b = model.linear_model(xvals, ider)
    # A c + b = y
    coefs, _, _, _ = np.linalg.lstsq(A, yvals - b, rcond=None)
    return coefs, A, b


def dict_to_string(errs, convert_values=1.0):
    return ''.join([f'{x} {y * convert_values: .5} ' for (x, y) in errs.items()])


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


# def map_linear_models_broken(source,target, xgrid = None, ngrid = None,
#                       out_of_range_degrees = [0,0]):
#     '''
#     THIS IS KEPT BECAUSE IT MAY BE RECOVERABLE. The piecewise function
#     does an internal transformation of coefficients that needs to be taken
#     account of in this approach, but current it not used

#     For models of the form:
#         source:  np.dot(A1,c1) + b1
#         target:  np.dot(A2,c2) + b2
#     This routine calculates the relation:
#         c2 = np.dot(X,c1) + y
#     resulting from the process of:
#         use source to generate values, ygrid, on xgrid
#         fit the target model to these ygrid values

#     Parameters
#     ----------
#     source : PairwiseLinearModel  (typically a sparse model)
#     target : PairwiseLinearModel  (typically a dense model)
#     xgrid : 1-D numpy array  (ignored if ngrid is provided)
#             note that xgrid must cover the full range of the models and be 
#             sufficiently dense that there is sufficient information to map
#             between the source and target models
#     ngrid  : int
#             creates xgrid wkth ngrid points, spanning union of the range of
#             the two models
#     out_of_range_degrees : list of 2 ints [deg_low, deg_high]
#             if the ranges of the models do not overlap, extrapolate into that
#             region using a polynomials of deg_low, for values that out of range
#             on the low side, and deg_high for values that are out of range on
#             the high side. A PolynomialModel is created with these powers, 
#             and they are joined with continuous derivatives up to the order
#             of the polynomial

#     Returns
#     -------
#     X : 2-D numpy array
#     y : 1-D numpy array

#     '''
#     models = [source,target]
#     rlows = np.zeros(2)
#     rhighs = np.zeros(2)
#     for imod, mod in enumerate(models):
#         rlows[imod], rhighs[imod] = mod.r_range()
#     rlow = np.min(rlows)
#     rhigh = np.max(rhighs)

#     if ngrid is not None:
#         xgrid = np.linspace(rlow,rhigh,ngrid)

#     tol = 1.0e-10
#     for imod, mod in enumerate(models):
#         if rlow + tol < rlows[imod]:
#             degree = out_of_range_degrees[0]
#             poly = PolynomialModel(np.arange(degree+1))
#             models[imod] = PiecewiseFunction(poly, models[imod], 
#                                               rlow, np.arange(degree))
#         if rhigh -tol > rhighs[imod]:
#             degree = out_of_range_degrees[1]
#             poly = PolynomialModel(np.arange(degree+1))
#             models[imod] = PiecewiseFunction(models[imod], poly,
#                                               rhigh, np.arange(degree))


#     A1,b1 = models[0].linear_model(xgrid)
#     A2,b2 = models[1].linear_model(xgrid)
#     # A2 c2 + b2 = A1 c1 + b1 
#     # A2 c2 = A1 c1 + (b1-b2)
#     # A c = b --> c = (A.T A)^-1 A.T b
#     # c2 = (A2.T A2)^-1 A2.T (A1 c1 + b1 - b2)
#     # c2 = X c1 + y
#     #   X = (A2.T A2)^-1 A2.T    A1
#     #   y = (A2.T A2)^-1 A2.T   (b1 - b2)
#     try:
#         t1 = np.dot(np.linalg.inv(np.dot(A2.T, A2)),A2.T)
#     except:
#         raise Exception('singular matrix in map_linear_models')
#     X = np.dot(t1, A1)
#     y = np.dot(t1, b1-b2)

#     return X,y


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
            self.spline_dict = spline_new_xvals(self.spline_dict, xeval, nder=ider + 1)

        return self.spline_dict['X'][ider], self.spline_dict['const'][ider]


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

    def __init__(self, Z):
        self.Z = Z
        self.par_dict = ParDict()

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


class PiecewiseFunction(PairwiseLinearModel):
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


class ModelZ:
    def __init__(self, models, locs, cutoffs, is_linear=True):
        self.models = models
        self.locs = locs
        self.cutoffs = cutoffs
        self.is_linear = is_linear

    def __getitem__(self, Z):
        return self.models[Z]

    def Zs(self):
        return list(self.models.keys())

    def nvars(self):
        return len(np.concatenate(list(self.locs.values()), 0))


class LinearLoss:
    def __init__(self, alpha, beta, const):
        self.alpha = alpha
        self.beta = beta
        self.const = const

    def nvars(self):
        return self.alpha.shape[1]

    def value(self, coefs):
        res = coefs.T.dot(self.alpha).dot(coefs) - 2 * self.beta.T.dot(coefs) + self.const
        return res

    def solve(self):
        res = np.linalg.lstsq(self.alpha, self.beta, rcond=None)[0]
        return res


def create_model_Z_from_models(models):
    locs = dict()
    cutoffs = dict()
    loc_start = 0
    allZ = []
    for Z, model in models.items():
        cutoffs[Z] = model.r_range()
        loc_end = loc_start + model.nvar()
        locs[Z] = np.arange(loc_start, loc_end)
        loc_start = loc_end
        allZ.extend(Z)
    # reference energy is one constant per Z, plus an overall constant
    loc_end = loc_start + len(set(allZ)) + 1
    locs['ref'] = np.arange(loc_start, loc_end)

    return ModelZ(models, locs, cutoffs)


def map_linear_models_Z(sparse_models_only, dense_model_Z, ngrid):
    """
    Create a sparse model_Z from a dense model_Z by
      1) calling map_linear_models() for the models corresponding to each Z
      2) setting the number of reference energy parameters of the sparse model
         equal to those of the dense model

    Parameters
    ----------
    sparse_models_only : dict
        this dict will become the "models" of the sparse_model_Z
        sparse_models_only[Z] = PairwiseLinearModel
    dense_model_Z : model_Z
    ngrid : int
        The xgrid for map_linear_models has ngrid points in the range specified
        by dense_model_Z.cutoffs

    Returns
    -------
    map_models : dict
        map_models[Z] = (Z,y) from call to map_linear_models() for that Z
    sparse_model_Z : model_Z
        in sparse_model_Z:
            models = sparse_models_only
            cutoffs is copied from dense_model_Z
            locs is created to map the dense to sparse coefs

    """
    # TODO: implement handling of disagreements between ranges
    dense = dense_model_Z.models
    cutoffs = dense_model_Z.cutoffs
    map_models = dict()
    loc_sparse = dict()
    loc_start = 0
    for Z in sparse_models_only.keys():
        (rlow, rhigh) = cutoffs[Z]
        rgrid = np.linspace(rlow, rhigh, ngrid)
        map_models[Z] = map_linear_models(sparse_models_only[Z], dense[Z], rgrid)
        Xmap, ymap = map_models[Z]
        if np.max(np.abs(ymap)) > 1.0e-12:
            print("warning: model map has a constant term")
        loc_end = loc_start + sparse_models_only[Z].nvar()
        loc_sparse[Z] = np.arange(loc_start, loc_end)
        loc_start = loc_end

    nvar_ref = len(dense_model_Z.locs['ref'])
    loc_end = loc_start + nvar_ref
    loc_sparse['ref'] = np.arange(loc_start, loc_end)
    # map is an identity function
    map_models['ref'] = (np.identity(nvar_ref), np.zeros([nvar_ref]))

    sparse_cutoffs = dict()
    for Z, mod in sparse_models_only.items():
        sparse_cutoffs[Z] = mod.r_range()
    sparse_model_Z = ModelZ(sparse_models_only, loc_sparse, sparse_cutoffs)
    return map_models, sparse_model_Z


def dense_coefs_from_sparse(map_models, coefs_sparse):
    X = scipy.sparse.block_diag([X for (X, y) in map_models.values()])
    X = X.todense()
    X = np.asarray(X)
    y = np.concatenate([y for (X, y) in map_models.values()], 0)
    return np.dot(X, coefs_sparse) + y


def sparse_loss_from_dense(map_models, dense_loss):
    # from LinearLoss.value()
    #   loss = d.T A d  -2 B.T d  + C0
    #      d = coefficients of the dense model
    #      A, B, C0 = alpha, beta, const
    # from map_linear_models()
    #   d = X c + y    ; c = coefficients of the sparse model
    # substituting d into the loss gives
    #   loss = (y.T + c.T X.T) A (X c + y) - 2 B.T (X c + y) + C0
    #   loss = c.T X.T A X c
    #        - 2 B.T X c + c.T X.T A y + y.T A X c 
    #        + y.T A y - 2 B.T y + C0
    # We can simplify the second line of the loss by using
    #    c.T X.T A y + y.T A X c = 2 y.T A X c
    # loss = c.T X.T A X c
    #        - 2 B.T X c +  2 y.T A X c 
    #        + y.T A y - 2 B.T y + C0
    # the second line can be written:
    #        - 2 beta.T c    with beta.T = B.T X - y.T A X 
    # So the new loss function has:
    #    alpha = X.T A X
    #    beta  = X.T (B - A.T y)
    #    const =  y.T A y - 2 B.T y + C0

    # Convert X,y from map_models to a block diagonal matrix
    A = dense_loss.alpha
    B = dense_loss.beta
    C0 = dense_loss.const
    # kernel crashes if I don't convert to dense?
    X = scipy.sparse.block_diag([X for (X, y) in map_models.values()])
    X = X.todense()
    X = np.asarray(X)
    y = np.concatenate([y for (X, y) in map_models.values()], 0)
    alpha = np.dot(X.T, np.dot(A, X))
    beta = np.dot(X.T, B - np.dot(A.T, y))
    const = np.dot(y.T, np.dot(A, y)) - 2.0 * np.dot(B.T, y) + C0

    linear_loss = LinearLoss(alpha, beta, const)
    return linear_loss


def fit_model_Z_to_xydata_linear(model_Z, xydata_Z):
    coefs = np.zeros(model_Z.nvars())
    for Z, model in model_Z.models.items():
        (rgrid, ygrid) = xydata_Z[Z]
        coefs[model_Z.locs[Z]], _, _ = fit_linear_model(model, rgrid, ygrid)

    coefs[model_Z.locs['ref']] = 0.0

    return coefs


def get_xydata_Z(model_Z, coefs, nvals=500, ider=0):
    models = model_Z.models
    cutoffs = model_Z.cutoffs
    loc = model_Z.locs
    xydata = dict()
    for Z, (rlow, rhigh) in cutoffs.items():
        x = np.linspace(rlow, rhigh, nvals)
        y = models[Z].vals(coefs[loc[Z]], x, ider)
        xydata[Z] = (x, y)
    return xydata


def plot_xydata_Z(xydata, syms=('bx', 'rx', 'gx', 'kx')):
    if not isinstance(xydata, list):
        xydata = [xydata]
    iplot = 1
    plt.figure(figsize=(30, 10))
    for Z in xydata[0].keys():
        plt.title(Z)
        plt.subplot(2, 5, iplot)
        for idata, xyZ in enumerate(xydata):
            plt.xlabel('Distance (A)')
            plt.ylabel('Potential (kcal/mol)')
            plt.ylim(-20, 20)
            (x, y) = xyZ[Z]
            plt.plot(x, y * HARTREE, syms[idata % len(syms)])
        iplot += 1


def diff_xydata_Z(xydata1, xydata2):
    xydiff = dict()
    errors_allZ = dict()
    for Z in xydata1.keys():
        errors_allZ[Z] = dict()
        x1 = xydata1[Z][0]
        x2 = xydata2[Z][0]
        if (x1.shape != x2.shape) or (np.max(np.abs(x2 - x1))) > 1.0e-10:
            raise ValueError('diff_allZ: x values do not agree')
        y1 = xydata1[Z][1]
        y2 = xydata2[Z][1]
        ydiff = y2 - y1
        xydiff[Z] = (x1, ydiff)
        errors_allZ[Z]['rms'] = np.sqrt(np.mean(np.square(ydiff)))
        errors_allZ[Z]['mae'] = np.mean(np.abs(ydiff))
        errors_allZ[Z]['max'] = np.max(np.abs(ydiff))
    err_summary = {
        'rms': np.mean([x['rms'] for x in errors_allZ.values()]),
        'mae': np.mean([x['rms'] for x in errors_allZ.values()]),
        'max': np.max([x['rms'] for x in errors_allZ.values()])}
    return xydiff, err_summary, errors_allZ


def get_data_type(specs):
    if not isinstance(specs, list):
        specs = [specs]
    res = []
    for spec in specs:
        if spec in ALIAS2TARGET.keys():
            res.append(ALIAS2TARGET[spec])
        elif spec in ALIAS2TARGET.values():
            res.append(spec)
        else:
            raise ValueError('get_data_type: spec ' + spec + ' not recognized')
    return res


def get_targets_from_dataset(target_type, fold, dataset_path):
    dtypes = get_data_type(target_type)
    targets = dict()
    dataset_type = get_dataset_type(dataset_path)

    if dataset_type == 'h5':
        with File(dataset_path, 'r') as dataset:
            for mol, conf_arr in fold.items():
                moldata = [dataset[mol][x][conf_arr] for x in dtypes]
                if len(moldata) == 1:
                    targets[mol] = moldata[0]
                else:
                    targets[mol] = moldata[0] - moldata[1]
    else:
        with open(dataset_path, 'rb') as f:
            dataset = pkl.load(f)
        for mol, conf_arr in fold.items():
            moldata = [dataset[mol][x][conf_arr] for x in dtypes]
            if len(moldata) == 1:
                targets[mol] = moldata[0]
            else:
                targets[mol] = moldata[0] - moldata[1]

    return targets


def get_predictions_from_dense(coefs_dense, fold, gammas_path):
    # map_models is kept for compatibility
    preds = dict()
    with File(gammas_path, 'r') as gammas:
        for mol, conf_arr in fold.items():
            gamma = gammas[mol]['gammas'][conf_arr]
            preds[mol] = np.dot(gamma, coefs_dense)
    return preds


def get_predictions_from_sparse(coefs_sparse, map_models, fold, gammas_path):
    coefs_dense = dense_coefs_from_sparse(map_models, coefs_sparse)
    preds = get_predictions_from_dense(coefs_dense, fold, gammas_path)
    return preds


def create_dense(target_type, fold, dataset_path, gammas_path):
    # if target_type is a list of length 2, the target is the difference
    #   target = target_type[0] - target_type[1]
    dtypes = get_data_type(target_type)
    if len(dtypes) not in [1, 2]:
        raise ValueError('create_dense: target_type is not of length 1 or 2')

    targets = get_targets_from_dataset(dtypes, fold, dataset_path)
    alpha = None
    beta = None
    loss0 = 0.0

    with File(gammas_path, 'r') as gammas:
        # Read model parameters from gammas
        Zs = [(zs[0], zs[1]) for zs in gammas['_INFO']['Zs']]
        dense_params = {'Zs': Zs,
                        'Xis': gammas['_INFO']['Xis'][()],
                        'R_cutoffs': {Z: (r[0], r[1]) for Z, r in zip(Zs, gammas['_INFO']['R_cutoffs'])},
                        'Nknots': {Z: n[0] for Z, n in zip(Zs, gammas['_INFO']['Nknots'])},
                        'Xknots': {Z: x for Z, x in zip(Zs, gammas['_INFO']['Xknots'])},
                        'Degrees': {Z: d[0] for Z, d in zip(Zs, gammas['_INFO']['Degrees'])}}
        for mol, conf_arr in fold.items():
            gamma = gammas[mol]['gammas'][conf_arr]
            target = targets[mol]
            if alpha is None:
                dense_nvars = gamma.shape[1]
                alpha = np.zeros([dense_nvars, dense_nvars])
                beta = np.zeros(dense_nvars)
            alpha += gamma.T.dot(gamma)
            beta += target.T.dot(gamma)
            loss0 += target.dot(target)

    nconfs = fold.nconfs()
    alpha /= nconfs
    beta /= nconfs
    loss0 /= nconfs

    dense = dict()
    loc = dict()
    loc_start = 0
    for Z in dense_params['Zs']:
        dense[Z] = SplineModel(
            {'xknots': dense_params['Xknots'][Z],
             'deg': dense_params['Degrees'][Z],
             'bconds': 'vanishing'})
        loc_end = loc_start + dense[Z].nvar()
        loc[Z] = np.arange(loc_start, loc_end)
        loc_start = loc_end
    # last elements of c are: overall constant, constants for each element
    loc['ref'] = np.arange(loc_start, loc_start + 1 + len(dense_params['Xis']))

    model_Z = ModelZ(dense, loc, dense_params['R_cutoffs'])
    loss_func = LinearLoss(alpha, beta, loss0)
    return model_Z, loss_func


def find_outliers(targets, preds, num_to_print=10):
    labels = dict()
    for mol, edata in targets.items():
        nconfig = len(edata)
        labels[mol] = [(mol, x) for x in range(nconfig)]

    ids = list(itertools.chain(*list(labels.values())))
    goal = np.concatenate([targets[x] for x in labels.keys()])
    pred = np.concatenate([preds[x] for x in labels.keys()])
    err = np.abs(goal - pred)
    isort = np.argsort(-err)
    ids = [ids[x] for x in isort]
    err = err[isort]
    for i1 in range(num_to_print):
        rms = np.sqrt(np.mean(np.square(err[i1:]))) * HARTREE
        mae = np.mean(np.abs(err[i1:])) * HARTREE
        maxerr = np.max(np.abs(err[i1:])) * HARTREE
        if i1 == 0:
            removed = 'include all'
        else:
            (mol, config) = ids[i1 - 1]
            removed = f'{mol}  {config}'
        errstr = f'rms {rms: .5} mae {mae: .5} max {maxerr: .5} kcal/mol'
        print(removed, errstr)
    return ids, err


def compare_target_pred(targets, preds):
    try:
        goal = np.concatenate([targets[x] for x in targets.keys()])
        pred = np.concatenate([preds[x] for x in targets.keys()])
        err = np.abs(goal - pred)
    except ValueError:
        err = np.nan
    errdict = {'rms': np.sqrt(np.mean(np.square(err))),
               'mae': np.mean(err),
               'max': np.max(err)}
    return errdict, err


if __name__ == "__main__":
    # def test_fit_linear_model():
    #     """
    #     fits a spline to a single exponential and a sum of two exponentials
    #
    #     """
    #     xlow, xhigh = 1.0, 5.0
    #     nknots = 6
    #     xknots = np.linspace(xlow, xhigh, nknots)
    #     xeval = np.linspace(xlow, xhigh, 20)
    #     config = {'xknots': xknots,
    #               'bconds': 'natural'}
    #     sp4 = SplineModel(config, xeval)
    #
    #     yeval = np.exp(-2.0 * xeval)
    #     c4, A4, b4 = fit_linear_model(sp4, xeval, yeval)
    #     ypred = np.dot(A4, c4) + b4
    #
    #     yknots = np.exp(-2.0 * xknots)
    #     plt.figure(2)
    #     plt.plot(xeval, yeval, 'r.')
    #     plt.plot(xknots, yknots, 'rx')
    #     plt.plot(xeval, ypred, 'bo')
    #
    #     xlow, xhigh = 1.0, 5.0
    #     exponents = [-1.2, -3.0]
    #     coefs = [3.0, 4.0]
    #     neval = 20
    #     xeval = np.linspace(xlow, xhigh, neval)
    #     yeval = np.zeros(xeval.shape)
    #     for c1, e1 in zip(coefs, exponents):
    #         yeval += c1 * np.exp(e1 * xeval)
    #
    #     nknots = 4
    #     xknots = np.linspace(xlow, xhigh, nknots)
    #     config = {'xknots': xknots,
    #               'bconds': 'natural'}
    #     sp4 = SplineModel(config, xeval)
    #
    #     c4, A4, b4 = fit_linear_model(sp4, xeval, yeval)
    #     ypred = np.dot(A4, c4) + b4
    #
    #     yknots = np.zeros(xknots.shape)
    #     for c1, e1 in zip(coefs, exponents):
    #         yknots += c1 * np.exp(e1 * xknots)
    #     plt.figure(2)
    #     plt.plot(xeval, yeval, 'r.')
    #     plt.plot(xknots, yknots, 'rx')
    #     plt.plot(xeval, ypred, 'bo')
    #
    #
    # def test_PiecewiseFunction():
    #     """
    #     Generates two exponential models and joins them together, then:
    #          - uses no joining conditions and plots the originals and the joined
    #            (this concatenates coefs for the two funcs)
    #          - applies joining conditions, generates random coefs and plots to
    #            - first see if join conditions seem ok
    #            - extract coefs of original functions to ensure they are recovered
    #          - joins in another exponential
    #     """
    #     f1 = ExponentialModel([0.1, 0.2, 0.3])
    #     f2 = ExponentialModel([1.0, 2.0, 3.0])
    #     c1 = np.random.uniform(0.5, 2.5, f1.nvar())
    #     c2 = np.random.uniform(0.5, 2.5, f2.nvar())
    #     c12 = np.concatenate((c1, c2))
    #     f12 = PiecewiseFunction(f1, f2, 1.0, [])
    #     rvals = np.linspace(0.0, 2.0, 100)
    #     np.random.shuffle(rvals)
    #     plt.figure(1)
    #     f1.plot(c1, 'g.', rvals=rvals)
    #     f2.plot(c2, 'b.', rvals=rvals)
    #     f12.plot(c12, 'rx', rvals=rvals)
    #
    #     f3 = PiecewiseFunction(f1, f2, 1, [0])
    #     plt.figure(2)
    #     c3 = np.random.uniform(0.5, 2.5, f3.nvar())
    #     f3.plot(c3, 'r.', rvals=rvals)
    #
    #     f4 = PiecewiseFunction(f1, f2, 0.5, [0, 1])
    #     plt.figure(3)
    #     c4 = np.random.uniform(0.5, 2.5, f4.nvar())
    #     f4.plot(c4, 'go', rvals=rvals)
    #     (c1orig, c2orig) = f4.original_coefs(c4)
    #     f1.plot(c1orig, 'r.', rvals=rvals)
    #     f2.plot(c2orig, 'b.', rvals=rvals)
    #
    #     # add in another exponential
    #     f5 = ExponentialModel([0.5, 0.8, 0.1])
    #     f6 = PiecewiseFunction(f4, f5, 1.0, [0, 1])
    #     plt.figure(4)
    #     c5 = np.random.uniform(-2.5, 2.5, f6.nvar())
    #     f6.plot(c5, 'g.', rvals=rvals)
    #
    #     plt.show()
    #
    #
    # def test_map_linear_models():
    #     """
    #     Generates a variety of sparse and dense models
    #             (these include splines with fixed boundary conditions, to
    #              test handling of the constant term in the linear model,
    #              i.e. A c + b  with b being non-zero.)
    #       maps the sparse onto the dense
    #       generates random coefficients for the sparse models
    #       uses the map to generate coefficients for the dense models
    #       plots the sparse and dense model, to see if the results are reasonable
    #          (reasonable means the dense model is fit, as well as possible, to the
    #           sparse model)
    #     """
    #
    #     xlow, xhigh = 0.0, 5.0
    #     ngrid = 300
    #     xgrid = np.linspace(xlow, xhigh, ngrid)
    #
    #     sparse = dict()
    #     dense = dict()
    #
    #     sparse['8 knot natural spline'] = \
    #         SplineModel(
    #             {'xknots': np.linspace(xlow, xhigh, 8),
    #              'bconds': 'natural'})
    #     sparse['8 knot fixed bc spline'] = \
    #         SplineModel(
    #             {'xknots': np.linspace(xlow, xhigh, 8),
    #              'bconds': [Bcond(0, 1, 10.0), Bcond(-1, 0, 1.0)]})
    #     sparse['4 exponent model'] = \
    #         ExponentialModel(np.linspace(0.3, 0.3, 5))
    #
    #     rmatch = 2.5
    #     f1 = ExponentialModel([0.1, 0.2, 0.3])
    #     f2 = SplineModel(
    #         {'xknots': np.linspace(rmatch, xhigh, 8),
    #          'bconds': 'natural'})
    #     sparse['3 exp + 8 knot spline'] = \
    #         PiecewiseFunction(f1, f2, rmatch, [0, 1])
    #
    #     dense['30 knot natural spline'] = \
    #         SplineModel(
    #             {'xknots': np.linspace(xlow, xhigh, 30),
    #              'bconds': 'natural'})
    #     dense['30 knot fixed bc spline'] = \
    #         SplineModel(
    #             {'xknots': np.linspace(xlow, xhigh, 30),
    #              'bconds': [Bcond(0, 1, 20.0), Bcond(-1, 0, 2.0)]})
    #
    #     rmatch = 2.5
    #     f1 = ExponentialModel([0.1, 0.15, 0.2, 0.25, 0.3])
    #     f2 = SplineModel(
    #         {'xknots': np.linspace(rmatch, xhigh, 16),
    #          'bconds': 'natural'})
    #     dense['5 exp + 16 knot spline'] = \
    #         PiecewiseFunction(f1, f2, rmatch, [0, 1])
    #
    #     ifig = 1
    #     nsamples = 5
    #     for sparse_name, mod1 in sparse.items():
    #         for dense_name, mod2 in dense.items():
    #             print('generating map for ', sparse_name, ' to ', dense_name)
    #             X, y = map_linear_models(mod1, mod2, xgrid)
    #             for i1 in range(nsamples):
    #                 c1 = np.random.randn(X.shape[1])
    #                 c2 = np.dot(X, c1) + y
    #                 plt.figure(ifig)
    #                 mod1.plot(c1, 'b-')
    #                 mod2.plot(c2, 'r-')
    #             plt.title(sparse_name + ' --> ' + dense_name)
    #             plt.show()
    #             ifig += 1
    #
    #
    # def test_fitting_dense_to_mio(ani1_path, gammas_path, show_plots=True,
    #                               outlier_analysis=True, exclude_outliers=True):
    #     print("***** TEST: fitting dense to mio total repulsive energies *****")
    #     target_type = 'pr'
    #     if exclude_outliers:
    #         exclude = {'H2O2': [180, 157]}
    #     else:
    #         exclude = None
    #
    #     dense_model_Z, dense_loss_func = \
    #         create_dense(target_type, ani1_path, gammas_path, exclude)
    #
    #     # Make a ModelZ that returns mio data. It doesn't use the coefs, so locs
    #     # is not used in any real way, but included to allow interface to work
    #     mio_model_Z = ModelZ({Z: MIOFunction(Z) for Z in dense_model_Z.models.keys()},
    #                          dense_model_Z.locs,
    #                          dense_model_Z.cutoffs)
    #     xydata_mio = get_xydata_Z(mio_model_Z, np.zeros(mio_model_Z.nvars()))
    #
    #     for ifit, fit_type in enumerate(['total E', 'xydata']):
    #         print('** Fitting dense to ' + fit_type + ' **')
    #         if fit_type == 'total E':
    #             c_dense = dense_loss_func.solve()
    #             ref_E_params = c_dense[dense_model_Z.locs['ref']]
    #         else:
    #             c_dense = fit_model_Z_to_xydata_linear(dense_model_Z, xydata_mio)
    #             ref_E_params = c_dense[dense_model_Z.locs['ref']]
    #
    #         dense_loss = dense_loss_func.value(c_dense)
    #         print(f'loss on total energy: rms loss {np.sqrt(dense_loss) * HARTREE: .7f} kcal/mol')
    #
    #         # TODO: incorporate Fold
    #         targets = get_targets_from_dataset(target_type, ani1_path, exclude)
    #         preds = get_predictions_from_dense(c_dense, None, gammas_path, exclude)
    #
    #         if outlier_analysis:
    #             print('Outlier analysis for ', target_type)
    #             configs_sorted, errs_sorted = find_outliers(targets, preds, 100)
    #
    #         err_dict, _ = compare_target_pred(targets, preds)
    #         print(f'Errors on total energy: {target_type} {dict_to_string(err_dict, HARTREE)} kcal/mol')
    #
    #         xydata_dense = get_xydata_Z(dense_model_Z, c_dense)
    #
    #         xydiff, errors, _ = diff_xydata_Z(xydata_mio, xydata_dense)
    #         print(f'Errors on xydata: {dict_to_string(errors, HARTREE)} kcal/mol')
    #
    #         if show_plots:
    #             plt.figure(100 * (ifit + 1))
    #             plot_xydata_Z([xydiff])
    #             plt.xlim(right=2.5)
    #             plt.title(fit_type)
    #             plt.show()
    #
    #
    # def test_map_from_sparse_with_restricted_range():
    #     # Testing ability to map from sparse model with restricted range
    #     range_dense = (0.0, 5.0)
    #     range_sparse = (0.0, 4.0)
    #     ngrid = 500
    #
    #     dense = SplineModel(
    #         {'xknots': np.linspace(range_dense[0], range_dense[1], 30),
    #          'bconds': 'natural'})
    #
    #     sparse = SplineModel(
    #         {'xknots': np.linspace(range_sparse[0], range_sparse[1], 8),
    #          'bconds': 'vanishing'})
    #
    #     ifig = 1
    #     nsamples = 5
    #     X, y = map_linear_models(sparse, dense, ngrid=ngrid)
    #     for i1 in range(nsamples):
    #         c1 = np.random.randn(X.shape[1])
    #         c2 = np.dot(X, c1) + y
    #         plt.figure(ifig)
    #         sparse.plot(c1, 'b-')
    #         dense.plot(c2, 'r-.')
    #     # plt.title(sparse_name+' --> '+dense_name)
    #     plt.show()
    #
    #
    # def plot_performance_vs_nknots():
    #     # Plot performance versus number of knots
    #     ngrid = 500
    #     nknots_values = np.arange(20, 50)
    #     loss_values = np.zeros(len(nknots_values))
    #     for iloss, nknots in enumerate(nknots_values):
    #         sparse_models_only = dict()
    #         for Z, (rlow, rhigh) in dense_model_Z.cutoffs.items():
    #             sparse_models_only[Z] = SplineModel(
    #                 {'xknots': np.linspace(rlow, rhigh, nknots),
    #                  'deg': 3,
    #                  'bconds': 'vanishing'})
    #         map_models, sparse_model_Z = \
    #             map_linear_models_Z(sparse_models_only, dense_model_Z, ngrid)
    #         sparse_loss_func = sparse_loss_from_dense(map_models, dense_loss_func)
    #         c_sparse = sparse_loss_func.solve()
    #         loss_values[iloss] = np.sqrt(sparse_loss_func.value(c_sparse)) * HARTREE
    #
    #     plt.figure(110)
    #     plt.plot(nknots_values, loss_values, 'ro')
    #     plt.show()
    #
    #
    # def test_map_sparse_to_dense():
    #     # map sparse to dense model using this size grid
    #     ngrid = 500
    #     # plot loss and MAE versus rmax, for these rmax values
    #     rmax_values = np.linspace(1.5, 4.0, 10)
    #     loss_values = np.zeros(len(rmax_values))
    #     mae_values = np.zeros(len(rmax_values))
    #     xydata_values = dict()
    #     for iloss, rmax in enumerate(rmax_values):
    #         # create a model_Z for splines with the reduced range
    #         sparse_models_only = dict()
    #         for Z, (rlow, rhigh) in dense_model_Z.cutoffs.items():
    #             if Z == (8, 8) and rmax < 3.0:
    #                 rmax_use = 3.0
    #             elif Z == (1, 1) and rmax < 3.0:
    #                 rmax_use = 3.0
    #             else:
    #                 rmax_use = rmax
    #             sparse_models_only[Z] = SplineModel(
    #                 {'xknots': np.linspace(rlow, rmax_use, 20),
    #                  'deg': 3,
    #                  'bconds': 'vanishing'})
    #         # map the sparse to the dense model
    #         map_models, sparse_model_Z = \
    #             map_linear_models_Z(sparse_models_only, dense_model_Z, ngrid)
    #         # transform the dense loss function to the sparse model coefficients
    #         sparse_loss_func = sparse_loss_from_dense(map_models, dense_loss_func)
    #         # solve for the coefficients of the sparse model
    #         c_sparse = sparse_loss_func.solve()
    #         # save the loss, evaluated from the loss function
    #         loss_values[iloss] = np.sqrt(sparse_loss_func.value(c_sparse)) * HARTREE
    #         # to calculate MAE, load the targets
    #         targets = get_targets_from_dataset(target_type, fold, ani1_path)
    #         # transform the coefs of the sparse model to the dense model
    #         c_dense = dense_coefs_from_sparse(map_models, c_sparse)
    #         # use the coefs of the dense model to calculate the predictions
    #         preds = get_predictions_from_dense(c_dense, None, fold, gammas_path)
    #         # get the error statistics
    #         err_dict, _ = compare_target_pred(targets, preds)
    #         print(f'rmax {rmax: .3f}: {target_type} {dict_to_string(err_dict, HARTREE)} kcal/mol')
    #         mae_values[iloss] = err_dict['mae'] * HARTREE
    #         xydata_values[rmax] = get_xydata_Z(sparse_model_Z, c_sparse)
    #     plt.figure(120)
    #     plt.plot(rmax_values, loss_values, 'bo')
    #     plt.plot(rmax_values, mae_values, 'ro')
    #     plt.show()
    #
    #     plt.figure(130)
    #     plot_xydata_Z(list(xydata_values.values()))
    #     plt.show()
    #
    #
    # def plot_performance_vs_rmax():
    #     # Plot performance versus max cutoff distance
    #     ngrid = 500
    #
    #     sparse_models_only = dict()
    #     for Z, (rlow, rhigh) in dense_model_Z.cutoffs.items():
    #         sparse_models_only[Z] = SplineModel(
    #             {'xknots': np.linspace(rlow, 3.2, 20),
    #              'deg': 3,
    #              'bconds': 'vanishing'})
    #     map_models, sparse_model_Z = \
    #         map_linear_models_Z(sparse_models_only, dense_model_Z, ngrid)
    #     mio_model_Z.cutoffs = sparse_model_Z.cutoffs
    #     xydata_mio2 = get_xydata_Z(mio_model_Z, np.zeros(mio_model_Z.nvars()))
    #     c_sparse = fit_model_Z_to_xydata_linear(sparse_model_Z, xydata_mio2)
    #     xydata_sparse = get_xydata_Z(sparse_model_Z, c_sparse)
    #     xydiff, errors, _ = diff_xydata_Z(xydata_mio, xydata_sparse)
    #     print(f'Errors on xydata: {dict_to_string(errors, HARTREE)} kcal/mol')
    #

    if os.getenv("USER") == "yaron":
        ani1_path = 'data/ANI-1ccx_clean_shifted.h5'
        gammas_path = 'data/gammas_50_5_extend.h5'
    elif os.getenv("USER") == "francishe":
        ani1_path = "/home/francishe/Downloads/ANI-1ccx_clean_shifted.h5"
        gammas_path = "/home/francishe/Downloads/gammas_50_5_extend.h5"
    else:
        raise ValueError("Invalid user")

    target_type = ['cc', 'pe']  # 'pr'
    exclude = {'H2O2': [180, 157]}

    with File(ani1_path, "r") as dataset:
        fold = Fold.from_dataset(dataset)
        fold -= Fold(exclude)

    dense_model_Z, dense_loss_func = create_dense(target_type, fold, ani1_path, gammas_path)

    c_dense = dense_loss_func.solve()
    dense_loss = dense_loss_func.value(c_dense)
    print(f'dense loss: rms loss {np.sqrt(dense_loss) * HARTREE: .7f} kcal/mol')

    ngrid = 500
    nknots = 50
    sparse_models_only = dict()
    for Z, (rlow, rhigh) in dense_model_Z.cutoffs.items():
        sparse_models_only[Z] = SplineModel(
            {'xknots': np.linspace(rlow, rhigh, nknots),
             'deg': 3,
             'bconds': 'vanishing'})

    map_models, sparse_model_Z = \
        map_linear_models_Z(sparse_models_only, dense_model_Z, ngrid)

    sparse_loss_func = sparse_loss_from_dense(map_models, dense_loss_func)

    c_sparse = sparse_loss_func.solve()
    sparse_loss = sparse_loss_func.value(c_sparse)
    print(f'sparse loss: rms loss {np.sqrt(sparse_loss) * HARTREE: .7f} kcal/mol')
    c_dense2 = dense_coefs_from_sparse(map_models, c_sparse)
    sparse_loss2 = dense_loss_func.value(c_dense2)
    print(f'sparse loss2: rms loss {np.sqrt(sparse_loss) * HARTREE: .7f} kcal/mol')

    # # MIO model
    # mio_model_Z = ModelZ({Z: MIOFunction(Z) for Z in dense_model_Z.models.keys()},
    #                      dense_model_Z.locs,
    #                      dense_model_Z.cutoffs)
    # xydata_mio = get_xydata_Z(mio_model_Z, np.zeros(mio_model_Z.nvars()))

    # coef_sparse = fit_model_to_xydata(sparse_modelZ, target_modelZ, coef_target, 500)

    # xydata_target = xydata_allZ(target_modelZ, coef_target)
    # xydata_sparse = xydata_allZ(sparse_modelZ, coef_sparse)

    # xydiff,err_summary, errors_allZ = diff_xydata_allZ(xydata_sparse, xydata_target)
    # for Z,err_dict in errors_allZ.items():
    #      print(f'xydiff {Z} {dict_to_string(err_dict,HARTREE)} kcal/mol')
    # print('fitting mio to 10 point spline')
    # print(f'xydiff mean over Z {dict_to_string(err_summary,HARTREE)} kcal/mol')

    # plot_xydata_allZ([xydata_target, xydata_sparse])
