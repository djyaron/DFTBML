from .model_v1 import LinearLoss, ModelZ
from scipy.linalg import block_diag

import cvxopt
import numpy as np


# TODO: consider packing alpha, beta, loc, etc into a single class,
#       then passing that class to Solver
# TODO: see Solver.solve
class Solver:
    def __init__(self, modelZ: ModelZ, loss_func: LinearLoss,
                 pens: dict = None, ngrid: int = 500,
                 show: bool = False) -> None:
        r"""Solve for the coefficient vector given a series of models and their loss function

        Args:
            modelZ (ModelZ): a series of models associated with pairwise interactions
            loss_func (LinearLoss): linear loss function corresponding to modelZ
            pens (dict): Penalty dictionary whose keys are penalty types and values are penalty values.
                1. Penalty dictionary is ignored when using least squares solver (lstsq).
                2. Penalties are treated as constraints and penalty values are ignored when using convex
                    solver (cvxopt)
                3. Available options:
                    a) "convex": enforcing positive second derivative
                    b) "monotonic": enforcing negative first derivative
                    c) "monoconv": applying "convex" and "monotonic" at the same time
            ngrid (int): number of grid points to evaluate the model and to apply constraints.
                Ignored when using least squares solver (lstsq)
            show (bool): show the current status of calculation

        Returns:
            None

        Examples:
            >>> from h5py import File
            >>> from deprecated.model_v1 import *
            >>> from fold import Fold
            >>> dataset_path, gammas_path = "dataset.h5", "gammas.h5"
            >>> target_type = "some_target"
            >>> with File(dataset_path,'r') as dataset:
            ...     data_fold = Fold.from_dataset(dataset)
            >>> dense_model_Z, dense_loss_func = create_dense(target_type, data_fold, dataset_path, gammas_path)
            >>> pens = {"convex": None}
            >>> solver = Solver(dense_model_Z, dense_loss_func, pens, ngrid=500, show=False)
            >>> c = solver.solve('cvxopt')
        """
        self.modelZ = modelZ
        self.Penalties = pens
        self.ngrid = ngrid
        self.show = show
        cvxopt.solvers.options['show_progress'] = self.show

        self.Zs = modelZ.Zs()  #: pairwise interactions in modelZ
        #: loc (dict): location of coefficients in the coefficient vector
        #:     corresponding to each pairwise interaction
        self.loc = modelZ.locs.copy()
        self.loc['spline'] = np.concatenate([arr for Z, arr in self.loc.items() if Z != 'ref'])
        self.loc['all'] = np.concatenate([self.loc['spline'], self.loc['ref']])
        #: Xgrid (dict): dense grid to evaluate each model and to apply constraints,
        #:     equidistantly distributed in the range spanned by the knots of each model
        self.Xgrid = {Z: np.linspace(mod.xknots[0], mod.xknots[-1], self.ngrid)  #: equidistant grid
                      for Z, mod in self.modelZ.models.items()}
        #: Dgrid (dict): zeroth to second derivative of each model evaluated on Xgrid
        self.Dgrid = {Z: np.array([mod.linear_model(self.Xgrid[Z], d)[0] for d in range(3)])
                      for Z, mod in self.modelZ.models.items()}
        self.alpha = loss_func.alpha
        self.beta = loss_func.beta
        self.c = None  #: coefficient vector, which will be solved later

    def solve(self, solver: str = 'cvxopt') -> np.ndarray:
        r"""Solve for the coefficient vector

        Args:
            solver (str): Available option:
                "lstsq": Least squares solver by np.linalg.lstsq
                "cvxopt": Quadratic programming solver by cvxopt.solvers.qp

        Returns:
            c (np.array): coefficient vector

        Raises:
            NotImplementedError: when available constraints and solvers are not
                chosen in the output.

        Todo:
            1) Point the data structure of the coefficient vector to our paper
            2) Scalable constraints? "-1" for monotonic decreasing, "+2" for
                convex? Allow multiple constraints simultaneously? A good starting
                point is the monoconv constraint. Also the order of Dgrid can also be
                dependent on the constraint specified.
        """
        if solver == 'lstsq':
            """
                Solve         A * c = b
            """
            self.c = np.linalg.lstsq(self.alpha, self.beta, rcond=None)[0]
            if self.show:
                print("Initial guess solved by least squares solver.")

        elif solver == 'cvxopt':
            """
                Minimize      (1/2) x^T P x + q^T x
                subject to    G x <= h    (inequality constraints)
                              A x == b    (equality constraints)
            """

            if 'monotonic' in self.Penalties.keys():
                """
                    Monotonic decreasing potentials by enforcing negative first derivatives 
                """
                P = self.alpha
                q = - self.beta
                D_list = [self.Dgrid[Z][1] for Z in self.Zs]
                D_list.append(np.zeros((len(self.loc['ref']), len(self.loc['ref']))))
                G = block_diag(*D_list)  # negative first derivatives
                h = np.zeros(G.shape[0])

                # convert to cvxopt matrix
                P = cvxopt.matrix(P)
                q = cvxopt.matrix(q)
                G = cvxopt.matrix(G)
                h = cvxopt.matrix(h)

                res = cvxopt.solvers.qp(P, q, G, h)
                self.c = np.array(res['x']).flatten()

            elif 'convex' in self.Penalties.keys():
                """
                    Convex potentials by enforcing positive second derivatives
                """
                P = self.alpha
                q = - self.beta
                D_list = [self.Dgrid[Z][2] for Z in self.Zs]
                D_list.append(np.zeros((len(self.loc['ref']), len(self.loc['ref']))))
                G = - block_diag(*D_list)  # positive second derivatives
                h = np.zeros(G.shape[0])

                # convert to cvxopt matrix
                P = cvxopt.matrix(P)
                q = cvxopt.matrix(q)
                G = cvxopt.matrix(G)
                h = cvxopt.matrix(h)

                res = cvxopt.solvers.qp(P, q, G, h)
                self.c = np.array(res['x']).flatten()

            elif 'monoconv' in self.Penalties.keys():
                """
                    Monotonic decreasing and convex potentials, by enforcing
                    positive second derivatives and negative first derivatives
                    at the same time
                """
                P = self.alpha
                q = - self.beta

                # Concatenated derivative matrices
                D_list = [self.Dgrid[Z][1] for Z in self.Zs]  # first derivative matrix
                D_list.extend([-self.Dgrid[Z][2] for Z in self.Zs])  # second derivative matrix
                D_list.append(np.zeros((len(self.loc['ref']), len(self.loc['ref']))))  # ref matrix
                D = block_diag(*D_list)

                # Transformation matrix: [spline ref] --> [spline spline ref]
                T_spl = np.eye(len(self.loc['spline']))  # shape: (spline, spline)
                T_ref = np.zeros((len(self.loc['ref']), len(self.loc['ref'])))  # (ref, ref)
                T_lower = block_diag(T_spl, T_ref)  # (spline + ref, spline + ref)
                T_upper = np.hstack([T_spl, np.zeros((len(self.loc['spline']), len(self.loc['ref'])))])  # (spline, spline + ref)
                T = np.vstack([T_upper, T_lower])  # (spline + spline, spline + ref)

                G = D.dot(T)
                h = np.zeros(G.shape[0])

                # convert to cvxopt matrix
                P = cvxopt.matrix(P)
                q = cvxopt.matrix(q)
                G = cvxopt.matrix(G)
                h = cvxopt.matrix(h)

                res = cvxopt.solvers.qp(P, q, G, h)
                self.c = np.array(res['x']).flatten()

            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        return self.c
