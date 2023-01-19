import cvxopt
import numpy as np
from scipy.linalg import block_diag

from .options import Options
from .target import Constraint

cvxopt.solvers.options['show_progress'] = False


class Solver:
    def __init__(self, opts: Options):
        self.opts = opts
        self.solver = self.opts.solver
        self.constr = Constraint(self.opts.constr)

    def solve(self, alpha: np.ndarray, beta: np.ndarray, loc: dict, deriv: list) -> np.ndarray:
        r"""

        Args:
            alpha:
            beta:
            loc:
            deriv: list
                An list of derivative matrices, from zeroth order to the maximal order (max_der).

        Returns:

        """
        if self.opts.solver == 'lstsq':
            return np.linalg.lstsq(alpha, beta, rcond=None)[0]

        elif self.opts.solver == 'cvxopt':
            """
                Minimize      (1/2) x^T P x + q^T x
                subject to    G x <= h    (inequality constraints)
                              A x == b    (equality constraints)
            """
            P = alpha
            q = -beta

            # Construct inequality constraints (G and h)
            # Construct derivative matrix (D) of the bases
            # TODO: check the expressions. Currently the solver is complaining about
            #       ValueError: Rank(A) < p or Rank([P; A; G]) < n
            D = [op * deriv[der] for der, op in self.constr.items()]
            D_ref = np.zeros((len(loc['ref']), len(loc['ref'])))
            D = block_diag(*D, D_ref)
            # Construct transformation matrix (T) of the coefficient vector to support multiple constraints
            T = self.Tmatrix(loc, repeat_spl=len(self.constr), ignore_ref=True)
            # Construct G and h
            G = -D.dot(T)
            h = np.zeros(G.shape[0])

            # Convert matrices and vectors to cvxopt matrix
            P = cvxopt.matrix(P)
            q = cvxopt.matrix(q)
            G = cvxopt.matrix(G)
            h = cvxopt.matrix(h)

            # Solve for coefficient vector (x)
            res = cvxopt.solvers.qp(P, q, G, h)
            c = np.array(res['x']).flatten()
            return c

        else:
            raise NotImplementedError("Solver is not supported")

    @staticmethod
    def Tmatrix(loc: dict, repeat_spl: int, ignore_ref: bool = True) -> np.ndarray:
        r"""Generate a transformation matrix to repeat the spline part of the coefficient vector

        Args:
            loc: dict
                Location dictionary corresponding to the coefficient vector
            repeat_spl: int
                Number of repeats of the spline part in the coefficient vector, equal to the number of
                constraints to be applied. E.g. '+2' -> repeat_spl = 1, '+2-1' -> repeat_spl = 2, etc.
            ignore_ref: bool
                When set to True (common case), reference energies are set to zero after transformation

        Returns:

        """
        spl_len = len(loc['spline'])
        ref_len = len(loc['ref'])
        repeat_block = np.hstack([np.eye(spl_len), np.zeros((spl_len, ref_len))])
        if ignore_ref:
            ref_block = np.hstack([np.zeros((ref_len, spl_len + ref_len))])
        else:
            ref_block = np.hstack([np.zeros((ref_len, spl_len)), np.eye(ref_len)])
        # repeat [repeat_block] [repeat_spl] times, then append [ref_block]
        T = np.vstack([*([repeat_block] * repeat_spl), ref_block])
        return T


if __name__ == '__main__':
    coef = np.arange(8)
    loc = {'spline': np.arange(5), 'ref': np.arange(5, 8)}
    T = Solver.Tmatrix(loc, 2, True)
    print(T)
    print(T.dot(coef))
