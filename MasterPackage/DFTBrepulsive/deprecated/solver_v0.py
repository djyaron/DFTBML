import cvxopt
import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import minimize


class Solver:
    def __init__(self, modelZ, loss_func,
                 pens=None, ngrid=500,
                 show=False):
        self.modelZ = modelZ
        self.loss_func = loss_func
        self.Penalties = pens
        self.ngrid = ngrid
        self.show = show
        cvxopt.solvers.options['show_progress'] = self.show
        self.Zs = modelZ.Zs()
        self.loc = modelZ.locs.copy()
        self.loc['spline'] = np.concatenate([arr for Z, arr in self.loc.items() if Z != 'ref'])
        self.loc['all'] = np.concatenate([self.loc['spline'], self.loc['ref']])
        self.Xgrid = {Z: np.linspace(mod.xknots[0], mod.xknots[-1], self.ngrid)  # equidistant grid
                      for Z, mod in self.modelZ.models.items()}
        self.Dgrid = {Z: np.array([mod.linear_model(self.Xgrid[Z], d)[0] for d in range(3)])
                      for Z, mod in self.modelZ.models.items()}
        self.alpha = loss_func.alpha
        self.beta = loss_func.beta
        self.enabled = [ptype for ptype, p in pens.items() if p] if pens else []
        self.c = None
        self.inflect_loc = None

    def get_monotonic_penalty(self, c):
        coef = self.Penalties['monotonic']
        penalty = 0
        for Z in self.Zs:
            cZ = c[self.loc[Z]]
            deriv = self.Dgrid[Z][1]
            p = cZ.dot(deriv)
            p = np.clip(p, None, 0)
            penalty += p.dot(p)
        penalty *= coef
        return penalty

    def get_convex_penalty(self, c):
        coef = self.Penalties['convex']
        penalty = 0
        for Z in self.Zs:
            cZ = c[self.loc[Z]]
            deriv = self.Dgrid[Z][2]
            p = cZ.dot(deriv)
            p = np.clip(p, None, 0)
            penalty += p.dot(p)
        penalty *= coef
        return penalty

    def get_smooth_penalty(self, c):
        coef = self.Penalties['smooth']
        penalty = 0
        for Z in self.Zs:
            cZ = c[self.loc[Z]]
            deriv = self.Dgrid[Z][2]
            p = cZ.dot(deriv)
            penalty += p.dot(p)
        penalty *= coef
        return penalty

    def get_inflect_loc(self):
        self.inflect_loc = dict()
        idx_inflect_b = self.loc['ref'][-1] + 1
        for iZ, Z in enumerate(self.Zs):
            self.inflect_loc[Z] = np.array([idx_inflect_b + iZ])
        idx_inflect_e = idx_inflect_b + len(self.Zs)
        self.inflect_loc['inflect'] = np.arange(idx_inflect_b, idx_inflect_e)

    def get_xinflect(self):
        c = np.zeros(self.inflect_loc['inflect'][-1] + 1)
        c[self.loc['spline']] = self.c[self.loc['spline']]
        c[self.loc['ref']] = self.c[self.loc['ref']]
        for Z in self.Zs:
            xgrid = self.Xgrid[Z]
            xinflect = (xgrid[0] + xgrid[-1]) / 2
            c[self.inflect_loc[Z]] = xinflect
        self.c = c

    def get_inflect_penalty(self, c):
        coef = self.Penalties['inflect']
        penalty = 0
        for Z in self.Zs:
            cZ = c[self.loc[Z]]
            deriv = self.Dgrid[Z][2]
            xgrid = self.Xgrid[Z]
            xinflectZ = c[self.inflect_loc[Z]]
            p = cZ.dot(deriv)
            p[xgrid < xinflectZ] *= -1
            p = np.clip(p, None, 0)
            penalty += p.dot(p)
        penalty *= coef
        return penalty

    def get_loss(self, c):
        c_penalty = c

        c_spl = c_penalty[self.loc['all']]
        loss = c_spl.T.dot(self.alpha).dot(c_spl) - 2 * self.beta.dot(c_spl)

        if 'monotonic' in self.enabled:
            monotonic_penalty = self.get_monotonic_penalty(c_penalty)
            loss += monotonic_penalty
        if 'convex' in self.enabled:
            convex_penalty = self.get_convex_penalty(c_penalty)
            loss += convex_penalty
        if 'smooth' in self.enabled:
            smooth_penalty = self.get_smooth_penalty(c_penalty)
            loss += smooth_penalty
        if 'inflect' in self.enabled:
            inflect_penalty = self.get_inflect_penalty(c_penalty)
            loss += inflect_penalty

        return loss

    def solve(self, solver='cvxopt'):
        if solver == 'lstsq':
            self.c = np.linalg.lstsq(self.alpha, self.beta, rcond=None)[0]
            if 'inflect' in self.enabled:
                self.get_inflect_loc()
                self.get_xinflect()
            if self.show:
                print("Initial guess solved by least squares solver.")

        elif solver == 'bfgs':
            self.solve('lstsq')
            res = minimize(self.get_loss, self.c)
            self.c = res.x
            if self.show:
                print(res.message)

        elif solver == 'slsqp':
            if 'convex' in self.Penalties.keys():
                G_list = [self.Dgrid[Z][2] for Z in self.Zs]
                G_list.append(np.zeros((len(self.loc['ref']), len(self.loc['ref']))))
                G = block_diag(*G_list)
            elif 'monotonic' in self.Penalties.keys():
                G_list = [self.Dgrid[Z][1] for Z in self.Zs]
                G_list.append(np.zeros((len(self.loc['ref']), len(self.loc['ref']))))
                G = - block_diag(*G_list)
            else:
                raise NotImplementedError

            loss = lambda x: x.T.dot(self.alpha).dot(x) - 2 * self.beta.dot(x)
            jac = lambda x: 2 * self.alpha.dot(x) - 2 * self.beta
            x0 = self.solve('lstsq')
            cons = {'type': 'ineq',
                    'fun': lambda x: G.dot(x),
                    'jac': lambda x: G}

            res = minimize(fun=loss, jac=jac, x0=x0, constraints=cons, method='SLSQP')
            self.c = res.x
            if self.show:
                print(res.message)

        elif solver == 'cvxopt':
            """
                Minimize      (1/2) x^T P x + q^T x
                subject to    gammas x <= h
                              A x == b
            """
            if 'monotonic' in self.Penalties.keys():
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
                P = self.alpha
                q = - self.beta

                # Concatenated derivative matrices
                D_list = [self.Dgrid[Z][1] for Z in self.Zs]  # first derivative matrix
                D_list.extend([-self.Dgrid[Z][2] for Z in self.Zs])  # second derivative matrix
                D_list.append(np.zeros((len(self.loc['ref']), len(self.loc['ref']))))  # ref matrix
                D = block_diag(*D_list)

                # Transformation matrix: [spline ref] --> [spline spline ref]
                T_spl = np.eye(len(self.loc['spline']))  # (spline, spline)
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

            elif 'inflect' in self.Penalties.keys():
                """
                    Inflection point    xinf (the iinf-th grid point specified by ngrid)
                    Constraints         f''(x) <= 0     x <  xinf
                                        f''(x) >= 0     x >= xinf
                """
                iinf = self.Penalties['inflect']  # iinf is a dict with Zs as keys and int as values
                P = self.alpha
                q = - self.beta
                D_list = [self.Dgrid[Z][2] for Z in self.Zs]
                D_list.append(np.zeros((len(self.loc['ref']), len(self.loc['ref']))))
                M_list = [np.concatenate([np.ones(iinf[Z]), -np.ones(self.ngrid - iinf[Z])]) for Z in self.Zs]
                M_list.append(np.ones(len(self.loc['ref'])))
                G = np.diag(np.concatenate(M_list)).dot(block_diag(*D_list))  # convex on x < xinf, concave on x >= xinf
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
        # return self.c[self.loc['all']]
