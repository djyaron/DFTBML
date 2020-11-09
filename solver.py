import numpy as np
from scipy.optimize import minimize

"""
Originally part of DFTBrepulsive project. USE AS REFERENCE ONLY!
"""


class Solver:
    def __init__(self, modelZ, loss_func,
                 penalties=None, ngrid=500,
                 trim=None, show=False):
        """

        Parameters
        ----------
        modelZ : ModelZ
        loss_func
        penalties : dict
        ngrid : int
        trim : dict
        show : bool
        """
        self.modelZ = modelZ
        self.loss_func = loss_func
        self.Penalties = penalties
        self.alpha = loss_func.alpha
        self.beta = loss_func.beta
        self.n_configs_total = loss_func.nvars()
        self.loc = modelZ.locs.copy()
        self.loc['spline'] = np.concatenate([arr for Z, arr in self.loc.items() if Z != 'ref'])
        self.ngrid = ngrid
        self.trim = trim
        self.show = show
        self.Zs = modelZ.Zs()
        self.Xgrid = {
            Z: np.linspace(mod.xknots[0], mod.xknots[-1], self.ngrid)
            for Z, mod in self.modelZ.models.items()}
        self.Dgrid = {Z: np.array([mod.linear_model(self.Xgrid[Z], 0)[0],
                                   mod.linear_model(self.Xgrid[Z], 1)[0],
                                   mod.linear_model(self.Xgrid[Z], 2)[0]])
                      for Z, mod in self.modelZ.models.items()}
        self.c = None
        self.inflect_loc = None
        self.output = None
        self.smooth_enabled = False
        self.monotonic_enabled = False
        self.convex_enabled = False
        self.inflect_enabled = False
        # TODO: remove H-H and O-O
        self.remove_HH = False
        self.remove_OO = False

        self.penalty_check()

    def penalty_check(self):
        try:
            if self.Penalties['smooth']:
                self.smooth_enabled = True
        except KeyError:
            pass
        except TypeError:
            pass
        try:
            if self.Penalties['monotonic']:
                self.monotonic_enabled = True
        except KeyError:
            pass
        except TypeError:
            pass
        try:
            if self.Penalties['convex']:
                self.convex_enabled = True
        except KeyError:
            pass
        except TypeError:
            pass
        try:
            if self.Penalties['inflect']:
                self.inflect_enabled = True
        except KeyError:
            pass
        except TypeError:
            pass

    def get_monotonic_penalty(self, c):
        lambda_monotonic = self.Penalties['monotonic']
        monotonic_penalty = 0
        for Z in self.Zs:
            cZ = c[self.loc[Z]] #Just pulling out the coefficients for use
            deriv = self.Dgrid[Z][1]
            p_monotonic = np.einsum('j,ij->i', cZ, deriv) #Functionally equivalent to np.dot(cZ, deriv.T)
            p_monotonic[p_monotonic > 0] = 0
            monotonic_penalty += np.einsum('i,i->', p_monotonic, p_monotonic)
        monotonic_penalty *= lambda_monotonic * self.n_configs_total
        return monotonic_penalty

    def get_convex_penalty(self, c):
        lambda_convex = self.Penalties['convex']
        convex_penalty = 0
        for Z in self.Zs:
            cZ = c[self.loc[Z]]
            deriv = self.Dgrid[Z][2]
            p_monotonic = np.einsum('j,ij->i', cZ, deriv)
            p_monotonic[p_monotonic > 0] = 0
            convex_penalty += np.einsum('i,i->', p_monotonic, p_monotonic)
        convex_penalty *= lambda_convex * self.n_configs_total
        return convex_penalty

    def get_smooth_penalty(self, c):
        lambda_smooth = self.Penalties['smooth']
        smooth_penalty = 0
        for Z in self.Zs:
            cZ = c[self.loc[Z]]
            deriv = self.Dgrid[Z][2]
            p_smooth = np.einsum('j,ij->i', cZ, deriv)
            smooth_penalty += np.einsum('i,i->', p_smooth, p_smooth)
        smooth_penalty *= lambda_smooth * self.n_configs_total
        return smooth_penalty

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
        lambda_inflect = self.Penalties['inflect']
        inflect_penalty = 0
        for Z in self.Zs:
            cZ = c[self.loc[Z]]
            deriv = self.Dgrid[Z][2]
            xgrid = self.Xgrid[Z]
            xinflectZ = c[self.inflect_loc[Z]]
            p_inflect = np.einsum('j,ij->i', cZ, deriv)
            p_inflect[xgrid < xinflectZ] *= -1
            p_inflect[p_inflect > 0] = 0
            inflect_penalty += np.einsum('i,i->', p_inflect, p_inflect)
        inflect_penalty *= lambda_inflect * self.n_configs_total
        return inflect_penalty

    def get_loss(self, c):
        c_penalty = c.copy()
        if self.remove_HH:
            c_HH = np.zeros_like(self.loc[(1, 1)])
            self.c = np.concatenate([c_HH, self.c])
            c_penalty = self.c.copy()
        if self.remove_OO:
            c_OO = np.zeros_like(self.loc[(8, 8)])
            self.c = np.insert(self.c, self.loc[(8, 8)][0], c_OO)
            c_penalty = self.c.copy()

        c_spl = c_penalty[np.concatenate([self.loc['spline'], self.loc['ref']])]
        loss = np.einsum('i,ij,j->', c_spl, self.alpha, c_spl) - 2 * np.einsum('i,i->', self.beta, c_spl)

        if self.smooth_enabled:
            smooth_penalty = self.get_smooth_penalty(c_penalty)
            loss += smooth_penalty
        if self.monotonic_enabled:
            monotonic_penalty = self.get_monotonic_penalty(c_penalty)
            loss += monotonic_penalty
        if self.inflect_enabled:
            inflect_penalty = self.get_inflect_penalty(c_penalty)
            loss += inflect_penalty
        if self.convex_enabled:
            convex_penalty = self.get_convex_penalty(c_penalty)
            loss += convex_penalty

        return loss

    def solve_lstsq(self):
        self.c = np.linalg.lstsq(self.alpha, self.beta, rcond=None)[0]
        if self.inflect_enabled:
            self.get_inflect_loc()
            self.get_xinflect()
        if self.show:
            print("Initial guess solved by least squares solver.")

    def solve(self):
        self.solve_lstsq()
        if self.inflect_enabled + self.monotonic_enabled + self.smooth_enabled + self.convex_enabled:
            if self.show:
                print("Applying penalties...")

            # TODO: delete HH and OO
            if self.remove_HH + self.remove_OO:
                to_delete = list()
                if self.remove_HH:
                    to_delete.append(self.loc[(1, 1)])
                if self.remove_OO:
                    to_delete.append(self.loc[(8, 8)])
                to_delete = np.array(to_delete).flatten()
                self.c = np.delete(self.c, to_delete)
            self.output = minimize(self.get_loss, self.c)
            self.c = self.output.x
            print(self.output.message)

            if self.remove_HH:
                c_HH = np.zeros_like(self.loc[(1, 1)])
                self.c = np.concatenate([c_HH, self.c])
            if self.remove_OO:
                c_OO = np.zeros_like(self.loc[(8, 8)])
                self.c = np.insert(self.c, self.loc[(8, 8)][0], c_OO)
        return self.c[np.concatenate([self.loc['spline'], self.loc['ref']])]


