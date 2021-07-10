from __future__ import annotations

from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import block_diag

from .consts import HARTREE
from .dataset import Dataset, Gammas
from .generator import Generator
from .options import Options
from .shifter import Shifter
from .skf import SKFSet
from .solver import Solver
from .spline import BSpline
from .util import padZ, Z2A


## WARNING: dense grid to apply constraint is hardcoded
class RepulsiveModel:
    def __init__(self, opts: Options):
        self.opts = opts
        self.Zs = self.opts.Zs
        self.atypes = Z2A(self.Zs)
        # Attributes below are determined when self.fit() is called
        self.loc = None
        self.alpha = None
        self.beta = None
        self.const = None
        self.gammas = None
        self.coef = None
        self.models = None
        self.target = None
        self.shifter = None

    def fit(self, dset: Dataset, target: str, gammas: Gammas = None,
            shift: bool = True, n_worker: int = 1, **kwargs):
        r"""

        Args:
            dset:
            target:
            gammas:
            shift:
            n_worker:
            **kwargs: dict
                'fit_intercept': bool = True
        Returns:

        """
        self.target = target
        # Compute gammas if not specified
        self.gammas = Generator(dset, self.opts).gammas(n_worker) if gammas is None else gammas
        self.loc = self.gammas.loc
        # Compute alpha, beta and const
        nvars = self.gammas.nvars()
        alpha = np.zeros((nvars, nvars))
        beta = np.zeros(nvars)
        const = 0
        if shift:
            self.shifter = Shifter(kwargs.get('fit_intercept', True))
            self.shifter.fit(dset, self.target)
            targets = self.shifter.shift(dset, self.target)
        else:
            targets = dset.extract(self.target)
        for mol in targets.mols():
            target_mol = targets[mol][self.target]
            gamma_mol = self.gammas[mol]['gammas']
            alpha += gamma_mol.T.dot(gamma_mol)
            beta += target_mol.T.dot(gamma_mol)
            const += target_mol.T.dot(target_mol)
        nconfs = targets.nconfs()
        self.alpha = alpha / nconfs
        self.beta = beta / nconfs
        self.const = const / nconfs
        # Solve for coef
        ## Create spline models
        self.models = {}
        for Z in self.Zs:
            spl = BSpline(self.opts.xknots[Z], self.opts.bconds[Z], self.opts.deg[Z], self.opts.maxder)
            self.models.update({Z: spl})
        ## Compute derivative matrices
        ## WARNING: hardcoded constr_grid (dense grid to apply constraints)
        constr_grid = {Z: np.linspace(c[0], c[-1], 500) for Z, c in self.opts.cutoff.items()}
        deriv = {Z: spl(constr_grid[Z], bases_only=True) for Z, spl in self.models.items()}
        ## Construct block diagonal derivative matrices
        deriv = [block_diag(*[d[i] for Z, d in deriv.items()]) for i in range(self.opts.maxder + 1)]
        ## Solve for coef
        solver = Solver(self.opts)
        self.coef = solver.solve(self.alpha, self.beta, self.loc, deriv)
        for Z in self.Zs:
            self.models[Z].coef = self.coef[self.loc[Z]]

    def predict(self, dset: Dataset, gammas: Gammas = None, n_worker: int = 1) -> Dataset:
        _gammas = gammas if gammas is not None else Generator(dset, self.opts).gammas(n_worker)
        preds = _gammas * self.coef
        if self.shifter is not None:
            preds = self.shifter.shift(preds, 'prediction', reverse=True)
        return preds

    def __call__(self, grid: Union[dict, np.ndarray]):
        r"""Evaluate models on a dense grid"""
        _grid = padZ(grid, self.Zs)
        # TODO: refactor this expression when spline.BSpline.__call__ is modified
        return {Z: self.models[Z](gr)[0] for Z, gr in _grid.items()}

    # TODO: refactor
    def plot(self, grid: Union[dict, np.ndarray]):
        _grid = padZ(grid, self.Zs)
        vals = self(_grid)
        res = {}
        for Z in self.Zs:
            x = _grid[Z]
            y = vals[Z]
            res[Z] = (x, y)
            plt.plot(x, y * HARTREE)
            plt.ylabel('kcal/mol')
            plt.xlabel('Angstrom')
            plt.title(Z)
            plt.ylim(0, 20)
            plt.show()
        return res

    def to_skf(self, skfdir_path, **kwargs):
        raise NotImplementedError

    def from_skf(self, skfdir_path: str, ngrid: int):
        # Determine Zs and atypes
        _skfset = SKFSet.from_dir(skfdir_path)
        _Zs = _skfset.Zs(ordered=False)
        _atypes = _skfset.atypes()
        # Load SKF
        # NOTE: use equidistant grid defined on the range of SKF repulsive
        coef_tmp = {}
        loc_tmp = {}
        for Z in _Zs:
            _skf = _skfset[Z]
            _grid = np.linspace(*_skf.range('R'), ngrid)
            _vals = _skf('R', _grid)['R']
            spl = BSpline(self.opts.xknots[Z], self.opts.bconds[Z], self.opts.deg[Z], self.opts.maxder)
            coef = spl.fit(_grid, _vals)
            coef_tmp[Z] = coef
            loc_tmp[Z] = len(coef)

        # Create self.loc
        idx_running = 0
        self.loc = {}
        ## Location of spline coefficients
        for Z, nvar in loc_tmp.items():
            idx_end = idx_running + nvar
            self.loc[Z] = np.arange(idx_running, idx_end)
            idx_running = idx_end
        self.loc['spline'] = np.unique(list(self.loc.values()))
        ## Location of reference energy coefficients
        if self.opts.ref != 'none':
            idx_ref_begin = idx_running
            if self.opts.ref == 'full':
                self.loc['const'] = np.array([idx_running])
                idx_running += 1
            ### Locations of atom counts
            for i, _atype in enumerate(_atypes):
                self.loc[_atype] = np.array([idx_running + i])
            idx_running += len(self.opts.atypes)
            idx_ref_end = idx_running
            self.loc['ref'] = np.arange(idx_ref_begin, idx_ref_end)
        else:
            self.loc['ref'] = np.array([])
        # Create self.coef
        self.coef = [coef_tmp[Z] for Z in _Zs]
        self.coef.append(np.zeros_like(self.loc['ref']))
        self.coef = np.concatenate(self.coef)


if __name__ == '__main__':
    pass