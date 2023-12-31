from __future__ import annotations

from typing import Union

import numpy as np
from scipy.linalg import block_diag

from .dataset import Dataset, Gammas
from .generator import Generator
from .options import Options
from .shifter import Shifter
from .skf import SKFSet
from .solver import Solver
from .spline import BSpline
from .util import padZ, Z2A, count_n_heavy_atoms, formatZ


# WARNING: dense grid to apply constraint is hardcoded
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

    def __call__(self, grid: Union[dict, np.ndarray]):
        r"""Evaluate models on a dense grid"""
        _grid = padZ(grid, self.Zs)
        return {Z: self.models[Z](gr)[0] for Z, gr in _grid.items()}

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
            targets = self.shifter.shift(dset, self.target, mode='-')
        else:
            targets = dset.extract(self.target)
        for mol in targets.mols():
            nHA = count_n_heavy_atoms(self.gammas[mol]['atomic_numbers'])
            target_mol = targets[mol][self.target] / nHA
            gamma_mol = self.gammas[mol]['gammas'] / nHA
            # target_mol = targets[mol][self.target]
            # gamma_mol = self.gammas[mol]['gammas']
            alpha += gamma_mol.T.dot(gamma_mol)
            beta += target_mol.T.dot(gamma_mol)
            const += target_mol.T.dot(target_mol)
        nconfs = targets.nconfs()
        self.alpha = alpha / nconfs
        self.beta = beta / nconfs
        self.const = const / nconfs
        # Solve for coef
        # Create spline models
        self.models = {}
        for Z in self.Zs:
            spl = BSpline(self.opts.xknots[Z], self.opts.bconds[Z], self.opts.deg[Z], self.opts.maxder)
            self.models.update({Z: spl})
        # Compute derivative matrices
        # WARNING: hardcoded constr_grid (dense grid to apply constraints)
        constr_grid = {Z: np.linspace(c[0], c[-1], 500) for Z, c in self.opts.cutoff.items()}
        deriv = {Z: spl(constr_grid[Z], bases_only=True) for Z, spl in self.models.items()}
        # Construct block diagonal derivative matrices
        deriv = [block_diag(*[d[i] for Z, d in deriv.items()]) for i in range(self.opts.maxder + 1)]
        # Solve for coef
        solver = Solver(self.opts)
        self.coef = solver.solve(self.alpha, self.beta, self.loc, deriv)
        for Z in self.Zs:
            self.models[Z].coef = self.coef[self.loc[Z]]

    def predict(self, dset: Dataset, gammas: Gammas = None, n_worker: int = 1) -> Dataset:
        _gammas = gammas if gammas is not None else Generator(dset, self.opts).gammas(n_worker)
        preds = _gammas * self.coef
        if self.shifter is not None:
            preds = self.shifter.shift(preds, 'prediction', mode='+')
        return preds

    def create_xydata(self, grid: Union[dict, np.ndarray], expand: bool = False) -> dict:
        r"""Generate a dictionary of grid-value pairs

        Args:
            grid: Union[dict, np.ndarray]
            expand: bool
                Generate xydata for Zs that Z[1] > Z[0].
                Set to True if the output xydata is used to generate SKFs

        Returns:
            dict
        """
        _grid = padZ(grid, self.Zs)
        _vals = self(_grid)
        _Zs = formatZ(self.Zs, unique=True, ordered=True, expand=expand)
        xydata = {Z: np.stack([_grid[tuple(sorted(Z))], _vals[tuple(sorted(Z))]], axis=1) for Z in _Zs}
        return xydata

    def ref_coef(self):
        r"""Retrieve the coefficients of the reference model

        Returns: dict

        """
        try:
            coef_ = self.coef[self.loc['ref']][1:]
        except KeyError:
            coef_ = np.zeros_like(self.atypes)
        try:
            intercept_ = self.coef[self.loc['const']]
        except KeyError:
            intercept_ = 0

        # Incorporate the coefficients of energy shifter
        if self.shifter is not None:
            coef_ += self.shifter.shifter.coef_.flatten()
            intercept_ += self.shifter.shifter.intercept_

        return {"coef": coef_, "intercept": intercept_}

    def ref_shift(self, dset: Dataset, target: str, mode: str = '-'):
        r"""Shift the target in the Dataset with trained reference energies

        Args:
            dset: Dataset
            target: str
            mode: str
                '+': returns the target plus the reference energies
                '-': returns the target minus the reference energies

        Returns: Dataset

        """
        ref_shifter = Shifter()
        ref_shifter.shifter.coef_, ref_shifter.shifter.intercept_ = \
            self.ref_coef().values()
        ref_shifter.atypes = self.atypes
        return ref_shifter.shift(dset, target, mode)

    def set_coef(self, coef: np.ndarray):
        r"""Set coefficient of RepulsiveModel and underlying splines"""
        self.coef = coef
        for Z in self.Zs:
            self.models[Z].coef = self.coef[self.loc[Z]]

    def get_ref_energy(self, atom_counts: np.ndarray) -> np.ndarray:
        r"""Predict the reference energy given atom counts

        Args:
            atom_counts: np.ndarray
                Rows: conformations; columns: number of atoms. Atom types must match self.atypes
        Returns:
            E_ref: np.ndarray:
                An array of reference energies (including the predictions of the shifter)

        """
        # Construct a shifter from self.coef
        ref_shifter = Shifter()
        ref_shifter.shifter.coef_, ref_shifter.shifter.intercept_ = \
            self.ref_coef().values()
        ref_shifter.atypes = self.atypes

        return ref_shifter.shifter.predict(atom_counts).flatten()

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
        # Location of spline coefficients
        for Z, nvar in loc_tmp.items():
            idx_end = idx_running + nvar
            self.loc[Z] = np.arange(idx_running, idx_end)
            idx_running = idx_end
        self.loc['spline'] = np.unique(list(self.loc.values()))
        # Location of reference energy coefficients
        if self.opts.ref != 'none':
            idx_ref_begin = idx_running
            if self.opts.ref == 'full':
                self.loc['const'] = np.array([idx_running])
                idx_running += 1
            # Locations of atom counts
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
