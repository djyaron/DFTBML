from __future__ import annotations

import re

from consts import CUTOFFS, BCONDS
from tfspline import Bcond
from typing import ItemsView, ValuesView, KeysView, Union, Iterable
from util import formatZ, padZ, Z2A

import numpy as np
import pickle as pkl


# TODO: support various types of models (current version only supports b-splines)
class Options:
    def __init__(self, opts: dict) -> None:
        r"""Hyperparameters of repulsive model

        Args:
            opts: dict
                 xknots: Union[list, tuple, np.ndarray, dict]
                    Knot sequence. When xknots is specified, nknots, dknots and cutoff are ignored.
                 nknots: Union[int, dict]
                    Number of knots. When nknots is specified, dknots is ignored.
                 dknots: Union[float, dict]
                    Knot density or the interval between adjacent knots.
                 cutoff: Union[list, tuple, np.ndarray, str, dict]
                    Closed interval of knot positions.
                 deg: Union[int, dict]
                    Degree of splines
                 bconds: Union[list, tuple, str, dict]
                    Boundary conditions
                 constr (optional): str
                    Apply convex constraint (positive second derivatives) by default
                 solver (optional): str
                    Available options: 'lstsq', 'cvxopt'
                    Use 'cvxopt' by default
                 ref (optional): str
                    Fit reference energy simultaneously. Default: 'full'
                    Available options: 'none', 'full', 'no_const'
                    Use 'full' by default
        """
        self.xknots = None
        self.nknots = None
        self.dknots = None
        self.cutoff = None
        self.deg = None
        self.bconds = None
        self.constr = '+2'
        self.solver = 'cvxopt'
        self.ref = 'full'
        # Attributes below are determined when Options.check() is called
        self.maxder = None  # highest order of derivatives of the splines to be calculated
        self.Zs = None  # atom pairs
        self.atypes = None  # atomic numbers
        self.__dict__.update(opts)
        self.check()

    def __getitem__(self, opt: str) -> Union[list, dict, tuple, str, int, np.ndarray]:
        return self.__dict__[opt]

    def __setitem__(self, opt: str, opt_val: Union[list, dict, tuple, str, int, np.ndarray]) -> None:
        self.__dict__[opt] = opt_val

    def __delitem__(self, opt: str) -> None:
        del self.__dict__[opt]

    def check(self) -> None:
        r"""Check the validity of parameters"""
        # Sort atom pairs in the parameters
        opts_tmp = {}
        for opt, opt_val in self.items():
            if isinstance(opt_val, dict):
                opts_tmp[opt] = {tuple(sorted(Z)): val for Z, val in opt_val.items()}
            else:
                opts_tmp[opt] = opt_val
        self.__dict__.update(opts_tmp)
        # Check the validity of parameters
        ## Check xknots, nknots, dknots and cutoff
        ## If xknots is not specified, check the validity of nknots and cutoff
        if self.xknots is None:
            ### Check nknots
            if self.nknots is None:
                #### If nknots is not specified, check the validity of dknots
                if not isinstance(self.dknots, (float, dict)):
                    raise TypeError(f"dknots is invalid")
            elif not isinstance(self.nknots, (int, dict)):
                raise TypeError(f"nknots is invalid")
            ### Check cutoff
            if not isinstance(self.cutoff, (str, list, tuple, np.ndarray, dict)):
                raise TypeError(f"cutoff is invalid")
            if isinstance(self.cutoff, str):
                if not self.cutoff in CUTOFFS.keys():
                    raise ValueError(f"cutoff is not recognized")
        else:
            if not isinstance(self.xknots, (list, tuple, np.ndarray, dict)):
                raise TypeError("xknots is invalid")
        ## Check other parameters
        ### Check deg
        if not isinstance(self.deg, (int, dict)):
            raise TypeError(f"deg is invalid")
        ### Check bconds
        if not isinstance(self.bconds, (str, list, tuple, dict)):
            raise TypeError(f"bconds is invalid")
        if isinstance(self.bconds, str):
            if not self['bconds'] in BCONDS.keys():
                raise ValueError(f"bconds is not recognized")
        ### Check constr
        if not isinstance(self.constr, str):
            raise TypeError(f"constr is invalid")
        ### Check solver
        if not isinstance(self.solver, str):
            raise TypeError(f"solver is invalid")
        ### Determine der
        try:
            self.maxder = max([int(i) for i in re.findall(r'[0-9]', self.constr)])
        except ValueError:
            raise ValueError(f"constr is invalid")
        ### Check ref
        if not isinstance(self.ref, str):
            raise TypeError(f"ref is invalid")
        if self.ref not in ('none', 'full', 'no_const'):
            raise ValueError(f"ref is not recognized")
        # Infer atom pairs (Zs) from input parameters
        if self.Zs is None and isinstance(self.cutoff, str):
            self.Zs = formatZ(CUTOFFS[self.cutoff].keys())
        for opt in ('xknots', 'nknots', 'dknots', 'cutoff', 'deg', 'bconds'):
            if isinstance(self.get(opt), dict):
                if self.Zs is None:
                    self.Zs = formatZ(self[opt].keys(), unique=True)
                else:
                    if self.Zs != formatZ(self[opt].keys(), unique=True):
                        raise ValueError("Atom pairs in parameters do not match")
        # Determine atom types when Zs is determined
        if self.Zs is not None:
            self.atypes = Z2A(self.Zs)

    def convert(self, Zs: Iterable) -> Options:
        r"""Convert current options to a detailed format based on atom pairs specified

        Args:
            Zs: Iterable
                Atom pairs
        """
        # Check if Zs matches self.Zs
        _Zs = formatZ(Zs)
        if self.Zs is not None and self.Zs != _Zs:
            raise ValueError("Atom pairs (Zs) do not match")

        res_opts = {}
        # Compute xknots from nknots/dknots and cutoff if necessary
        if self.xknots is None:  # then nknots/dknots and cutoff must have been specified
            if isinstance(self.cutoff, str):
                res_opts['cutoff'] = {Z: CUTOFFS[self.cutoff][Z] for Z in _Zs}
            else:
                res_opts['cutoff'] = padZ(self.cutoff, _Zs)
            if self.get('nknots') is None:
                res_opts['dknots'] = padZ(self.dknots, _Zs)
                res_opts['xknots'] = {Z: np.arange(res_opts['cutoff'][Z][0], res_opts['cutoff'][Z][-1], res_opts['dknots'][Z])
                                      for Z in _Zs}
            else:
                res_opts['nknots'] = padZ(self.nknots, _Zs)
                res_opts['xknots'] = {Z: np.linspace(res_opts['cutoff'][Z][0], res_opts['cutoff'][Z][-1], res_opts['nknots'][Z])
                                      for Z in _Zs}
        else:
            res_opts['xknots'] = padZ(self.xknots, _Zs)
        # Convert deg
        res_opts['deg'] = padZ(self.deg, _Zs)
        # Convert bconds
        if self.bconds in BCONDS.keys():
            res_opts['bconds'] = padZ(BCONDS[self.bconds], _Zs)
        else:
            res_opts['bconds'] = padZ(self.bconds, _Zs)
        # Copy constr, solver and der
        res_opts.update({opt: self[opt] for opt in ('constr', 'solver', 'maxder')})
        # Copy Zs and determine atypes
        res_opts['Zs'] = _Zs
        res_opts['atypes'] = Z2A(res_opts)

        return Options(res_opts)

    def get(self, opt: str) -> Union[dict, tuple, str, np.ndarray, Bcond, None]:
        return self.__dict__.get(opt)

    def items(self) -> ItemsView:
        return self.__dict__.items()

    def keys(self) -> KeysView:
        return self.__dict__.keys()

    def save(self, opts_path: str) -> None:
        with open(opts_path, 'wb') as f:
            pkl.dump(self, f)

    def values(self) -> ValuesView:
        return self.__dict__.values()

    @classmethod
    def load(cls, opts_path: str) -> Options:
        with open(opts_path, 'rb') as f:
            return pkl.load(f)
