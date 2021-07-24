from __future__ import annotations

import os
import re
from typing import ItemsView, KeysView, ValuesView, Iterable, Tuple, Union, Dict

import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.interpolate import PPoly, interp1d, CubicSpline

from .consts import ANGSTROM2BOHR, ATOM2SYM, SYM2ATOM
from .util import formatZ, path_check, Z2A

SKF_BLOCKS = {'header': ('comment', 'gridDist', 'nGridPoints', 'atomic_info', 'poly_info'),
              'HS': ('H', 'S'),
              'spline': ('nInt', 'cutoff', 'exp_coef', 'cub_coef', 'fif_coef'),
              'doc': ('doc',)}


class SKF:
    def __init__(self, skf_data: dict):
        r"""Handler of single Slater-Koster file

        Args:
            skf_data: dict
                SKF data dictionary, whose keys are the following class attributes

        Notes:
            Length unit: Bohr
            Energy unit: Hartree
        """
        # Basic information
        self.homo = None
        self.extend = None
        self.Z = None
        # Header block
        self.comment = None
        self.gridDist = None
        self.nGridPoints = None
        self.atomic_info = None
        self.poly_info = None
        # Hamiltonian (H) and overlap integral (S) block
        self.H = None
        self.S = None
        # Spline block
        self.nInt = None
        self.cutoff = None
        self.exp_coef = None
        self.cub_coef = None
        self.fif_coef = None
        # Documentation (doc) block
        self.doc = None
        # Update class attributes with given SKF data
        self.__dict__.update(skf_data)

    def __getitem__(self, entry: str):
        r"""Look up SKF data

        Args:
            entry: str
                Supported entries:
                    SKF major blocks
                        'header'
                        'HS'
                        'spline'
                        'doc'
                    Dense grids defined in the SKF
                        'intGrid'
                        'splGrid'
                    Atomic information
                        'E': On-site energies for angular momenta
                        'E?': E for a specific angular momentum, e.g. 'Es'
                        'SPE': Spin polarization error
                        'U': Hubbard U for angular momenta
                        'U?': U for a specific angular momentum, e.g. 'Up'
                        'f': Occupations for angular momenta
                        'f?': Occupation for a specific angular momentum, e.g. 'fd'
                        'mass': atomic mass
                    Other entries
                        'homo':
                        'extend':
                        'Z':
                        'comment':
                        'gridDist':
                        'nGridPoints':
                        'atomic_info':
                        'poly_info':
                        'H':
                        'S':
                        'nInt':
                        'cutoff':
                        'exp_coef':
                        'cub_coef':
                        'fif_coef':

        Raises:
            KeyError
        """
        # SKF major blocks (header, HS, spline and doc)
        if entry in SKF_BLOCKS.keys():
            return {e: self.__dict__[e] for e in SKF_BLOCKS[entry]}
        # Dense grids defined in the SKF
        elif entry == 'intGrid':
            return {entry: self.intGrid()}
        elif entry == 'splGrid':
            return {entry: self.splGrid()}
        else:
            # Atomic information (energies, SPE, Hubbard U, orbital occupations and atomic mass)
            try:
                if re.match(r'^[EfU]$', entry):
                    cols = [ent for ent in self.atomic_info.columns if re.match(r'^' + entry + r'[a-z]$', ent)]
                    return self.atomic_info[cols]
                elif re.match(r'^[EfU][a-z]$', entry):  # e.g. 'Es', 'fd', 'Up'
                    return self.atomic_info[entry].value
                elif entry == 'SPE':
                    return self.atomic_info[entry].value
                else:
                    pass
            except (AttributeError, TypeError):
                raise KeyError(entry)
            if entry == 'mass':
                return self.poly_info[entry].value
        return self.__dict__[entry]

    def __setitem__(self, entry: str, value):
        if entry in SKF_BLOCKS.keys():
            self.__dict__.update(value)
        else:
            self.__dict__.update({entry: value})

    def __delitem__(self, entry: str):
        if entry in SKF_BLOCKS.keys():
            for e in SKF_BLOCKS[entry]:
                del self.__dict__[e]
        else:
            del self.__dict__[entry]

    def __call__(self, entry: str, grid: np.ndarray = None) -> dict:
        r"""Look up or evaluate H, S and R on a grid

        Args:
            entry: str
                Supported entries:
                    Integral tables
                        NOTE: Interpolate using cubic spline if a grid is given
                        'H': Hamiltonian matrix elements
                        'H???': H for a specific two-center interaction, e.g. 'Hss0' (ss-sigma)
                        'S': Overlap matrix elements
                        'S???': S for a specific two-center interaction, e.g. 'Sff3' (ff-phi)
                    Repulsive potential
                        NOTE: Evaluate on a given grid (must be specified)
                        'R': Repulsive potential

            grid: np.ndarray
                Evaluate repulsive potential on a grid

        Returns:
            res: dict

        Raises:
            ValueError: when requesting 'R' without specifying a dense grid
        """
        res = {}
        # Integral tables
        if re.match(r'^[HS]$', entry):
            if grid is None:
                res.update({entry: self[entry]})
            else:
                interpRes = {}
                # Out-of-range grid points
                int_start = self.intGrid()[0]
                int_end = self.intGrid()[-1]
                interpGrid = grid[(int_start <= grid) & (grid <= int_end)]
                underGrid = grid[grid < int_start]
                overGrid = grid[grid > int_end]
                underVal = np.empty_like(underGrid) * np.nan
                overVal = np.empty_like(overGrid) * np.nan
                for ent, val in self[entry].items():
                    interpFunc = interp1d(x=self.intGrid(), y=val, kind='cubic')
                    interpVal = np.concatenate([underVal, interpFunc(interpGrid), overVal])
                    interpRes.update({ent: interpVal})
                res.update({entry: pd.DataFrame(interpRes)})
        elif re.match(r'^[HS][a-z]{2}\d$', entry):  # e.g. 'Hpd1', 'Sff3'
            if grid is None:
                res.update({entry: self[entry[0]][entry]})  # e.g. 'Hpd1' -> self['H']['Hpd1']
            else:
                # Out-of-range grid points
                int_start = self.intGrid()[0]
                int_end = self.intGrid()[-1]
                interpGrid = grid[(int_start <= grid) & (grid <= int_end)]
                underGrid = grid[grid < int_start]
                overGrid = grid[grid > int_end]
                underVal = np.empty_like(underGrid) * np.nan
                overVal = np.empty_like(overGrid) * np.nan
                interpFunc = interp1d(x=self.intGrid(), y=self[entry[0]][entry], kind='cubic')
                interpVal = np.concatenate([underVal, interpFunc(interpGrid), overVal])
                res.update({entry: interpVal})
        # Repulsive potentials
        elif re.match(r'^R$', entry):
            if grid is None:
                raise ValueError("Grid must be specified to evaluate repulsive potentials")
            # Use polynomial repulsive potential when spline is not defined
            if self.splGrid() is None:
                # NOTE: In SKF:
                #           rep = sum(c[i] (rcut - r)**i) for i in range(2, 10)
                #       In numpy.polynomial.Polynomial:
                #           f = sum(c[i] x**i), where i = 0, 1, 2, ...
                #       So we need to zero-pad c0 and c1
                poly_coef = self.poly_info.loc[:, 'c2':'c9']  # c2 ~ c9
                poly_coef = np.array([0, 0, *poly_coef.values.flatten()])  # zero-pad c0 and c1
                # Create and evaluate polynomial potential
                poly_func = Polynomial(poly_coef)
                rcut = self.poly_info['rcut'][0]
                res.update({entry: poly_func(rcut - grid)})
            else:
                # Evaluate exponential
                exp_start = 0.0
                exp_end = self.cub_coef['start'][0]
                expGrid = grid[(exp_start <= grid) & (grid < exp_end)]
                a1 = self.exp_coef['a1'][0]
                a2 = self.exp_coef['a2'][0]
                a3 = self.exp_coef['a3'][0]
                expVal = np.exp(-a1 * expGrid + a2) + a3

                # Evaluate cubic spline
                # x is the vector of breaking points (knots)
                x = np.unique(self.cub_coef.loc[:, 'start':'end'])
                cub_start = x[0]
                cub_end = x[-1]
                # NOTE: In SKF:
                #           cub_val = sum(c[i] (r - r0)**i) for i in range(4)
                #       In scipy.interpolate.PPoly:
                #           S = sum(c[m, i] * (xp - x[i])**(k-m) for m in range(k+1))
                #       So we need to reverse the order of coefficients
                c = self.cub_coef.loc[:, 'c0':]  # slice the coefficient block
                c = c.iloc[:, ::-1].values.T  # reverse the order of the columns
                # Slice the grid points in the range of cubic splines
                cubGrid = grid[(cub_start <= grid) & (grid < cub_end)]
                # Create and evaluate cubic splines
                cubSpl = PPoly(c, x)
                cubVal = cubSpl(cubGrid)

                # Evaluate 5-th order spline
                x = np.unique(self.fif_coef.loc[:, 'start':'end'])
                fif_start = x[0]
                fif_end = x[-1]
                c = self.fif_coef.loc[:, 'c0':]
                c = c.iloc[:, ::-1].values.T
                fifGrid = grid[(fif_start <= grid) & (grid < fif_end)]
                fifSpl = PPoly(c, x)
                fifVal = fifSpl(fifGrid)

                # Out-of-range grid points
                underGrid = grid[grid < exp_start]
                overGrid = grid[grid >= fif_end]
                underVal = np.empty_like(underGrid) * np.nan
                overVal = np.empty_like(overGrid) * np.nan

                # Concatenate
                res.update({entry: np.concatenate([underVal, expVal, cubVal, fifVal, overVal])})

        return res

    def intGrid(self):
        return self.gridDist * np.arange(1, self.nGridPoints + 1)

    def splGrid(self) -> Union[np.ndarray, None]:
        for value in self['spline'].values():
            if value is None:
                return None
        return np.array([*self.cub_coef['start'], self.fif_coef['end'].iloc[-1]])

    def range(self, entry: str) -> tuple:
        r"""Look up the range of a given entry

        Args:
            entry: str
                Supported entries:
                    'H'
                    'H???'
                    'S'
                    'S???'
                    'HS'
                    'R'
                    'exp...': case insensitive. E.g. 'exp', 'Exp', 'Exponential', etc.
                    'cub...': case insensitive. E.g. 'cub', 'Cub', 'Cubic', etc.
                    'fif...': case insensitive. E.g. 'fif', 'Fif', 'Fifth', etc.


        Returns:
            tuple
        """
        if entry in ('H', 'S', 'HS') or re.match(r'^[HS][a-z]{2}\d$', entry):
            return self.intGrid()[0], self.intGrid()[-1]
        elif re.match(r'^R$', entry):
            if self.splGrid() is None:
                return 0.0, self.poly_info['rcut'][0]
            else:
                return 0.0, self.cutoff
        else:
            try:
                if re.match(r'(?i)^exp', entry):
                    return 0.0, self.cub_coef['start'][0]
                elif re.match(r'(?i)^cub', entry):
                    return self.cub_coef['start'][0], self.fif_coef['start'][0]
                elif re.match(r'(?i)^fif', entry):
                    return self.fif_coef['start'][0], self.fif_coef['end'][0]
                else:
                    raise KeyError("Spline is not defined")
            except TypeError:
                raise KeyError("Spline is not defined")

    def to_file(self, skf_path):
        writer = _SKFWriter()
        writer.write_skf(self, skf_path)

    @classmethod
    def from_data(cls, skf_data: dict) -> SKF:
        return SKF(skf_data)

    @classmethod
    def from_file(cls, skf_path: str) -> SKF:
        reader = _SKFReader()
        return reader.read_skf(skf_path)


class SKFSet:
    def __init__(self, skfs: dict):
        r"""SKF set handler with dictionary-like interface

        Args:
            skfs:
        """
        self.skfs = skfs

    def __getitem__(self, Z: tuple) -> SKF:
        return self.skfs[Z]

    def __setitem__(self, Z: tuple, skf: SKF):
        self.skfs[Z] = skf

    def __delitem__(self, Z: tuple):
        del self.skfs[Z]

    def items(self) -> ItemsView:
        return self.skfs.items()

    def keys(self) -> KeysView:
        return self.skfs.keys()

    def values(self) -> ValuesView[SKF]:
        return self.skfs.values()

    def update(self, item: Dict[tuple, Union[SKF, dict]]):
        r"""Update the SKFs in SKFSet

        Args:
            item: Dict[tuple, Union[SKF, dict]]
                Keys are tuple of atom pairs (Z); values can be SKFs or
                major SKF blocks (header, HS, spline and doc). In the
                second case, the content in the SKFs will be updated by
                the specified SKF blocks

        """
        for Z, content in item.items():
            if isinstance(content, SKF):
                self.skfs[Z] = content
            else:
                for entry, block in content.items():
                    self.skfs[Z][entry] = block

    def atypes(self) -> tuple:
        return Z2A(self.Zs())

    def Zs(self, ordered: bool = True) -> tuple:
        r"""Retrieve pairwise interactions (Zs) given by SKFs

        Args:
            ordered: bool
                Sort each Z tuple when set to False.
                For integral tables: set to True
                For repulsive potentials: set to False

        Returns:
            _Zs: tuple

        """
        _Zs = tuple(self.skfs.keys())
        _Zs = formatZ(_Zs, unique=True, ordered=ordered)
        return _Zs

    def range(self, entry: str) -> dict:
        if re.match(r'^[HS]$', entry) or re.match(r'^[HS][a-z]{2}\d$', entry):
            ordered = True
        else:
            ordered = False
        return {Z: self[Z].range(entry) for Z in self.Zs(ordered)}

    @classmethod
    def from_data(cls, skfset_data: dict) -> SKFSet:
        r"""Create a SKFSet using SKF data

        Args:
            skfset_data: dict

        Returns: SKFSet
        """
        skfs = {Z: SKF.from_data(skf_data) for Z, skf_data in skfset_data.items()}
        skfset = SKFSet(skfs)
        return skfset

    @classmethod
    def from_dir(cls, skfdir_path: str, exclude_atypes: tuple = ()) -> SKFSet:
        r"""Load a SKF set from directory

        Args:
            skfdir_path: str
            exclude_atypes: tuple

        Returns: SKFSet
        """
        skfs = {}
        for dirpath, dirnames, filenames in os.walk(skfdir_path):
            for filename in filenames:
                if not filename.endswith('.skf'):
                    continue
                skf_path = os.path.join(dirpath, filename)
                skf = SKF.from_file(skf_path)
                Z = skf.Z
                try:
                    for atype in Z:
                        assert atype not in exclude_atypes
                except AssertionError:
                    continue
                skfs[skf.Z] = skf
        skfset = SKFSet(skfs)
        return skfset

    def to_file(self, skfdir_path: str):
        path_check(skfdir_path)
        for skf in self.skfs.values():
            filename = f"{ATOM2SYM[skf.Z[0]]}-{ATOM2SYM[skf.Z[1]]}.skf"
            skf_path = os.path.join(skfdir_path, filename)
            skf.to_file(skf_path)


class SKFBlockCreator:
    def __init__(self):
        r"""Create SKF blocks from values"""
        pass

    @staticmethod
    def create_header_HS_block(Z: tuple, header_HS_data: dict) -> dict:
        r"""Create the header block and the HS block for a single SKF

        Args:
            Z: tuple
                Atom pair in tuple, e.g. (1, 6) for hydrogen-oxygen SKF
            header_HS_data: dict
                A dictionary of SKF data required for the header block and HS block
                Structure
                ===========================================================================
                'comment': str
                    The first line of a SKF in extended format, starting with '@'
                    Will be set to '@\n' if not specified
                'gridDist': float
                    The distance between the grid points of the integral table
                'atomic_info': dict
                    Energies, SPE, Hubbard U and occupations in the ground state of an atom
                    Required by homonuclear SKF and is omitted otherwise
                    The keys in the dictionary should be
                    --------------------------------------------------------
                    Simple format:      Ed Ep Es SPE    Ud Up Us    fd fp fs
                    Extended format: Ef Ed Ep Es SPE Uf Ud Up Us ff fd fp fs
                    --------------------------------------------------------
                'mass': float
                    Atomic mass in atomic mass units
                    Required by homonuclear SKF and is set to zero otherwise
                'poly_coef': np.ndarray, shape = (8,)
                    Polynomial coefficients in an array of length 8
                    Required when splines are not defined and is set to zeros otherwise
                'rcut': float
                    Polynomial cutoff
                    Required when splines are not defined and is set to zero otherwise
                'HS': dict
                    Hamiltonian and overlap integral matrix elements.
                    Each value of the dictionary is an 1-D np.ndarray of matrix elements
                    The keys of the dictionary will be
                    ------------------------------------------------------------------
                    Simple format:   Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsd0 Hsp0 Hss0
                                     Sdd0 Sdd1 Sdd2 Spd0 Spd1 Spp0 Spp1 Ssd0 Ssp0 Sss0
                    Extended format: Hff0 Hff1 Hff2 Hff3 Hdf0 Hdf1 Hdf2 Hdd0 Hdd1 Hdd2
                                     Hpf0 Hpf1 Hpd0 Hpd1 Hpp0 Hpp1 Hsf0 Hsd0 Hsp0 Hss0
                                     Sff0 Sff1 Sff2 Sff3 Sdf0 Sdf1 Sdf2 Sdd0 Sdd1 Sdd2
                                     Spf0 Spf1 Spd0 Spd1 Spp0 Spp1 Ssf0 Ssd0 Ssp0 Sss0
                    ------------------------------------------------------------------
                ===========================================================================
        Returns:
            header_HS: dict
                Packed header and HS block. Pass to SKF.__setitem__ to create an SKF
                Structure:
                    {'header': dict
                     'HS': dict
                     'homo': bool
                     'extend': bool
                     'Z': tuple}
        """
        data = header_HS_data
        homo = Z[0] == Z[1]
        extend = len(data['HS'].keys()) == 40

        # Assemble the header block
        comment = data.get('comment', '@\n') if extend else None
        gridDist = data['gridDist']
        nGridPoints = len(data['HS']['Hdd0'])
        atomic_info = pd.DataFrame(data['atomic_info'], index=['value']) if homo else None
        poly_entries = ('mass', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'rcut',
                        'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10')
        poly_data = np.concatenate([data['mass'] if homo else np.array([0.0]),
                                    data.get('poly_coef', np.zeros(8, dtype=np.float64)),
                                    data.get('rcut', 0.0),
                                    np.zeros(10)])
        poly_info = pd.DataFrame(dict(zip(poly_entries, poly_data)), index=['value'])
        header = {'comment': comment,
                  'gridDist': gridDist,
                  'nGridPoints': nGridPoints,
                  'atomic_info': atomic_info,
                  'poly_info': poly_info}

        # Assemble the HS block
        if extend:
            H_entries = ('Hff0', 'Hff1', 'Hff2', 'Hff3', 'Hdf0', 'Hdf1', 'Hdf2', 'Hdd0', 'Hdd1', 'Hdd2',
                         'Hpf0', 'Hpf1', 'Hpd0', 'Hpd1', 'Hpp0', 'Hpp1', 'Hsf0', 'Hsd0', 'Hsp0', 'Hss0')
            S_entries = ('Sff0', 'Sff1', 'Sff2', 'Sff3', 'Sdf0', 'Sdf1', 'Sdf2', 'Sdd0', 'Sdd1', 'Sdd2',
                         'Spf0', 'Spf1', 'Spd0', 'Spd1', 'Spp0', 'Spp1', 'Ssf0', 'Ssd0', 'Ssp0', 'Sss0')
        else:
            H_entries = ('Hdd0', 'Hdd1', 'Hdd2', 'Hpd0', 'Hpd1', 'Hpp0', 'Hpp1', 'Hsd0', 'Hsp0', 'Hss0')
            S_entries = ('Sdd0', 'Sdd1', 'Sdd2', 'Spd0', 'Spd1', 'Spp0', 'Spp1', 'Ssd0', 'Ssp0', 'Sss0')
        H = pd.DataFrame({entry: data['HS'][entry] for entry in H_entries})
        S = pd.DataFrame({entry: data['HS'][entry] for entry in S_entries})
        HS = {'H': H,
              'S': S}

        # Assemble the output
        header_HS = {'header': header,
                     'HS': HS,
                     'homo': homo,
                     'extend': extend,
                     'Z': Z}
        return header_HS

    @staticmethod
    def create_spline_block(exp_coef: np.ndarray, spl_xydata: np.ndarray) -> dict:
        r"""Create a spline block from xydata

        Args:
            exp_coef: np.ndarray
                Coefficients of short-range exponential potential
            spl_xydata: np.ndarray
                n_grid x 2 matrix, of which the 1st and 2nd column is
                the grid points (unit: Angstroms)
                and the spline potential (unit: Hartrees) evaluated on the grid

        Returns:
            spline : dict
        """

        splGrid = spl_xydata[:, 0] * ANGSTROM2BOHR
        spl_val = spl_xydata[:, 1]
        nInt = len(spl_xydata) - 1
        cutoff = splGrid[-1]

        # Create _exp_coef DataFrame
        exp_entries = ('a1', 'a2', 'a3')
        _exp_coef = pd.DataFrame(dict(zip(exp_entries, exp_coef)), index=['value'])

        # Fit xydata to cubic spline and format the results
        spl_coef = CubicSpline(splGrid, spl_val).c
        spl_coef = spl_coef[::-1].T
        spl_ints = np.array([splGrid[:-1], splGrid[1:]]).T
        _cub_fif_coef = np.concatenate([spl_ints, spl_coef], axis=1)

        # Create _cub_coef DataFrame
        cub_entries = ('start', 'end', 'c0', 'c1', 'c2', 'c3')
        _cub_coef = pd.DataFrame(_cub_fif_coef[:-1], columns=cub_entries)

        # Create _fif_coef DataFrame
        fif_entries = ('start', 'end', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5')
        # Zero-pad cubic spline coefficients to 5-th order
        _fif_coef = [*_cub_fif_coef[-1], 0, 0]
        _fif_coef = pd.DataFrame(dict(zip(fif_entries, _fif_coef)), index=['value'])

        spline = {'nInt': nInt,
                  'cutoff': cutoff,
                  'exp_coef': _exp_coef,
                  'cub_coef': _cub_coef,
                  'fif_coef': _fif_coef}

        return spline

    @staticmethod
    def create_doc_block(doc: list) -> list:
        return doc


class _SKFReader:
    def read_skf(self, skf_path: str) -> SKF:
        r"""Create SKF from file

        Args:
            skf_path: str

        Returns: SKF

        Notes:
        =========================================================================================
        SKF structure
        -----------------------------------------------------------------------------------------
        Header
        -----------------------------------------------------------------------------------------
        Line 0 (Optional)                     : Comment starting with '@'. If specified,
                                                the SKF follows the extended format

        Line 1                                : gridDist nGridPoints nShells (optional)

        # Atomic information, presents only in homonuclear SKFs
        Line 2 (Simple)                       : Ed Ep Es SPE Ud Up Us fd fp fs
        Line 2 (Extended)                     : Ef Ed Ep Es SPE Uf Ud Up Us ff fd fp fs

        # Atomic mass and repulsive polynomial coefficients (optional)
        Line 3                                : mass (homonuclear) / placeholder (heteronuclear)
                                                c2 c3 c4 c5 c6 c7 c8 c9 rcut
                                                d1 d2 d3 d4 d5 d6 d7 d8 d9 d10 (all placeholders)

        # Tables of Hamiltonian matrices (H) and overlap matrices (S)
        Line 4 -> (nGridPoints + 3) (Simple)  : Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsd0 Hsp0 Hss0
                                                Sdd0 Sdd1 Sdd2 Spd0 Spd1 Spp0 Spp1 Ssd0 Ssp0 Sss0
        Line 4 -> (nGridPoints + 3) (Extended): Hff0 Hff1 Hff2 Hff3 Hdf0 Hdf1 Hdf2 Hdd0 Hdd1 Hdd2
                                                Hpf0 Hpf1 Hpd0 Hpd1 Hpp0 Hpp1 Hsf0 Hsd0 Hsp0 Hss0
                                                Sff0 Sff1 Sff2 Sff3 Sdf0 Sdf1 Sdf2 Sdd0 Sdd1 Sdd2
                                                Spf0 Spf1 Spd0 Spd1 Spp0 Spp1 Ssf0 Ssd0 Ssp0 Sss0

        # An empty line may present before the spline block
        -----------------------------------------------------------------------------------------
        Spline (optional)
        -----------------------------------------------------------------------------------------
        Line 0                                : 'Spline'
        Line 1                                : nInt cutoff

        # Exponential coefficients
        Line 2                                : a1 a2 a3

        # Cubic spline coefficients
        Line 3 -> (nInt + 1)                  : start end c0 c1 c2 c3

        # 5-th order spline coefficients
        Line (nInt + 2)                       : start end c0 c1 c2 c3 c4 c5
        -----------------------------------------------------------------------------------------
        Documentation (optional)
        -----------------------------------------------------------------------------------------
        Line 0:                               : '<Documentation>'
        Line -1:                              : '</Documentation>'
        =========================================================================================
        """

        # Retrieve basic information
        basic_info = self._basic_info(skf_path)
        # Split SKF into blocks (header, integral table / HS, spline, documentation / doc
        blocks = self._split_blocks(skf_path)
        # Parse each block
        header, HS = self._parse_header_HS(blocks, basic_info)
        spline = self._parse_spline(blocks, basic_info)
        doc = self._parse_doc(blocks)
        # Create SKF object
        res = SKF({**{entry: basic_info[entry] for entry in ('Z', 'homo', 'extend')},
                   **header, **HS, **spline, **doc})
        return res

    @staticmethod
    def _basic_info(skf_path: str) -> dict:
        r"""Extract basic information from SKF

        Args:
            skf_path: str

        Returns:
            dict
        """
        # Check if the SKF describes homonuclear interaction
        atom_pair = os.path.splitext(os.path.basename(skf_path))[0].split('-')
        Z = tuple(SYM2ATOM[sym] for sym in atom_pair)
        homo = Z[0] == Z[1]
        # Check if the SKF is in extended format (line 1)
        extend = open(skf_path, 'r').readline().startswith('@')
        # Determine available H and S entries
        if extend:
            H_entries = ('Hff0', 'Hff1', 'Hff2', 'Hff3', 'Hdf0', 'Hdf1', 'Hdf2', 'Hdd0', 'Hdd1', 'Hdd2',
                         'Hpf0', 'Hpf1', 'Hpd0', 'Hpd1', 'Hpp0', 'Hpp1', 'Hsf0', 'Hsd0', 'Hsp0', 'Hss0')
            S_entries = ('Sff0', 'Sff1', 'Sff2', 'Sff3', 'Sdf0', 'Sdf1', 'Sdf2', 'Sdd0', 'Sdd1', 'Sdd2',
                         'Spf0', 'Spf1', 'Spd0', 'Spd1', 'Spp0', 'Spp1', 'Ssf0', 'Ssd0', 'Ssp0', 'Sss0')
        else:
            H_entries = ('Hdd0', 'Hdd1', 'Hdd2', 'Hpd0', 'Hpd1', 'Hpp0', 'Hpp1', 'Hsd0', 'Hsp0', 'Hss0')
            S_entries = ('Sdd0', 'Sdd1', 'Sdd2', 'Spd0', 'Spd1', 'Spp0', 'Spp1', 'Ssd0', 'Ssp0', 'Sss0')
        # Determine atomic information entries (line 2)
        atomic_entries = None
        if homo:
            atomic_entries = ('Ef', 'Ed', 'Ep', 'Es', 'SPE', 'Uf', 'Ud', 'Up', 'Us', 'ff', 'fd', 'fp', 'fs') \
                if extend else \
                ('Ed', 'Ep', 'Es', 'SPE', 'Ud', 'Up', 'Us', 'fd', 'fp', 'fs')
        # Determine polynomial coefficients (line 3)
        poly_entries = ('mass', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'rcut',
                        'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10')
        # Exponential coefficients
        exp_entries = ('a1', 'a2', 'a3')
        # Cubic spline coefficients
        cub_entries = ('start', 'end', 'c0', 'c1', 'c2', 'c3')
        # 5-th order spline coefficients
        fif_entries = ('start', 'end', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5')
        # Collect information in a dictionary
        basic_info = {'Z': Z,
                      'homo': homo,
                      'extend': extend,
                      'H_entries': H_entries,
                      'S_entries': S_entries,
                      'atomic_entries': atomic_entries,
                      'poly_entries': poly_entries,
                      'exp_entries': exp_entries,
                      'cub_entries': cub_entries,
                      'fif_entries': fif_entries}
        return basic_info

    @staticmethod
    def _split_blocks(skf_path: str) -> dict:
        r"""Read and split SKF into content blocks

        Args:
            skf_path: str

        Returns:
            dict
        """
        # Read SKF by lines and remove empty lines
        skf_raw = open(skf_path, 'r').readlines()
        skf_raw = list(filter(lambda line: line.strip(), skf_raw))

        # Determine the range of each block
        file_start = 0
        file_end = len(skf_raw)
        # Range of documentation block
        doc_start = None
        doc_end = None
        p_doc = re.compile(r'<Documentation>')
        for line_num, line in enumerate(skf_raw):
            if p_doc.match(line):
                doc_start = line_num
                doc_end = file_end + 1
                break
        doc_block = None if doc_start is None else skf_raw[doc_start:doc_end]
        # Range of spline block
        spline_start = None
        spline_end = None
        p_spline = re.compile(r'(?i)\bspline\b')
        for line_num, line in enumerate(skf_raw[:doc_start]):
            if p_spline.match(line):
                spline_start = line_num
                spline_end = doc_start
                break
        spline_block = None if spline_start is None else skf_raw[spline_start: spline_end]
        # Range of header block
        header_start = file_start
        header_end = spline_start
        header_block = skf_raw[header_start:header_end]

        # Assemble blocks into a dictionary
        blocks = {'header_block': header_block,
                  'spline_block': spline_block,
                  'doc_block': doc_block}
        return blocks

    def _parse_header_HS(self, blocks: dict, basic_info: dict) -> Tuple[dict, dict]:
        r"""Parse the header and HS block of SKF

        Args:
            blocks: dict
            basic_info: dict

        Returns:
            dict
        """
        homo = basic_info.get('homo')
        extend = basic_info.get('extend')
        H_entries = basic_info.get('H_entries')
        S_entries = basic_info.get('S_entries')
        HS_entries = H_entries + S_entries
        atomic_entries = basic_info.get('atomic_entries')
        poly_entries = basic_info.get('poly_entries')
        header_block = blocks.get('header_block')

        # Process header block sequentially
        # Line 0: comment starting with '@'
        line_num = 0
        if extend:
            comment = header_block[0]
            line_num += 1
        else:
            comment = None

        # Line 1: gridDist, nGridPoints and nShell (optional)
        gridDist, nGridPoints = self._parse_line(header_block[line_num])[:2]  # discard nShell
        nGridPoints = int(nGridPoints)
        line_num += 1

        # Line 2: atomic information
        if homo:
            atomic_info = dict(zip(atomic_entries, self._parse_line(header_block[line_num])))
            atomic_info = pd.DataFrame(atomic_info, index=['value'])
            line_num += 1
        else:
            atomic_info = None

        # Line 3: atomic mass and repulsive polynomial coefficients
        poly_info = dict(zip(poly_entries, self._parse_line(header_block[line_num])))
        poly_info = pd.DataFrame(poly_info, index=['value'])
        line_num += 1

        # Last nGridPoints lines: integral table of Hamiltonian and overlap matrices
        # NOTE: Old SKFs have multiline placeholders before the integral table.
        #       Slicing lines from the end gets rid of these placeholders easily.
        HS_block = header_block[-nGridPoints:]
        HS_val = pd.DataFrame([self._parse_line(line) for line in HS_block], columns=HS_entries)
        H, S = HS_val[list(H_entries)], HS_val[list(S_entries)]

        header = {'comment': comment,
                  'gridDist': gridDist,
                  'nGridPoints': nGridPoints,
                  'atomic_info': atomic_info,
                  'poly_info': poly_info}
        HS = {'H': H,
              'S': S}
        return header, HS

    def _parse_spline(self, blocks: dict, basic_info: dict) -> dict:
        r"""Parse spline block

        Args:
            blocks: dict
            basic_info: dict

        Returns:
            dict
        """
        spline_block = blocks.get('spline_block')
        exp_entries = basic_info.get('exp_entries')
        cub_entries = basic_info.get('cub_entries')
        fif_entries = basic_info.get('fif_entries')

        if spline_block is None:
            nInt = None
            cutoff = None
            exp_coef = None
            cub_coef = None
            fif_coef = None
        else:
            # Parse spline block sequentially
            # nInt and cutoff
            nInt, cutoff = self._parse_line(spline_block[1])
            nInt = int(nInt)
            # Exponential coefficients
            exp_coef = dict(zip(exp_entries, self._parse_line(spline_block[2])))
            exp_coef = pd.DataFrame(exp_coef, index=['value'])
            # Cubic spline coefficients
            assert len(spline_block[3:-1]) == nInt - 1, "nInt does not match the number of intervals of cubic splines"
            cub_coef = pd.DataFrame([self._parse_line(line) for line in spline_block[3:-1]], columns=cub_entries)
            # 5-th order spline coefficients
            fif_coef = dict(zip(fif_entries, self._parse_line(spline_block[-1])))
            fif_coef = pd.DataFrame(fif_coef, index=['value'])

        spline = {'nInt': nInt,
                  'cutoff': cutoff,
                  'exp_coef': exp_coef,
                  'cub_coef': cub_coef,
                  'fif_coef': fif_coef}
        return spline

    @staticmethod
    def _parse_doc(blocks: dict) -> dict:
        r"""Parse documentation block

        Args:
            blocks: dict

        Returns:
            dict
        """
        doc_block = blocks.get('doc_block')
        return {'doc': doc_block}

    @staticmethod
    def _parse_line(line: str) -> list:
        r"""Split a single line of SKF and convert the content to float

        Args:
            line: str

        Returns:
            list
        """
        segments = list(filter(None, re.split(r'\s|,', line)))  # split by space and comma
        res = []
        for segment in segments:
            if '*' in segment:  # e.g. '9*0.0'
                rep, val = segment.split('*')
                res.extend(int(rep) * [float(val)])
            else:
                res.append(float(segment))
        return res


class _SKFWriter:
    def __init__(self):
        r"""Write SKF objects to file"""
        self.header = None
        self.HS = None
        self.spline = None
        self.doc = None

    def write_skf(self, skf: SKF, skf_path: str) -> None:
        r"""Write an SKF object to an Slater-Koster file in plain text

        Args:
            skf: SKF
            skf_path: str

        """
        self.header = self._construct_header(skf)
        self.HS = self._construct_HS(skf)
        self.spline = self._construct_spline(skf)
        self.doc = self._construct_doc(skf)
        path_check(skf_path)
        with open(skf_path, 'w') as f:
            f.writelines([*self.header, *self.HS, *self.spline, *self.doc])

    def _construct_header(self, skf) -> list:
        # Line 0: comment if the SKF is in extended format
        header = [skf.comment] if skf.extend else []
        # Line 1
        header.append(self._construct_line([skf.gridDist, skf.nGridPoints]))
        # Line 2: atomic info if the SKF is homonuclear
        if skf.homo:
            header.append(self._construct_line(skf.atomic_info.values.flatten()))
        # Line 3: atomic mass, cutoff and polynomial coefficients
        header.append(self._construct_line(skf.poly_info.values.flatten()))
        return header

    def _construct_HS(self, skf) -> list:
        HS_vals = pd.concat([skf.H, skf.S], axis=1).values
        HS = [self._construct_line(HS_val, indent=4, exp=True) for HS_val in HS_vals]
        return HS

    def _construct_spline(self, skf) -> list:
        spline = ['Spline\n',
                  self._construct_line([skf.nInt, skf.cutoff]),
                  self._construct_line(skf.exp_coef.values.flatten()),
                  *[self._construct_line(cub_val) for cub_val in skf.cub_coef.values],
                  self._construct_line(skf.fif_coef.values.flatten())]
        return spline

    @staticmethod
    def _construct_doc(skf) -> list:
        doc = skf.doc
        return doc

    @staticmethod
    def _construct_line(num_array: Iterable, sep='    ', indent=0, exp=False) -> str:
        r"""Convert an array of numbers to

        Args:
            num_array: Iterable
                Array of numbers
            sep: str
                Separator. 4 spaces as default
            indent: int
                Number of spaces starting the line
            exp: bool
                Format number into exponentials

        Returns:

        """
        if exp:
            # np.float64 supports at most 15 decimal digits
            line_content = [f"{num:.15e}" for num in num_array]
        else:
            line_content = []
            for number in num_array:
                if isinstance(number, float):
                    # np.float64 supports at most 15 decimal digits
                    line_content.append(f"{number:.15f}")
                else:
                    line_content.append(f"{number}")
        line = f"{''.join([' '] * indent)}{sep.join(line_content)}\n"
        return line


if __name__ == '__main__':
    mio_dirpath = './slakos/mio-1-1/'
    mioOO_path = os.path.join(mio_dirpath, 'O-O.skf')
    mioNO_path = os.path.join(mio_dirpath, 'N-O.skf')

    mioOO = SKF.from_file(mioOO_path)
    mioNO = SKF.from_file(mioNO_path)
    mio = SKFSet.from_dir(mio_dirpath)

    """
    Example of creating an SKF
    ==========================
    Z = (..., ...)
    header_HS_data = {'gridDist': ...,
                      'atomic_info': {'Es': ...,
                                      ...},
                      'mass': ...,
                      'HS': {'Hss0': ...,
                             ...}}
    exp_coef = np.array([..., ..., ...])
    spl_xydata = np.array([...])
    doc = [...]
    
    creator = SKFBlockCreator()
    header_HS_block = creator.create_header_HS_block(Z, header_HS_data)
    spline_block = creator.create_spline_block(exp_coef, spl_xydata)
    doc_block = creator.create_doc_block(doc)
    
    skf = SKF(skf_data={})
    skf['homo'] = header_HS_block['homo']
    skf['extend'] = header_HS_block['extend']
    skf['Z'] = header_HS_block['Z']
    skf['header'] = header_HS_block['header']
    skf['HS'] = header_HS_block['HS']
    skf['spline'] = spline_block
    skf['doc'] = doc_block
    """
