from __future__ import annotations

import os
import re
from typing import ItemsView, KeysView, ValuesView, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial
from scipy.interpolate import PPoly, interp1d

from .consts import ANGSTROM2BOHR, ATOM2SYM, HARTREE, SYM2ATOM
from .util import formatZ, path_check, Z2A

# WARNING: ANGSTROM2BOHR

class SKF:
    def __init__(self, skf_data: dict):
        self.homo = None
        self.extend = None
        self.Z = None
        self.H_entries = None
        self.S_entries = None
        self.atomic_entries = None
        self.poly_entries = None
        self.exp_entries = None
        self.cub_entries = None
        self.fif_entries = None
        self.comment = None
        self.gridDist = None
        self.nGridPoints = None
        self.intGrid = None
        self.atomic_info = None
        self.poly_info = None
        self.H = None
        self.S = None
        self.nInt = None
        self.cutoff = None
        self.splGrid = None
        self.exp_coef = None
        self.cub_coef = None
        self.fif_coef = None
        self.doc = None
        self.__dict__.update(skf_data)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__.update({key, value})

    def __delitem__(self, key):
        del self.__dict__[key]

    def __call__(self, entry: str, grid: np.ndarray = None) -> dict:
        r"""Look up SKF data or evaluate H, S and R on a grid

        Args:
            entry: str
                Acceptable entries:
                    Atomic information
                        'E': On-site energies for angular momenta
                        'E?': E for a specific angular momentum, e.g. 'Es'
                        'SPE': Spin polarization error
                        'U': Hubbard U for angular momenta
                        'U?': U for a specific angular momentum, e.g. 'Up'
                        'f': Occupations for angular momenta
                        'f?': Occupation for a specific angular momentum, e.g. 'fd'
                    Integral tables
                        NOTE: Interpolate using cubic spline if a grid is given
                        'H': Hamiltonian matrix elements
                        'H???': H for a specific two-center interaction, e.g. 'Hss0' (ss-sigma)
                        'S': Overlap matrix elements
                        'S???': S for a specific two-center interaction, e.g. 'Sff3' (ff-phi)
                    Repulsive potential
                        NOTE: Evaluate on a given grid (must be specified)
                        'R': Repulsive potential
                    Other information:
                        'comment': First-line comment in the original SKF
                        'gridDist': Distance between the grid points of the integral table
                        'nGridPoints': Number of points in the integral table
                        'nInt': Number of (subsequent) intervals described by splines
                        'cutoff': Cutoff of repulsive spline

            grid: np.ndarray
                Evaluate repulsive potential on a grid

        Returns:
            res: dict

        """
        res = {}
        # Atomic information
        if re.match(r'^[EfU]$', entry):
            cols = [ent for ent in self.atomic_entries if re.match(r'^' + entry + r'[a-z]$', ent)]
            res.update({entry: self.atomic_info[cols]})
        elif re.match(r'^[EfU][a-z]$', entry):  # e.g. 'Es', 'fd', 'Up'
            res.update({entry: self.atomic_info[entry]})
        elif re.match(r'^SPE$', entry):
            res.update({entry: self.atomic_info[entry]})
        elif re.match(r'^mass$', entry):
            res.update({entry: self.poly_info[entry]})
        # Integral tables
        elif re.match(r'^[HS]$', entry):
            if grid is None:
                res.update({entry: self[entry]})
            else:
                interpRes = {}
                # Out-of-range grid points
                int_start = self.intGrid[0]
                int_end = self.intGrid[-1]
                interpGrid = grid[(int_start <= grid) & (grid <= int_end)]
                underGrid = grid[grid < int_start]
                overGrid = grid[grid > int_end]
                underVal = np.empty_like(underGrid) * np.nan
                overVal = np.empty_like(overGrid) * np.nan
                for ent, val in self[entry].items():
                    interpFunc = interp1d(x=self.intGrid, y=val, kind='cubic')
                    interpVal = np.concatenate([underVal, interpFunc(interpGrid), overVal])
                    interpRes.update({ent: interpVal})
                res.update({entry: pd.DataFrame(interpRes)})
        elif re.match(r'^[HS][a-z]{2}\d$', entry):  # e.g. 'Hpd1', 'Sff3'
            if grid is None:
                res.update({entry: self[entry[0]][entry]})  # e.g. 'Hpd1' -> self['H']['Hpd1']
            else:
                # Out-of-range grid points
                int_start = self.intGrid[0]
                int_end = self.intGrid[-1]
                interpGrid = grid[(int_start <= grid) & (grid <= int_end)]
                underGrid = grid[grid < int_start]
                overGrid = grid[grid > int_end]
                underVal = np.empty_like(underGrid) * np.nan
                overVal = np.empty_like(overGrid) * np.nan
                interpFunc = interp1d(x=self.intGrid, y=self[entry[0]][entry], kind='cubic')
                interpVal = np.concatenate([underVal, interpFunc(interpGrid), overVal])
                res.update({entry: interpVal})
        # Repulsive potentials
        elif re.match(r'^R$', entry):
            if grid is None:
                raise ValueError("Grid must be specified to evaluate repulsive potentials")
            # Use polynomial repulsive potential when spline is not defined
            if self['splGrid'] is None:
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
                expFunc = lambda x: np.exp(-a1 * x + a2) + a3
                expVal = expFunc(expGrid)

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

        # Others
        else:
            res.update({entry: self[entry]})

        return res

    def range(self, entry: str) -> tuple:
        r"""Look up the range of a given entry

        Args:
            entry: str

        Returns:
            tuple
        """
        if re.match(r'^[HS]$', entry) or re.match(r'^[HS][a-z]{2}\d$', entry):
            return self.intGrid[0], self.intGrid[-1]
        elif re.match(r'^R$', entry):
            if self['splGrid'] is None:
                return 0.0, self.poly_info['rcut'][0]
            else:
                return 0.0, self.cutoff
        elif re.match(r'(?i)^exp', entry):
            if self['splGrid'] is None:
                raise ValueError("Spline is not defined")
            else:
                return 0.0, self.cub_coef['start'][0]
        elif re.match(r'(?i)^cub', entry):
            if self['splGrid'] is None:
                raise ValueError("Spline is not defined")
            else:
                return self.cub_coef['start'][0], self.fif_coef['start'][0]
        elif re.match(r'(?i)^fif', entry):
            if self['splGrid'] is None:
                raise ValueError("Spline is not defined")
            else:
                return self.fif_coef['start'][0], self.fif_coef['end'][0]

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


# TODO: access of all the entries in all skfs
class SKFSet:
    def __init__(self, skfs: dict):
        self.skfs = skfs

    def __getitem__(self, Z):
        return self.skfs[Z]

    def items(self) -> ItemsView:
        return self.skfs.items()

    def keys(self) -> KeysView:
        return self.skfs.keys()

    def values(self) -> ValuesView[SKF]:
        return self.skfs.values()

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
    def from_dir(cls, skfdir_path: str) -> SKFSet:
        skfs = {}
        for dirpath, dirnames, filenames in os.walk(skfdir_path):
            for filename in filenames:
                if not filename.endswith('.skf'):
                    continue
                skf_path = os.path.join(dirpath, filename)
                skf = SKF.from_file(skf_path)
                skfs[skf.Z] = skf
        skfset = SKFSet(skfs)
        return skfset

    def to_dir(self, skfdir_path: str):
        path_check(skfdir_path)
        for skf in self.skfs:
            filename = f"{ATOM2SYM[skf.Z[0]]}-{ATOM2SYM[skf.Z[1]]}.skf"
            skf_path = os.path.join(skfdir_path, filename)
            skf.to_file(skf_path)


# SKF blocks from values
# TODO: implementation
class _SKFConstructor:
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def _construct_HS_block(model):
        raise NotImplementedError

    @staticmethod
    def _construct_spline_block(spl_xydata: np.ndarray) -> dict:
        r"""Construct a spline block from xydata

        Args:
            spl_xydata: np.ndarray
                n_grid x 2 matrix, of which the 1st and 2nd column is
                the grid points and the spline values evaluated on the grid,
                respectively

        Returns:
            spline : dict
        """
        spl_grid = spl_xydata[:, 0]
        spl_vals = spl_xydata[:, 1]

        spline = {'nInt': nInt,
                  'cutoff': cutoff,
                  'splGrid': splGrid,
                  'exp_coef': exp_coef,
                  'cub_coef': cub_coef,
                  'fif_coef': fif_coef}
        return spline


# SKF data from file
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
        header = self._parse_header(blocks, basic_info)
        spline = self._parse_spline(blocks, basic_info)
        doc = self._parse_doc(blocks)
        # Create SKF object
        res = SKF({**basic_info, **header, **spline, **doc})
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
        H_entries = ('Hff0', 'Hff1', 'Hff2', 'Hff3', 'Hdf0', 'Hdf1', 'Hdf2', 'Hdd0', 'Hdd1', 'Hdd2',
                     'Hpf0', 'Hpf1', 'Hpd0', 'Hpd1', 'Hpp0', 'Hpp1', 'Hsf0', 'Hsd0', 'Hsp0', 'Hss0') \
            if extend else \
            ('Hdd0', 'Hdd1', 'Hdd2', 'Hpd0', 'Hpd1', 'Hpp0', 'Hpp1', 'Hsd0', 'Hsp0', 'Hss0')
        S_entries = ('Sff0', 'Sff1', 'Sff2', 'Sff3', 'Sdf0', 'Sdf1', 'Sdf2', 'Sdd0', 'Sdd1', 'Sdd2',
                     'Spf0', 'Spf1', 'Spd0', 'Spd1', 'Spp0', 'Spp1', 'Ssf0', 'Ssd0', 'Ssp0', 'Sss0') \
            if extend else \
            ('Sdd0', 'Sdd1', 'Sdd2', 'Spd0', 'Spd1', 'Spp0', 'Spp1', 'Ssd0', 'Ssp0', 'Sss0')
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
        ## Range of documentation block
        doc_start = None
        doc_end = None
        p_doc = re.compile(r'<Documentation>')
        for line_num, line in enumerate(skf_raw):
            if p_doc.match(line):
                doc_start = line_num
                doc_end = file_end + 1
                break
        doc_block = None if doc_start is None else skf_raw[doc_start:doc_end]
        ## Range of spline block
        spline_start = None
        spline_end = None
        p_spline = re.compile(r'(?i)\bspline\b')
        for line_num, line in enumerate(skf_raw[:doc_start]):
            if p_spline.match(line):
                spline_start = line_num
                spline_end = doc_start
                break
        spline_block = None if spline_start is None else skf_raw[spline_start: spline_end]
        ## Range of header block
        header_start = file_start
        header_end = spline_start
        header_block = skf_raw[header_start:header_end]

        # Assemble blocks into a dictionary
        blocks = {'header_block': header_block,
                  'spline_block': spline_block,
                  'doc_block': doc_block}
        return blocks

    def _parse_header(self, blocks: dict, basic_info: dict) -> dict:
        r"""Parse header block of SKF

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
        ## Line 0: comment starting with '@'
        line_num = 0
        if extend:
            comment = header_block[0]
            line_num += 1
        else:
            comment = None
        ## Line 1: gridDist, nGridPoints and nShell (optional)
        gridDist, nGridPoints = self._parse_line(header_block[line_num])[:2]  # discard nShell
        nGridPoints = int(nGridPoints)
        intGrid = gridDist * np.arange(1, nGridPoints+1)
        line_num += 1
        ## Line 2: atomic information
        if homo:
            atomic_info = dict(zip(atomic_entries, self._parse_line(header_block[line_num])))
            atomic_info = pd.DataFrame(atomic_info, index=['value'])
            line_num += 1
        else:
            atomic_info = None
        ## Line 3: atomic mass and repulsive polynomial coefficients
        poly_info = dict(zip(poly_entries, self._parse_line(header_block[line_num])))
        poly_info = pd.DataFrame(poly_info, index=['value'])
        line_num += 1
        ## Last nGridPoints lines: integral table of Hamiltonian and overlap matrices
        ## NOTE: Old SKFs have multiline placeholders before the integral table.
        ##       Slicing lines from the end gets rid of these placeholders easily.
        HS_block = header_block[-nGridPoints:]
        HS = pd.DataFrame([self._parse_line(line) for line in HS_block], columns=HS_entries)
        H, S = HS[list(H_entries)], HS[list(S_entries)]

        header = {'comment': comment,
                  'gridDist': gridDist,
                  'nGridPoints': nGridPoints,
                  'intGrid': intGrid,
                  'atomic_info': atomic_info,
                  'poly_info': poly_info,
                  'H': H,
                  'S': S}
        return header

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
            splGrid = None
            exp_coef = None
            cub_coef = None
            fif_coef = None
        else:
            # Parse spline block sequentially
            ## nInt and cutoff
            nInt, cutoff = self._parse_line(spline_block[1])
            nInt = int(nInt)
            ## Exponential coefficients
            exp_coef = dict(zip(exp_entries, self._parse_line(spline_block[2])))
            exp_coef = pd.DataFrame(exp_coef, index=['value'])
            ## Cubic spline coefficients
            assert len(spline_block[3:-1]) == nInt - 1, "nInt does not match the number of intervals of cubic splines"
            cub_coef = pd.DataFrame([self._parse_line(line) for line in spline_block[3:-1]], columns=cub_entries)
            ## 5-th order spline coefficients
            fif_coef = dict(zip(fif_entries, self._parse_line(spline_block[-1])))
            fif_coef = pd.DataFrame(fif_coef, index=['value'])
            ## Spline grid
            splGrid = np.array([*cub_coef['start'], fif_coef['end'].iloc[-1]])

        spline = {'nInt': nInt,
                  'cutoff': cutoff,
                  'splGrid': splGrid,
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
        segments = list(filter(None, re.split('\s|,', line)))  # split by space and comma
        res = []
        for segment in segments:
            if '*' in segment:  # e.g. '9*0.0'
                rep, val = segment.split('*')
                res.extend(int(rep) * [float(val)])
            else:
                res.append(float(segment))
        return res


# SKF data to file
class _SKFWriter:
    def __init__(self):
        self.skf_text = None

    def write_skf(self, skf: SKF, skf_path: str):
        header = self._construct_header(skf)
        spline = self._construct_spline(skf)
        doc = self._construct_doc(skf)
        self.skf_text = {'header': header,
                         'spline': spline,
                         'doc': doc}
        with open(skf_path, 'w') as file:
            for line in [*header, *spline, *doc]:
                if line is not None:
                    file.write(line)

    def _construct_header(self, skf) -> list:
        header = [skf.comment,
                  self._construct_line([skf.gridDist, skf.nGridPoints]),
                  self._construct_line(skf.atomic_info.values),
                  self._construct_line(skf.poly_info.values)]
        HS_vals = pd.concat([skf.H, skf.S], axis=1).values
        HS = [self._construct_line(HS_val) for HS_val in HS_vals]
        return header

    def _construct_HS(self, skf) -> list:
        HS = [self._construct_line()]

    @staticmethod
    def _construct_spline(skf) -> list:
        raise NotImplementedError

    @staticmethod
    def _construct_doc(skf) -> list:
        raise NotImplementedError

    @staticmethod
    def _construct_line(line: Iterable, sep=" ") -> str:
        return sep.join([str(s) for s in line]) + '\n'


if __name__ == '__main__':
    # mioHH_path = './slakos/mio-0-1/H-H.skf'
    # auAuH_path = './slakos/auorg-1-1/Au-H.skf'
    #
    # mioHH = SKF.from_file(mioHH_path)
    # auAuH = SKF.from_file(auAuH_path)
    #
    # mio_path = './slakos/mio-0-1/'
    # auorg_path = './slakos/auorg-1-1/'
    #
    # mio = SKFSet.from_dir(mio_path)
    # auorg = SKFSet.from_dir(auorg_path)
    #
    # print(mio.range('R'))
    # print(auorg.Zs(ordered=True))

    skfset_path = '/export/home/hanqingh/Documents/DFTBrepulsive/SKF/au_full'
    skfset = SKFSet.from_dir(skfset_path)
    for Z, skf in skfset.items():
        plt.plot(skf.splGrid * ANGSTROM2BOHR, skf('Hss0', None)['R'] * HARTREE)
        plt.ylim(-2, 15)
        plt.title(Z)
        plt.xlabel(r"Distance ($\mathrm{\AA}$)")
        plt.ylabel(r"Repulsive potential (kcal/mol)")
        plt.savefig(os.path.join(skfset_path, f"{Z}.png"))
