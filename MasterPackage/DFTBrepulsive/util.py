from __future__ import annotations

import os
import pickle as pkl
from _pickle import UnpicklingError
from collections import Counter
from itertools import combinations
from time import time
from typing import Optional, Iterable

import matplotlib as mpl
from h5py import File

from .consts import ATOM2SYM


class Timer:

    def __init__(self, prompt: Optional[str] = None) -> None:
        r"""Print the wall time of the execution of a code block

        Args:
            prompt (str): Specify the prompt in the output. Optional.

        Returns:
            None

        Examples:
            >>> from util import Timer
            >>> with Timer("TEST"):
            ...     # CODE_BLOCK
            Wall time of TEST: 0.0 seconds
            >>> from util import Timer
            >>> with Timer():
            ...     # CODE_BLOCK
            Wall time: 0.0 seconds
        """
        self.prompt = prompt

    def __enter__(self) -> Timer:
        self.start = time()  # <-- Record the time when Timer is called
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end = time()  # <-- Record the time when Timer exits
        self.seconds = self.end - self.start  # <-- Compute the time interval
        if self.prompt:
            print(f"Wall time of {self.prompt}: {self.seconds:.1f} seconds")
        else:
            print(f"Wall time: {self.seconds:.1f} seconds")


def count_n_heavy_atoms(atomic_numbers):
    counts = sum([c for a, c in dict(Counter(atomic_numbers)).items() if a > 1])
    return counts


def path_check(file_path: str) -> None:
    r"""Check if the specified path exists, if not then create the parent directories recursively

    Args:
        file_path: Path to be checked

    Returns:
        None

    Examples:
        >>> from util import path_check
        >>> file_path_1 = "EXISTING_DIR/NOT_EXISTING_FILE"
        >>> path_check(file_path_1)
        # No output
        >>> file_path_2 = "NOT_EXISTING_DIR/NOT_EXISTING_FILE"
        >>> path_check(file_path_2)
        # "/NOT_EXISTING_DIR" is created
    """
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  #: Recursively create the parent directories of the file


def mpl_default_setting():
    mpl.rcParams['axes.labelsize'] = 'large'
    mpl.rcParams['figure.subplot.hspace'] = 0.3
    mpl.rcParams['figure.subplot.wspace'] = 0.3
    mpl.rcParams['figure.titlesize'] = 'large'

    mpl.rcParams['font.family'] = ['Arial']
    mpl.rcParams['font.size'] = 16

    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['legend.loc'] = 'upper right'

    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['xtick.labelsize'] = 'large'
    mpl.rcParams['xtick.major.size'] = 0
    mpl.rcParams['xtick.minor.size'] = 0
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['ytick.labelsize'] = 'large'
    mpl.rcParams['ytick.major.size'] = 0
    mpl.rcParams['ytick.minor.size'] = 0

    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.format'] = 'png'
    mpl.rcParams['savefig.transparent'] = False
    mpl.rcParams['savefig.bbox'] = 'tight'


def expandZ(item: dict) -> dict:
    # formatZ will make sure item.keys() are Zs
    Zs_expanded = formatZ(item.keys(), unique=True, ordered=True, expand=True)
    res = {}
    for Z in Zs_expanded:
        try:
            res[Z] = item[Z]
        except KeyError:
            try:
                res[Z] = item[tuple(reversed(Z))]
            except KeyError:
                res[Z] = item[tuple(sorted(Z))]
    return res


def padZ(item, Zs: Iterable) -> dict:
    Zs_initial = formatZ(Zs, unique=False, ordered=True, expand=False)
    Zs_formatted = formatZ(Zs, unique=True, ordered=True, expand=False)
    assert Zs_initial == Zs_formatted, "Zs must be unique"
    if isinstance(item, dict):
        if Zs_formatted == formatZ(item.keys(), unique=True, ordered=True, expand=False):
            return item
    return {Z: item for Z in Zs_formatted}


def formatZ(Zs: Iterable, unique: bool = True, ordered: bool = False, expand: bool = False) -> tuple:
    r"""Convert Zs to a sorted tuple of tuples

    Args:
        Zs: Iterable
        unique: bool
            Remove duplicated Zs
        ordered: bool
            Sort each Z tuple when set to False.
            For integral tables: set to True
            For repulsive potentials: set to False
        expand: bool
            Only valid when ordered is set to True
            Add atom pairs in reversed order if they do not exist
            E.g. Input: ((1, 1), (1, 6), (1, 7))
                 Output: ((1, 1), (1, 6), (1, 7), (6, 1), (7, 1))

    Returns:
        res: tuple
    """
    if ordered:
        _Zs = tuple(tuple(Z) for Z in Zs)
    else:
        _Zs = tuple(tuple(sorted(Z)) for Z in Zs)
    if unique:
        _Zs = tuple(set(Z for Z in _Zs))
    res = tuple(sorted(Z for Z in _Zs))
    if expand and ordered:
        _res = list(res)
        for Z in res:
            if Z[0] != Z[1]:
                _res.append((Z[1], Z[0]))
        res = formatZ(_res, unique, ordered)
    # Sanity check
    for Z in res:
        assert len(Z) == 2
        for atom in Z:
            assert atom in ATOM2SYM.keys()
    return res


def Z2A(Zs: Iterable) -> tuple:
    r"""Determine atom types from Zs"""
    res = set()
    for Z in Zs:
        res.update(Z)
    return tuple(sorted(res))


def Zs_from_opts(opts: dict):
    Zs = opts['model_settings']['low_end_correction_dict'].keys()
    return formatZ(Zs, unique=True, ordered=False)