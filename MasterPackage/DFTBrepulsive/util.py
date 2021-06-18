from __future__ import annotations

import os
import pickle as pkl
from _pickle import UnpicklingError
from collections import Counter
from time import time
from typing import Optional, Iterable

import matplotlib as mpl
from h5py import File


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


def get_dataset_type(dataset_path: str) -> str:
    r"""Check the dataset type (HDF5 or pickle) and the validity of the dataset

    Args:
        dataset_path: Path to the dataset

    Returns:
        dataset_type: "h5" or "pkl"

    Examples:
        >>> from util import get_dataset_type
        >>> dataset_path_1 = "DIR/valid_dataset.h5"
        >>> get_dataset_type(dataset_path_1)
        'h5'
        >>> dataset_path_2 = "DIR/corrupted_dataset.pkl"
        >>> get_dataset_type(dataset_path_2)
        ValueError: DIR/corrupted_dataset.pkl is not a valid pickle dataset

    Raises:
        FileNotFoundError: if the dataset does not exist
        ValueError: if dataset cannot be opened
        NotImplementedError: if the extension of the dataset is not one of the following
                             '.hdf', '.h4', '.hdf4', '.he2', '.h5', '.hdf5', '.he5',
                             '.pkl', '.pickle', '.p'
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not exists")

    ext = os.path.splitext(dataset_path)[-1].lower()
    if ext in ('.hdf', '.h4', '.hdf4', '.he2', '.h5', '.hdf5', '.he5'):
        try:
            with File(dataset_path, 'r'):
                dataset_type = 'h5'
        except OSError:
            raise ValueError(f"{dataset_path} is not a valid HDF5 dataset")
    elif ext in ('.pkl', '.pickle', '.p'):
        try:
            with open(dataset_path, 'rb') as f:
                pkl.load(f)
                dataset_type = 'pkl'
        except UnpicklingError:
            raise ValueError(f"{dataset_path} is not a valid pickle dataset")
    else:
        raise NotImplementedError(f"{dataset_path} is not a supported dataset")

    return dataset_type


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


def formatZ(Zs: Iterable, unique: bool = True, ordered: bool = False) -> tuple:
    r"""Convert Zs to a sorted tuple of tuples

    Args:
        Zs: Iterable
        unique: bool
            Remove duplicated Zs
        ordered: bool
            Sort each Z tuple when set to False.
            For integral tables: set to True
            For repulsive potentials: set to False

    Returns:
        tuple

    """
    if ordered:
        _Zs = tuple(tuple(Z) for Z in Zs)
    else:
        _Zs = tuple(tuple(sorted(Z)) for Z in Zs)
    if unique:
        _Zs = tuple(set(Z for Z in _Zs))
    return tuple(sorted(Z for Z in _Zs))

def Z2A(Zs: Iterable) -> tuple:
    r"""Determine atom types from Zs"""
    res = set()
    for Z in Zs:
        res.update(Z)
    return tuple(sorted(res))

def padZ(item, Zs: Iterable) -> dict:
    _Zs = formatZ(Zs)
    if isinstance(item, dict):
        if _Zs == formatZ(item.keys()):
            return item
    return {Z: item for Z in _Zs}
