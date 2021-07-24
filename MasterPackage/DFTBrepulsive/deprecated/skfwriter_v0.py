# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:40:17 2020

@author: Frank

Module for writing skf files, which are basically text files that adhere
to a specific format as described in the slater-koster file format document

Right now, the writer only handles simple format since we are not doing
angular momenta up to f, only up to d. It is assumed that for all
skf files constructed in this module, the simple format is being used

TODO:
    1) Need to go closer with the ranges for the repulsive models
"""

import numpy as np
import os
import pickle as pkl
from consts import ANGSTROM2BOHR, CUTOFFS
from scipy.interpolate import CubicSpline
from typing import List, Dict

# TODO: Generate SKF for dense/sparse model (using a selector)

Array = np.ndarray
ATOM_NUMS = {1: 'H', 6: 'C', 8: 'O', 7: 'N', 79: 'Au'}
REF_DIR = '../slakos/auorg-1-1'


def load_file_content(Z: tuple, ref_direc: str, atom_nums: Dict) -> List[List[str]]:
    r"""Loads the necessary file content

    Arguments:
        Z (tuple): Atomic numbers of elements whose interaction is being modeled
        ref_direc (str): The reference directory
        atom_nums (Dict): Dictionary mapping atomic number to their symbols

    Returns:
        content (List[List[str]]): The content of the reference skf file
            as a 2D list of strings
    """
    elem1, elem2 = Z
    atom1, atom2 = atom_nums[elem1], atom_nums[elem2]
    full_path = os.path.join(ref_direc, f"{atom1}-{atom2}.skf")
    with open(full_path, "r") as file:
        content = file.read().splitlines()
        content = list(map(lambda x: x.split(), content))
        return content


def compute_spline_repulsive(res_path: str, Zs: tuple) -> (Array, Array, float):
    r"""Computes the repulsive spline coefficients for the repulsive block

    Arguments:
        res_path (str): The path of saved results
        Zs (tuple): The elements whose repulsive interaction is concerned

    Returns:
        coeffs (Array): An array of coefficients for the cubic splines of
            each segment fit to bohr radius on the x-axis
        rgrid (Array): The distance array that the spline spans, in angstroms
        cutoff (float): The cutoff distance for the spline in angstroms

    Notes: To get the coefficients of the correct form, we perform a scipy interpolate
        with CubicSplines to get the coeffs.
    """

    with open(res_path, 'rb') as file:
        res = pkl.load(file)

    _Zs = tuple(sorted(Zs))
    # param_grid = res['search_grid']

    # TODO: SKF for dense or sparse model
    xydata = res['all_xydata']['sparse_xydata'][0][0][0][_Zs]
    # xydata = res['all_xydata']['dense_xydata'][0][_Zs]

    rgrid = xydata[0]
    r_vals = xydata[1]
    # WARNING: HARDCODED CUTOFF!
    cutoff = CUTOFFS['au_full'][_Zs][1]
    # cutoff = CUTOFFS[param_grid['rmax'][0]][_Zs][1]

    # Obtain the spline, but given the file format we must
    # fit the spline with units of bohr radii vs hartree
    spl = CubicSpline(rgrid * ANGSTROM2BOHR, r_vals)
    return spl.c, rgrid, cutoff


def assemble_spline_body_block(coeffs: Array, rgrid: Array) -> List:
    r"""Generates the necessary format for the coefficients for the spline

    Arguments:
        coeffs (Array): The array of spline coefficients of shape (k, m) where
            k is the degree and m is the number of intervals
        rgrid (Array): The distances that the spline spans in angstroms

    Returns:
        spline_block (List): The correctly formatted spline block; each row of
            the block corresponds to one line

    Notes: The slater-koster file format specifies that the last spline in the
        block should be a fifth degree spline, but for this purpose we will
        do cubic splines. Thus, the last line is padded with 0s
    """
    rgrid = rgrid * ANGSTROM2BOHR  # Convert from angstrom to bohr
    intervals = np.array([rgrid[:-1], rgrid[1:]]).T
    assert (len(intervals) == coeffs.shape[1])
    rows = []
    for index, interval in enumerate(intervals):
        curr_coeffs = list(coeffs[:, index])
        curr_coeffs.reverse()  # Reversal of coefficients in accordance with scipy docs
        rows.append([interval[0], interval[1], *curr_coeffs])
    rows[-1].extend([0, 0])
    return rows


def assemble_spline_header(content: List, ngrid: int, cutoff: float) -> List:
    r"""Constructs the spline block header
    Arguments:
        content (List): The original contents of the reference file as a list of lists
        ngrid (int): The number of grid points
        cutoff (float): The cutoff distance

    Returns:
        Header (List): A list of the spline header, where each row is one
            line in the spline header

    The main trick is to get the close-range coefficients out
    """
    line1 = ['Spline']
    line2 = [ngrid - 1, cutoff * ANGSTROM2BOHR]  # The number of intervals is equal to gridpoints - 1
    index = 0
    for i in range(len(content)):
        if len(content[i]) == 1 and content[i][0] == 'Spline':
            index = i + 2
            break
    line3 = content[index]
    return [line1, line2, line3]


def combine_list_to_str(strlst: List, sep: str = "  ") -> str:
    r"""Combines a list of strings into a single string

    Arguments:
        strlst (List): A list of strings to combine
        sep (str): The separator for each string element. Defaults to two spaces

    Returns:
        combined_str (str): The combined string with the given separator
    """
    return sep.join(strlst) + "\n"


def write_single_skf_file(res_path: str, save_dir: str, Z: tuple, atom_nums: Dict,
                          ref_direc: str, str_sep: str = "  ",
                          spline_ngrid: int = 50, ext: str = None) -> None:
    r"""Write the skf file for a single atom pair

    Arguments:
        save_dir:
        res_path: Path to results
        Z (tuple): Atom pair to write the skf file for
        atom_nums (Dict): The dictionary mapping atom numbers to their symbols
        ref_direc (str): The relative path to the directory containing all skf files
        str_sep (str): The separator for each line
        spline_ngrid (int): The number of gridpoints ot use for the splines
        ext (str): Additional save path (e.g., to a directory or something).
            Defaults to None

    Returns:
        None

    Notes: For the repulsive splines, the intervals go from xlow -> cutoff, where
        cutoff is the value assigned to the spline during initialization (this only
        applies to joined spline models) and xlow is the lowest distance for the
        model across all the data

        The coefficients a1, a2, a3 for the very near-range repulsive interaction,
        described by Exp[-a1 * r + a2] + a3 are copied from the initial SKF file

        For the fifth order spline at the end, the 4th and 5th coefficients are set to
        0 so all splines are technically cubic splines

        The number of intervals is equal to spline_ngrid - 1

        The SPE field is not copied and defaults to zero, since the spin polarization
        error is not interpreted by DFTB+

        In the case of heteronuclear interactions, the mass field in line 2 is
        defaulted as the mass of the first element in the pair. This does not
        matter since mass is a placeholder in heteronuclear cases and is not interpreted

        If d-orbital values exist for the hubbard parameters, these are copied over
        from the original file

        Because we are not fitting S, the S-block of all skf files are pulled from their
        respective reference files
    """

    elem1, elem2 = Z
    atom1, atom2 = atom_nums[elem1], atom_nums[elem2]
    full_path = os.path.join(ref_direc, f"{atom1}-{atom2}.skf")
    with open(full_path, "r") as file:
        content_line = file.readlines()
        content = load_file_content(Z, ref_direc, atom_nums)

    HS_block = []
    for line in content_line:
        if 'spline' in line or 'Spline' in line:
            break
        HS_block.append(line)

    # Dealing with the spline datablock and header
    spline_coeffs, spline_grid, cutoff = compute_spline_repulsive(res_path, Z)
    spline_block = assemble_spline_body_block(spline_coeffs, spline_grid)
    # spline_header = assemble_spline_header(content, spline_ngrid, cutoff)
    spline_header = assemble_spline_header(content, len(spline_grid), cutoff)

    # Final file assembly
    elem1, elem2 = Z
    atom1, atom2 = atom_nums[elem1], atom_nums[elem2]
    save_file_name = f"{atom1}-{atom2}.skf" if ext is None else os.path.join(ext, f"{atom1}-{atom2}.skf")
    save_path = os.path.join(save_dir, save_file_name)
    with open(save_path, "w+") as handle:
        for line in HS_block:
            handle.write(line)

        for line in spline_header:
            write_line = list(map(lambda x: str(x), line))
            combined_str = combine_list_to_str(write_line, str_sep)
            handle.write(combined_str)

        for line in spline_block:
            write_line = list(map(lambda x: str(x), line))
            combined_str = combine_list_to_str(write_line, str_sep)
            handle.write(combined_str)

        handle.close()


def main(res_path: str, save_dir: str, atom_nums: dict = ATOM_NUMS, ref_direc: str = REF_DIR,
         str_sep: str = "  ", spline_ngrid: int = 50, ext: str = None) -> None:
    r"""Main method for writing out all the skf files for the given set of models

    Arguments:
        save_dir:
        res_path: Path to results
        atom_nums:
        ref_direc:
        str_sep (str): The separator for each line, defaults to two spaces
        spline_ngrid (int): The number of grid points to use for the splines.
            Defaults to 50
        ext (str): Additional save path (e.g., to a directory or something).
            Defaults to None

    Returns:
        None

    Notes: See notes for write_single_skf_file
    """
    with open(res_path, 'rb') as file:
        res = pkl.load(file)

    Zs = list(res['all_xydata']['sparse_xydata'][0][0][0].keys())
    for Z in Zs:
        write_single_skf_file(res_path, save_dir, Z, atom_nums, ref_direc, str_sep, spline_ngrid, ext)
        if Z[0] != Z[1]:
            _Z = (Z[1], Z[0])
            write_single_skf_file(res_path, save_dir, _Z, atom_nums, ref_direc, str_sep, spline_ngrid, ext)

if __name__ == "__main__":
    from util import path_check

    # res_path = "/home/francishe/Downloads/SKF/cv_nknots.pkl"
    # save_dir = "/home/francishe/Downloads/SKF/ANI_skf/"
    # path_check(save_dir)
    # main(res_path, save_dir)

    # res_path = '/home/francishe/Documents/DFTBrepulsive/Au_cv/Au_cv (cvxopt, nknots=50, deg=3, rmax=au_short~au_short, ptype=convex)/Au_cv_rmax.pkl'
    # res_path = '/home/francishe/Documents/DFTBrepulsive/Au_cv/Au_cv (lstsq, nknots=50, deg=3, rmax=au_short~au_short, ptype=None)/Au_cv_rmax.pkl'
    res_path = '/home/francishe/Documents/DFTBrepulsive/a1K_full_xydata.pkl'
    # save_dir = "/home/francishe/Documents/DFTBrepulsive/SKF/aed_convex/"
    save_dir = "/home/francishe/Documents/DFTBrepulsive/SKF/a1k/"
    # Modify line 75 to select between dense/sparse model
    path_check(save_dir)

    main(res_path, save_dir)
