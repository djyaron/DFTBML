# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:40:17 2020

@author: Frank

Module for writing skf files, which are basically text files that adhere
to a specific format as described in the slater-koster file format document

Right now, the writer only handles simple format since we are not doing
angular momenta up to f, only up to d. It is assumed that for all
skf files constructed in this module, the simple format is being used
"""

"""
TODO:
    1) Figure out how to deal with the required spline format
    2) Assemble all information into files
    3) Test out the skfwriter
"""
from batch import Model
import torch
import numpy as np
Array = np.ndarray
from typing import Union, List, Optional, Dict, Any, Literal
import os, os.path
from functools import partial

#%% Some constants
atom_nums = {
    6 : 'C',
    1 : 'H',
    8 : 'O',
    7 : 'N',
    79 : 'Au'
    }

atom_masses = {
    6 : 12.01,
    1 : 1.008,
    8 : 15.999,
    7 : 14.007,
    79 : 196.967
    }
ref_direct = 'auorg-1-1'

#%% Header, H, and S block
def generate_grid(grid_dist: float, ngrid: int, include_endpoint: bool = True) -> Array:
    r"""Generates a grid with a specified grid_dist and the specified number of grid points
    
    Arguments:
        grid_dist (float): The distance between each grid point
        ngrid (int): The number of grid points
        include_endpoint (bool): Whether to include the endpoint or not. Defaults
            to true
    
    Returns:
        rgrid (Array): The distances to evaluate each two-center interaction
            at
    
    Notes: Not using np.linspace for precision concerns and the fact that we 
        want to be able to specify a grid distance
    """
    res = [grid_dist + i * grid_dist for i in range(ngrid if include_endpoint else ngrid - 1)]
    return np.array(res)

def extract_elem_pairs(all_models: Dict) -> List[tuple]:
    r"""Obtains all the element pairs whose interactions are being modeled
    
    Arguments:
        all_models (Dict): Dictionary contianing references to all the models
            being used
    
    Returns:
        pair_lst (list): List of all element pairs as tuples
    """
    double_mods = filter(lambda mod_spec : not isinstance(mod_spec, str) and len(mod_spec.Zs) == 2, 
                         all_models.keys())
    return list(set(map(lambda x : x.Zs, double_mods)))

def extract_S(elems: tuple, atom_nums: Dict, ref_direc: str) -> (Array, float, int):
    r"""Returns a list of all the overlap operator values for a given pair
    
    Arguments:
        elems (tuple): The tuple representing the two elements interacting
        atom_nums (Dict): Dictionary mapping element symbols to their numbers
        ref_direc (str): String representing the reference directory containing
            the given skf files
    
    Returns:
        s_vals (Array): An array of the values for the overlap operator taken from
            the original skf file
        grid_dist (float): The distance needed for the grids
        ngrid (int): The number of grid points to use.
        
    
    Notes: Because we are not fitting the overlap operator, the values for S
        are copied from the original SKF files. The only way for the reference S
        values to be valid is if the same grid distance and ngridpoints are used.
        That's why the grid distance and ngrid are returned, so that the 
        correct number of H values can be made.
    """
    elem1, elem2 = elems
    file_str = os.path.join(ref_direc, f"{atom_nums[elem1]}-{atom_nums[elem2]}.skf")
    with open(file_str, "r") as file:
        content = file.read().splitlines()
    content = list(map(lambda x : x.split(), content))
    grid_dist, ngrid = float(content[0][0]), int(content[0][1])
    startline = 3 if elem1 == elem2 else 2
    #pull out the data block
    datablock = content[startline : startline + ngrid]
    s_vals = list(map(lambda line : line[10:], datablock))
    assert(len(s_vals) == ngrid)
    return np.array(s_vals), grid_dist, ngrid

def get_yvals(model_spec: Model, rgrid: Array, all_models: Dict) -> Array:
    r"""Given a grid of distances, computes the H values using the given model
    
    Arguments:
        rgrid (Array): Array of distances to get values for
        model_spec (Model): The named tuple for the current model
        all_models (Dict): Dictionary referencing all existing models
    
    Returns:
        y_vals (Array): The values computed for each distance in rgrid
    
    Notes: This only works with spline models, not the off-diagonal model.
        However, since the skf files only need the two-center interactions for
        the H and S operators, this is fine.
    """
    spline_model = all_models[model_spec]
    dgrids_consts = spline_model.pairwise_linear_model.linear_model(rgrid, 0)
    model_variables = spline_model.get_variables().detach().numpy()
    if hasattr(spline_model, "joined"):
        fixed_coefs = spline_model.get_fixed().detach().numpy()
        model_variables = np.concatenate((model_variables, fixed_coefs))
    y_vals = np.dot(dgrids_consts[0], model_variables) + dgrids_consts[1]
    return y_vals 

def determine_index(model_spec: Model) -> int:
    r"""Determines the appropriate index of the element based on the orbital type
    
    Arguments: 
        model_spec (Model): The named tuple describing the interaction
    
    Returns:
        index (int): The index in the array
    
    Notes: Based on the Slater-Koster file format for the data, there is a specific
        ordering of the orbitals for the H elements. Refer to the pdf for more 
        information
    
    TODO: If we end up dealing with dd overlaps, need to add that in. For now, 
        sticking only to s and p
    """
    orb_type = model_spec.orb
    if '_' in orb_type:
        orbs, sym = orb_type.split('_')
        orb1, orb2 = orbs[0], orbs[1]
    else:
        orb1, orb2 = orb_type[0], orb_type[1]
        sym = None
    
    if (orb1, orb2) == ('s', 's'):
        return 9
    elif (orb1, orb2) in [('s', 'p'), ('p', 's')]:
        return 8
    elif (orb1, orb2) == ('p', 'p'):
        if sym == 'pi':
            return 6
        elif sym == 'sigma':
            return 5
        elif sym is None: #No symmetry is treated as sigma (should never happen)
            return 5

def compute_H(elems : tuple, all_models: Dict, grid_dist: float, ngrid: int,
              ignore_d: bool = True) -> Array:
    r"""Computes the H-values for each of the two-center interactions
    
    Arguments:
        elems (tuple): The elements concerned in the two-center interaction
        all_models (Dict): Dictionary referencing all the models used
        grid_dist (float): The grid distance to use
        ngrid (int): The number of grid points to use
        ignore_d (bool): Whether or not to ignore d-orbital interactions.
            Defaults to True
    
    Returns:
        h_vals (Array): The values for the Hamiltonian, ordered correctly by interaction
    
    Notes: Will compute the necessary two-body interactions; if not specified, 
        the interactions between two orbitals are assumed to be a sigma interaction
    """
    predicate = lambda mod_spec : not isinstance(mod_spec, str) and mod_spec.Zs == elems and mod_spec.oper == 'H'
    matching_mods = filter(predicate, all_models.keys())
    rgrid = generate_grid(grid_dist, ngrid)
    yval_partial = partial(get_yvals, rgrid = rgrid, all_models = all_models)
    vals_and_ind = map(lambda mod : (yval_partial(mod), determine_index(mod)), matching_mods)
    result = np.zeros((ngrid, 10))
    for val_arr, ind in vals_and_ind:
        result[:, ind] = val_arr
    return result

def construct_datablock(h_block: Array, s_block: Array) -> Array:
    r"""Constructs the datablock for the two-center skf file for the given elements
    
    Arguments:
        h_block (Array): The block containing the hamiltonian values
        s_block (Array): The block containing the overlap values
    
    Returns:
        datablock (Array): An array of shape (ngrid, 20) that contains all the
            data elements as described in the skf file format
    """
    datablock = np.hstack((h_block, s_block))
    return datablock.astype('float64')

def obtain_occupation_ds(elems: tuple, ref_direc: str, atom_nums: Dict) -> (Array, Array):
    r"""Gets the ground state orbital occupation for the elements from the reference SKF file
    
    Arguments:
        elems (tuple): A tuple of the atomic numbers of the elements concerned
        ref_direc (str): The relative path to the directory with the original skf files
        atom_nums (Dict): Dictionary mapping atomic numbers to symbols
    
    Returns:
        occupations (Array): The orbital occupations at ground state as a list
        Ed (Array): On-site energy for the d orbital
        Ud (Array): Hubbard parameter for the d orbital
    """
    filename = os.path.join(ref_direc, f"{atom_nums[elems[0]]}-{atom_nums[elems[0]]}.skf")
    with open(filename, "r") as file:
        content = file.read().splitlines()
        second_line = content[1]
        secondline_elems = second_line.split()
        return (np.array(secondline_elems[7:]).astype('float64'),
                np.array(secondline_elems[0]).astype('float64'),
                np.array(secondline_elems[4]).astype('float64'))
        

def construct_header(elems : tuple, all_models: Dict, atom_nums: Dict, atom_masses: Dict, 
                     grid_dist: float, ngrid: int, ref_direc: str,
                     ignore_d: bool = True) -> List:
    r"""Constructs the header block for the two-center skf file for the given elements
    
    Arguments:
        elems (tuple): The elements in this interaction
        all_models (Dict): The dictionary containing references for all models
        atom_nums (Dict): Dictionary mapping element symbols to atomic numbers
        atom_masses (Dict): Dictionary mapping element number to mass in amu
        grid_dist (float): The grid spacing
        ngrid (int): The number of grid points to use
        ref_direc(str): The relative path to the directory of original skf files
        ignore_d (bool): Whether or not to ignore d orbitals. Defaults to True
    
    Returns:
        headerblock (List): The first few lines of the file. Each line is a separate
            list
    
    Notes: The header is constructed differently for the homonuclear and heteronuclear cases.
        Exact differences are elaborated on in the skf format guide. The first line
        will be a list but the remaining lines will all be np arrays.
        
        In the homonuclear case, we need the occupations of the orbitals at 
        ground state. Rather than hard-coding these, just look them up from the original 
        skf file.
    """
    elem1, elem2 = elems
    line1 = [f"{grid_dist}", f"{ngrid}"]
    mass = np.array([atom_masses[elem1]])
    remainder = np.zeros(19)
    massline = np.hstack((mass, remainder))
    if elem1 == elem2:
        # Short-circuits to prevent asking for p models for H. The original d values are fine
        Ep = 0.0 if elem1 == 1 else all_models[Model('H', (elem1, ), 'p')].variables[0].item()
        Es = all_models[Model('H', (elem1, ), 's')].variables[0].item()
        Up = 0.0 if elem1 == 1 else all_models[Model('G', (elem1, ), 'pp')].variables[0].item()
        Us = all_models[Model('G', (elem1, ), 'ss')].variables[0].item()
        occupations, Ed_orig, Ud_orig = obtain_occupation_ds(elems, ref_direc, atom_nums)
        Ed = Ed_orig if ignore_d else all_models[Model('H', (elem1, ), 'd')].variables[0].item()
        Ud = Ud_orig if ignore_d else all_models[Model('G', (elem1, ), 'dd')].variables[0].item()
        Erun = np.array([Ed, Ep, Es])
        Urun = np.array([Ud, Up, Us])
        SPE = np.array([0.0])
        secondline = np.hstack((Erun, SPE, Urun, occupations)).astype('float64')
        return [line1, secondline, massline]
    else:
        return [line1, massline]

#%% Spline block








#%% File Assembly 


if __name__ == "__main__":
    pass