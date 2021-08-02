# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:40:17 2020

@author: Frank

Module for writing skf files, which are basically text files that adhere
to a specific format as described in the slater-koster file format document

Right now, the writer only handles simple format since we are not doing
angular momenta up to f, only up to d. It is assumed that for all
skf files constructed in this module, the simple format is being used

TODO: NEED TO TEST NEW SKF CODE AFTER NEW DFTBREPULSIVE CODE INTEGRATED!
"""

import numpy as np
Array = np.ndarray
from typing import List, Dict
import os, os.path, pickle
from functools import partial
from scipy.interpolate import CubicSpline
from MasterConstants import ANGSTROM2BOHR, Model, H_entries, S_entries, atom_header_entries
from DFTBrepulsive import SKF, SKFBlockCreator

#%% Header, H, and S block
def load_file_content(elems: tuple, ref_direc: str, atom_nums: Dict) -> List[List[str]]:
    r"""Loads the necessary file content
    
    Arguments:
        elems (tuple): Atomic numbers of elements whose interaction is being modeled
        ref_direc (str): The reference directory
        atom_nums (Dict): Dictionary mapping atomic number to their symbols
    
    Returns:
        content (List[List[str]]): The content of the reference skf file 
            as a 2D list of strings
    """
    elem1, elem2 = elems
    atom1, atom2 = atom_nums[elem1], atom_nums[elem2]
    full_path = os.path.join(ref_direc, f"{atom1}-{atom2}.skf")
    with open(full_path, "r") as file:
        content = file.read().splitlines()
        content = list(map(lambda x : x.split(), content))
        return content
    
def get_grid_info(content: List[List[str]]) -> (float, int):
    r"""Gets the necessary grid distance and number of grid points for the H,S block
    
    Arguments:
        content (List[List[str]]): The contents of a file as a 2D list
    
    Returns:
        (grid_dist, ngrid) (float, int): The grid distance and the number of grid points
            for the H and S block
        
    Notes: The grid distance is in units of bohr radius, but all calculations 
        in the model are done using angstroms
    """
    return (float(content[0][0]), int(content[0][1]))

def generate_grid(grid_dist: float, ngrid: int, include_endpoint: bool = True) -> Array:
    r"""Generates a grid with a specified grid_dist and the specified number of grid points
    
    Arguments:
        grid_dist (float): The distance between each grid point, in bohr radii
        ngrid (int): The number of grid points
        include_endpoint (bool): Whether to include the endpoint or not. Defaults
            to true
    
    Returns:
        rgrid (Array): The distances to evaluate each two-center interaction
            at, in angstroms
    
    Notes: Not using np.linspace for precision concerns and the fact that we 
        want to be able to specify a grid distance
        
        To convert from bohr-radii to angstroms, we divide the resulting grid
        by the factor ANGSTROM2BOHR, imported from dftb
    """
    res = [grid_dist + i * grid_dist for i in range(ngrid if include_endpoint else ngrid - 1)]
    res = np.array(res) / ANGSTROM2BOHR
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

def extract_S_content(elems: tuple, content: List[List[str]], ngrid: int) -> Array:
    r"""Grabs the S-block from the content of the original skf file
    ngrid here is the second result from get_grid_info
    """
    elem1, elem2 = elems
    startline = 3 if elem1 == elem2 else 2
    datablock = content[startline : startline + ngrid]
    svals = list(map(lambda line : line[10:], datablock))
    assert(len(svals) == ngrid)
    return np.array(svals)

def get_yvals(model_spec: Model, rgrid: Array, all_models: Dict) -> Array:
    r"""Given a grid of distances, computes the H values using the given model
    
    Arguments:
        rgrid (Array): Array of distances to get values for, must be in angstroms
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
    model_variables = spline_model.get_variables().detach().cpu().numpy()
    if hasattr(spline_model, "joined"):
        fixed_coefs = spline_model.get_fixed().detach().cpu().numpy()
        model_variables = np.concatenate((model_variables, fixed_coefs))
    y_vals = np.dot(dgrids_consts[0], model_variables) + dgrids_consts[1]
    # Instead of 0's, pad the front dummy values with the first non-zero 
    # values to allow for smoother spline interpolation
    ind = 0
    for index, elem in enumerate(y_vals):
        if elem != 0:
            ind = index
            break
    y_vals[:ind] = y_vals[ind]
    # Plot the values as a scatter to see what's being written to the skf
    # files as a debugging step
    # fig, ax = plt.subplots()
    # ax.scatter(rgrid, y_vals)
    # ax.set_title(f"{model_spec.oper}, {model_spec.Zs}, {model_spec.orb}")
    # plt.show()
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

def compute_S(elems: tuple, all_models: Dict, grid_dist: float, ngrid : int, 
              ignore_d: bool = True) -> Array:
    r"""Computes values for the overlap operator block, S
    
    Arguments:
        elems (tuple): The elements concerned in the two-center interaction
        all_models (Dict): Dictionary referencing all the models used
        grid_dist (float): The grid distance to use, in units of bohr radii
        ngrid (int): The number of grid points to use
        ignore_d (bool): Whether or not to ignore d-orbital interactions. 
            Defaults to True
    
    Returns:
        s_vals (Array): The values for the overlapoperator, ordered
            by the interaction. Order specified in slater-koster file format
    
    Notes: Logic is very similar to that of compute_H
    """
    predicate = lambda mod_spec : not isinstance(mod_spec, str) and\
        (mod_spec.oper == 'S') and (mod_spec.Zs == elems or mod_spec.Zs == (elems[1], elems[0]))\
            and (mod_spec.orb in ['ss', 'pp_pi', 'pp_sigma'])
    # Or ensure equivalence of Zs but the orbital type is not symmetric
    predicate2 = lambda mod_spec : not isinstance(mod_spec, str) and\
        (mod_spec.oper == 'S') and (mod_spec.Zs == elems) and (mod_spec.orb in ['sp'])
    matching_mods = filter(lambda x : (predicate(x) or predicate2(x)), all_models.keys())
    rgrid = generate_grid(grid_dist, ngrid) #rgrid here is in angstroms
    yval_partial = partial(get_yvals, rgrid = rgrid, all_models = all_models)
    vals_and_ind = map(lambda mod : (yval_partial(mod), determine_index(mod)), matching_mods)
    result = np.zeros((ngrid, 10))
    for val_arr, ind in vals_and_ind:
        result[:, ind] = val_arr
    return result

def compute_H(elems : tuple, all_models: Dict, grid_dist: float, ngrid: int,
              ignore_d: bool = True) -> Array:
    r"""Computes the H-values for each of the two-center interactions
    
    Arguments:
        elems (tuple): The elements concerned in the two-center interaction
        all_models (Dict): Dictionary referencing all the models used
        grid_dist (float): The grid distance to use, in units of bohr radii
        ngrid (int): The number of grid points to use
        ignore_d (bool): Whether or not to ignore d-orbital interactions.
            Defaults to True
    
    Returns:
        h_vals (Array): The values for the Hamiltonian, ordered correctly by interaction
    
    Notes: Will compute the necessary two-body interactions; if not specified, 
        the interactions between two orbitals are assumed to be a sigma interaction
    """
    #Account for the reverse order as well to get all the values 
    # between the two elements
    # Either flip the Zs but the orbital type must be symmetric
    predicate = lambda mod_spec : not isinstance(mod_spec, str) and\
        (mod_spec.oper == 'H') and (mod_spec.Zs == elems or mod_spec.Zs == (elems[1], elems[0]))\
            and (mod_spec.orb in ['ss', 'pp_pi', 'pp_sigma'])
    # Or ensure equivalence of Zs but the orbital type is not symmetric
    predicate2 = lambda mod_spec : not isinstance(mod_spec, str) and\
        (mod_spec.oper == 'H') and (mod_spec.Zs == elems) and (mod_spec.orb in ['sp'])
    matching_mods = filter(lambda x : (predicate(x) or predicate2(x)), all_models.keys())
    rgrid = generate_grid(grid_dist, ngrid) #rgrid here is in angstroms
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

def obtain_occupation_ds(content: List) -> (Array, Array):
    r"""Gets the ground state orbital occupation for the elements from the reference SKF file
    
    Arguments:
        elems (tuple): A tuple of the atomic numbers of the elements concerned
        content (List): The split lines of the file
        atom_nums (Dict): Dictionary mapping atomic numbers to symbols
    
    Returns:
        occupations (Array): The orbital occupations at ground state as a list
        Ed (Array): On-site energy for the d orbital
        Ud (Array): Hubbard parameter for the d orbital
    
    Notes: Hard-coded indexing works with simple format SKF
    """
    second_line = content[1]
    return (np.array(second_line[7:]).astype('float64'),
            np.array(second_line[0]).astype('float64'),
            np.array(second_line[4]).astype('float64'))
        

def construct_header(elems : tuple, all_models: Dict, atom_masses: Dict, 
                     grid_dist: float, ngrid: int, content: List,
                     ignore_d: bool = True) -> List:
    r"""Constructs the header block for the two-center skf file for the given elements
    
    Arguments:
        elems (tuple): The elements in this interaction
        all_models (Dict): The dictionary containing references for all models
        atom_masses (Dict): Dictionary mapping element number to mass in amu
        grid_dist (float): The grid spacing
        ngrid (int): The number of grid points to use
        content (List): The content of the reference skf file
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
        occupations, Ed_orig, Ud_orig = obtain_occupation_ds(content) #NOTE: this will have to change when dealing with elements with angular momenta greater than p
        Ed = Ed_orig if ignore_d else all_models[Model('H', (elem1, ), 'd')].variables[0].item()
        Ud = Ud_orig if ignore_d else all_models[Model('G', (elem1, ), 'dd')].variables[0].item()
        Erun = np.array([Ed, Ep, Es])
        Urun = np.array([Ud, Up, Us])
        SPE = np.array([0.0]) #Not interpreted, assumed to be unimportant
        secondline = np.hstack((Erun, SPE, Urun, occupations)).astype('float64')
        return [line1, secondline, massline]
    else:
        return [line1, massline]

#%% Spline block

def compute_spline_repulsive(elems: tuple, all_models: Dict, ngrid: int = 50) -> (Array, Array, float):
    r"""Computes the repulsive spline coefficients for the repulsive block
    
    Arguments:
        elems (tuple): The elements whose repulsive interaction is concerned
        all_models (Dict): Dictionary referencing all the models
        ngrid (int): The number of gridpoints to use for the repulsive splines. 
            Defaults to 50
    
    Returns:
        coeffs (Array): An array of coefficients for the cubic splines of 
            each segment fit to bohr radius on the x-axis
        rgrid (Array): The distance array that the spline spans, in angstroms
        cutoff (float): The cutoff distance for the spline in angstroms
        
    Notes: To get the coefficients of the correct form, we perform a scipy interpolate
        with CubicSplines to get the coeffs.
    """
    R_mods = [mod for mod in all_models.keys() if isinstance(mod, Model) and mod.oper == 'R' and\
              (mod.Zs == elems or mod.Zs == (elems[1], elems[0]))]
    assert(len(R_mods) == 1) # Should only be one repulsive mod per atom pair
    r_model = all_models[R_mods[0]]
    xlow, xhigh = r_model.pairwise_linear_model.r_range()
    cutoff = r_model.cutoff #Use the cutoff distance from the model itself (only works for joined splines)
    rgrid = np.linspace(xlow, cutoff, ngrid) #rgrid here is in angstroms
    r_vals = get_yvals(R_mods[0], rgrid, all_models)
    #Obtain the spline, but given the file format we must
    # fit the spline with units of bohr radii vs hartree
    spl = CubicSpline(rgrid * ANGSTROM2BOHR, r_vals)
    # fig, ax = plt.subplots()
    #Plot out the cubic spline fit for reference
    # ax.plot(rgrid * ANGSTROM2BOHR, spl(rgrid * ANGSTROM2BOHR))
    # ax.set_title(f"{elems}, rep spline ref")
    # plt.show()
    #Coefficients of the spline
    assert(spl.c.shape[1] == ngrid - 1)
    return spl.c, rgrid, cutoff

def compute_spline_repulsive_vals(elems: tuple, all_models: Dict, ngrid: int = 50) -> (Array, Array):
    r"""Same functionality as compute_spline_repulsive, but returns the rgrid and 
        the values for each point in rgrid instead of spline coefficients. Again, 
        used with the new DFTBRepulsive SKF functionalities.
        
    Arguments:
        elems (tuple): The elements whose repulsive interaction is concerned
        all_models (Dict): Dictionary referencing all the models
        ngrid (int): The number of gridpoints to use for the repulsive splines. 
            Defaults to 50
    
    Returns:
        rgrid (Array): The rgrid that was used, in angstroms
        r_vals (Array): The values from evaluating the repulsive model
            on the given rgrid
    
    Notes: Even though this is code duplication, the idea is to make the 
        old repulsive models backward compatible with the new DFTBRepulsive 
        SKF functionalities
    """
    R_mods = [mod for mod in all_models.keys() if isinstance(mod, Model) and mod.oper == 'R' and\
              (mod.Zs == elems or mod.Zs == (elems[1], elems[0]))]
    assert(len(R_mods) == 1) # Should only be one repulsive mod per atom pair
    r_model = all_models[R_mods[0]]
    xlow, xhigh = r_model.pairwise_linear_model.r_range()
    cutoff = r_model.cutoff #Use the cutoff distance from the model itself (only works for joined splines)
    rgrid = np.linspace(xlow, cutoff, ngrid) #rgrid here is in angstroms
    r_vals = get_yvals(R_mods[0], rgrid, all_models)
    return rgrid, r_vals

# def compute_spline_repulsive_new(elems: tuple, all_models: Dict, ngrid: int = 50) -> (Array, Array, float):
#     r"""Computes the repulsive spline coefficients for the repulsive block 
#         using the new repulsive model
        
#     Arguments, Returns: identical to compute_spline_repulsive
    
#     Notes: This is based on the new repulsive model, i.e. integration with
#         DFTBrepulsive. The built-in method to the new repulsive model, 
#         obtain_spline_vals(), is used to get the values out using a 
#         given rgrid. 
#     """
#     #obtain the cutoff and range
#     r_low, r_high = all_models['rep'].obtain_range(elems)
#     cutoff = all_models['rep'].obtain_cutoff(elems)
#     rgrid = np.linspace(r_low, cutoff, ngrid)
#     r_vals = all_models['rep'].obtain_spline_vals(rgrid, elems)
#     spl = CubicSpline(rgrid * ANGSTROM2BOHR, r_vals)
#     assert(spl.c.shape[1] == ngrid - 1)
#     return spl.c, rgrid, cutoff

def compute_spline_repulsive_new(elems: tuple, all_models: Dict) -> Dict:
    r"""With the new repulsive model, the spline repulsive block should 
        already be present in all_models['rep'].
    
    Arguments:
        elems (tuple): The elements currently being written.
        all_models (Dict): Dictionary of all trained models used during the 
            run
    
    Returns:
        spl_block (Dict): The spline block for the given element. 
    """
    elems_rev = (elems[-1], elems[0])
    spl_block = all_models['rep'].spl_block[elems] if elems in all_models['rep'].spl_block\
        else all_models['rep'].spl_block[elems_rev]
    return spl_block

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
    rgrid = rgrid * ANGSTROM2BOHR #Convert from angstrom to bohr
    intervals = [(rgrid[i], rgrid[i+1]) for i in range(len(rgrid) - 1)]
    assert(len(intervals) == coeffs.shape[1])
    rows = []
    for index, interval in enumerate(intervals):
        curr_coeffs = list(coeffs[:, index])
        curr_coeffs.reverse() #Reversal of coefficients in accordance with scipy docs
        rows.append([interval[0], interval[1]] + curr_coeffs)
    rows[-1].extend([0, 0])
    return rows

def assemble_spline_header(rgrid: Array, content: List, ngrid : int, cutoff: float) -> List:
    r"""Constructs the spline block header
   
    Arguments:
        rgrid (Array): The distances spanned by the spline
        content (List): The original contents of the reference file as a list of lists
        ngrid (int): The number of grid points
        cutoff (float): The cutoff distance
    
    Returns:
        Header (List): A list of the spline header, where each row is one 
            line in the spline header
    
    The main trick is to get the close-range coefficients out
    """
    line1 = ['Spline']
    line2 = [ngrid - 1, cutoff * ANGSTROM2BOHR] #The number of intervals is equal to gridpoints - 1
    index = 0
    for i in range(len(content)):
        if len(content[i]) == 1 and content[i][0] == 'Spline':
            index = i + 2
            break
    line3 = content[index]
    return [line1, line2, line3]

#%% File Assembly (Old method)  
def combine_list_to_str(strlst: List, sep: str = "  ") -> str:
    r"""Combines a list of strings into a single string
    
    Arguments:
        strlst (List): A list of strings to combine
        sep (str): The separator for each string element. Defaults to two spaces
    
    Returns:
        combined_str (str): The combined string with the given separator
    """
    return sep.join(strlst) + "\n"

def write_single_skf_file(elems: tuple, all_models: Dict, atom_nums: Dict,
                          atom_masses: Dict, compute_S_block: bool, ref_direc: str,
                          rep_mode: str, str_sep: str = "  ",
                          spline_ngrid: int = 50, ext: str = None) -> None:
    r"""Write the skf file for a single atom pair
    
    Arguments:
        elems (tuple): Atom pair to write the skf file for 
        all_models (Dict): Dictionary containing references to all models
        atom_nums (Dict): The dictionary mapping atom numbers to their symbols
        atom_masses (Dict): The dictionary mapping atom numbers to their masses
        compute_S_block (bool): Whether or not to compute values for the overlap operator S.
        ref_direc (str): The relative path to the directory containing all skf files
        rep_mode (str): One of 'new' and 'old', indicates which method should be used
            to compute the repulsive spline blocks
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
    #Dealing with the H,S datablock
    content = load_file_content(elems, ref_direc, atom_nums)
    grid_dist, ngrid = get_grid_info(content) #grid_dist in bohr here
    #WARNING: HARDCODE TO INCREASE GRID DENSITY FOR SKFS, REMOVE LATER
    # grid_dist /= 2
    # ngrid *= 2 
    if compute_S_block: #When fitting S
        print("Computing S block")
        s_block = compute_S(elems, all_models, grid_dist, ngrid)
    else:
        print("Extracting S block")
        s_block = extract_S_content(elems, content, ngrid)
    h_block = compute_H(elems, all_models, grid_dist, ngrid)
    HS_datablock = construct_datablock(h_block, s_block)
    HS_header = construct_header(elems, all_models, atom_masses, grid_dist, ngrid, content)
    
    #Dealing with the spline datablock and header
    if rep_mode == 'old':
        print("Calling compute_spline_repulsive")
        spline_coeffs, spline_grid, cutoff = compute_spline_repulsive(elems, all_models, spline_ngrid)
        spline_block = assemble_spline_body_block(spline_coeffs, spline_grid)
        spline_header = assemble_spline_header(spline_grid, content, spline_ngrid, cutoff)
    elif rep_mode == 'new':
        print("Calling compute_spline_repulsive_new")
        spline_block = compute_spline_repulsive_new(elems, all_models)
        spline_header = spline_block[:3]
        spline_block = spline_block[3:]
    else:
        raise ValueError("Unrecognized repulsive setting")
    
    #Final file assembly
    elem1, elem2 = elems
    atom1, atom2 = atom_nums[elem1], atom_nums[elem2]
    save_file_name = f"{atom1}-{atom2}.skf" if ext is None else os.path.join(ext, f"{atom1}-{atom2}.skf")
    with open(save_file_name, "w+") as handle:
        for line in HS_header:
            write_line = list(map(lambda x : str(x), line))
            combined_str = combine_list_to_str(write_line, str_sep)
            handle.write(combined_str)
        
        for line in HS_datablock:
            write_line = list(map(lambda x : str(x), line))
            combined_str = combine_list_to_str(write_line, str_sep)
            handle.write(combined_str)
        
        for line in spline_header:
            write_line = list(map(lambda x : str(x), line))
            combined_str = combine_list_to_str(write_line, str_sep)
            handle.write(combined_str)
            
        for line in spline_block:
            write_line = list(map(lambda x : str(x), line))
            combined_str = combine_list_to_str(write_line, str_sep)
            handle.write(combined_str)
        
        handle.close()

def main(all_models: Dict, atom_nums: Dict, atom_masses: Dict, compute_S_block: bool,
         ref_direc: str, rep_mode: str, opts: Dict, str_sep: str = "  ", spline_ngrid: int = 50, ext: str = None) -> None:
    r"""Main method for writing out all the skf files for the given set of models
    
    Arguments:
        all_models (Dict): Dictionary containing references to all models
        atom_nums (Dict): The dictionary mapping atom numbers to their symbols
        atom_masses (Dict): The dictionary mapping atom numbers to their masses
        compute_S (bool): Whether to compute values for overlap operator or
            extract them from reference file
        ref_direc (str): The relative path to the directory containing all skf files
        rep_mode (str): One of 'new' and 'old', indicates which method should be used
            to compute the repulsive spline blocks
        opts (Dict): The dictionary containing the hyperparameter settings for 
            DFTBrepulsive. 
        str_sep (str): The separator for each line, defaults to two spaces
        spline_ngrid (int): The number of gridpoints to use for the repulsive splines.
            Defaults to 50
        ext (str): Additional save path (e.g., to a directory or something). 
            Defaults to None
    
    Returns:
        None
    
    Notes: See notes for write_single_skf_file
    """
    elem_pairs = extract_elem_pairs(all_models)
    #If the new repulsive model is being used, generate the spline block from 
    #   the DFTBrepulsive model immediately.
    if rep_mode == 'new':
        if (all_models['rep'].mode is not None) and (all_models['rep'].mode == 'internal'):
            all_models['rep'].update_mod_coefs()
        all_models['rep'].calc_spline_block(opts)
        print("Saving reference energy parameters")
        ref_dict = all_models['rep'].get_ref_ener_info(False)
        filename = "ref_params.p" if ext is None else os.path.join(ext, "ref_params.p")
        with open(filename, 'wb') as handle:
            pickle.dump(ref_dict, handle)
    for pair in elem_pairs:
        write_single_skf_file(pair, all_models, atom_nums, atom_masses, compute_S_block, 
                              ref_direc, rep_mode, str_sep, spline_ngrid, ext)

#%% File Assembly (New method, using DFTBrepulsive)

#TODO: Implement me!

"""
These functions will use the functions written in the Header, H, and S block to assemble the
dictionaries necessary for the SKF functions in the DFTBrepulsive backend. 
"""

def obtain_HS_dict(H_block: Array, S_block: Array) -> Dict:
    r"""Generates the dictionary formatted input for the Hamiltonian and overlap
        operator integral elements.
    
    Arguments: 
        H_block (Array): The Hamiltonian integral block, of shape (ngrid, 10)
        S_block (Array): The overlap integral block, of shape (ngrid, 10)
    
    Returns:
        HS_dict (Dict): The dictionary that decomposes the H and S blocks 
            into their integral labels. 
    
    Notes: This method only works with the simple format of the SKF files, not
        the extended format. This function depends on the compute_H and computer_S
        methods.
    """
    HS_dict = dict()
    for key in H_entries:
        HS_dict[key] = H_block[:, H_entries[key]]
    for key in S_entries:
        HS_dict[key] = S_block[:, S_entries[key]]
    return HS_dict

def get_header_dict(header_block: List[List], homo: bool) -> Dict:
    r"""Generates the dictionary containing the input for the header block of the 
        SKFs.
    
    Arguments: 
        header_block (List[List]): The header block used in SKF files. Contains
            the first few lines.
        homo (bool): Whether this is for a homonuclear SKF or heteronuclear.
            
    Returns:
        grid_dist (float): The grid distance
        atom_info_dict (Dict): The dictionary with the requisite entries for the 
            header block.
        mass (float): The mass, only relevant when dealing with the homonuclear case.
        poly_coef (Array): The array of polynomial coefficients. They are 
            defaulted as 0 because splines are defined for repulsive. 
        rcut (float): 0, since repulsive is explained by splines
        
    Notes: This function depends on the construct_header method from above,
        and decomposes the output from this method. The first three lines 
        provided have the following formats:
            1) grid_dist, ngrid
            2) Ed, Ep, Es, SPE, Ud, Up, Us, fd, fp, fs
            3) mass, c1, c2, c3, ...
        For line 3, the polynomial coefficients c2 -> c9, rcut, and d1 -> d10 are
        all set as zero. The DFTBrepulsive SKF backend excludes the second line
        (mapping to atom_info_dict), so assuming passing in a dummy dict of zeros is 
        acceptable.
    """
    grid_dist = float(header_block[0][0])
    poly_coef = np.zeros(8)
    rcut = 0
    
    if homo:
        atom_info_dict = {
            key : val for key, val in zip(atom_header_entries, header_block[1])
            }
        mass = header_block[2][0]
    else: 
        atom_info_dict = {
            key : 0 for key in atom_header_entries
            }
        mass = header_block[1][0]
    
    return grid_dist, atom_info_dict, mass, poly_coef, rcut

def assemble_final_header_dict(H_block: Array, S_block: Array, header_block: List[List], homo: bool) -> Dict:
    r"""Uses obtain_HS_dict() and atom_info_dict() to make final dictionary
        for use in creating SKFS
    
    Arguments;
        H_block (Array): The Hamiltonian integral table
        S_block (Array): The overlap integral table
        header_block (List[List]): Nested list for the first three lines of the
            skf files.
        homo (bool): Whether we're dealing with a homonuclear case
    
    Returns:
        final_dict (Dict): The dictionary containing the various elements 
            needed by DFTBrepulsive's SKF class to generate SKFs.
    """
    HS_dict = obtain_HS_dict(H_block, S_block)
    grid_dist, atom_info_dict, mass, poly_coef, rcut = get_header_dict(header_block, homo)
    
    #Assemble all the pieces together
    final_dict = {
        
        'gridDist' : grid_dist,
        'atomic_info' : atom_info_dict,
        'mass' : mass,
        'poly_coef' : poly_coef,
        'rcut' : rcut,
        'HS' : HS_dict
        
        }
    
    return final_dict

def get_exp_coef(spline_header: List) -> Array:
    r"""Depends upon the method assemble_spline_header() to get the 
        short-range exopnential coefficients
    
    Arguments:
        spline_header (List): The spline header block constructed by 
            assemble_spline_header()
    
    Returns:
        exp_coefs (Array): The array of short-range exponential coefficients
    
    Notes: Takes in the header generated by assemble_spline_header(). The
        third line is what we're interested in.
    """
    assert(len(spline_header) == 3)
    assert(len(spline_header[2]) == 3) #Should have three coefficients for the exponential expression
    
    #Coefficients are strings here, need them to be floats, do a type-cast
    exp_coefs = spline_header[2]
    exp_coefs = list(map(lambda x : float(x), exp_coefs))
    return np.array(exp_coefs).astype('float64')

def get_spl_xydata(all_models: Dict, rep_mode: str, all_elems: List[tuple], spl_ngrid: int = 500) -> Array:
    r"""Computes the spline xydata for the repulsive potential block
    
    Arguments:
        all_models (Dict): The dictionary containing all the models
        rep_mode (str): Which repulsive mode is being used
        all_elems (tuple): The elements whose interaction is being addressed 
            (e.g. (1, 6) for H-C)
        spl_ngrid (int): The number of grid points ot use for evaluating the 
            spline repulsive. Defaults to 500 for a dense grid.
    
    Returns:
        xy_data (Dict): The xy_data required to construct the spline block
    
    Notes: This part is tricky because it requires casing on the repulsive mode. 
        If dealing with the new repulsive, need to invoke the create_xydata() method
        of the RepulsiveModel backend. If dealing with old repulsive, need to get the 
        xydata for the element from compute_spline_repulsive_vals().
        
        Because the new DFTBRepulsive repulsive model has the create_xydata for 
        all Zs, to prevent redundant computation, going to do all Zs together for 
        both old and new repulsive.
        
        The reverse atom pairs should be present in all_elems, and they should
        also be present in the xydata dictionary generated by the 
        create_xydata() method.
    
    Raises:
        ValueError if rep_mode is not 'new' or 'old'
    """
    if (rep_mode == 'old'):
        print("Using old repulsive")
        xy_data = dict()
        for Zs in all_elems:
            rgrid, r_vals = compute_spline_repulsive_vals(Zs, all_models, spl_ngrid)
            assert(len(rgrid) == len(r_vals))
            data = np.zeros((len(rgrid), 2))
            data[:, 0] = rgrid #rgrid is in angstroms
            data[:, 1] = r_vals
            xy_data[Zs] = data
        return xy_data
    elif (rep_mode == 'new'):
        print("Using new repulsive")
        #We know then that an instance of DFTBRepulsiveModel exists in all_models
        grid_dict = {elems : np.linspace(v[0], v[1], spl_ngrid) for elems, v in all_models['rep'].mod.opts.cutoff.items()}
        xy_data = all_models['rep'].mod.create_xydata(grid_dict, expand = True) #Setting expand to True ensures that the reverse element pairs are included
        #These asserts should pass because with the inclusion of the 
        #   reflexive pair, the keys should be equivalent
        assert(set(all_elems) == set(xy_data.keys()))
        return xy_data
    else:
        raise ValueError("Unrecognized repulsive method!")
        
def write_ref_information(all_models: Dict, rep_mode: str, dest: str) -> None:
    r"""Writes out the reference information for the model to a file called
        ref_params.p using pickle
    
    Arguments:
        all_models (Dict): The dictionary containing all the trained models
        rep_mode (str): The repulsive mode to use
        dest (str): Where to save the reference infomration
        
    Returns:
        None
    
    Notes: Reference parameters should be dictionaries that contain the following 
        keys:
            'coef' : Coefficients for the atoms
            'intercept' : The constant term
            'atype_ordering' : The ordering of the atomic numbers (should be sorted)
    """
    if rep_mode == 'new':
        curr_mod = all_models['rep']
        ref_info = curr_mod.get_ref_ener_info(False)
    elif rep_mode == 'old':
        curr_mod = all_models['Eref']
        ref_info = curr_mod.get_ref_ener_info()
    
    with open(os.path.join(dest, "ref_params.p"), 'wb') as handle:
        pickle.dump(ref_info, handle)
    
    print("Reference parameters saved")

def write_skfs(all_models: Dict, atom_nums: Dict, atom_masses: Dict, compute_S_block: bool, 
               ref_direc: str, rep_mode: str, dest: str = None, spl_ngrid: int = 500) -> None:
    r"""Master method for writing out SKFs using the new DFTBRepulsive framework
    
    Arguments:
        all_models (Dict): The dictionary containing all the models
        atom_nums (Dict): The dictionary mapping atom numbers to their symbols
        atom_masses (Dict): The dictionary mapping atom numbers to their masses
        compute_S_block (bool): Whether or not to compute values for the overlap operator S.
        ref_direc (str): The relative path to the directory containing all skf files
        rep_mode (str): One of 'new' and 'old', indicates which method should be used
            to compute the repulsive spline blocks
        dest (str): The location where the SKF will be saved. Defaults to 
            None, so will be saved at same level as main module
        spl_ngrid (int): The number of grid points ot use for evaluating the 
            spline repulsive. Defaults to 500 for a dense grid.
    
    Returns:
        None
    
    Notes: This method uses the methods defined in this block, previous methods defined in the 
        first block, and the DFTBRepulsive SKF backend to write out SKF files
        using the new SKF classes. 
    """
    elem_pairs = extract_elem_pairs(all_models)
    #Need to construct the HS block and the 
    master_Z_dict = dict()
    for Zs in elem_pairs:
        master_Z_dict[Zs] = dict()
        content = load_file_content(Zs, ref_direc, atom_nums)
        grid_dist, ngrid = get_grid_info(content)
        if compute_S_block:
            print("Computing S block")
            s_block = compute_S(Zs, all_models, grid_dist, ngrid)
        else:
            print("Extracting S block")
            s_block = extract_S_content(Zs, content, ngrid)
        h_block = compute_H(Zs, all_models, grid_dist, ngrid)
        HS_header = construct_header(Zs, all_models, atom_masses, grid_dist, ngrid, content)
        header_dict = assemble_final_header_dict(h_block, s_block, HS_header, Zs[0] == Zs[1])
        #We only care about the exponential coefficients, which come from the refernce
        #   content. As such, the other arguments are dummies.
        spline_header = assemble_spline_header([], content, spl_ngrid, 0.0)
        exp_coefs = get_exp_coef(spline_header)
        master_Z_dict[Zs]['header_HS_data'] = header_dict
        master_Z_dict[Zs]['exp_coef'] = exp_coefs
    #master_Z_dict now has all the necessary header and electronic integral table information, but still 
    #   need to obtain the repulsive information
    xy_data = get_spl_xydata(all_models, rep_mode, elem_pairs, spl_ngrid)
    assert(set(xy_data.keys()) == set(master_Z_dict.keys()))
    
    #Now create all the SKF objects and write them to the necessary location
    for Zs in master_Z_dict:
        # print(Zs)
        creator = SKFBlockCreator()
        header_HS_data, exp_coef, spl_xydata = master_Z_dict[Zs]['header_HS_data'],\
            master_Z_dict[Zs]['exp_coef'], xy_data[Zs]
        # print(header_HS_data['mass'])
        header_HS_block = creator.create_header_HS_block(Zs, header_HS_data)
        spline_block = creator.create_spline_block(exp_coef, spl_xydata)
        doc_block = creator.create_doc_block([]) #No need for documentation right now
        skf = SKF(skf_data = {})
        skf['homo'] = header_HS_block['homo']
        skf['extend'] = header_HS_block['extend']
        skf['Z'] = header_HS_block['Z']
        skf['header'] = header_HS_block['header']
        skf['HS'] = header_HS_block['HS']
        skf['spline'] = spline_block
        skf['doc'] = doc_block
        elem1, elem2 = Zs
        atom1, atom2 = atom_nums[elem1], atom_nums[elem2]
        save_file_name = f"{atom1}-{atom2}.skf" if dest is None else os.path.join(dest, f"{atom1}-{atom2}.skf")
        skf.to_file(save_file_name)
    
    #Write the reference energy information
    write_ref_information(all_models, rep_mode, dest)
    
    print("All SKFs written!")

#%% Main block
if __name__ == "__main__":
    pass