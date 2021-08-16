# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 18:17:58 2021

@author: fhu14

Used to generate a parameter dictionary from a set of SKF files. This 
circumvents the need to copy and paste/depend on external file structures.
"""
#%% Imports, definitions
import pickle, os
from DFTBrepulsive import SKFSet
from SKF import SkfInfo
from MasterConstants import atom_nums
from typing import List, Dict

#%% Code behind

def generate_pardict(skf_dir: str, elements: List[int]) -> Dict:
    r"""Generates a parameter dictionary of SKFInfo objects 
        from a given set of SKF files.
    
    Arguments:
        skf_dir (str): The path to the directory containing the SKF files.
        elements (List[int]): The list of atomic numbers that the parameter
            dictionary needs to contain
    
    Returns:
        par_dict (Dict): The parameter dictionary.
    
    Notes: The spin parameters are not used by DFTBpy or the model during
        training, but are included here.
    """
    elements = [atom_nums[num] for num in elements]
    parDict = {el1 + '-' + el2: SkfInfo(el1 + '-' + el2, paramPath = skf_dir)
               for el1 in elements for el2 in elements}
    
    # Spin constants (parameter W, doublju)
    # H:
    #      -0.07174
    if 'H' in elements:
        parDict['H-H'].SetAtomProp('Wss', -0.07174)
    # C:
    #      -0.03062     -0.02505
    #      -0.02505     -0.02265
    if 'C' in elements:
        parDict['C-C'].SetAtomProp('Wss', -0.03062)
        parDict['C-C'].SetAtomProp('Wsp', -0.02505)
        parDict['C-C'].SetAtomProp('Wpp', -0.02265)
    # N:
    #      -0.03318     -0.02755
    #      -0.02755     -0.02545
    if 'N' in elements:
        parDict['N-N'].SetAtomProp('Wss', -0.03318)
        parDict['N-N'].SetAtomProp('Wsp', -0.02755)
        parDict['N-N'].SetAtomProp('Wpp', -0.02545)
    # O:
    #      -0.03524     -0.02956
    #      -0.02956     -0.02785
    if 'O' in elements:
        parDict['O-O'].SetAtomProp('Wss', -0.03524)
        parDict['O-O'].SetAtomProp('Wsp', -0.02956)
        parDict['O-O'].SetAtomProp('Wpp', -0.02785)
    # S:
    #      -0.02137     -0.01699     -0.01699
    #      -0.01699     -0.01549     -0.01549
    #      -0.01699     -0.01549     -0.01549
    if 'S' in elements:
        parDict['S-S'].SetAtomProp('Wss', -0.02137)
        parDict['S-S'].SetAtomProp('Wsp', -0.01699)
        parDict['S-S'].SetAtomProp('Wsd', -0.01699)
        parDict['S-S'].SetAtomProp('Wpp', -0.01549)
        parDict['S-S'].SetAtomProp('Wpd', -0.01549)
        parDict['S-S'].SetAtomProp('Wdd', -0.01549)
    # Au:
    #      -0.01304     -0.01304   -0.00525
    #      -0.01304     -0.01304   -0.00525
    #      -0.00525     -0.00525   -0.01082
    if 'Au' in elements:
        parDict['Au-Au'].SetAtomProp('Wss', -0.01304)
        parDict['Au-Au'].SetAtomProp('Wsp', -0.01304)
        parDict['Au-Au'].SetAtomProp('Wsd', -0.00525)
        parDict['Au-Au'].SetAtomProp('Wpp', -0.01304)
        parDict['Au-Au'].SetAtomProp('Wpd', -0.00525)
        parDict['Au-Au'].SetAtomProp('Wdd', -0.01082)
    
    return parDict


