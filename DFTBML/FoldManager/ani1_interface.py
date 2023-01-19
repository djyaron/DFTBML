# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:21:16 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import Union, List, Dict
from h5py import File
import numpy as np
import collections
from copy import deepcopy

#%% Code behind

def get_data_type(specs: Union[List[str], str]) -> List[str]:
    r"""Obtains the corresponding ANI dataset keys from the keys specified
        in specs
    
    Arguments:
        specs (Union[List[str], str]): The abbreviated keys used to refer
            to specific ANI dataset keys.
    
    Returns:
        res (list[str]): The list of ANI keys corresponding to the 
            given keys in specs.
    """
    if not isinstance(specs,list):
        specs = [specs]
    ANI1TYPES = {'dt': 'dftb.energy',  # Dftb Total
                'de': 'dftb.elec_energy',  # Dftb Electronic
                'dr': 'dftb.rep_energy',  # Dftb Repulsive
                'pt': 'dftb_plus.energy',  # dftb Plus Total
                'pe': 'dftb_plus.elec_energy',  # dftb Plus Electronic
                'pr': 'dftb_plus.rep_energy',  # dftb Plus Repulsive
                'hd': 'hf_dz.energy',  # Hf Dz
                'ht': 'hf_tz.energy',
                'hq': 'hf_qz.energy',
                'wd': 'wb97x_dz.energy',  # Wb97x Dz
                'wt': 'wb97x_tz.energy',
                'md': 'mp2_dz.energy',  # Mp2 Dz
                'mt': 'mp2_tz.energy',
                'mq': 'mp2_qz.energy',
                'td': 'tpno_ccsd(t)_dz.energy',  # Tpno Dz
                'nd': 'npno_ccsd(t)_dz.energy',  # Npno Dz
                'nt': 'npno_ccsd(t)_tz.energy',
                'cc': 'ccsd(t)_cbs.energy'}
    res = []
    for spec in specs:
        if spec in ANI1TYPES.keys():
            res.append( ANI1TYPES[spec] )
        elif spec in ANI1TYPES.values():
            res.append( spec )
        else:
            res.append(spec) #To handle additional things
    return res

def get_targets_from_h5file(data_specs: Union[List[str], str], ani1_path: str, 
                            exclude: dict = None) -> dict:
    r"""Obtains the necessary target information from the dataset stored at ani1_path
    
    Arguments:
        data_specs (Union[List[str], str]): A string or list of strings encoding the 
            data fields that should be extracted.
        ani1_path (str): The string indicating the relative or total path to 
            the h5 dataset file.
        exclude (dict): Contains keys to exclude. Defaults to None.
        
    Returns:
        target_molecs (dict): A dictionary mapping the molecule name to the
            corresponding data for that molecule as specified in data_specs
    """
    if exclude == None:
        exclude = dict()
    dtypes = get_data_type(data_specs)
    target_molecs = dict()
    with File(ani1_path, 'r') as ani1data:
        for mol, gdata in ani1data.items():
            moldata = [gdata[x][()] for x in dtypes]
            if mol in exclude.keys():
                if exclude[mol] == 'all':
                    continue
                moldata = [np.delete(x, exclude[mol],0) for x in moldata]     
            # if len(moldata) == 1:
            target_molecs[mol] = moldata
            # Not sure why this subtraction exists, ignore for now...
            # else:
            #     target_molecs[mol] = [moldata[0] - moldata[1]]
            #     target_molecs[mol] += moldata[2:]
    return target_molecs

#Fix the ani1_path for now
ani1_path = 'data/ANI-1ccx_clean_fullentry.h5'

def get_ani1data(allowed_Z: List[int], heavy_atoms: List[int], max_config: int, 
                 target: Dict[str, str], ani1_path: str = ani1_path, exclude: List[str] = []) -> List[Dict]:
    r"""Extracts data from the ANI-1 data files
    
    Arguments:
        allowed_Z (List[int]): Include only molecules whose elements are in
            this list
        heavy_atoms (List[int]): Include only molecules for which the number
            of heavy atoms is in this list
        max_config (int): Maximum number of configurations included for each
            molecule.
        target (Dict[str,str]): entries specify the targets to extract
            key: target_name name assigned to the target
            value: key that the ANI-1 file assigns to this target
        ani1_path (str): The relative path to the data file. Defaults to
            'data/ANI-1ccx_clean_fullentry.h5'
        exclude (List[str], optional): Exclude these molecule names from the
            returned molecules
            Defaults to [].
            
    Returns:
        molecules (List[Dict]): Each Dict contains the data for a single
            molecular structure:
                {
                    'name': str with name ANI1 assigns to this molecule type
                    'iconfig': int with number ANI1 assignes to this structure
                    'atomic_numbers': List of Zs
                    'coordinates': numpy array (:,3) with cartesian coordinates
                    'targets': Dict whose keys are the target_names in the
                        target argument and whose values are numpy arrays
                        with the ANI-1 data
                        
    Notes: The ANI-1 data h5 files are indexed by a molecule name. For each
        molecule, the data is stored in arrays whose first dimension is the
        configuration number, e.g. coordinates(iconfig,atom_num,3). This
        function treats each configuration as its own molecular structure. The
        returned dictionaries include the ANI1-name and configuration number
        in the dictionary, along with the data for that individual molecular
        structure.
    """
    print(f"data file path is {ani1_path}")
    target_alias, h5keys = zip(*target.items())
    target_alias, h5keys = list(target_alias), list(h5keys)
    all_zs = get_targets_from_h5file('atomic_numbers', ani1_path)
    all_coords =  get_targets_from_h5file('coordinates', ani1_path)
    all_targets = get_targets_from_h5file(h5keys, ani1_path)
    if exclude is None:
        exclude = []
    batches = list()
    for name in all_zs.keys():
        if name in exclude:
            continue
        zs = all_zs[name][0] #Extract np array of the atomic numbers
        zcount = collections.Counter(zs)
        ztypes = list(zcount.keys())
        zheavy = [x for x in ztypes if x > 1]
        nheavy = sum([zcount[x] for x in zheavy])
        ztypes.sort()
        zkeep = deepcopy(allowed_Z)
        zkeep.sort()
        if any([zz not in allowed_Z for zz in ztypes]):
            continue
        if nheavy not in heavy_atoms:
            continue
        nconfig = all_coords[name][0].shape[0] #Extract the np array of the atomic coordinates
        for iconfig in range(min(nconfig, max_config)):
            batch = dict()
            batch['name'] = name
            batch['iconfig'] = iconfig
            batch['atomic_numbers'] = zs
            batch['coordinates'] = all_coords[name][0][iconfig,:,:]
            batch['targets'] = dict()
            for i in range(len(target_alias)):
                targ_key = target_alias[i]
                batch['targets'][targ_key] = all_targets[name][i][iconfig]
            batches.append(batch)
    return batches

def get_ani1data_boosted(allowed_Z: List[int], heavy_atoms: List[int], target_atoms: List[int],
                         criterion: str, max_config: int,
                         boosted_config: int, target: Dict[str, str], 
                         ani1_path: str = ani1_path, exclude: List[str] = []) -> List[Dict]:
    r"""Extracts data from the ANI-1 data files
    
    Arguments:
        allowed_Z (List[int]): Include only molecules whose elements are in
            this list
        heavy_atoms (List[int]): Include only molecules for which the number
            of heavy atoms is in this list
        target_atoms (List[int]): List of atomic numbers for atom that need
            more representation. For example, if O needs more representation, 
            target_atoms will include [8...]
        criterion (str): The requirement for boosted molecules, one of 'any' or 'all'.
            If 'any', any molecule that contains at least one of the target atoms
            is boosted. If 'all', then only molecule that contain all the target atoms
            are boosted.
        max_config (int): Maximum number of configurations included for each
            molecule by default.
        boosted_config (int): Maximum number of configurations included for 
            each molecule if they contain elements contained in the list.
        target (Dict[str,str]): entries specify the targets to extract
            key: target_name name assigned to the target
            value: key that the ANI-1 file assigns to this target
        ani1_path (str): The relative path to the data file. Defaults to
            'data/ANI-1ccx_clean_fullentry.h5'
        exclude (List[str], optional): Exclude these molecule names from the
            returned molecules
            Defaults to [].
            
    Returns:
        molecules (List[Dict]): Each Dict contains the data for a single
            molecular structure:
                {
                    'name': str with name ANI1 assigns to this molecule type
                    'iconfig': int with number ANI1 assignes to this structure
                    'atomic_numbers': List of Zs
                    'coordinates': numpy array (:,3) with cartesian coordinates
                    'targets': Dict whose keys are the target_names in the
                        target argument and whose values are numpy arrays
                        with the ANI-1 data
                        
    Notes: The ANI-1 data h5 files are indexed by a molecule name. For each
        molecule, the data is stored in arrays whose first dimension is the
        configuration number, e.g. coordinates(iconfig,atom_num,3). This
        function treats each configuration as its own molecular structure. The
        returned dictionaries include the ANI1-name and configuration number
        in the dictionary, along with the data for that individual molecular
        structure.
    """
    print(f"data file path is {ani1_path}")
    target_alias, h5keys = zip(*target.items())
    target_alias, h5keys = list(target_alias), list(h5keys)
    all_zs = get_targets_from_h5file('atomic_numbers', ani1_path)
    all_coords =  get_targets_from_h5file('coordinates', ani1_path)
    all_targets = get_targets_from_h5file(h5keys, ani1_path)
    if exclude is None:
        exclude = []
    batches = list()
    for name in all_zs.keys():
        if name in exclude:
            continue
        zs = all_zs[name][0] #Extract np array of the atomic numbers
        zcount = collections.Counter(zs)
        ztypes = list(zcount.keys())
        zheavy = [x for x in ztypes if x > 1]
        nheavy = sum([zcount[x] for x in zheavy])
        ztypes.sort()
        zkeep = deepcopy(allowed_Z)
        zkeep.sort()
        if any([zz not in allowed_Z for zz in ztypes]):
            continue
        if nheavy not in heavy_atoms:
            continue
        #Check if any/all of the target elements are contained in the current molecule
        if criterion == 'any':
            target_present = any([z in ztypes for z in target_atoms])
        elif criterion == 'all':
            target_present = all([z in ztypes for z in target_atoms])
        target_config = boosted_config if target_present else max_config
        nconfig = all_coords[name][0].shape[0] #Extract the np array of the atomic coordinates
        for iconfig in range(min(nconfig, target_config)):
            batch = dict()
            batch['name'] = name
            batch['iconfig'] = iconfig
            batch['atomic_numbers'] = zs
            batch['coordinates'] = all_coords[name][0][iconfig,:,:]
            batch['targets'] = dict()
            for i in range(len(target_alias)):
                targ_key = target_alias[i]
                batch['targets'][targ_key] = all_targets[name][i][iconfig]
            batches.append(batch)
    return batches