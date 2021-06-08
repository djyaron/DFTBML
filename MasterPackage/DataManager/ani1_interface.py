# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:21:16 2021

@author: fhu14
"""
#%% Imports, definitions
from typing import Union, List
from h5py import File
import numpy as np

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