#Module for running the xTB version of DFTB+ for the purpose of comparing it against our in-house methods

#%% Imports, definitions

import re, os
from Elements import ELEMENTS
import numpy as np
import numbers
from MasterConstants import valence_dict
from typing import List, Dict
from subprocess import call
from .dftbplus import read_detailed_out
Array = np.ndarray

dftb_exec = os.path.join(os.getcwd(), "../../../dftbp_xtb/dftbplus-21.2.x86_64-linux/bin/dftb+") #New version of DFTB+ with xTB methods included

#%% Code behind

def write_dftb_infile_xTB(Zs: List[int], rcart_angstroms: Array, 
                      file_path: str, method: str, 
                      DFTBparams_overrides: dict = {}):
    r"""
    Write DFTB HSD input file (dftb_hsd.in) for single point calculation.
    
    Arguments:
        Zs (List[int]): element numbers for atoms in the molecule
        rcart_angstroms (array[]): [natom,3] array with cartesian coordinates
          in anstroms
        file_path (str): path to the output file (e.g. 'scratch/dftb_in.hsd')
        method (str): the xTB method to us (e.g. GFN1-xTB)
        
        DFTBparams_overrides (dict): dict to override these default params
           'ShellResolvedSCC' : True 
                 If True, s,p,d.. have separate Hubbard parameters
           'FermiTemp' : None
                 Temperature in kelvin (must be a number)
    Raises:
        ValueError: If keys in DFTBparams_overrides are not supported
                    or 'FermiTemp' is not either None or a number
                 
    """
    # Default DFTB parameters
    DFTBparams = {'ShellResolvedSCC': True,
                  'FermiTemp': None}
    # Check and implement requested overides to DFTB parameters
    if any([x not in DFTBparams for x in DFTBparams_overrides]):
        unsupported = [x for x in DFTBparams_overrides if x not in DFTBparams]
        raise ValueError('Unsupported DFTB parameters ' +
                        ' '.join(unsupported))
    DFTBparams.update(DFTBparams_overrides)
    # For convenience. The angstroms is attached to variable name to prevent
    # use of a.u. (DFTB+ uses a.u. for most quantities, so use of a.u. here may
    # be a common error.)
    rcart = rcart_angstroms
    #HSD input requires list of element types, and their max ang momentum, only
    #for elements in this molecule.
    Ztypes = np.unique(Zs)
    # map from Z to the "type" given in the TypeNames argument of the HSD file
    ZtoType = {Z: (i + 1) for i, Z in enumerate(Ztypes)}
    with open(file_path, 'w') as dftbfile:
        dftbfile.write(r'Geometry = {' + '\n')
        line = r'  TypeNames = {'
        for Z in Ztypes:
            line += r' "' + ELEMENTS[Z].symbol + r'" '
        line += r'}'
        dftbfile.write(line + '\n')
        dftbfile.write(r'   TypesAndCoordinates [Angstrom] = {' + '\n')
        for iatom, Z in enumerate(Zs):
            line = r'      ' + str(ZtoType[Z])
            line += ' %.8f' % rcart[iatom, 0] + ' %.8f' % rcart[iatom, 1] \
                    + ' %.8f' % rcart[iatom, 2]
            dftbfile.write(line + '\n')
        dftbfile.write(r'   }' + '\n')
        dftbfile.write(r'}' + '\n')
        dftbfile.write(
            r'Driver = {}' + '\n' +
            r'Hamiltonian = xTB {' + '\n' +
            r'   Method = "' + method + '"' + '\n')
        #The CM5 charges are included because the model is being trained against
        #CM5 charges. Also, including the mulliken analysis and CM5 charges 
        #ensures the dipole moments are outputted in detailed.out (although not sure
        #how to do proper conversion to Debye)
        dftbfile.write(
            r'}' + '\n' +
            r'Options { WriteChargesAsText = Yes }' + '\n' +
            r'Analysis {' + '\n' +
            r'   CalculateForces = No' + '\n' +
            r'   MullikenAnalysis = Yes' + '\n' +
            r'   CM5{}' + '\n' +
            r'}' + '\n')
        # A windows executable is only available for version 17.1
        #  https://sites.google.com/view/djmolplatform/get-dftb-17-1-windows
        # and this version uses the OrbitalResolvedSCC keywork, instead of
        # ShellResolvedSCC. We use parserversion = 5, so that more recent 
        # versions of DFTB+ will use OrbitalResolvedSCC
        dftbfile.write(
            r'ParserOptions {' + '\n'
                                 r'   Parserversion = 5' + '\n'
                                                           r'}'
        )

def run_xTB_dftbp(dataset: List[Dict], method: str) -> None:
    r"""Method for running xTB-enabled DFTB+ with molecules from the ANI-1 set
    
    Arguments:
        dataset (List[Dict]): The list of molecule dictionaries to run
        method (str): The method to use for the xTB Hamiltonian
    
    Returns:
        None
    
    Notes: Right now, the xTB Hamiltonian is very bare-bones and does not include 
        playing around with the additional options available to the Hamiltonian. 
        These are enumerated in the manual on DFTB+, and may be investigated further
        (requires version 21.2)
    """
    
    print(f"Running xTB enabled DFTB+ with method {method}")
    
    scratch_dir = "dftbscratch"
    dftb_in_file = os.path.join(scratch_dir, 'dftb_in.hsd')
    dftb_outfile = os.path.join(scratch_dir, 'detailed.out')
    default_res_file = os.path.join(scratch_dir, 'dftb.out')
    
    for imol, mol in enumerate(dataset):
        print('starting', imol, mol['name'])
        Zs = mol['atomic_numbers']
        rcart = mol['coordinates']
        
        natom = len(Zs)
        cart = np.zeros([natom, 4])
        cart[:, 0] = Zs
        for ix in range(3):
            cart[:, ix + 1] = rcart[:, ix]
        charge = 0
        mult = 1
        
        write_dftb_infile_xTB(Zs, rcart, dftb_in_file, method)
        
        with open(default_res_file, 'w') as f:
            res2 = dict()
            res = call(dftb_exec, cwd = scratch_dir, stdout = f, shell = False)
            dftb_res = read_detailed_out(dftb_outfile)
            res2['t'] = dftb_res['t']
            for key in ['e', 'r', 'disp']:
                try:
                    res2[key] = dftb_res[key]
                except:
                    pass
            #Not going to worry about charges right now, just looking at total energies
            
        mol['pzero'] = res2
    
    print("Done with running all the molecules")
