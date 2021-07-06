# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:07:56 2021

@author: fhu14
"""

#%% Imports, definitions
import re
from Elements import ELEMENTS
import numpy as np
import numbers

from typing import List

Array = np.ndarray

#%% Code behind

def named_re(name: str, respec: str,
             before: str = 'none', after: str = 'require') -> str:
    r"""
    Wrap a regular expression in a block that gives it a name.
    
    Arguments:
        respec (string): regexp to be wrapped
        name (string): name to be assigned to the parsed results:
                (?P<name> respec)
        before (string): whitespace requirements before respec with choices
            'none'   : do not allow any whitespace
            'allow' : accept but do not require whitespace
            'require' : require whitespace
        after(string): whitespace requirements after respec with same choices
            as for the after variable

    Returns:            
        final_regexp: wrapped version of the respec
    """
    ws = {'none': r'',
          'allow': r'\s*',
          'require': r'\s+'}
    res = ws[before] + "(?P<" + name + ">" + respec + ")" + ws[after]

    return res

def write_dftb_infile(Zs: List[int], rcart_angstroms: Array, 
                      file_path: str, skf_dir: str,
                      DFTBparams_overrides: dict = {}):
    r"""
    Write DFTB HSD input file (dftb_hsd.in) for single point calculation.
    
    Arguments:
        Zs (List[int]): element numbers for atoms in the molecule
        rcart_angstroms (array[]): [natom,3] array with cartesian coordinates
          in anstroms
        file_path (str): path to the output file (e.g. 'scratch/dftb_in.hsd')
        skf_dir (str): directory with the SKF files (should not end in / or \)
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
            r'Hamiltonian = DFTB {' + '\n' +
            r'   Scc = Yes' + '\n')
        if DFTBparams['ShellResolvedSCC']:
            # Using OrbitalResolved for backward compatibility
            # This is why the version number below is 5
            dftbfile.write(r'   OrbitalResolvedSCC = Yes' + '\n')
        else:
            dftbfile.write(r'   OrbitalResolvedSCC = No' + '\n')
        dftbfile.write(
            r'   SlaterKosterFiles = Type2FileNames {' + '\n' +
            r'      Prefix = "' + skf_dir + r'/"' + '\n' +
            r'      Separator = "-"' + '\n' +
            r'      Suffix = ".skf"' + '\n' +
            r'   }' + '\n' +
            r'   MaxAngularMomentum {' + '\n'
        )
        # required because DFTB+ wants ang momentum listed only for elements
        # actually in the molecule. The block field of ELEMENT indicates
        # the s,p,d,f label of the valence orbitals, which works for elements
        # currently being studied.
        # TODO: How does DFTB+ handle an element like Ca, for which block = 's' 
        for Z in Ztypes:
            ele = ELEMENTS[Z]
            dftbfile.write(r'      ' + ele.symbol
                           + r' = "' + ele.block + '"\n')
        dftbfile.write(
            r'   }' + '\n')

        if DFTBparams['FermiTemp'] is not None:
            if not isinstance(DFTBparams['FermiTemp'], numbers.Number):
                raise ValueError('FermiTemp not a number')
            dftbfile.write(
                r'   Filling = Fermi {' + '\n' +
                r'       Temperature[K] = ' + str(DFTBparams['FermiTemp']) + '\n' +
                r'       }' + '\n'
            )

        dftbfile.write(
            r'}' + '\n' +
            r'Options {}' + '\n' +
            r'Analysis {' + '\n' +
            r'   CalculateForces = No' + '\n' +
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


def read_dftb_out(file_path: str) -> dict:
    r"""
    Parse a dftb.out file.

    Args:
        file_path (str): path to dftb.out file

    Returns:
        results(dict): parsed results. Currently includes:
            Eev: total energy in eV
            Ehartree: total energy in Hartree
    
    Raises:
        Exception('DFTB calculation failed') 
    
    """
    # From: https://stackoverflow.com/questions/12643009/
    anyfloat = r"[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)"
    # Parsing line of this type:
    # Total Energy:                      -33.6613043365 H         -915.9707 eV
    energy_string = r'Total\sEnergy:' \
                    + named_re('Ehartree', anyfloat, 'require', 'allow') + r'H' \
                    + named_re('Eev', anyfloat, 'allow', 'allow') + r'eV'
    energy_re = re.compile(energy_string)

    with open(file_path, 'rb') as file_in:
        lines = file_in.readlines()
    lines = [line.decode("utf-8").strip() for line in lines]

    results = dict()
    found = False
    for line in lines:
        if energy_re.match(line):
            energy_dict = energy_re.match(line).groupdict()
            results['Ehartree'] = float(energy_dict['Ehartree'])
            results['Eev'] = float(energy_dict['Eev'])
            found = True
    if not found:
        results['Ehartree'] = np.nan
        results['Eev'] = np.nan
        raise Exception('DFTB Calculation Failed')
    return results