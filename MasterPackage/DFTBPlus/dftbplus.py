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
from MasterConstants import valence_dict

from typing import List, Dict

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

def write_dispersion_block(use_UFF_params: bool = True, dispersion_mode: str = "LJ") -> str:
    r"""Writes out the dispersion block according to the DFTB+ manual's
        specification for the input
    
    Arguments:
        use_UFF_params (bool): Whether to use the parameters specific by the
            universal force field (UFF). Defaults to True for consistency
            with the DFTB_Layer code.
        dispersion_mode (str): The form of the dispersion to use. Defaults to 
            "LJ" for LennardJones. 
    
    Returns:
        dispersion_block (str): The dispersion block ready to be written to the 
            input files.
    
    Notes: The DFTB+ manual indicates that you can specify the set of 
        parameters to use, but for convenience, only going to use the default UFF
        parameters
    """
    dispersion_block = ""
    line1 = "   Dispersion = LennardJones{\n"
    line2 = "      Parameters = UFFParameters{}\n"
    line3 = "   }\n"
    dispersion_block = line1 + line2 + line3
    return dispersion_block

def write_dftb_infile(Zs: List[int], rcart_angstroms: Array, 
                      file_path: str, skf_dir: str, dispersion: bool = False,
                      DFTBparams_overrides: dict = {}):
    r"""
    Write DFTB HSD input file (dftb_hsd.in) for single point calculation.
    
    Arguments:
        Zs (List[int]): element numbers for atoms in the molecule
        rcart_angstroms (array[]): [natom,3] array with cartesian coordinates
          in anstroms
        file_path (str): path to the output file (e.g. 'scratch/dftb_in.hsd')
        skf_dir (str): directory with the SKF files (should not end in / or \)
        dispersion (bool): Whether to include the dispersion block. Defaults to 
            False.
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
        
        #Need to add in Dispersion block to the Hamiltonian block
        if dispersion:
            dispersion_block = write_dispersion_block()
            dftbfile.write(dispersion_block)

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

def read_detailed_out(file_path: str) -> dict:
    r"""Parse a detailed.out file
    
    Arguments:
        file_path (str): path to detailed.out file
    
    Returns:
        results (Dict): Dictionary containing the total energy,
            total electronic energy, and total repulsive energy. 
    """
    name_mapping = {
        'total energy:' : 't',
        'total electronic energy:' : 'e',
        'repulsive energy:' : 'r',
        'dispersion energy' : 'disp'
        }
    result_dict = dict()
    triggered = False
    content = open(file_path).read().splitlines()
    for i in range(len(content)):
        #Remove the extraneous whitespace and fix capitalization
        content[i] = content[i].strip().lower()
    for line in content[::-1]:
        for start in name_mapping:
            if line.startswith(start) and (name_mapping[start] not in result_dict):
                #Only care about Ha values
                try:
                    curr_line = line.split()
                    # print(curr_line)
                    H_index = curr_line.index('h')
                    val_ind = H_index - 1
                    result_dict[name_mapping[start]] = float(curr_line[val_ind])
                    triggered = True
                except:
                    pass
    if (not triggered):
        raise ValueError("DFTB+ calculation failed!")
    return result_dict


def parse_charges_dat(charge_filename: str, rcart_angstrom: Array, Zs: Array, valence_correction: bool = True,
                  val_dict: Dict = None) -> Array:
    r"""Parses out the charge information from the charges.dat file output
        from DFTB+
    
    Arguments:
        charge_filename (str): The path to the charges.dat file
        rcart_angstrom (Array): The array of cartesian coordinates for the molecule
        Zs (Array): The atomic number of atoms in the molecule
        valence_correction (bool): Whether to correct charges or not. Defaults
            to True
        val_dict (Dict): The dictionary used for the valence correction.
            Defaults to None
        
    Returns:
        charges (Array): The parsed charges from the DFTB+ output. May be
            valence corrected depending on the valence_correction toggle
    
    Notes:
        A valence correction takes the following form:
            
            Q_corrected = Q_calc - Q_neutral
        
        Where Q_calc comes from the parsed charges.dat file and Q_neutral is 
        interpreted as the number of valence electrons in the atom. Because 
        the DFTB+ output file sums up all electrons across all the orbitals 
        by default (i.e. not orbital resolved), the charge fluctuation is 
        easy to calculate. Just to be safe, the charges for all the spin 
        channels are summed together just in case charges are orbital resolved
        
        This correction mirrors the behavior in DFTBML, where the on-atom
        charges computed are a sum of individual orbital charge fluctuations 
        (Q_atom = sum_{orb} dQ_{orb}), and each dQ is computed as the difference:
            
            dQ_{orb} = Q_orb - Q_orb_0
        
        The charges in DFTBML are trained to a charge model 5 (cm5) target
        which are an extension of the Hirshfeld population analysis. 
        
        The charges parsed out should be in the same order as the Natoms fed
        in the input into DFTB+. Later in the calculation of the dipole,
        the matrix multiplication should work b/w coordinates and charges.

        Technically, this information can be obtained from the output file,
        but this method works too since the calculation is consistent for parsing
        from charges.dat.

        In the DFTBLayer paper, the charge fluctuations are actually computed as follows:
            
            dQ_{orb} = - (Q_orb - Q_orb_0) = Q_orb_0 - Q_orb

        Hence, the multiplication by -1 for consistency.
    """
    assert(len(rcart_angstrom) == len(Zs))
    if valence_correction:
        assert(val_dict is not None)
    with open(charge_filename, "r+") as file:
        charge_content = file.read().splitlines()
    n_atom = rcart_angstrom.shape[0] #(Natom, 3)
    assert(rcart_angstrom.shape[1] == 3)
    charge_lines = charge_content[len(charge_content) - n_atom:]
    charges = []
    for line in charge_lines:
        elems = line.split()
        flt_mapped = list(map(lambda x : float(x), elems))
        charges.append(sum(flt_mapped))
    if valence_correction:
        syms = [ELEMENTS[z].symbol for z in Zs]
        valences = np.array([val_dict[elem] for elem in syms])
        charges -= valences
        charges *= -1 #Need to multiply by negative 1 to maintain consistency with DFTBLayer/DFTBPy
    return np.array(charges)

def parse_dipole(output_file: str, pattern: str, unit: str = 'Debye') -> Array:
    r"""Uses regex to parse out the dipoles from the output file using a given regex pattern.

    Arguments:
        output_file (str): The output file to parse dipoles from
        pattern (str): The pattern to use for parsing out dipoles
        unit (str): The dipole unit of interest. Defaults to 'Debye',
            one of 'Debye' or 'au'

    Returns:
        Dipole (Array): The dipole represented as a (3,) np array

    Raises;
        ValueError if the dipole matching fails

    Notes: This is the dipole calculated by DFTB+, so not the ESP dipole
        internally generated for training.  
    """
    assert(unit in ['Debye', 'au'])
    #assert(unit == "Debye")
    dipole_matcher = re.compile(pattern)
    content = open(output_file, 'r').read()
    #The dipole regex pattern can be used with findall
    dipole_result = dipole_matcher.findall(content)
    try:
        assert(dipole_result != [])
    except:
        raise ValueError("Regex dipole parsing failed!")
    assert(len(dipole_result) == 2) #There should be two dipoles
    chosen_dipole = dipole_result[0] if unit in dipole_result[0] else dipole_result[1]
    dip_splt = chosen_dipole.split()
    dipole_elems = []
    for elem in dip_splt:
        try:
            val = float(elem)
            dipole_elems.append(val)
        except:
            pass #Handle trying to convert non-numerical elements to numerical
    assert(len(dipole_elems) == 3) #There should only be three cartesian components
    return np.array(dipole_elems)

def parse_charges_output(output_file: str, pattern: str) -> Array:
    r"""Parses the charges from the output file depending on the pattern

    Arguments:
        output_file (str): The output file to parse the charges from
        pattern (str): The regex pattern to use when parsing out charges

    Returns:
        charges (Array): The array of parsed charges represented as a np array

    Raises:
        ValueError if the regex charge parsing does not work (i.e. the regex 
            search returns None)

    Notes: The CM5 charges are contained in the body of the output file detailed.out 
        Will also have to ensure that the charges are sign-corrected, as mentioned in 
        the doc string of parse_charges_dat. This function can also parse out the 
        gross atomic charges depending on the pattern fed in as the second argument. 
    """
    charges_matcher = re.compile(pattern)
    content = open(output_file, 'r').read()
    charge_result = charges_matcher.search(content)
    try:
        assert(charge_result is not None)
    except:
        raise ValueError("Regex charge parsing failed!")
    start, end = charge_result.span()
    charge_str = charge_result.string[start : end]
    splt_lines = charge_str.strip().splitlines()
    #Skips the header (first two lines)
    atom_lines = splt_lines[2:]
    charges = np.array([float(elem.split()[1]) for elem in atom_lines])
    #Actually no need to multiply by negative one here since defintition is consistent
    #with valence - charge giving dQ
    return charges


def compute_ESP_dipole(charges: Array, rcart_angstrom: Array) -> Array:
    r"""Computes the ESP dipoles from DFTB+ computed charges
    
    Arguments:
        charges (Array): The parsed out charges from the DFTB+ output
        rcart_angstroms (Array): The (Natom, 3) array of atomic positions in angstroms
    
    Returns:
        dipole (Array): Array of shape (3, 1) or just (3,) containing the 
            computed dipole from the charges and positions
        
    Notes:
        The ESP dipole is computed as R @ q where R is the (3, Natom) matrix of
        atomic positions and q is the vector of Natom charges. Since R is (3, Natom) 
        and q is (Natom,), the resulting dipole vector should have dimension 
        (3,)
    """
    assert(len(rcart_angstrom) == len(charges))
    assert(rcart_angstrom.shape[1] == 3)
    assert(len(rcart_angstrom.shape) == 2) #2 dimensional array
    assert(len(charges.shape) == 1) #1 dimensional array
    return np.dot(rcart_angstrom.T, charges)
