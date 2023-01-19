#Module for running xTB-enabled DFTB+ on ANI-1 organic molecules

#%% Imports, definitions
from DFTBPlus import run_xTB_dftbp
import pickle, os, shutil
import time
import json
import argparse
from DFTBPlus import compute_results_torch, calc_format_target_reports
from typing import List, Dict
from MasterConstants import atom_nums
from subprocess import call
scratch_dir = "xtbscratch"

#Some global variables for calc_format_target_reports
error_metric = "MAE"
tot_ener_targ = "Etot"
#No dipole ESP
pairwise_targs = [['dipole', 'dipole'],
        ['charges', 'charges']]
fit_fresh_ref_ener = True #Always, GFNn-xTB does not have a reference energy 
allowed_Zs = [1,6,7,8] #C H N O
dipole_conversion = True #Yes, parsed dipole is in units of Debye
prediction_dipole_unit = "Debye" 
target_dipole_unit = "eA" #For ANI-1 molecules it's eA
exclusion_threshold = 20 #20 standard deviations for outlier exclusions
num_failed_molecules = 0 #Assuming no molecules fail




#%% Code behind

#This is the method that works with the internal implementation of GFN in DFTB+ but 
#   the second method works with the new version of xtb

def xTB_analyze_molecs(test_set_path: str, method: str, save_location: str) -> None:
    # # # # # # # # # raise NotImplementedError("Implement method xTB_analyze_molecs")
    
    #Prevents overwriting the test set with the new set of molecules which contain the DFTB+/xTB results
    assert(test_set_path != save_location)

    with open(test_set_path, 'rb') as handle:
        dataset = pickle.load(handle)

    print(f"Testing on dataset with total of {len(dataset)} molecules with method {method}")
    run_xTB_dftbp(dataset, method)
    print("xTB DFTB+ information added to molecules")
    
    #Compute the differnce in energy using a fresh reference energy fit
    diffs, result = compute_results_torch(dataset, "Etot", [1,6,7,8], error_metric = "MAE")

    with open(save_location + ".p", 'wb') as handle:
        pickle.dump(dataset, handle)
        pickle.dump(diffs, handle)

    print(f"The resulting difference in MAE is {result}")
    print("The augmented dataset and individual differences have been saved")

    with open(save_location + ".txt", 'w') as handle:
        handle.write(f"The resulting difference in MAE is {result} Ha\n")
        handle.write(f"The resulting difference in MAE is {result * 627.5} kcal/mol\n")

    print("Numerical result saved")

#Method to use wih the new xtb code that is installed via anaconda (the xtb executable exists in the Linux environment)

def write_molecule_to_xyz(mol_dict: Dict, iteration: int) -> None:
    r"""Writes the given molecule dictionary to an input geometry file in the xyz form

    Arguments:
        mol_dict (Dict): The molecule dictionary
        iteration (int): The current iteration of the calculation being run

    Returns:
        None
    """
    filename = os.path.join(os.getcwd(), f"xtb_in_{iteration}.xyz")
    #Assert that the coordinates are formatted as (Natom, 3) for xyz format
    assert(len(mol_dict['atomic_numbers']) == mol_dict['coordinates'].shape[0])
    num_atoms = len(mol_dict['atomic_numbers'])
    with open(filename, 'w') as handle:
        handle.write(f"{num_atoms}\n") #Total number of atoms
        handle.write(f"{mol_dict['name']} {mol_dict['iconfig']}\n") #Comment line
        for i, elem_num in enumerate(mol_dict['atomic_numbers'], 0): #Iterate and write each line
            sym = atom_nums[elem_num]
            x, y, z = mol_dict['coordinates'][i,:]
            if i == len(mol_dict['atomic_numbers']) - 1:
                line = f"{sym} {x} {y} {z}" #No new line for the final line
            else:
                line = f"{sym} {x} {y} {z}\n"
            handle.write(line)

def del_xyz_file(iteration: int) -> None:
    r"""Removes an xyz file after the calculation is completed

    Arguments:
        mol_dict (Dict): The molecule dictionary that was previously used for writing an 
            xyz file
        iteration (int): The iteration (identifier number) to be removed

    Returns:
        None

    Notes:
        The filename for the xyz input is of the form xtb_in_{iteration}.xyz
    """
    targ_file_name = f"xtb_in_{iteration}.xyz"
    full_path_name = os.path.join(os.getcwd(), targ_file_name)
    os.remove(targ_file_name)
    print(f"Successfully removed input file for calculation iteration {iteration + 1}")

def run_xTB_calculation(method: str) -> None:
    r"""Runs the xTB calculation assuming the existence of an .xyz input file

    Arguments:
       method (str): The type of xTB parameterization to use, one of GFN2-xTB or GFN1-xTB

    Returns:
        None
    
    Notes:
        Performs the xTB calculation using subprocess call and scans the scratch directory
            for the version of the .xyz input file. There is also an .inp file contained which
            instructs xtb to write out all physically relevant quantities in an json file. This 
            input file is standardized to be called coord.inp.
    """
    files = os.listdir(os.getcwd())
    xyz_file_name = list(filter(lambda x : x.split('.')[-1] == 'xyz', files))
    #There should only be one xyz file for running, xyz files should be removed after calculations
    assert(len(xyz_file_name) == 1)
    xyz_file_name = xyz_file_name[0]
    if method == 'GFN2-xTB':
        res = call(['xtb', '--input', 'coord.inp', '--gfn', '2', xyz_file_name])
    elif method == 'GFN1-xTB':
        res = call(['xtb', '--input', 'coord.inp', '--gfn', '1', xyz_file_name])

def xTB_true_analysis(test_set_path: str, method: str, save_location: str) -> None:
    r"""Runs an xTB analysis using the anaconda installation of xtb.

    Arguments:
        test_set_path (str): The path to the test set molecules saved as a pickle file
        method (str): The xTB calculation method to use. Should be one of GFN1-xTB or GFN2-xTB
        save_location (str): The name to save things with. This also doubles as the experiment label
            when passing into calc_format_target_reports

    Returns:
        None

    Notes: The entry point for calc_format_target_reports is right after the dataset has been augmented
        with the values from xTB calculations. 
    """
    
    assert(test_set_path != save_location)

    with open(test_set_path, 'rb') as handle:
        dataset = pickle.load(handle)
    print(f"The number of molecules being analyzed is {len(dataset)}")
    #Jump to the scratch directory for performing calculations
    #   this allows the invocation of xtb from the command line
    print(f"Changing current working directory to {scratch_dir}")
    os.chdir(scratch_dir)

    start = time.time()

    for i, mol_dict in enumerate(dataset, 0):
        print(f"Starting calculation for {(mol_dict['name'], mol_dict['iconfig'])}, molecule {i}")
        write_molecule_to_xyz(mol_dict, i) #Write molecule to correct input form
        run_xTB_calculation(method) #Run the calculation
        del_xyz_file(i) #remvoe the file right after to prevent overlaps
        #The output is a json file that can be treated as a dictionary. However, we will only focus on total energy right now
        #   although other properties are available
        resulting_file = 'xtbout.json'
        if os.path.isfile(resulting_file):
            with open(resulting_file, 'r') as handle:
                jdict = json.load(handle)
            #The pzero key is used because all the analysis code was built for DFTB+ previously
            mol_dict['pzero'] = {}
            mol_dict['pzero']['t'] = jdict['total energy']
            mol_dict['pzero']['dipole'] = jdict['dipole']
            mol_dict['pzero']['charges'] = jdict['partial charges']
            mol_dict['pzero']['conv'] = True #Mark convergence
            os.remove("xtbout.json")
        else:
            mol_dict['pzero'] = {}
            mol_dict['pzero']['conv'] = False #Mark nonconvergence


    print("Analysis completed")
    end = time.time()

    with open(save_location + ".p", "wb") as handle:
        pickle.dump(dataset, handle)

    original_len = len(dataset)
    good_mols = [mol for mol in dataset if mol['pzero']['conv']]
    final_len = len(good_mols)

    #Entry point for calc_format_target_reports
    #The destination can just be os.getcwd() because we did a chdir() call previously
    calc_format_target_reports(save_location, os.getcwd(), good_mols, error_metric, tot_ener_targ, pairwise_targs, fit_fresh_ref_ener, 
            allowed_Zs, ref_params = None, dipole_conversion = dipole_conversion, prediction_dipole_unit = prediction_dipole_unit,
            target_dipole_unit = target_dipole_unit, exclusion_threshold = exclusion_threshold, num_failed_molecules = original_len - final_len)
    
    #Do some copy operations as well
    #TODO: Do the copy operations after some simple testing
    src = os.path.join(os.getcwd(), "analysis.txt")
    dst = os.path.join(os.getcwd(), f"{save_location}.txt")
    print("Copying over the analysis file")
    shutil.copy(src, dst)

    print("Analysis finished") 

    #Old analysis code, comment it out
    '''
    diffs, result = compute_results_torch(dataset, "Etot", [1,6,7,8], error_metric = "MAE") #Might need to expand to other elements beyond 1,6,7,8
    with open(save_location + ".p", "wb") as handle:
        pickle.dump(dataset, handle)
        pickle.dump(diffs, handle)

    print(f"The resulting difference in MAE is {result}")
    print("The augmented dataset and individual differences have been saved")
    
    with open(save_location + ".txt", "w") as handle:
        handle.write(f"The resulting difference in MAE is {result} Ha\n")
        handle.write(f"The resulting difference in MAE is {result * 627.5} kcal/mol\n")
        handle.write(f"The elapsed time for {len(dataset)} molecule calculations is {end - start} seconds")

    print("Numerical result saved")
    '''

#%% Main block

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test_set", help = "Name of the test set to use")
    parser.add_argument("xtb_version", help = "Version of the xtb code to use, either 'old' (DFTB+ internal implementation) or 'new' (conda xtb version)")
    parser.add_argument("method", help = "Name of the xTB method to use. One of 'GFN1-xTB' or 'GFN2-xTB")
    parser.add_argument("out_name", help = "Name of the output files to write; the augmented dataset will be saved to out_name.p, the numerical result will be saved to out_name.txt")

    args = parser.parse_args()
    test_set_path, xtb_version, method, save_location = args.test_set, args.xtb_version, args.method, args.out_name
    assert(method in ["GFN1-xTB", "GFN2-xTB"])
    assert(xtb_version == 'new')

    if xtb_version == 'old':
        print("Using DFTB+ implementation of xTB")
        print("Using method xTB_analyze_molecs")
        xTB_analyze_molecs(test_set_path, method, save_location)
    elif xtb_version == 'new':
        print("Using conda installed xTB version")
        print("Using method xTB_true_analysis")
        #The method argument from the parser is ignored for the new version, default is GFN2-xTB
        xTB_true_analysis(test_set_path, method, save_location)




